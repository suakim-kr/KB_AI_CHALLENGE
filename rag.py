# ai_agent/rag.py
# -*- coding: utf-8 -*-
import os
import re
from datetime import datetime
from typing import Dict, Any, List, Tuple, Optional, Union

import numpy as np
import pandas as pd
from dateutil import tz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity
import google.generativeai as genai
from pathlib import Path

from config import GOOGLE_API_KEY, GEN_MODEL, POLICY_CSV, LOCAL_TZ, RAG_TOP_K_DEFAULT
from prompts import RAG_SYSTEM, RAG_DETAIL

from dotenv import load_dotenv
load_dotenv(override=True)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
POLICY_CSV = os.path.join(BASE_DIR, "data", "930_preprocessed.csv")

class RAGRunner:
    """
    개선된 TF-IDF 기반 로컬 RAG + Gemini 답변
    - 향상된 검색 품질 (다양성 확보, 중복 제거)
    - 정확한 멀티턴 상태 관리
    - 개선된 번호 선택 로직 (완전 재구성)
    """

    RESET_PHRASES = ("상담 종료", "종료", "그만", "새로 시작", "초기화")

    INTENT_SLOTS = {
        "deadline": ["deadline"],
        "overview": ["business_overview"],
        "eligibility": ["eligibility_content"],
        "amount": ["amount"],
        "method": ["application_method"],
        "contact": ["contact_info", "phone", "org_dept", "agency"],
        "link": ["link", "url"],
    }

    SYS_PROMPT: str = RAG_SYSTEM
    DETAIL_PROMPT: str = RAG_DETAIL

    def __init__(
        self,
        top_k: int = RAG_TOP_K_DEFAULT,
        initial_state: Optional[Dict[str, Any]] = None,
        csv_path: Optional[str | Path] = None,
        model_name: Optional[str] = None,
    ):
        # LLM 설정
        if not GOOGLE_API_KEY:
            raise RuntimeError("GOOGLE_API_KEY가 설정되어 있지 않습니다 (.env 또는 환경변수).")
        genai.configure(api_key=GOOGLE_API_KEY)
        self.model_name = model_name or GEN_MODEL

        # 데이터 & 인덱스 준비
        self.csv_path: Path = Path(csv_path) if csv_path else Path(POLICY_CSV)
        self.df = self._load_df(self.csv_path)
        self.docs = [self._build_doc_row(r) for _, r in self.df.iterrows()]
        
        # 개선된 TF-IDF 설정 (더 넓은 범위, 다양성 확보)
        self.vectorizer = TfidfVectorizer(
            analyzer="char",
            ngram_range=(1, 4),  # 1~4글자 n-gram으로 확장
            min_df=1,
            max_df=0.95,  # 너무 흔한 단어 제거
            max_features=200_000,  # 특성 수 증가
            sublinear_tf=True,  # TF 정규화
        )
        self.tfidf = self.vectorizer.fit_transform(self.docs)

        # tz
        self.local_tz = tz.gettz(LOCAL_TZ)

        # 멀티턴 상태 - 완전히 재구성된 구조
        self.state = {
            "selected_idx": None,  # 현재 선택된 정책 인덱스
            "candidates": [],  # 검색 후보 리스트 (실제 인덱스)
            "candidate_display_map": {},  # {표시번호: 실제인덱스}
            "pending_selection": False,  # 사용자 선택 대기 중인지
            "last_reset": None,
            "detail_welcome": True,
            "last_query": "",
        }
        if initial_state:
            self.load_state(initial_state)

        self.top_k = top_k

    # ---------- 상태 직렬화 ----------
    def save_state(self) -> Dict[str, Any]:
        """현재 멀티턴 상태를 dict로 반환"""
        return dict(self.state)

    def load_state(self, state: Dict[str, Any]) -> None:
        """외부에서 불러온 상태를 현재 인스턴스에 반영"""
        for k in self.state.keys():
            if k in state:
                self.state[k] = state[k]

    # ---------- 데이터/전처리 ----------
    @staticmethod
    def _norm_date(x: str) -> str:
        x = str(x).strip()
        if not x or x.lower() in ["nan", "none"]:
            return ""
        x = x.replace(".", "-").replace("/", "-")
        try:
            return datetime.fromisoformat(x[:10]).date().isoformat()
        except Exception:
            return x

    def _load_df(self, path: Path) -> pd.DataFrame:
        df = pd.read_csv(path)
        df = df.drop("index", axis=1, errors="ignore")
        if "deadline" in df.columns:
            df["deadline"] = df["deadline"].map(self._norm_date)
        return df

    @staticmethod
    def _build_doc_row(row: pd.Series) -> str:
        link = row["link"] if str(row.get("link", "")).strip() else row.get("url", "")
        return (
            f"[정책명]{row.get('title','')} [지역]{row.get('region','')} "
            f"[기관]{row.get('agency','')} [태그]{row.get('tag','')} [마감]{row.get('deadline','')}\n"
            f"[개요]{row.get('business_overview','')}\n"
            f"[대상]{row.get('eligibility_content','')}\n"
            f"[신청]{row.get('application_method','')}\n"
            f"[연락]{row.get('contact_info','')} {row.get('phone','')}\n"
            f"[링크]{link}"
        )

    # ---------- 개선된 검색 ----------
    def _search(self, query: str, top_n: Optional[int] = None, diversity_weight: float = 0.3) -> Tuple[List[int], np.ndarray]:
        """개선된 검색: 다양성과 관련성을 모두 고려"""
        top_n = top_n or self.top_k
        
        # 쿼리 토큰 추출
        tokens = re.findall(r"[가-힣A-Za-z0-9]+", query)
        
        # 메타데이터 부스팅 (기존 로직 개선)
        booster: List[str] = []
        query_boost_factor = 1.0
        
        if tokens:
            for col in ["title", "agency", "region", "tag"]:
                if col in self.df.columns:
                    # 정확 매칭과 부분 매칭 구분
                    exact_mask = self.df[col].astype(str).str.lower().str.contains(
                        "|".join([f"\\b{t.lower()}\\b" for t in tokens]), 
                        case=False, na=False, regex=True
                    )
                    partial_mask = self.df[col].astype(str).str.contains(
                        "|".join(tokens), case=False, na=False
                    )
                    
                    if exact_mask.any():
                        booster += self.df.loc[exact_mask, col].astype(str).head(2).tolist()
                        query_boost_factor = 2.0
                    elif partial_mask.any():
                        booster += self.df.loc[partial_mask, col].astype(str).head(3).tolist()
                        query_boost_factor = 1.5

        # 부스팅된 쿼리
        q_boost = query + (" " + " ".join(booster) if booster else "")
        
        # TF-IDF 유사도 계산
        q_vec = self.vectorizer.transform([q_boost])
        sims = cosine_similarity(q_vec, self.tfidf).flatten()
        
        # 부스팅 적용
        sims *= query_boost_factor
        
        # 초기 후보 선정 (더 많이)
        candidate_count = min(top_n * 3, len(sims))
        initial_idxs = sims.argsort()[-candidate_count:][::-1]
        
        # 다양성을 고려한 최종 선정
        final_idxs = self._diversify_results(initial_idxs, sims, top_n, diversity_weight)
        
        return final_idxs, sims

    def _diversify_results(self, candidates: List[int], sims: np.ndarray, top_n: int, diversity_weight: float) -> List[int]:
        """결과 다양성 확보"""
        if len(candidates) <= top_n:
            return candidates.tolist()
        
        selected = [candidates[0]]  # 최고 유사도는 반드시 포함
        remaining = candidates[1:].tolist()
        
        while len(selected) < top_n and remaining:
            best_idx = None
            best_score = -1
            
            for idx in remaining:
                # 관련성 점수
                relevance_score = sims[idx]
                
                # 다양성 점수 (선택된 것들과의 차이)
                diversity_score = 0
                for sel_idx in selected:
                    # 지역, 기관, 태그 등의 차이를 점수화
                    diversity_score += self._calculate_diversity(idx, sel_idx)
                
                diversity_score /= len(selected)
                
                # 최종 점수 (관련성 + 다양성)
                final_score = (1 - diversity_weight) * relevance_score + diversity_weight * diversity_score
                
                if final_score > best_score:
                    best_score = final_score
                    best_idx = idx
            
            if best_idx is not None:
                selected.append(best_idx)
                remaining.remove(best_idx)
            else:
                break
        
        return selected

    def _calculate_diversity(self, idx1: int, idx2: int) -> float:
        """두 정책 간의 다양성 점수 계산"""
        row1, row2 = self.df.loc[idx1], self.df.loc[idx2]
        
        diversity = 0
        # 지역이 다르면 +1
        if str(row1.get('region', '')).strip() != str(row2.get('region', '')).strip():
            diversity += 1
        # 기관이 다르면 +1  
        if str(row1.get('agency', '')).strip() != str(row2.get('agency', '')).strip():
            diversity += 1
        # 태그가 다르면 +0.5
        if str(row1.get('tag', '')).strip() != str(row2.get('tag', '')).strip():
            diversity += 0.5
        
        return diversity / 2.5  # 정규화

    def _get_search_results(self, query: str, top_n: Optional[int] = None) -> List[int]:
        """검색 결과만 반환 (간단한 버전)"""
        idxs, _ = self._search(query, top_n=top_n)
        return idxs

    # ---------- 의도/컨텍스트 ----------
    @staticmethod
    def _parse_intent(q: str) -> str:
        x = q.lower()
        if any(k in x for k in ["마감", "마감일", "deadline", "언제", "데드라인", "언제까지"]): return "deadline"
        if any(k in x for k in ["대상", "자격", "요건", "eligibility"]): return "eligibility"
        if any(k in x for k in ["금액", "얼마", "지원금", "amount"]): return "amount"
        if any(k in x for k in ["신청", "접수", "방법", "method", "how"]): return "method"
        if any(k in x for k in ["연락", "문의", "전화", "담당", "contact"]): return "contact"
        if any(k in x for k in ["링크", "원문", "자세히", "어디서", "url"]): return "link"
        if any(k in x for k in ["개요", "요약", "overview"]): return "overview"
        return "overview"

    @staticmethod
    def _build_detail_context(row: pd.Series) -> str:
        link = row["link"] if str(row.get("link", "")).strip() else row.get("url", "")
        return (
            f"[title]{row.get('title','')}\n"
            f"[business_overview]{row.get('business_overview','')}\n"
            f"[eligibility_content]{row.get('eligibility_content','')}\n"
            f"[link]{link}"
        )

    @staticmethod
    def _build_context(row: pd.Series, intent: str) -> str:
        link = row["link"] if str(row.get("link", "")).strip() else row.get("url", "")

        if intent == "deadline":
            main = f"[마감일]{row.get('deadline') or '정보 없음'}"
        elif intent == "eligibility":
            main = f"[지원대상]{row.get('eligibility_content') or '정보 없음'}"
        elif intent == "amount":
            amt = row.get("amount")
            if isinstance(amt, (int, float)) and str(amt) != "nan":
                try:
                    amt = f"{float(amt):,.0f}원"
                except Exception:
                    pass
            main = f"[지원금]{amt or '정보 없음'}"
        elif intent == "method":
            main = f"[신청방법]{row.get('application_method') or '정보 없음'}"
        elif intent == "contact":
            main = f"[연락처]{row.get('contact_info','')} / {row.get('phone','')} ({row.get('org_dept') or row.get('agency')})"
        elif intent == "link":
            main = f"[원문]{link or '정보 없음'}"
        else:
            main = f"[개요]{row.get('business_overview') or '정보 없음'}"

        meta = (
            f"[정책명]{row.get('title','')} [지역]{row.get('region','')} [기관]{row.get('agency','')}\n"
            f"[마감]{row.get('deadline','')}\n"
            f"[링크]{link}"
        )
        return f"{main}\n{meta}"

    # ---------- LLM 호출 ----------
    def _llm_generate(self, prompt: str) -> str:
        resp = genai.GenerativeModel(self.model_name).generate_content(prompt)
        return (getattr(resp, "text", "") or "").strip()

    def _llm_answer_with_template(self, context_text: str, template: str) -> str:
        prompt = f"{template}\n\n[컨텍스트]\n{context_text}"
        return self._llm_generate(prompt)

    def _llm_answer(self, user_query: str, context_text: str, title: str, deadline: str) -> str:
        prompt = (
            f"{self.SYS_PROMPT}\n\n"
            f"[사용자 질문]\n{user_query}\n\n"
            f"[컨텍스트]\n{context_text}\n"
        )
        return self._llm_generate(prompt)

    # ---------- 상태 관리 ----------
    def reset(self):
        """상태 완전 초기화"""
        self.state.update({
            "selected_idx": None,
            "candidates": [],
            "candidate_display_map": {},
            "pending_selection": False,
            "last_reset": datetime.now().isoformat(),
            "detail_welcome": True,
            "last_query": "",
        })

    def _extract_number_selection(self, text: str) -> Optional[int]:
        """번호 선택 추출 - 완전 재구성된 로직"""
        patterns = [
            r"(\d+)번",
            r"번호\s*(\d+)",
            r"\((\d+)\)",
            r"^(\d+)$",
            r"(\d+)\s*선택"
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text.strip())
            if match:
                num = int(match.group(1))
                # candidate_display_map에서 실제 인덱스 찾기
                if num in self.state["candidate_display_map"]:
                    return self.state["candidate_display_map"][num]
        return None

    # ---------- 퍼블릭 엔드포인트 - 완전 재구성 ----------
    def answer(self, user_query: str, force_pick: bool = False, pick_idx: Optional[int] = None) -> Dict[str, Any]:
        """완전히 재구성된 답변 로직"""
        q = (user_query or "").strip()
        self.state["last_query"] = q

        # 종료 명령
        if any(p in q for p in self.RESET_PHRASES):
            self.reset()
            return {"route": "rag", "reply": "상담을 종료했습니다. 추가로 궁금한 정책이 있으면 말씀해주세요."}

        # 강제 선택 (외부에서 특정 정책 선택)
        if force_pick and pick_idx is not None and 0 <= pick_idx < len(self.df):
            self.state["selected_idx"] = pick_idx
            self.state["candidates"] = []
            self.state["candidate_display_map"] = {}
            self.state["pending_selection"] = False
            self.state["detail_welcome"] = True

        # 1단계: 선택 대기 중 - 번호 입력 처리
        if self.state["pending_selection"] and self.state["candidates"]:
            selected_idx = self._extract_number_selection(q)
            
            if selected_idx is not None:
                # 선택 완료
                self.state["selected_idx"] = selected_idx
                self.state["candidates"] = []
                self.state["candidate_display_map"] = {}
                self.state["pending_selection"] = False
                self.state["detail_welcome"] = True
                
                # 선택된 정책의 상세 정보 제공
                row = self.df.loc[selected_idx]
                ctx = self._build_detail_context(row)
                answer = self._llm_answer_with_template(ctx, self.DETAIL_PROMPT)
                self.state["detail_welcome"] = False
                answer += "\n\n추가로 궁금한 점이 있으면 말씀해주세요!"
                answer += "\n(상담 종료를 원하시면 '상담 종료'라고 입력해 주세요.)"
                
                return {
                    "route": "rag",
                    "reply": answer,
                    "policy": {
                        "idx": int(selected_idx),
                        "title": row.get("title", ""),
                        "region": row.get("region", ""),
                        "deadline": row.get("deadline", ""),
                        "agency": row.get("agency", ""),
                        "link": row["link"] if str(row.get("link","")).strip() else row.get("url", ""),
                    },
                }
            else:
                # 잘못된 입력 - 다시 선택 요청
                return {
                    "route": "rag",
                    "reply": (
                        "올바른 번호를 입력해 주세요. 예: '1번', '2', '3번'\n\n"
                        "다시 검색하려면 새로운 키워드를 입력하거나, "
                        "상담을 종료하려면 '상담 종료'라고 입력해 주세요."
                    ),
                }

        # 2단계: 정책이 이미 선택됨 - 세부 질문 처리
        if self.state["selected_idx"] is not None:
            row = self.df.loc[self.state["selected_idx"]]
            
            # 첫 방문시 환영 메시지
            if self.state.get("detail_welcome", True):
                ctx = self._build_detail_context(row)
                answer = self._llm_answer_with_template(ctx, self.DETAIL_PROMPT)
                self.state["detail_welcome"] = False
            else:
                # 세부 질문 처리
                intent = self._parse_intent(q)
                ctx = self._build_context(row, intent)
                answer = self._llm_answer(q, ctx, row.get("title",""), row.get("deadline",""))
                if intent == "deadline":
                    answer += "\n\n이메일 알림을 설정해 드릴까요?"

            answer += "\n\n(상담 종료를 원하시면 '상담 종료'라고 입력해 주세요.)"

            return {
                "route": "rag",
                "reply": answer,
                "policy": {
                    "idx": int(self.state["selected_idx"]),
                    "title": row.get("title", ""),
                    "region": row.get("region", ""),
                    "deadline": row.get("deadline", ""),
                    "agency": row.get("agency", ""),
                    "link": row["link"] if str(row.get("link","")).strip() else row.get("url", ""),
                },
            }

        # 3단계: 새로운 검색
        search_results = self._get_search_results(q, top_n=self.top_k)
        
        if not search_results:
            return {
                "route": "rag",
                "reply": "관련 정책을 찾지 못했어요. 정책명을 좀 더 구체적으로 말씀해 주세요.",
                "candidates": [],
            }

        # 검색 결과가 1개인 경우 - 바로 선택
        if len(search_results) == 1:
            self.state["selected_idx"] = search_results[0]
            self.state["candidates"] = []
            self.state["candidate_display_map"] = {}
            self.state["pending_selection"] = False
            self.state["detail_welcome"] = True
            
            # 바로 상세 정보 제공하지 않고 다음 답변에서 처리하도록
            return self.answer("", force_pick=False, pick_idx=None)

        # 검색 결과가 여러개인 경우 - 선택지 제공
        self.state["candidates"] = search_results
        self.state["candidate_display_map"] = {i+1: idx for i, idx in enumerate(search_results)}
        self.state["pending_selection"] = True
        self.state["selected_idx"] = None  # 아직 선택되지 않음
        
        # 선택지 표시
        preview_lines = []
        for display_num, real_idx in self.state["candidate_display_map"].items():
            row = self.df.loc[real_idx]
            marker = "🏆 " if display_num == 1 else "   "  # 첫 번째는 가장 관련성 높음
            preview_lines.append(
                f"{marker}{display_num}번: {row.get('title', '')} (지역: {row.get('region', '')}, 마감: {row.get('deadline', '정보없음')})"
            )
        
        preview = "\n".join(preview_lines)
        return {
            "route": "rag",
            "reply": (
                "검색된 정책들입니다. 자세히 알고 싶은 정책의 번호를 입력해 주세요:\n\n"
                f"{preview}\n\n"
                "※ 1번이 가장 관련성이 높습니다.\n"
                "※ 상담 종료를 원하시면 '상담 종료'라고 입력해 주세요."
            ),
            "candidates": search_results,
        }

    # 검색 결과만 보고 싶을 때
    def search(self, query: str, k: Optional[int] = None) -> List[Dict[str, Any]]:
        """검색 결과만 반환"""
        idxs, sims = self._search(query, top_n=k or self.top_k)
        out = []
        for i in idxs:
            out.append({
                "idx": int(i),
                "score": float(sims[i]),
                "title": self.df.loc[i, "title"],
                "region": self.df.loc[i, "region"],
                "deadline": self.df.loc[i, "deadline"],
                "agency": self.df.loc[i, "agency"],
            })
        return out