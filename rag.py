# ai_agent/rag.py
# -*- coding: utf-8 -*-
import re
from datetime import datetime
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import pandas as pd
from dateutil import tz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import google.generativeai as genai
from pathlib import Path

from .config import GOOGLE_API_KEY, GEN_MODEL, POLICY_CSV, LOCAL_TZ, RAG_TOP_K_DEFAULT
from .prompts import RAG_SYSTEM, RAG_DETAIL

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
POLICY_CSV = os.path.join(BASE_PATH, "data", "930_preprocessed.csv")

class RAGRunner:
    """
    TF-IDF 기반 로컬 RAG (CSV → 인덱스) + Gemini 답변.
    - 프롬프트 외부 파일 사용 (prompts/rag_prompt.txt 등)
    - Path 기반 경로 처리 (OS 독립성)
    - 멀티턴 상태 내장 (save_state/load_state로 외부 저장 가능)
    - 라우팅 통합 시 `answer()`와 `search()` 인터페이스 그대로 사용
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

    # 1) 프롬프트 외부 파일 로딩
    SYS_PROMPT: str = load_rag_prompt()
    DETAIL_PROMPT: str = load_rag_detail_prompt()

    def __init__(
        self,
        top_k: int = RAG_TOP_K_DEFAULT,
        initial_state: Optional[Dict[str, Any]] = None,  # 3) 멀티턴 상태 외부 주입 가능
        csv_path: Optional[str | Path] = None,           # 2) Path 기반 경로
        model_name: Optional[str] = None,
    ):
        # LLM 설정
        if not GOOGLE_API_KEY:
            raise RuntimeError("GOOGLE_API_KEY가 설정되어 있지 않습니다 (.env 또는 환경변수).")
        genai.configure(api_key=GOOGLE_API_KEY)
        self.model_name = model_name or GEN_MODEL

        # 데이터 & 인덱스 준비 (2: Path 통일)
        self.csv_path: Path = Path(csv_path) if csv_path else Path(POLICY_CSV)
        self.df = self._load_df(self.csv_path)
        self.docs = [self._build_doc_row(r) for _, r in self.df.iterrows()]
        self.vectorizer = TfidfVectorizer(
            analyzer="char",
            ngram_range=(2, 3),
            min_df=1,
            max_features=100_000,
        )
        self.tfidf = self.vectorizer.fit_transform(self.docs)

        # tz
        self.local_tz = tz.gettz(LOCAL_TZ)

        # 3) 멀티턴 상태
        self.state = {
            "selected_idx": None,
            "candidates": [],
            "pending_action": None,
            "last_reset": None,
            "detail_welcome": True,
        }
        if initial_state:
            # 외부 세션에 저장된 상태를 주입하고 싶을 때 사용
            self.load_state(initial_state)

        self.top_k = top_k

    # ---------- 상태 직렬화 (3) ----------
    def save_state(self) -> Dict[str, Any]:
        """현재 멀티턴 상태를 dict로 반환 (Streamlit session_state 등에 저장용)."""
        return dict(self.state)

    def load_state(self, state: Dict[str, Any]) -> None:
        """외부에서 불러온 상태를 현재 인스턴스에 반영."""
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

    # ---------- 검색 ----------
    def _search(self, query: str, top_n: Optional[int] = None) -> Tuple[List[int], np.ndarray]:
        top_n = top_n or self.top_k
        tokens = re.findall(r"[가-힣A-Za-z0-9]+", query)
        booster: List[str] = []
        if tokens:
            for col in ["title", "agency", "region", "tag"]:
                if col in self.df.columns:
                    mask = self.df[col].astype(str).str.contains("|".join(tokens), case=False, na=False)
                    if mask.any():
                        booster += self.df.loc[mask, col].astype(str).head(3).tolist()
        q_boost = query + (" " + " ".join(booster) if booster else "")

        q_vec = self.vectorizer.transform([q_boost])
        sims = linear_kernel(q_vec, self.tfidf).flatten()
        idxs = sims.argsort()[-top_n:][::-1].tolist()
        return idxs, sims

    def _pick_one_policy(self, query: str, top_n: Optional[int] = None):
        idxs, sims = self._search(query, top_n=top_n)
        if not idxs:
            return None, []

        q_norm = re.sub(r"[\s\[\]\(\)]+", "", query.lower())
        # 정확 일치
        for i in idxs:
            t_norm = re.sub(r"[\s\[\]\(\)]+", "", str(self.df.loc[i, "title"]).lower())
            if t_norm and t_norm == q_norm:
                return i, []
        # 부분 일치
        for i in idxs:
            ti = str(self.df.loc[i, "title"]).lower()
            if ti and (ti in query.lower() or query.lower() in ti):
                return i, []
        # 유사도 최상 + 후보
        chosen = idxs[0]
        cands = [i for i in idxs if i != chosen][:3]
        return chosen, cands

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
        self.state.update({
            "selected_idx": None,
            "candidates": [],
            "pending_action": None,
            "last_reset": datetime.now().isoformat(),
            "detail_welcome": True,
        })

    @staticmethod
    def _extract_first_index(text: str) -> Optional[int]:
        m = re.search(r"(\d+)", text)
        return int(m.group(1)) if m else None

    # ---------- 퍼블릭 엔드포인트 ----------
    def answer(self, user_query: str, force_pick: bool = False, pick_idx: Optional[int] = None) -> Dict[str, Any]:
        q = (user_query or "").strip()

        # 종료 명령
        if any(p in q for p in self.RESET_PHRASES):
            self.reset()
            return {"route": "rag", "reply": "상담을 종료했습니다. 추가로 궁금한 정책이 있으면 말씀해주세요."}

        # 번호 선택 처리
        idx_from_text = self._extract_first_index(q)
        if self.state["candidates"] and idx_from_text is not None:
            if idx_from_text in [self.state["selected_idx"]] + self.state["candidates"]:
                self.state["selected_idx"] = idx_from_text
                self.state["candidates"] = []
                self.state["pending_action"] = None

        if force_pick and pick_idx is not None and 0 <= pick_idx < len(self.df):
            self.state["selected_idx"] = pick_idx
            self.state["candidates"] = []
            self.state["pending_action"] = None

        chosen = self.state["selected_idx"]

        # 미확정 → 검색
        if chosen is None:
            chosen, cands = self._pick_one_policy(q, top_n=self.top_k)
            if chosen is None:
                return {
                    "route": "rag",
                    "reply": "관련 정책을 찾지 못했어요. 정책명을 좀 더 구체적으로 말씀해 주세요.",
                    "candidates": [],
                }

            self.state["selected_idx"] = chosen
            self.state["candidates"] = cands or []

            if self.state["candidates"]:
                preview = "\n".join([
                    f"- ({i}) {self.df.loc[i, 'title']} / {self.df.loc[i, 'region']} / 마감 {self.df.loc[i, 'deadline']}"
                    for i in [chosen] + self.state["candidates"]
                ])
                return {
                    "route": "rag",
                    "reply": (
                        "찾으시는 정책이 아래에 있나요? 자세히 알고 싶은 정책 번호를 입력해 주세요.\n"
                        f"{preview}\n\n"
                        "※ 상담 종료를 원하시면 '상담 종료'라고 입력해 주세요."
                    ),
                    "candidates": [chosen] + self.state["candidates"],
                }

        # 확정됨 → DETAIL 또는 슬롯 응답
        row = self.df.loc[self.state["selected_idx"]]

        if self.state.get("detail_welcome", True):
            ctx = self._build_detail_context(row)
            answer = self._llm_answer_with_template(ctx, self.DETAIL_PROMPT)
            self.state["detail_welcome"] = False
        else:
            intent = self._parse_intent(q)
            ctx = self._build_context(row, intent)
            answer = self._llm_answer(q, ctx, row.get("title",""), row.get("deadline",""))
            if intent == "deadline":
                answer += "\n이메일 알림을 설정해 드릴까요?"

        answer += "\n(상담 종료를 원하시면 '상담 종료'라고 입력해 주세요.)"

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

    # 검색 결과만 보고 싶을 때
    def search(self, query: str, k: Optional[int] = None) -> List[Dict[str, Any]]:
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