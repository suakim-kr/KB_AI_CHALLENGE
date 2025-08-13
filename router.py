# ai_agent/router.py
# -*- coding: utf-8 -*-
"""
LangChain 기반 Router
- 규칙 기반 하드 라우팅 + LLM 백업 라우팅
- rag / t2sql / emailer 세 가지 도구를 일관된 인터페이스로 호출
- 간단한 세션 상태(state) 관리: 마지막 RAG 정책 선택 등
"""

from __future__ import annotations
import os, re, json
from typing import Literal, Optional, Dict, Any

from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

# ---- 내부 모듈 연결 ----
# 상대 경로/패키지 구조에 맞게 import 경로를 조정하세요.
from .rag import RAGRunner
from .t2sql import (
    materialize_tables_simple,
    route_to_table_node,
    generate_sql_query_node,
    execute_sql_query_node,
    generate_answer_node,
    GraphState,
    POLICY_CSV as T2SQL_POLICY_CSV,
    PRODUCT_CSV as T2SQL_PRODUCT_CSV,
    DB_URI as T2SQL_DB_URI,
)
from .emailer import send_policy_email_oneoff_html, validate_email

from langgraph.graph import StateGraph, END
import pandas as pd


ToolName = Literal["rag", "t2sql", "email"]

# ------------------------------------------------------------
# 0) 간단한 유틸
# ------------------------------------------------------------
_EMAIL_WORDS = ("메일", "email", "이메일", "보내", "발송", "알림", "구독", "notify", "notification", "subscribe")
_POLICY_WORDS = ("정책", "지원금", "보조금", "사업", "공고", "지원사업", "사업공고")
_PRODUCT_WORDS = ("보증", "대출", "융자", "보증료", "보증료율", "금리", "금융상품", "한도", "담보", "보증기간")
_STRUCT_FILTER_HINTS = ("이상", "이하", "초과", "미만", "%", "원", "만원", "억원")

_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")

def _has_any(text: str, words: tuple[str, ...]) -> bool:
    q = re.sub(r"\s+", "", text or "").lower()
    return any(w in q for w in [w.lower() for w in words])

def _looks_structured(text: str) -> bool:
    """숫자/단위/비교표현이 섞여 있으면 Text2SQL에 유리하다고 판단"""
    if re.search(r"\d", text or ""):
        return True
    return _has_any(text, _STRUCT_FILTER_HINTS)

def _extract_email(text: str) -> Optional[str]:
    # 문장 속 이메일 추출
    m = re.search(r"[^<\s]+@[^>\s]+\.[^\s]+", text or "")
    if m and validate_email(m.group(0)):
        return m.group(0)
    return None

# ------------------------------------------------------------
# 1) LLM 백업 라우터
# ------------------------------------------------------------
_LLM_ROUTER_PROMPT = """다음 사용자 요청을 분류하여 사용할 도구를 하나만 고르세요.
선택지는 다음 중 하나입니다:
- rag: 상품이나 정책에 대해서 설명/요약/세부내용/지원 자격 및 대상/마감일 등 컨텍스트 질의응답
- t2sql: 조건 필터링(이상/이하/숫자/%)으로 최적의 정책/상품을 조회/추천
- email: 관련 정보를 메일 발송/알림/구독/이메일 관련 요청

출력은 소문자 키워드(rag|t2sql|email)만 한 단어로 답하세요.

[사용자 요청]
{question}
"""

def llm_route(question: str, model: Optional[str] = None) -> ToolName:
    llm = ChatOpenAI(model=(model or "gpt-4o-mini"), temperature=0)
    chain = (lambda x: _LLM_ROUTER_PROMPT.format(question=x)) | llm | StrOutputParser()
    out = (chain.invoke(question) or "").strip().lower()
    if "email" in out:
        return "email"
    if "t2sql" in out or "sql" in out:
        return "t2sql"
    return "rag"

# ------------------------------------------------------------
# 2) 하드 라우팅
# ------------------------------------------------------------
def hard_route(question: str) -> Optional[ToolName]:
    q = question or ""
    # 이메일 관련이 명시되면 최우선
    if _has_any(q, _EMAIL_WORDS) or _extract_email(q):
        return "email"
    # 정책/상품 키워드 + 구조적 필터 힌트 → t2sql
    pol_hit = _has_any(q, _POLICY_WORDS)
    prod_hit = _has_any(q, _PRODUCT_WORDS)
    if (pol_hit or prod_hit) and _looks_structured(q):
        return "t2sql"
    # 설명/요약/상세/마감 등 자연어 QA → rag
    if pol_hit or prod_hit:
        return "rag"
    # 기본값은 None (LLM 백업)
    return None

# ------------------------------------------------------------
# 3) T2SQL 앱 컴파일러 (t2sql.main의 그래프를 재구성)
# ------------------------------------------------------------
def _build_t2sql_app(db_uri: str = T2SQL_DB_URI) -> tuple[StateGraph, Any]:
    # CSV 로드 (인코딩은 원본 모듈 설정과 동일하게)
    policies = pd.read_csv(T2SQL_POLICY_CSV)
    product  = pd.read_csv(T2SQL_PRODUCT_CSV, encoding="euc-kr")
    # DuckDB 테이블 머터리얼라이즈
    db = materialize_tables_simple(policies, product, db_uri)

    # LangGraph 워크플로우 구성
    workflow = StateGraph(GraphState)
    workflow.add_node("route_to_table", route_to_table_node)
    workflow.add_node("generate_sql", generate_sql_query_node)
    workflow.add_node("execute_sql", lambda s: execute_sql_query_node(s, db))
    workflow.add_node("generate_answer", generate_answer_node)

    workflow.set_entry_point("route_to_table")
    workflow.add_edge("route_to_table", "generate_sql")
    workflow.add_edge("generate_sql", "execute_sql")
    workflow.add_edge("execute_sql", "generate_answer")
    workflow.add_edge("generate_answer", END)
    app = workflow.compile()
    return app, db  # db는 필요 시 추가 활용

def _ask_t2sql(app, question: str) -> Dict[str, Any]:
    final_state: Dict[str, Any] = {}
    for event in app.stream({"question": question}):
        # 마지막 노드의 상태에 answer가 들어있음
        for _node, payload in event.items():
            final_state.update(payload)
    return {
        "route": "t2sql",
        "answer": final_state.get("answer", ""),
        "raw": {
            "target_table": final_state.get("target_table"),
            "sql_query": final_state.get("sql_query"),
            "sql_queries": final_state.get("sql_queries", {}),
            "db_result": final_state.get("db_result"),
            "db_results": final_state.get("db_results", {}),
            "no_result": final_state.get("no_result", False),
            "error": final_state.get("error", ""),
        },
    }

# ------------------------------------------------------------
# 4) Router 클래스
# ------------------------------------------------------------
class AgentRouter:
    """
    세 도구를 감싸는 일관된 엔진.
    - decide(): 어떤 도구를 쓸지 결정
    - run(): 결정 + 실행 결과 반환
    - 상태(state)에는 최소한으로 다음 키를 사용:
        - "rag": RAGRunner 인스턴스 내부 상태 사용 + 마지막 선택 정책 정보
        - "sender_email", "app_password", "smtp_host", "smtp_port" (이메일 override 용)
        - "last_policy" : 최근 RAG 응답의 policy 딕셔너리 (이메일로 보내기 쉽게)
    """

    def __init__(
        self,
        llm_router_model: Optional[str] = None,
        rag_runner: Optional[RAGRunner] = None,
        t2sql_app=None,
    ):
        self.llm_router_model = llm_router_model
        self.rag = rag_runner or RAGRunner()
        # T2SQL 그래프 준비
        if t2sql_app is None:
            self.t2sql_app, _db = _build_t2sql_app()
        else:
            self.t2sql_app = t2sql_app

    # ---- 의사결정 ----
    def decide(self, question: str) -> ToolName:
        # 1) 하드 라우팅
        choice = hard_route(question)
        if choice:
            return choice
        # 2) 백업: LLM
        return llm_route(question, model=self.llm_router_model)

    # ---- 실행기 ----
    def run(self, question: str, state: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        question: 사용자 입력
        state: 세션 상태 dict (선택)
        returns: {"route": "rag|t2sql|email", "reply": "...", ...}
        """
        state = dict(state or {})
        tool: ToolName = self.decide(question)

        if tool == "rag":
            out = self.rag.answer(question)
            # 마지막 정책을 상태에 저장 (이메일로 바로 보내기 좋게)
            if isinstance(out, dict) and out.get("policy"):
                state["last_policy"] = out["policy"]
            return {
                "route": "rag",
                "reply": out.get("reply", ""),
                "policy": out.get("policy"),
                "candidates": out.get("candidates", []),
                "state": state,
            }

        if tool == "t2sql":
            out = _ask_t2sql(self.t2sql_app, question)
            # Text2SQL은 여러 건을 뽑을 수 있어, 이메일 원클릭을 위해서는
            # 사용자가 특정 정책을 골라주면 좋지만, 여기서는 답변만 반환
            return {
                "route": "t2sql",
                "reply": out.get("answer", ""),
                "raw": out.get("raw", {}),
                "state": state,
            }

        # tool == "email"
        # 이메일은 두 케이스:
        #  A) 질문에 수신자 이메일이 함께 옴 → 그 주소로 발송
        #  B) 최근 RAG에서 본 'last_policy'를 사용해 발송 (수신자는 질문에서 추출 or state에서 받기)
        to_addr = _extract_email(question) or state.get("email_to")
        if not to_addr:
            return {
                "route": "email",
                "error": "no_recipient",
                "reply": "수신자 이메일을 찾지 못했어요. '받는사람 example@domain.com'처럼 이메일 주소를 포함해 주세요.",
                "state": state,
            }

        # 어떤 정책을 메일로 보낼지: 최근 RAG의 last_policy 우선
        policy = state.get("last_policy")

        # 안전장치: 최소 제목/링크라도 있어야 보낼 가치가 있음
        if not policy or not (policy.get("title") or policy.get("link") or policy.get("url")):
            return {
                "route": "email",
                "error": "no_policy_context",
                "reply": "보낼 정책 컨텍스트가 없어요. 먼저 관심 정책을 찾아본 뒤 '이메일로 보내줘'라고 해주세요.",
                "state": state,
            }

        try:
            send_policy_email_oneoff_html(
                to_addr=to_addr,
                policy=policy,
                # state에 들어있는 발신 설정으로 override 가능
                state={
                    "sender_email": state.get("sender_email"),
                    "app_password": state.get("app_password"),
                    "smtp_host": state.get("smtp_host"),
                    "smtp_port": state.get("smtp_port"),
                },
            )
            return {
                "route": "email",
                "reply": f"✅ {to_addr}로 이메일을 보냈어요.",
                "policy": policy,
                "state": state,
            }
        except Exception as e:
            return {
                "route": "email",
                "error": "send_failed",
                "reply": f"이메일 발송에 실패했어요: {e}",
                "state": state,
            }


# ------------------------------------------------------------
# 5) 간단 데모 (옵션)
# ------------------------------------------------------------
if __name__ == "__main__":
    """
    빠른 수동 테스트:
    - 환경변수: OPENAI_API_KEY / (이메일 발송 시 SENDER_EMAIL, APP_PASSWORD 등)
    - RAG CSV와 T2SQL CSV 경로는 각 모듈 기본값 사용
    """
    router = AgentRouter()

    demo_qs = [
        "서울 청년 창업 지원금 1000만원 이상",
        "보증료율 1% 이하인 운전자금 대출 알려줘",
        "경북 창업지원사업 마감 언제야?",
        "이 정책 메일로 보내줘 chae@example.com",
    ]

    session_state: Dict[str, Any] = {}

    for q in demo_qs:
        print("\n==========================")
        print("Q:", q)
        res = router.run(q, state=session_state)
        session_state = res.get("state", session_state)

        print("Route:", res.get("route"))
        if "reply" in res:
            print("Reply:\n", res["reply"])
        if "raw" in res:
            print("Raw:", json.dumps(res["raw"], ensure_ascii=False, indent=2))
        if "policy" in res and res["policy"]:
            print("Policy:", json.dumps(res["policy"], ensure_ascii=False, indent=2))