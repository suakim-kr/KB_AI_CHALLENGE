# ai_agent/router.py
# -*- coding: utf-8 -*-
"""
LangChain 기반 Router (LLM 라우팅 전용)
- rag / t2sql / emailer 세 가지 도구를 일관된 인터페이스로 호출
- 상태(state): 최근 RAG 정책(last_policy), 이메일 발신 설정 등
"""

from __future__ import annotations
import re, json
from typing import Literal, Optional, Dict, Any, Tuple

from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
import pandas as pd

# ---- 내부 모듈 연결 ----
# (패키지 상단에서 실행: python -m ai_agent.router)
from rag import RAGRunner
from t2sql import (
    materialize_tables_sqlite,
    route_to_table_node,
    generate_sql_query_node,
    execute_sql_query_node,
    generate_answer_node,
    GraphState,
    POLICY_CSV as T2SQL_POLICY_CSV,
    PRODUCT_CSV as T2SQL_PRODUCT_CSV,
    SQLITE_DB as T2SQL_SQLITE_DB,
)
from emailer import send_policy_email_oneoff_html, validate_email

ToolName = Literal["rag", "t2sql", "email"]

# ------------------------------------------------------------
# 0) 유틸 (이메일 주소 추출만 유지)
# ------------------------------------------------------------
def _extract_email(text: str) -> Optional[str]:
    m = re.search(r"[^<\s]+@[^>\s]+\.[^\s]+", text or "")
    if m and validate_email(m.group(0)):
        return m.group(0)
    return None

# ------------------------------------------------------------
# 1) LLM 라우터 (단일 소스 오브 트루스)
# ------------------------------------------------------------
_LLM_ROUTER_PROMPT = """다음 사용자 요청을 반드시 하나로 분류해 도구 이름만 출력하세요.

선택지 정의(배타적):
- rag  : 특정 정책 타이틀/사업/공고의 설명, 요약, 세부, 자격요건, 제출서류, 신청방법, 마감일, 링크/문의 등 '상세 파악' 질의
- t2sql: '목록/추천/탐색/검색'과 같이 여러 상품/정책 후보를 조건으로 필터·정렬·비교하는 질의
         (예: 대상, 지역, 금리/한도/보증료율 수치 비교, 이상/이하/초과/미만, TOP N, ~원/만원 등 수치 포함)
- email: 이메일/구독/발송/알림 관련 요청 또는 이메일 주소(@ 포함)가 주된 목적일 때

충돌 규칙:
- "추천/찾아줘//조건/필터/정렬"이 있으면 t2sql 우선.
- 특정 '이 정책/이거/방금 정책'의 자격/서류/마감/설명 이나 '세부/자세히/상세히/구체적' 등은 rag.
- 이메일 발송/구독이 핵심이면 email.

출력 형식:
rag | t2sql | email  (소문자 한 단어만)

[사용자 요청]
{question}

예시:
- "보증료율 1% 이하 운전자금 대출 추천해줘" -> t2sql
- "파주시 2025년 2차 ~ 자세히 알려줘" -> rag
- "이 정책 메일로 보내줘 example@domain.com" -> email
- "청년 대상, 금리 4% 이하, 한도 5천 이상 추천" -> t2sql
- "방금 보여준 정책 자격요건 알려줘" -> rag
"""

def llm_route(question: str, model: Optional[str] = None) -> ToolName:
    llm = ChatOpenAI(model=(model or "gpt-4o-mini"), temperature=0)
    out = (llm.invoke(_LLM_ROUTER_PROMPT.format(question=question)).content or "").strip().lower()
    return "rag" if out not in ("rag", "t2sql", "email") else out

# ------------------------------------------------------------
# 2) T2SQL 앱 컴파일러 (SQLite)
# ------------------------------------------------------------
def _build_t2sql_app() -> tuple[StateGraph, Any]:
    policies = pd.read_csv(T2SQL_POLICY_CSV)
    product  = pd.read_csv(T2SQL_PRODUCT_CSV, encoding="euc-kr")
    db = materialize_tables_sqlite(policies, product, db_path=T2SQL_SQLITE_DB)

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
    return app, db

def _ask_t2sql(app, question: str) -> Dict[str, Any]:
    final_state: Dict[str, Any] = {}
    for event in app.stream({"question": question}):
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
# 3) Router
# ------------------------------------------------------------
class AgentRouter:
    """
    - decide(): LLM만으로 라우팅
    - run(): 실행 + 상태 업데이트(last_policy 등)
    """

    def __init__(
        self,
        llm_router_model: Optional[str] = None,
        rag_runner: Optional[RAGRunner] = None,
        t2sql_app=None,
    ):
        self.llm_router_model = llm_router_model
        self.rag = rag_runner or RAGRunner()
        self.t2sql_app = t2sql_app or _build_t2sql_app()[0]

    def decide(self, question: str) -> ToolName:
        # ✅ 오직 LLM 라우팅만 사용
        return llm_route(question, model=self.llm_router_model)

    def run(self, question: str, state: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        state = dict(state or {})
        tool: ToolName = self.decide(question)

        if tool == "rag":
            out = self.rag.answer(question)
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
            return {"route": "t2sql", "reply": out.get("answer", ""), "raw": out.get("raw", {}), "state": state}

        # tool == "email"
        to_addr = _extract_email(question) or state.get("email_to")
        if not to_addr:
            return {
                "route": "email",
                "error": "no_recipient",
                "reply": "수신자 이메일을 찾지 못했어요. '받는사람 example@domain.com'처럼 이메일 주소를 포함해 주세요.",
                "state": state,
            }

        policy = state.get("last_policy")
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
                state={
                    "sender_email": state.get("sender_email"),
                    "app_password": state.get("app_password"),
                    "smtp_host": state.get("smtp_host"),
                    "smtp_port": state.get("smtp_port"),
                },
            )
            return {"route": "email", "reply": f"✅ {to_addr}로 이메일을 보냈어요.", "policy": policy, "state": state}
        except Exception as e:
            return {"route": "email", "error": "send_failed", "reply": f"이메일 발송에 실패했어요: {e}", "state": state}

# ------------------------------------------------------------
# 4) 로컬 데모
# ------------------------------------------------------------
if __name__ == "__main__":
    router = AgentRouter()
    demo_qs = [
        "2025년 횡성군 청년 농업인 아카데미 교육생 모집 마감일일 언제야?",
        "보증료율 1% 이하 운전자금 대출 추천해줘",
        "이 정책 메일로 보내줘 example@example.com",
    ]
    session_state: Dict[str, Any] = {}
    for q in demo_qs:
        print("\n==========================")
        print("Q:", q)
        res = router.run(q, state=session_state)
        session_state = res.get("state", session_state)
        print("Route:", res.get("route"))
        if "reply" in res: print("Reply:\n", res["reply"])
        if "raw" in res:   print("Raw:", json.dumps(res["raw"], ensure_ascii=False, indent=2))
        if "policy" in res and res["policy"]:
            print("Policy:", json.dumps(res["policy"], ensure_ascii=False, indent=2))
