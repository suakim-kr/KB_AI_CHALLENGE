# ai_agent/graph_agent.py
# -*- coding: utf-8 -*-
"""
상위 라우터까지 LangGraph로 감싼 얇은 래퍼 + 세션 자동 저장/복원
- 기존 router.AgentRouter(룰+LLM 하이브리드)를 노드로 호출
- 프롬프트는 기존 모듈(router/rag/t2sql/emailer)에 있는 그대로 사용 (수정 없음)
- session_id(옵션)로 상태를 자동 관리. 미지정 시 'default' 세션을 사용.
"""

from __future__ import annotations
from typing import TypedDict, Optional, Dict, Any, Literal
from langgraph.graph import StateGraph, END
from .router import AgentRouter

Tool = Literal["rag", "t2sql", "email"]

class AgentState(TypedDict, total=False):
    question: str
    route: Tool
    reply: str
    raw: Dict[str, Any]
    policy: Dict[str, Any]
    error: str
    session: Dict[str, Any]
    session_id: str  # 추가: 세션 자동 관리용 식별자

# ---- In-memory 세션 스토어 ----
class _SessionStore:
    def __init__(self):
        self._mem: Dict[str, Dict[str, Any]] = {}

    def get(self, sid: str) -> Dict[str, Any]:
        return self._mem.setdefault(sid, {})

    def set(self, sid: str, session: Dict[str, Any]) -> None:
        self._mem[sid] = dict(session or {})

    def reset(self, sid: str) -> None:
        self._mem.pop(sid, None)

_SESSION_STORE = _SessionStore()

# 단일 라우터 인스턴스 재사용
router_singleton = AgentRouter()

def _sid(s: AgentState) -> str:
    return (s.get("session_id") or "default").strip() or "default"

def _load_session_into_state(state: AgentState) -> None:
    # state.session이 비어있으면 store에서 로드
    if not state.get("session"):
        state["session"] = _SESSION_STORE.get(_sid(state))

def _persist_session_from_state(state: AgentState) -> None:
    if "session" in state:
        _SESSION_STORE.set(_sid(state), state["session"] or {})

def decide_node(state: AgentState) -> AgentState:
    _load_session_into_state(state)
    q = state.get("question", "")
    chosen = router_singleton.decide(q)
    state["route"] = chosen
    _persist_session_from_state(state)
    return state

def rag_node(state: AgentState) -> AgentState:
    _load_session_into_state(state)
    q = state.get("question", "")
    sess = state.get("session", {}) or {}
    res = router_singleton.run(q, sess)

    state["reply"] = res.get("reply", "")
    if res.get("policy"):
        state["policy"] = res["policy"]
    state["session"] = res.get("state", sess)
    _persist_session_from_state(state)
    return state

def t2sql_node(state: AgentState) -> AgentState:
    _load_session_into_state(state)
    q = state.get("question", "")
    sess = state.get("session", {}) or {}
    res = router_singleton.run(q, sess)

    state["reply"] = res.get("reply", "")
    state["raw"] = res.get("raw", {})
    state["session"] = res.get("state", sess)
    _persist_session_from_state(state)
    return state

def email_node(state: AgentState) -> AgentState:
    _load_session_into_state(state)
    q = state.get("question", "")
    sess = state.get("session", {}) or {}
    res = router_singleton.run(q, sess)

    state["reply"] = res.get("reply", "")
    if res.get("error"):
        state["error"] = res["error"]
    if res.get("policy"):
        state["policy"] = res["policy"]
    state["session"] = res.get("state", sess)
    _persist_session_from_state(state)
    return state

# ========= 그래프 구성 =========
workflow = StateGraph(AgentState)
workflow.add_node("decide", decide_node)
workflow.add_node("rag", rag_node)
workflow.add_node("t2sql", t2sql_node)
workflow.add_node("email", email_node)
workflow.set_entry_point("decide")

def _route_selector(s: AgentState) -> Tool:
    r = s.get("route", "rag")
    return r if r in ("rag", "t2sql", "email") else "rag"

workflow.add_conditional_edges(
    "decide",
    _route_selector,
    {"rag": "rag", "t2sql": "t2sql", "email": "email"},
)

workflow.add_edge("rag", END)
workflow.add_edge("t2sql", END)
workflow.add_edge("email", END)

graph_app = workflow.compile()

# ========= 편의 함수 (선택) =========
def reset_session(session_id: str = "default") -> None:
    """세션 상태 초기화"""
    _SESSION_STORE.reset(session_id)

def set_email_sender(
    sender_email: str,
    app_password: str,
    session_id: str = "default",
    smtp_host: Optional[str] = None,
    smtp_port: Optional[int] = None,
) -> None:
    """발신자 설정을 세션에 저장"""
    sess = _SESSION_STORE.get(session_id)
    sess.update({
        "sender_email": sender_email,
        "app_password": app_password,
        **({"smtp_host": smtp_host} if smtp_host else {}),
        **({"smtp_port": smtp_port} if smtp_port else {}),
    })
    _SESSION_STORE.set(session_id, sess)

# ========= 로컬 테스트 =========
if __name__ == "__main__":
    sid = "demo"
    reset_session(sid)
    set_email_sender("you@gmail.com", "<앱비번>", session_id=sid)

    demo_inputs = [
        {"question": "경북 창업지원사업 마감 언제야?", "session_id": sid},
        {"question": "서울 청년 창업 지원금 1000만원 이상", "session_id": sid},
        {"question": "보증료율 1% 이하 운전자금 대출 알려줘", "session_id": sid},
        {"question": "이 정책 메일로 보내줘 example@example.com", "session_id": sid},
    ]

    final: AgentState = {}
    for s in demo_inputs:
        print("\n==========================")
        print("Q:", s["question"])
        for event in graph_app.stream(s):
            for node, payload in event.items():
                print(f"[{node}] -> keys: {list(payload.keys())}")
                final.update(payload)
        print("Route:", final.get("route"))
        print("Reply:", final.get("reply"))
        if final.get("error"):
            print("Error:", final["error"])
        if final.get("policy"):
            print("Policy title:", final["policy"].get("title"))