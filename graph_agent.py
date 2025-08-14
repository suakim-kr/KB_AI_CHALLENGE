# ai_agent/graph_agent.py
# -*- coding: utf-8 -*-
"""
상위 라우터까지 LangGraph로 감싼 얇은 래퍼 + 세션 자동 저장/복원
+ T2SQL → RAG 연계 지원 + 멀티스텝 워크플로우
"""

from __future__ import annotations
import os, re
from typing import TypedDict, Optional, Dict, Any, Literal
from langgraph.graph import StateGraph, END
from router import AgentRouter  # 패키지 실행 시 from .router import AgentRouter 권장

from dotenv import load_dotenv
load_dotenv(override=True)

Tool = Literal["rag", "t2sql", "email", "followup"]

class AgentState(TypedDict, total=False):
    question: str
    original_question: str  # 최초 질문 보존
    route: Tool
    reply: str
    raw: Dict[str, Any]
    policy: Dict[str, Any]
    error: str
    session: Dict[str, Any]
    session_id: str
    email_to: str
    # 연계 관련 추가 필드
    followup_needed: bool    # 추가 대화 필요 여부
    t2sql_result: str       # T2SQL 결과 보존
    context_history: list   # 대화 컨텍스트 히스토리
    step_count: int         # 현재 스텝 수 (무한루프 방지)

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

# 단일 라우터 인스턴스 재사용 (RAG 내부 멀티턴 상태도 인스턴스 내에서 유지됨)
router_singleton = AgentRouter()

def _sid(s: AgentState) -> str:
    return (s.get("session_id") or "default").strip() or "default"

def _load_session_into_state(state: AgentState) -> None:
    if not state.get("session"):
        state["session"] = _SESSION_STORE.get(_sid(state))

def _persist_session_from_state(state: AgentState) -> None:
    if "session" in state:
        _SESSION_STORE.set(_sid(state), state["session"] or {})

# ---- 간단 이메일 추출 (검증은 router/emailer에서 진행) ----
_EMAIL_RE = re.compile(r"[^<\s]+@[^>\s]+\.[^\s]+")

def _extract_email(text: str) -> Optional[str]:
    if not text:
        return None
    m = _EMAIL_RE.search(text)
    return m.group(0) if m else None

# ---- 추가 대화 필요 여부 판단 ----
def _needs_followup(reply: str, route: Tool) -> bool:
    """결과를 보고 추가 대화가 필요한지 판단"""
    if not reply:
        return False
        
    # T2SQL 결과에서 추가 질문을 유도하는 키워드
    followup_indicators = [
        "더 자세히", "추가로", "구체적으로", "자세한 내용", "더 알고 싶다면",
        "이 중에서", "선택해", "어떤 것이", "더 궁금한", "관련 정책"
    ]
    
    return (route == "t2sql" and 
            any(indicator in reply for indicator in followup_indicators))

# ========= LangGraph 노드들 =========

def preprocess_node(state: AgentState) -> AgentState:
    """전처리 및 초기 설정"""
    _load_session_into_state(state)
    q = state.get("question", "") or ""
    sess = state.get("session", {}) or {}
    
    # 초기 질문 보존 (최초 진입시만)
    if not state.get("original_question"):
        state["original_question"] = q
    
    # 컨텍스트 히스토리 초기화
    if not state.get("context_history"):
        state["context_history"] = []
    
    # 스텝 카운터 (디버깅용)
    state["step_count"] = state.get("step_count", 0) + 1
    
    # 컨텍스트 히스토리 슬라이딩 윈도우 (최근 5개만 유지)
    context_history = state.get("context_history", [])
    if len(context_history) > 5:
        state["context_history"] = context_history[-5:]  # 최신 5개만 유지

    # 이메일 자동 추출
    found_email = _extract_email(q)
    if found_email:
        sess["email_to"] = found_email

    # 상위 state의 policy ↔ session['last_policy'] 동기화
    if "policy" in state and state["policy"]:
        sess.setdefault("last_policy", state["policy"])

    state["session"] = sess
    state["email_to"] = sess.get("email_to", "")
    _persist_session_from_state(state)
    return state

def decide_node(state: AgentState) -> AgentState:
    """라우팅 결정"""
    _load_session_into_state(state)
    q = state.get("question", "") or ""
    sess = state.get("session", {}) or {}

    # 1) 기본 LLM 라우팅
    chosen = router_singleton.decide(q)

    # 2) 멀티턴 보강 라우팅 오버라이드
    if sess.get("pending_email"):
        if sess.get("email_to") or _extract_email(q):
            chosen = "email"

    if chosen != "email" and (sess.get("last_policy") and (sess.get("email_to") or _extract_email(q))):
        chosen = "email"

    state["route"] = chosen
    _persist_session_from_state(state)
    return state

def rag_node(state: AgentState) -> AgentState:
    """RAG 처리"""
    _load_session_into_state(state)
    q = state.get("question", "")
    sess = state.get("session", {}) or {}

    # 컨텍스트 히스토리에 T2SQL 결과가 있으면 질문에 추가 (최근 3개만 사용)
    context_history = state.get("context_history", [])
    enhanced_question = q
    
    if context_history:
        # 이전 T2SQL 결과를 컨텍스트로 활용 (최근 3개만 사용해서 프롬프트 길이 제한)
        recent_context = context_history[-3:]  # 최근 3개만
        context_str = "\n".join([f"[이전 검색 결과] {item}" for item in recent_context])
        enhanced_question = f"{context_str}\n\n[현재 질문] {q}"

    res = router_singleton.run(enhanced_question, sess)

    # 일반 출력
    reply = res.get("reply", "")
    state["reply"] = reply
    if res.get("policy"):
        state["policy"] = res["policy"]
        sess["last_policy"] = res["policy"]

    # RAG 결과도 히스토리에 추가 (정책명만 간단히)
    if reply and res.get("policy"):
        policy_title = res["policy"].get("title", "")
        if policy_title:
            context_history.append(f"RAG 결과: {policy_title} 정책 상세 조회")
            state["context_history"] = context_history

    # 라우터가 돌려준 세션 병합
    state["session"] = res.get("state", sess)
    state["email_to"] = state["session"].get("email_to", "")
    
    # RAG는 보통 후속 질문이 자연스럽게 가능하므로 followup_needed는 기본 False
    state["followup_needed"] = False
    
    _persist_session_from_state(state)
    return state

def t2sql_node(state: AgentState) -> AgentState:
    """T2SQL 처리 + 후속 대화 판단"""
    _load_session_into_state(state)
    q = state.get("question", "")
    sess = state.get("session", {}) or {}

    res = router_singleton.run(q, sess)
    reply = res.get("reply", "")
    
    state["reply"] = reply
    state["raw"] = res.get("raw", {})
    state["session"] = res.get("state", sess)
    state["email_to"] = state["session"].get("email_to", "")
    
    # T2SQL 결과를 컨텍스트 히스토리에 저장 (간단하게 요약)
    if reply:
        context_history = state.get("context_history", [])
        # 너무 길면 자르고 핵심만 저장
        summary = reply[:300] + "..." if len(reply) > 300 else reply
        context_history.append(f"T2SQL 검색결과: {summary}")
        state["context_history"] = context_history
        state["t2sql_result"] = reply
    
    # 후속 대화 필요 여부 판단
    state["followup_needed"] = _needs_followup(reply, "t2sql")
    
    _persist_session_from_state(state)
    return state

def email_node(state: AgentState) -> AgentState:
    """이메일 처리"""
    _load_session_into_state(state)
    q = state.get("question", "")
    sess = state.get("session", {}) or {}

    # 혹시 이번 턴에서 이메일을 추가로 입력했으면 반영
    found_email = _extract_email(q)
    if found_email:
        sess["email_to"] = found_email

    res = router_singleton.run(q, sess)

    state["reply"] = res.get("reply", "")
    if res.get("error"):
        state["error"] = res["error"]
    if res.get("policy"):
        state["policy"] = res["policy"]
        sess["last_policy"] = res["policy"]

    # 에러 케이스에 따라 pending_email 토글
    if res.get("error") == "no_recipient":
        sess["pending_email"] = True
    else:
        sess["pending_email"] = False

    state["session"] = res.get("state", sess)
    state["email_to"] = state["session"].get("email_to", "")
    state["followup_needed"] = False  # 이메일 후에는 일반적으로 종료
    
    _persist_session_from_state(state)
    return state

def followup_handler_node(state: AgentState) -> AgentState:
    """후속 대화 안내"""
    reply = state.get("reply", "")
    
    # T2SQL 결과에 후속 대화 안내 추가
    followup_msg = (
        "\n\n💬 위 결과 중에서 더 자세히 알고 싶은 정책이 있으시면 "
        "정책명을 말씀해 주세요. 마감일, 신청방법 등 상세 정보를 안내드릴게요!"
    )
    
    state["reply"] = reply + followup_msg
    state["followup_needed"] = False  # 안내 메시지 추가했으므로 더이상 필요없음
    return state

# ========= 라우팅 로직 =========
def _route_selector(s: AgentState) -> Tool:
    r = s.get("route", "rag")
    return r if r in ("rag", "t2sql", "email") else "rag"

def _continuation_router(s: AgentState) -> str:
    """실행 후 다음 단계 결정"""
    # 에러가 있으면 종료
    if s.get("error"):
        return "end"
    
    # 후속 대화 필요하면 followup 노드로
    if s.get("followup_needed", False):
        return "followup"
    
    return "end"

# ========= 그래프 구성 =========
workflow = StateGraph(AgentState)
workflow.add_node("preprocess", preprocess_node)
workflow.add_node("decide", decide_node)
workflow.add_node("rag", rag_node)
workflow.add_node("t2sql", t2sql_node)
workflow.add_node("email", email_node)
workflow.add_node("followup", followup_handler_node)

workflow.set_entry_point("preprocess")

workflow.add_edge("preprocess", "decide")
workflow.add_conditional_edges(
    "decide",
    _route_selector,
    {"rag": "rag", "t2sql": "t2sql", "email": "email"},
)

# 각 도구 실행 후 조건부 라우팅
workflow.add_conditional_edges(
    "rag", 
    _continuation_router,
    {"end": END, "followup": "followup"}
)

workflow.add_conditional_edges(
    "t2sql", 
    _continuation_router,
    {"end": END, "followup": "followup"}
)

workflow.add_conditional_edges(
    "email", 
    _continuation_router,
    {"end": END, "followup": "followup"}
)

workflow.add_edge("followup", END)

graph_app = workflow.compile()

# ========= 편의 함수 =========
def reset_session(session_id: str = "default") -> None:
    _SESSION_STORE.reset(session_id)

def set_email_sender(
    sender_email: str,
    app_password: str,
    session_id: str = "default",
    smtp_host: Optional[str] = None,
    smtp_port: Optional[int] = None,
) -> None:
    sess = _SESSION_STORE.get(session_id)
    sess.update({
        "sender_email": sender_email,
        "app_password": app_password,
        **({"smtp_host": smtp_host} if smtp_host else {}),
        **({"smtp_port": smtp_port} if smtp_port else {}),
    })
    _SESSION_STORE.set(session_id, sess)

def run_conversation(question: str, session_id: str = "default") -> Dict[str, Any]:
    """대화 실행 헬퍼 함수"""
    final_state = {}
    for event in graph_app.stream({"question": question, "session_id": session_id}):
        for node, payload in event.items():
            final_state.update(payload)
    return final_state

# ========= 로컬 REPL =========
if __name__ == "__main__":
    import sys

    sid = os.getenv("AGENT_SESSION_ID", "demo")
    reset_session(sid)

    sender_email = os.getenv("SENDER_EMAIL")
    app_password = os.getenv("APP_PASSWORD")
    if sender_email and app_password:
        set_email_sender(sender_email, app_password, session_id=sid)

    print("=== Enhanced Graph Agent (REPL) ===")
    print("T2SQL → RAG 연계 지원 + 멀티스텝 워크플로우")
    print("엔터만 누르면 종료됩니다. (세션:", sid, ")")
    print()
    print("🔍 예시 시나리오:")
    print("1) '경북 창업지원 정책 알려줘'  (→ T2SQL 검색)")
    print("2) '첫 번째 정책 자세히'        (→ RAG 상세 정보)")
    print("3) 'user@example.com으로 보내줘' (→ 이메일 발송)")
    print()

    try:
        while True:
            q = input("Q> ").strip()
            if not q:
                print("종료합니다. 안녕! 👋")
                break

            final = run_conversation(q, sid)

            route = final.get("route", "?")
            reply = final.get("reply", "")
            error = final.get("error")
            policy = final.get("policy")
            email_to = final.get("email_to", "")
            step_count = final.get("step_count", 0)
            context_history = final.get("context_history", [])

            print(f"\n[Route] {route} (Step {step_count})")
            if email_to:
                print(f"[Session.email_to] {email_to}")
            if context_history:
                print(f"[Context] {len(context_history)} items in history")
            if error:
                print(f"[Error] {error}")
            if reply:
                print("\n[Reply]")
                print(reply)

            if os.getenv("AGENT_DEBUG", "0") == "1":
                import json as _json
                debug_info = {
                    "step_count": step_count,
                    "context_history": context_history,
                    "followup_needed": final.get("followup_needed", False),
                    "raw": final.get("raw", {})
                }
                print("\n[Debug]")
                print(_json.dumps(debug_info, ensure_ascii=False, indent=2))

            print()

    except (KeyboardInterrupt, EOFError):
        print("\n종료합니다. 안녕! 👋")
        sys.exit(0)