# -*- coding: utf-8 -*-
import os, json, html, time
import streamlit as st
from uuid import uuid4
from dotenv import load_dotenv

APP_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = APP_DIR
load_dotenv(override=True)

# ---- graph agent import
from graph_agent import graph_app, reset_session  # set_email_sender 제거

# =========================
# KB 컬러 & CSS 테마
# =========================
KB_YELLOW = "#ffbc01"
KB_GRAY = "#60584d"

CUSTOM_CSS = f"""
<style>
:root {{
  --kb-yellow: {KB_YELLOW};
  --kb-gray: {KB_GRAY};
}}

html, body, [data-testid="stAppViewContainer"] {{
  background: linear-gradient(180deg, #fff9e6 0%, #ffffff 18%);
  color: var(--kb-gray);
}}

h1, h2, h3, h4, h5, h6 {{
  color: var(--kb-gray);
  letter-spacing: -0.2px;
}}

a {{
  color: #5e4a00;
}}

[data-testid="stHeader"] {{
  background: linear-gradient(90deg, var(--kb-yellow), #ffd866);
}}

.kb-hero {{
  background: #fff7d1;
  border: 1px solid #f2e4bf;
  border-radius: 16px;
  padding: 14px 16px;
  margin-bottom: 8px;
  box-shadow: 0 2px 10px rgba(96,88,77,.06);
}}

.kb-hero .title {{
  font-weight: 800;
  font-size: 1.25rem;
  margin: 0;
  color: var(--kb-gray);
}}

.kb-hero .subtitle {{
  margin-top: 4px;
  color: #7a7064;
  font-size: 0.95rem;
}}

[data-testid="stSidebar"] > div:first-child {{
  background: #fff7d1;
  border-right: 2px solid #ffe08a;
}}

.stButton > button {{
  background: var(--kb-yellow);
  color: #2f281f;
  border: 0;
  border-radius: 12px;
  padding: 0.6rem 1rem;
  font-weight: 700;
  box-shadow: 0 4px 0 #e0a500;
}}
.stButton > button:hover {{
  filter: brightness(0.95);
  transform: translateY(-1px);
  box-shadow: 0 6px 0 #e0a500;
}}

[data-testid="stChatInput"] textarea {{
  border: 2px solid #ffe08a;
  border-radius: 12px;
}}

[data-testid="stChatMessage"] {{
  border-radius: 16px;
  padding: 0.75rem 1rem;
  border: 1px solid #f2e4bf;
  background: #fffdf4;
}}
[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] p {{
  margin-bottom: 0;
}}

.kb-card {{
  border: 1px solid #f2e4bf;
  border-radius: 16px;
  padding: 1rem 1.1rem;
  background: #fffbe6;
  box-shadow: 0 2px 10px rgba(96,88,77,.08);
  margin-top: 8px;
}}
.kb-badges {{
  margin-bottom: 8px;
}}
.kb-badge {{
  display: inline-block;
  background: var(--kb-gray);
  color: #fff;
  border-radius: 999px;
  padding: 2px 10px;
  font-size: 0.75rem;
  margin-right: 6px;
}}
.kb-meta {{
  color: #7a7064;
  font-size: 0.92rem;
  margin-top: 4px;
}}
.kb-link a {{
  color: #5e4a00;
  text-decoration: none;
  border-bottom: 1px dashed #b38b00;
}}
.kb-link a:hover {{ text-decoration: underline; }}

.kb-divider {{
  height: 1px; background: #f2e4bf; margin: 10px 0 8px;
}}

/* 로딩 스피너 스타일 */
.kb-loading {{
  display: flex;
  align-items: center;
  gap: 12px;
  padding: 12px 16px;
  background: #fff7d1;
  border: 1px solid #f2e4bf;
  border-radius: 12px;
  margin: 8px 0;
}}

.kb-spinner {{
  width: 20px;
  height: 20px;
  border: 2px solid #f2e4bf;
  border-top: 2px solid var(--kb-yellow);
  border-radius: 50%;
  animation: spin 1s linear infinite;
}}

@keyframes spin {{
  0% {{ transform: rotate(0deg); }}
  100% {{ transform: rotate(360deg); }}
}}

.kb-loading-text {{
  color: var(--kb-gray);
  font-weight: 500;
}}

/* 타이핑 애니메이션 */
.kb-typing {{
  display: inline-block;
}}

.kb-typing::after {{
  content: '';
  animation: typing 1.5s infinite;
}}

@keyframes typing {{
  0%, 20% {{ content: ''; }}
  40% {{ content: '.'; }}
  60% {{ content: '..'; }}
  80%, 100% {{ content: '...'; }}
}}
</style>
"""

# =========================
# Streamlit 기본 설정
# =========================
st.set_page_config(
    page_title="KB AI Agent",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
st.markdown(
    f"""
    <div class="kb-hero">
      <div class="title">🤖 KB AI Agent</div>
      <div class="subtitle">정책·보증·대출 정보를 간단한 질문으로 찾아드려요.</div>
    </div>
    """,
    unsafe_allow_html=True
)

# =========================
# 세션 ID (내부 관리만)
# =========================
if "session_id" not in st.session_state:
    st.session_state.session_id = f"st-{uuid4().hex[:8]}"

# =========================
# Sidebar: 사용자 친화 메뉴
# =========================
with st.sidebar:
    st.header("메뉴")
    if st.button("🧹 새 대화"):
        # 1) LangGraph 세션 메모리 초기화
        reset_session(st.session_state.session_id)
        # 2) UI 히스토리 초기화
        st.session_state.messages = []
        # 3) Streamlit 버전에 따라 안전하게 재실행
        if hasattr(st, "rerun"):
            st.rerun()

# =========================
# Chat 히스토리 초기화/출력
# =========================
if "messages" not in st.session_state:
    st.session_state.messages = []

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        if isinstance(m["content"], str):
            st.markdown(m["content"])
        else:
            st.write(m["content"])

# =========================
# 유틸: graph_app 호출 (스트리밍 버전)
# =========================
def call_agent_with_streaming(question: str, session_id: str, status_placeholder):
    """에이전트를 호출하면서 실시간으로 상태를 업데이트"""
    final = {}
    
    # 단계별 상태 메시지
    status_messages = [
        "🔍 질문을 분석하고 있어요",
        "📚 관련 정보를 찾고 있어요", 
        "🤖 답변을 생성하고 있어요",
        "✨ 마무리하고 있어요"
    ]
    
    current_step = 0
    
    for i, event in enumerate(graph_app.stream({"question": question, "session_id": session_id})):
        # 주기적으로 상태 메시지 업데이트
        if i % 2 == 0 and current_step < len(status_messages):
            status_placeholder.markdown(
                f"""
                <div class="kb-loading">
                    <div class="kb-spinner"></div>
                    <div class="kb-loading-text">{status_messages[current_step]}<span class="kb-typing"></span></div>
                </div>
                """,
                unsafe_allow_html=True
            )
            current_step = min(current_step + 1, len(status_messages) - 1)
        
        for _, payload in event.items():
            final.update(payload)
    
    return final

def call_agent(question: str, session_id: str):
    """기존 방식 유지 (fallback용)"""
    final = {}
    for event in graph_app.stream({"question": question, "session_id": session_id}):
        for _, payload in event.items():
            final.update(payload)
    return final

def esc(x):  # HTML escape
    return html.escape(str(x or ""))

def render_policy_card(policy: dict):
    if not policy:
        return
    title   = esc(policy.get("title", ""))
    region  = esc(policy.get("region", ""))
    deadline= esc(policy.get("deadline", "") or "정보 없음")
    agency  = esc(policy.get("agency", ""))
    link    = policy.get("link", "") or policy.get("url", "")

    link_html = f'<div class="kb-link"><a href="{esc(link)}" target="_blank">🔗 원문 링크</a></div>' if link else ""

    st.markdown(
        f"""
        <div class="kb-card">
          <div class="kb-badges">
            <span class="kb-badge">정책</span>
          </div>
          <div><strong>🗂️ 정책명</strong>: {title}</div>
          <div class="kb-meta">📍 {region} &nbsp;|&nbsp; 🏢 {agency}</div>
          <div class="kb-divider"></div>
          <div><strong>⏰ 마감</strong>: {deadline}</div>
          {link_html}
        </div>
        """,
        unsafe_allow_html=True,
    )

# =========================
# 입력 처리
# =========================
user_text = st.chat_input("메시지를 입력하세요… (Enter로 전송)")

if user_text:
    # 1) 사용자 메시지 표시
    with st.chat_message("user"):
        st.markdown(user_text)
    st.session_state.messages.append({"role": "user", "content": user_text})

    # 2) 어시스턴트 응답 영역 생성
    with st.chat_message("assistant"):
        # 로딩 상태 표시용 플레이스홀더
        status_placeholder = st.empty()
        content_placeholder = st.empty()
        
        # 초기 로딩 메시지
        status_placeholder.markdown(
            """
            <div class="kb-loading">
                <div class="kb-spinner"></div>
                <div class="kb-loading-text">🚀 요청을 처리하고 있어요<span class="kb-typing"></span></div>
            </div>
            """,
            unsafe_allow_html=True
        )
        
        try:
            # 3) 에이전트 호출 (스트리밍으로 상태 업데이트)
            final = call_agent_with_streaming(user_text, st.session_state.session_id, status_placeholder)
            
            # 로딩 상태 제거
            status_placeholder.empty()
            
            # 결과 표시
            reply = final.get("reply", "")
            policy = final.get("policy")
            
            # 답변 렌더링
            if reply:
                content_placeholder.markdown(reply)
            
            # 정책 카드 렌더링
            if policy:
                render_policy_card(policy)
                
            # 히스토리에 저장
            st.session_state.messages.append({"role": "assistant", "content": reply or ""})
            
        except Exception as e:
            # 에러 발생 시 로딩 상태 제거하고 에러 메시지 표시
            status_placeholder.empty()
            content_placeholder.error(f"죄송합니다. 처리 중 오류가 발생했습니다: {str(e)}")
            st.session_state.messages.append({"role": "assistant", "content": "처리 중 오류가 발생했습니다."})