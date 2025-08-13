# -*- coding: utf-8 -*-
"""
Text2SQL 기반 정책/금융상품 검색기 - SQLite 버전 (DuckDB 문제 해결용)
"""

import os
import re
import sqlite3
from typing import TypedDict, Dict, List, Any

import pandas as pd
from sqlalchemy import create_engine, text

from langchain_openai import ChatOpenAI
from langchain_core.output_parsers.string import StrOutputParser
from langchain_community.utilities import SQLDatabase
from langgraph.graph import StateGraph, END
from prompts.t2sql_prompts import (ROUTING_PROMPT, POLICIES_SCHEMA_DOC, PRODUCT_SCHEMA_DOC, POLICY_SQL_PROMPT, PRODUCT_SQL_PROMPT, SINGLE_SUMMARY_PROMPT, BOTH_SUMMARY_PROMPT)

from dotenv import load_dotenv
load_dotenv(override=True)

# =========================
# 1) 환경 설정
# =========================
print("=== SQLite 버전 T2SQL 시작 ===")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
POLICY_CSV = os.path.join(BASE_DIR, "data", "final_detailed_data_preprocessed.csv")
PRODUCT_CSV = os.path.join(BASE_DIR, "data", "haedream_products_preprocessed2.csv")

# SQLite DB 파일
SQLITE_DB = os.path.join(BASE_DIR, "kb_ai_agent.sqlite")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY가 설정되어 있지 않습니다.")

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
print("✓ 환경 설정 완료")

# =========================
# 2) SQLite 연결 및 초기화
# =========================
def materialize_tables_sqlite(
    policies_df: pd.DataFrame,
    product_df: pd.DataFrame,
    db_path: str = SQLITE_DB
) -> SQLDatabase:
    """
    DataFrame → SQLite TABLE 생성
    """
    print("SQLite 테이블 생성 시작...")
    
    # 기존 DB 파일 삭제 (새로 시작)
    if os.path.exists(db_path):
        try:
            os.remove(db_path)
            print("기존 DB 파일 삭제됨")
        except:
            print("DB 파일 삭제 실패 (사용중)")
    
    try:
        # SQLite 연결
        conn = sqlite3.connect(db_path)
        print("✓ SQLite 연결 성공")
        
        # 테이블 정리
        conn.execute("DROP TABLE IF EXISTS policies")
        conn.execute("DROP TABLE IF EXISTS product")
        print("✓ 기존 테이블 정리")
        
        # policies 테이블 생성
        print("정책 테이블 생성 중...")
        
        # 데이터 전처리 (NULL 값 처리)
        policies_clean = policies_df.copy()
        policies_clean = policies_clean.fillna({
            'index': 0,
            'title': '',
            'region': '',
            'amount': 0,
            'deadline': '1900-01-01',
            'link': '',
            'agency': '',
            'business_overview': '',
            'eligibility_content': '',
            'application_method': '',
            'tag': '',
            'org_dept': '',
            'phone': '',
            'url': ''
        })
        
        # pandas to_sql 사용 (더 안정적)
        policies_clean.to_sql('policies', conn, if_exists='replace', index=False)
        print("✓ 정책 테이블 생성 완료")
        
        # product 테이블 생성
        print("상품 테이블 생성 중...")
        
        product_clean = product_df.copy()
        product_clean = product_clean.fillna({
            'product_name': '',
            'product_overview': '',
            'eligibility': '',
            'region': '',
            'application_agency': '',
            'application_period': '',
            'max_support_amount': 0,
            'fund_purpose': '',
            'gurantee_fee_min': 0,
            'gurantee_fee_max': 0,
            'gurantee_period': '',
            'preferntial_terms': ''
        })
        
        product_clean.to_sql('product', conn, if_exists='replace', index=False)
        print("✓ 상품 테이블 생성 완료")
        
        # 테이블 확인
        policies_count = conn.execute("SELECT COUNT(*) FROM policies").fetchone()[0]
        product_count = conn.execute("SELECT COUNT(*) FROM product").fetchone()[0]
        print(f"✓ 테이블 생성 확인 - policies: {policies_count}개, product: {product_count}개")
        
        conn.close()
        
    except Exception as e:
        print(f"✗ SQLite 테이블 생성 실패: {e}")
        raise
    
    # SQLAlchemy 엔진 생성
    try:
        db_uri = f"sqlite:///{db_path}"
        engine = create_engine(db_uri)
        sql_db = SQLDatabase(engine=engine)
        print("✓ SQLDatabase 객체 생성 성공")
        return sql_db
    except Exception as e:
        print(f"✗ SQLDatabase 객체 생성 실패: {e}")
        raise

# =========================
# 3) 그래프 상태 및 함수들 (기존과 동일)
# =========================
class GraphState(TypedDict, total=False):
    question: str
    target_table: str
    sql_query: str
    db_result: str
    answer: str
    error: str
    no_result: bool
    sql_queries: Dict[str, str]
    db_results: Dict[str, str]

# 라우팅 함수들 (기존과 동일)
_POLICY_KEYS  = ("지원금","보조금","정책","사업","공고","지원사업","사업공고")
_PRODUCT_KEYS = ("보증","대출","융자","보증료","보증료율","금리","금융","한도","담보","보증한도","융자한도","보증기간")

def hard_route(question: str) -> str:
    """키워드 기반 라우팅"""
    q = re.sub(r"\s+", "", str(question))
    policy_hit  = any(k in q for k in _POLICY_KEYS)
    product_hit = any(k in q for k in _PRODUCT_KEYS)
    if policy_hit and product_hit:
        return "both"
    if policy_hit:
        return "policies"
    if product_hit:
        return "product"
    return ""

def route_to_table_node(state: GraphState) -> GraphState:
    question = state["question"]
    choice = hard_route(question)
    if choice:
        state["target_table"] = choice
        return state
    router = ROUTING_PROMPT | llm | StrOutputParser()
    target_table = router.invoke({"question": question}).strip().lower()
    state["target_table"] = "product" if "product" in target_table else "policies"
    return state

def _cleanup_sql(sql: str) -> str:
    """SQL 정리"""
    s = re.sub(r"^```(?:sql)?\s*", "", sql.strip(), flags=re.I)
    s = re.sub(r"\s*```$", "", s).strip()
    s = re.sub(r"\s+", " ", s).strip()
    return s[:-1].strip() if s.endswith(";") else s

def _gen_sql_for(table_name: str, question: str) -> str:
    """프롬프트 → SQL 생성"""
    if table_name == "product":
        prompt, schema = PRODUCT_SQL_PROMPT, PRODUCT_SCHEMA_DOC
    else:
        prompt, schema = POLICY_SQL_PROMPT, POLICIES_SCHEMA_DOC

    sql = (prompt | llm | StrOutputParser()).invoke({"schema": schema, "question": question})
    sql = _cleanup_sql(sql)

    # 가드레일
    if not re.match(r"^\s*SELECT\b", sql, re.I):
        raise ValueError("Invalid SQL: not a SELECT")
    if not re.search(rf"\bFROM\s+{table_name}\b", sql, re.I):
        raise ValueError(f"Invalid SQL: wrong FROM (expected {table_name})")
    forbidden = ["INSERT","UPDATE","DELETE","DROP","ALTER","ATTACH","PRAGMA","CREATE","JOIN"]
    if any(re.search(rf"\b{kw}\b", sql, re.I) for kw in forbidden):
        raise ValueError("Invalid SQL: forbidden keyword detected")
    return sql

def generate_sql_query_node(state: GraphState) -> GraphState:
    tbl = state["target_table"]
    try:
        if tbl != "both":
            state["sql_query"] = _gen_sql_for(tbl, state["question"])
            state["error"], state["no_result"] = "", False
        else:
            state["sql_queries"] = {
                "policies": _gen_sql_for("policies", state["question"]),
                "product":  _gen_sql_for("product",  state["question"]),
            }
            state["error"], state["no_result"] = "", False
    except Exception as e:
        print(f"SQL 생성 오류: {e}")
        state["error"] = "sql_generation_failed"
        state["no_result"] = True
    return state

def _run_and_check(db: SQLDatabase, sql_query: str) -> (str, bool):
    """SQL 실행 및 결과 확인"""
    try:
        result_str = db.run(sql_query)
        with db._engine.connect() as conn:
            cnt = conn.execute(text(f"SELECT COUNT(*) AS c FROM ({sql_query}) AS sub")).scalar()
        is_empty = (int(cnt or 0) == 0)
        return result_str, is_empty
    except Exception as e:
        print(f"SQL 실행 오류: {e}")
        return "", True

def execute_sql_query_node(state: GraphState, db: SQLDatabase) -> GraphState:
    tbl = state["target_table"]
    try:
        if tbl != "both":
            sql_query = state.get("sql_query", "")
            if not sql_query:
                state.update(error="no_sql", no_result=True)
                return state
            result_str, is_empty = _run_and_check(db, sql_query)
            state.update(db_result=result_str, no_result=is_empty, error="")
        else:
            sqls = state.get("sql_queries", {})
            res: Dict[str, str] = {}
            empty_flags: Dict[str, bool] = {}
            for tb, q in sqls.items():
                r, empty = _run_and_check(db, q)
                res[tb] = r
                empty_flags[tb] = empty
            state.update(
                db_results=res,
                no_result=all(empty_flags.values()) if empty_flags else True,
                error=""
            )
    except Exception as e:
        print(f"SQL 실행 오류: {e}")
        state.update(error="sql_execution_failed", no_result=True)
    return state

def generate_answer_node(state: GraphState) -> GraphState:
    if state.get("no_result") or state.get("error"):
        state["answer"] = (
            "조건에 맞는 정책/상품을 찾지 못했어요. 아래처럼 더 구체적으로 알려주시면 다시 찾아볼게요.\n\n"
            "• 분야: 정책, 지원금, 보조금, 보증, 융자, 대출\n"
            "• 대상: 청년, 창업, 1인기업, 소상공인\n"
            "• 지역: 서울, 부산, 경기, 전국 (세부 지역 가능)\n"
            "• 조건: 500만원 이상, 1억원 이하, 보증료 1% 미만\n"
            "예) '서울 청년 창업 지원금 1000만원 이상', '보증료 1% 이하 소상공인 대출'"
        )
        return state

    tbl = state["target_table"]
    try:
        if tbl != "both":
            ans = (SINGLE_SUMMARY_PROMPT | llm | StrOutputParser()).invoke({
                "question": state["question"],
                "db_result": state.get("db_result", "")
            })
            state["answer"] = ans
        else:
            db_results = state.get("db_results", {})
            ans = (BOTH_SUMMARY_PROMPT | llm | StrOutputParser()).invoke({
                "question": state["question"],
                "policies_str": db_results.get("policies", ""),
                "product_str":  db_results.get("product",  "")
            })
            state["answer"] = ans
    except Exception as e:
        print(f"답변 생성 오류: {e}")
        state["answer"] = "답변 생성 중 오류가 발생했습니다."
    return state

# =========================
# 4) 메인 함수
# =========================
def main():
    print("\n=== SQLite 버전 T2SQL 메인 시작 ===")
    
    try:
        # CSV 로드
        print("CSV 파일 로드 중...")
        policies = pd.read_csv(POLICY_CSV)
        product = pd.read_csv(PRODUCT_CSV, encoding="euc-kr")
        print(f"✓ CSV 로드 성공 - 정책: {len(policies)}, 상품: {len(product)}")
        
        # SQLite DB 생성
        db = materialize_tables_sqlite(policies, product)
        print("✓ SQLite DB 생성 완료")
        
        # 간단한 테스트 쿼리
        test_result = db.run("SELECT COUNT(*) FROM policies")
        print(f"✓ 테스트 쿼리 성공: {test_result}")
        
        # LangGraph 워크플로우 구성
        print("LangGraph 워크플로우 구성 중...")
        workflow = StateGraph(GraphState)
        
        # 노드 추가
        workflow.add_node("route_to_table", route_to_table_node)
        workflow.add_node("generate_sql_query", generate_sql_query_node)
        workflow.add_node("execute_sql_query", lambda state: execute_sql_query_node(state, db))
        workflow.add_node("generate_answer", generate_answer_node)
        
        # 엣지 추가
        workflow.set_entry_point("route_to_table")
        workflow.add_edge("route_to_table", "generate_sql_query")
        workflow.add_edge("generate_sql_query", "execute_sql_query")
        workflow.add_edge("execute_sql_query", "generate_answer")
        workflow.add_edge("generate_answer", END)
        
        app = workflow.compile()
        print("✓ LangGraph 워크플로우 생성 완료")
        
        # # 테스트 질문들
        # questions = [
        #     "청년 농업인을 위한 지원금 100만원 받고 싶어",
        #     "보증료율 1% 이하인 사업자 대출 상품 알려줘",
        #     "경북에서 운전자금으로 사용할 보증료율 1% 이하로 알려줘",
        # ]
        
        # print("\n=== 테스트 질문 처리 시작 ===")
        # for i, q in enumerate(questions, 1):
        #     print(f"\n--- 질문 {i}: {q} ---")
        #     try:
        #         for event in app.stream({"question": q}):
        #             for key, value in event.items():
        #                 if key == "generate_answer" and 'answer' in value:
        #                     print(f"[답변] {value['answer']}")
        #     except Exception as e:
        #         print(f"질문 처리 중 오류: {e}")
        
        # print("\n=== 모든 테스트 완료 ===")
        
    except Exception as e:
        print(f"메인 함수 실행 중 오류: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
    # print("\n프로그램 종료")
    # input("Enter를 눌러서 종료...")