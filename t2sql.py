# -*- coding: utf-8 -*-
"""
Text2SQL 기반 정책/금융상품 검색기
- DuckDB 테이블 구성
- LangGraph 기반 SQL 생성 → 실행 → 요약
"""

import os
import re
from typing import TypedDict, Dict, List, Any

import pandas as pd
import duckdb
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
# CSV 경로 (요청한 형태 그대로)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
POLICY_CSV = os.path.join(BASE_DIR, "data", "final_detailed_data_preprocessed.csv")
PRODUCT_CSV = os.path.join(BASE_DIR, "data", "haedream_products_preprocessed2.csv")

# OpenAI 키 (환경변수에서 읽음)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY가 설정되어 있지 않습니다. 환경변수에 설정하세요.")
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# DuckDB URI: .env의 T2SQL_DB_URI 우선, 없으면 기본값 사용
DB_URI = os.getenv("T2SQL_DB_URI")

# =========================
# 2) DuckDB 연결 및 초기화화
# =========================
def materialize_tables_simple(
    policies_df: pd.DataFrame,
    product_df: pd.DataFrame,
    db_uri: str = DB_URI
) -> SQLDatabase:
    """
    DataFrame → DuckDB TABLE 생성
    """
    con = duckdb.connect(db_uri.replace("duckdb:///", ""))
    try:
        # 정리
        con.execute("DROP VIEW IF EXISTS policies;")
        con.execute("DROP VIEW IF EXISTS product;")
        # DF 등록
        con.register("policies_df", policies_df)
        con.register("product_df", product_df)

        # policies 테이블
        con.execute("""
            CREATE OR REPLACE TABLE policies AS
            SELECT
                CAST("index" AS BIGINT)                 AS "index",
                CAST(title AS TEXT)                     AS title,
                CAST(region AS TEXT)                    AS region,
                CAST(amount AS DOUBLE)                  AS amount,
                CAST(deadline AS DATE)                  AS deadline,  -- YYYY-MM-DD
                CAST(link AS TEXT)                      AS link,
                CAST(agency AS TEXT)                    AS agency,
                CAST(business_overview AS TEXT)         AS business_overview,
                CAST(eligibility_content AS TEXT)       AS eligibility_content,
                CAST(application_method AS TEXT)        AS application_method,
                CAST(tag AS TEXT)                       AS tag,
                CAST(org_dept AS TEXT)                  AS org_dept,
                CAST(phone AS TEXT)                     AS phone,
                CAST(url AS TEXT)                       AS url
            FROM policies_df;
        """)

        # product 테이블
        con.execute("""
            CREATE OR REPLACE TABLE product AS
            SELECT
                CAST(product_name AS TEXT)              AS product_name,
                CAST(product_overview AS TEXT)          AS product_overview,
                CAST(eligibility AS TEXT)               AS eligibility,
                CAST(region AS TEXT)                    AS region,
                CAST(application_agency AS TEXT)        AS application_agency,
                CAST(application_period AS TEXT)        AS application_period,
                CAST(max_support_amount AS DOUBLE)      AS max_support_amount,
                CAST(fund_purpose AS TEXT)              AS fund_purpose,
                CAST(gurantee_fee_min AS DOUBLE)        AS gurantee_fee_min,
                CAST(gurantee_fee_max AS DOUBLE)        AS gurantee_fee_max,
                CAST(gurantee_period AS TEXT)           AS gurantee_period,
                CAST(preferntial_terms AS TEXT)         AS preferntial_terms
            FROM product_df;
        """)
    finally:
        con.close()

    engine = create_engine(db_uri)
    return SQLDatabase(engine=engine)

# =========================
# 4) 그래프 상태
# =========================
class GraphState(TypedDict, total=False):
    question: str
    target_table: str            # 'policies' | 'product' | 'both'
    sql_query: str               # 단일 경로
    db_result: str               # 단일 경로
    answer: str
    error: str
    no_result: bool
    sql_queries: Dict[str, str]  # {'policies': '...', 'product': '...'}
    db_results: Dict[str, str]   # {'policies': '[...]', 'product': '[...]'}

# =========================
# 5) 라우팅 & SQL 생성
# =========================
_POLICY_KEYS  = ("지원금","보조금","정책","사업","공고","지원사업","사업공고")
_PRODUCT_KEYS = ("보증","대출","융자","보증료","보증료율","금리","금융상품","한도","담보","보증한도","융자한도","보증기간")

def hard_route(question: str) -> str:
    """키워드 기반 라우팅: 정책/상품 동시 존재 시 both"""
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
    # LLM 라우팅 백업
    router = ROUTING_PROMPT | llm | StrOutputParser()
    target_table = router.invoke({"question": question}).strip().lower()
    state["target_table"] = "product" if "product" in target_table else "policies"
    return state

def _cleanup_sql(sql: str) -> str:
    """```sql ...``` 제거 + 공백 정리 + 세미콜론 제거"""
    s = re.sub(r"^```(?:sql)?\s*", "", sql.strip(), flags=re.I)
    s = re.sub(r"\s*```$", "", s).strip()
    s = re.sub(r"\s+", " ", s).strip()
    return s[:-1].strip() if s.endswith(";") else s

def _gen_sql_for(table_name: str, question: str) -> str:
    """프롬프트 → SQL 생성 (테이블별 프롬프트 사용)"""
    if table_name == "product":
        prompt, schema = PRODUCT_SQL_PROMPT, PRODUCT_SCHEMA_DOC
    else:
        prompt, schema = POLICY_SQL_PROMPT, POLICIES_SCHEMA_DOC

    sql = (prompt | llm | StrOutputParser()).invoke({"schema": schema, "question": question})
    sql = _cleanup_sql(sql)

    # 가드레일: SELECT-only + FROM 검증 + 위험 키워드 차단
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
    if tbl != "both":
        state["sql_query"] = _gen_sql_for(tbl, state["question"])
        state["error"], state["no_result"] = "", False
        return state
    # both: 두 쿼리 생성
    state["sql_queries"] = {
        "policies": _gen_sql_for("policies", state["question"]),
        "product":  _gen_sql_for("product",  state["question"]),
    }
    state["error"], state["no_result"] = "", False
    return state

# =========================
# 6) SQL 실행
# =========================
def _run_and_check(db: SQLDatabase, sql_query: str) -> (str, bool):
    """문자열 결과와 empty 여부 반환"""
    result_str = db.run(sql_query)
    try:
        with db._engine.connect() as conn:
            cnt = conn.execute(text(f"SELECT COUNT(*) AS c FROM ({sql_query}) AS sub")).scalar()
        is_empty = (int(cnt or 0) == 0)
    except Exception:
        is_empty = result_str.strip() in ("[]", "")
    return result_str, is_empty

def execute_sql_query_node(state: GraphState, db: SQLDatabase) -> GraphState:
    tbl = state["target_table"]
    if tbl != "both":
        sql_query = state.get("sql_query", "")
        if not sql_query:
            state.update(error="invalid_sql", no_result=True)
            return state
        result_str, is_empty = _run_and_check(db, sql_query)
        state.update(db_result=result_str, no_result=is_empty, error="")
        return state

    # both: 두 쿼리 실행
    sqls = state.get("sql_queries", {})
    res: Dict[str, str] = {}
    empty_flags: Dict[str, bool] = {}
    try:
        for tb, q in sqls.items():
            r, empty = _run_and_check(db, q)
            res[tb] = r
            empty_flags[tb] = empty
        state.update(
            db_results=res,
            no_result=all(empty_flags.values()) if empty_flags else True,
            error=""
        )
        return state
    except Exception:
        state.update(db_results={}, error="invalid_sql", no_result=True)
        return state

# =========================
# 7) 답변 생성 (요약 프롬프트 import 사용)
# =========================
def generate_answer_node(state: GraphState) -> GraphState:
    if state.get("no_result"):
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
    if tbl != "both":
        ans = (SINGLE_SUMMARY_PROMPT | llm | StrOutputParser()).invoke({
            "question": state["question"],
            "db_result": state.get("db_result", "")
        })
        state["answer"] = ans
        return state

    # both
    db_results = state.get("db_results", {})
    ans = (BOTH_SUMMARY_PROMPT | llm | StrOutputParser()).invoke({
        "question": state["question"],
        "policies_str": db_results.get("policies", ""),
        "product_str":  db_results.get("product",  "")
    })
    state["answer"] = ans
    return state

# =========================
# 8) 메인 
# =========================
def main():
    # CSV 로딩
    policies = pd.read_csv(POLICY_CSV)
    product  = pd.read_csv(PRODUCT_CSV, encoding="euc-kr")

    # DuckDB 테이블 생성
    db = materialize_tables_simple(policies, product, DB_URI)

    # LangGraph 워크플로우
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

    # 데모 질의
    questions = [
        "청년 농업인을 위한 지원금 100만원 받고 싶어",
        "보증료율 1% 이하인 사업자 대출 상품 알려줘",
        "경북에서 운전자금으로 사용할 보증료율 1% 이하로 알려줘",
    ]
    for q in questions:
        print("\n==============================")
        print("Q:", q)
        for event in app.stream({"question": q}):
            for key, value in event.items():
                print(f"--- Node: {key} ---")
                print(value)
                if 'answer' in value:
                    print(f"\n[최종 답변]\n{value['answer']}")

if __name__ == "__main__":
    main