# -*- coding: utf-8 -*-
"""
Text2SQL용 프롬프트 모음 (원문 그대로)
"""

from langchain_core.prompts import ChatPromptTemplate

# -----------------------------
# 1) 테이블 라우팅 프롬프트
# -----------------------------
ROUTING_PROMPT = ChatPromptTemplate.from_template(
    """
    당신은 사용자의 질문 의도를 파악하여 어떤 데이터베이스 테이블을 사용해야 할지 결정하는 전문가입니다.
    사용자의 질문에 가장 적합한 테이블 이름을 다음 중에서 하나만 골라주세요.

    1. `policies`: 정부나 지자체의 **정책 자금, 지원금, 보조금, 사업** 관련 질문일 때 선택합니다.
       (예: "창업 지원금 알려줘", "서울시 소상공인 정책 찾아줘", "청년 지원 사업 있나요?")

    2. `product`: 금융 기관의 **보증, 대출, 융자 상품** 관련 질문일 때 선택합니다.
       (예: "사업자 대출 보증 상품 알려줘", "보증료 저렴한 상품 찾아줘", "5천만원 융자 가능할까?")

    사용자 질문: "{question}"

    가장 적합한 테이블 이름만 출력하세요. (policies 또는 product)
    """
)

# -----------------------------
# 2) 스키마 문서 (설명)
# -----------------------------
POLICIES_SCHEMA_DOC = """
Table: policies
(정부/지자체에서 지원하는 정책자금, 지원금, 보조금 정보)
- index (BIGINT): 고유 ID
- title (TEXT): 정책명
- region (TEXT): 지원 지역 (예: '서울', '경기', '전국')
- amount (DOUBLE): 지원 금액 (숫자)
- deadline (DATE): 마감일 (YYYY-MM-DD 형식)
- business_overview (TEXT): 사업 개요
- eligibility_content (TEXT): 자격 상세 내용
- application_method (TEXT): 신청 방법
- tag (TEXT): 관련 키워드 태그 (예: '청년', '여성', '창업')
- org_dept (TEXT): 담당 부서
- phone (TEXT): 담당자 연락처
- url (TEXT): 상세 정보 URL
"""

PRODUCT_SCHEMA_DOC = """
Table: product
(금융 기관의 보증/융자 상품 정보)
- product_name (TEXT): 상품명
- product_overview (TEXT): 상품 개요
- eligibility (TEXT): 지원 대상 및 자격
- region (TEXT): 취급 지역
- application_agency (TEXT): 신청 기관
- application_period (TEXT): 신청 기간 (예: '상시', '2025-01-01 ~ 2025-12-31')
- max_support_amount (DOUBLE): 최대 지원(보증) 한도 금액
- fund_purpose (TEXT): 자금 용도
- gurantee_fee_min (DOUBLE): 최소 보증료율 (%)
- gurantee_fee_max (DOUBLE): 최대 보증료율 (%)
- gurantee_period (TEXT): 보증 기간 (예: '5년')
- preferntial_terms (TEXT): 우대 조건
"""

# -----------------------------
# 3) SQL 생성 프롬프트
# -----------------------------
POLICY_SQL_PROMPT = ChatPromptTemplate.from_template(
    """
    당신은 `policies` 테이블에서 DuckDB SQL을 작성하는 전문가입니다.
    아래 테이블 설명을 참고하여 사용자 질문에 맞는 SQL 쿼리를 작성하세요.

    {schema}

    작성 지침:
    - 반드시 `SELECT` 쿼리만 생성하고, `FROM policies`만 사용하세요.
    - 기본적으로 `title, business_overview, amount, url, org_dept, phone` 컬럼을 포함하세요.
    - 개요는 business_overview를 참고하여 상세하면서도 명료하게 작성해주세요.
    - 사용자의 질의에서 키워드를 뽑은 후 필터링 조건 걸기 (예: 청년 창업인 -> 키워드: 청년, 창업, **키워드는 절대 수치가 될 수 없음**)
    - 태그나 제목에서 조건 필터링:
      예: WHERE (tag ILIKE '%청년%' OR title ILIKE '%청년%')
    - 여러 키워드면 각 키워드를 (tag ILIKE '%키워드%' OR title ILIKE '%키워드%') 로 만들고 AND로 연결
    - region은 부분일치: WHERE region ILIKE '%지역명%'
    - 한국어 금액(예: 500만원, 1.2억)은 각각 5000000, 120000000으로 해석해 amount 비교식에 사용
    - 기본적으로 `ORDER BY deadline ASC` 순으로 정렬하고 금액 조건이 있으면 amount로 정렬하세요.
    - 결과는 최대 3개로 제한하세요 (`LIMIT 3`).
    - SQL 쿼리만 출력하세요.

    사용자 질문: {question}
    """
)

PRODUCT_SQL_PROMPT = ChatPromptTemplate.from_template(
    """
    당신은 `product` 테이블에서 DuckDB SQL을 작성하는 전문가입니다.
    아래 테이블 설명을 참고하여 사용자 질문에 맞는 SQL 쿼리를 작성하세요.

    {schema}

    작성 지침:
    - 반드시 `SELECT` 쿼리만 생성하고, `FROM product`만 사용하세요.
    - 기본적으로 `product_name, product_overview, eligibility, max_support_amount, application_agency` 컬럼을 포함하세요.
    - 사용자의 질의에서 키워드를 뽑은 후 필터링 조건 걸기 (예: 청년 창업인 -> 키워드: 청년, 창업, **키워드는 절대 수치가 될 수 없음**)
    - 여러 키워드면 각 키워드를 (title ILIKE '%키워드%') 로 만들고 AND로 연결
    - 제목에서 조건 필터링:
      예: WHERE (tag ILIKE '%청년%' OR title ILIKE '%청년%')
    - 여러 키워드면 각 키워드를 (tag ILIKE '%키워드%') OR (title ILIKE '%키워드%') 로 만들고 AND로 연결
    - region은 부분일치: WHERE region ILIKE '%지역명%'
    - 자금용도 (fund_purpose) 필터링 또는 질문 시 제공
    - 보증료율 (gurantee_fee_min, gurantee_fee_max), 보증기간(gurantee_period) 필터링 또는 질문 시 제공
    - 한국어 금액(예: 500만원, 1.2억)은 각각 5000000, 120000000으로 해석해 amount 비교식에 사용
    - 개요는 product_overview를 참고하여 상세하면서도 명료하게 작성해주세요.
    - 금액 조건이 있으면 `ORDER BY max_support_amount DESC` 순으로 정렬하세요.
    - 결과는 최대 3개로 제한하세요 (`LIMIT 3`).
    - SQL 쿼리만 출력하세요.

    사용자 질문: {question}
    """
)

# -----------------------------
# 4) 결과 요약 프롬프트
# -----------------------------
SINGLE_SUMMARY_PROMPT = ChatPromptTemplate.from_template(
    """
    당신은 소상공인 정책/금융 상품 안내 챗봇입니다.
    아래 질문과 DB 조회 결과를 바탕으로 한국어로 간결히 안내하세요.

    질문: "{question}"
    DB 결과(문자열): "{db_result}"

    지침:
    - 최대 3건, 1., 2., 3. 번호 매겨 요약.
    - 각 항목에 개요, 대상/자격, 금액/한도, 지역/기관(가능 시), URL/담당부서/전화(가능 시)를 포함.
    - 질의에 보증기간, 보증료율 등 포함되면 해당 정보도 포함 (있을 때만).
    - DB에 없는 내용은 만들지 말 것. SQL은 노출하지 말 것.
    """
)

BOTH_SUMMARY_PROMPT = ChatPromptTemplate.from_template(
    """
    당신은 소상공인 정책/금융 상품 안내 챗봇입니다.
    아래 질문과 DB 조회 결과(정책/상품 두 섹션)를 바탕으로 한국어로 간결히 안내하세요.

    질문: "{question}"
    [정책 결과]: "{policies_str}"
    [상품 결과]: "{product_str}"

    지침:
    - 두 섹션으로 나눠 요약:
      1) [정책 지원금] (policies) 최대 3건
      2) [금융 상품] (product) 최대 3건
    - 각 항목은 제목과 핵심 정보(개요, 금액/한도, 대상/지역, 기관/연락처/URL 등 DB에 있는 것만)를 번호로 정리.
    - 질의에 보증기간, 보증료율 등 포함되면 해당 정보도 포함 (데이터에 존재할 때).
    - **SQL 결과로 나온 것만 사용, 새로운 내용 추가 금지**
    - SQL은 노출하지 말 것.
    """
)

# 맨 아래 __all__에 두 항목 추가
__all__ = [
    "ROUTING_PROMPT",
    "POLICIES_SCHEMA_DOC",
    "PRODUCT_SCHEMA_DOC",
    "POLICY_SQL_PROMPT",
    "PRODUCT_SQL_PROMPT",
    "SINGLE_SUMMARY_PROMPT",
    "BOTH_SUMMARY_PROMPT",
]