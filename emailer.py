# emailer.py
# -*- coding: utf-8 -*-

"""
이메일 발송 유틸
- 환경변수 기본값 사용 + AI Agent state(dict)로 발신자/앱비번/SMTP 설정을 덮어쓰기 가능
- 정책(공고) 안내용 HTML 본문 생성 + 단건 발송 래퍼 제공
"""

import os
import re
import smtplib
from datetime import datetime
from typing import Mapping, Any, Optional, Tuple
from email.mime.text import MIMEText

from dotenv import load_dotenv
load_dotenv(override=True)

# ---- SMTP 기본 설정 (환경변수) ----
SMTP_HOST = os.getenv("SMTP_HOST", "smtp.gmail.com")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
SENDER_EMAIL = os.getenv("SENDER_EMAIL", "")
APP_PASSWORD = os.getenv("APP_PASSWORD", "")

# ---- 이메일 형식 검증 ----
_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
def validate_email(addr: str) -> bool:
    return bool(_EMAIL_RE.match((addr or "").strip()))

# ---- D-day 계산 ----
def _today_kst():
    try:
        import zoneinfo
        KST = zoneinfo.ZoneInfo("Asia/Seoul")
        return datetime.now(tz=KST).date()
    except Exception:
        return datetime.now().date()

def _parse_deadline_date(policy: Mapping[str, Any]):
    val = policy.get("deadline")
    if hasattr(val, "isoformat"):
        return val
    s = str(val or "").strip()
    if not s:
        return None
    s = s.replace(".", "-").replace("/", "-")
    try:
        return datetime.fromisoformat(s[:10]).date()
    except Exception:
        return None

def compute_d_day_tag(policy: Mapping[str, Any]) -> str:
    dl = _parse_deadline_date(policy)
    if dl is None:
        return ""
    delta = (dl - _today_kst()).days
    if delta > 0:
        return f"[D-{delta}]"
    elif delta == 0:
        return "[D-0]"
    else:
        return f"[D+{abs(delta)}]"

# ---- HTML 본문 생성 ----
def build_interest_email_html(policy: Mapping[str, Any]) -> Tuple[str, str]:
    service_name = "정책알리미"  # 하드코딩
    title = str(policy.get("title") or "").strip()
    region = str(policy.get("region") or "").strip()
    overview = str(policy.get("business_overview") or "").strip()
    agency = str(policy.get("agency") or policy.get("org_dept") or "").strip()
    elig = str(policy.get("eligibility_content") or "").strip()
    org_dept = str(policy.get("org_dept") or "").strip()
    phone = str(policy.get("phone") or "").strip()
    url = str(policy.get("url") or "").strip()
    link = str(policy.get("link") or "").strip() or url

    deadline_val = policy.get("deadline")
    if hasattr(deadline_val, "isoformat"):
        deadline_raw = deadline_val.isoformat()
    else:
        deadline_raw = str(deadline_val or "").strip()

    d_tag = compute_d_day_tag(policy)
    d_prefix = (d_tag + " ") if d_tag else ""
    subject = f"{d_prefix}{title or '관심 공고 안내'} 신청 알림".strip()

    html = f"""
    <div style="font-family: 'Segoe UI', Arial, sans-serif; font-size:15px; line-height:1.7; color:#2d2d2d; background-color:#fafafa; padding:20px;">
      <div style="max-width: 640px; margin: auto; background: #ffffff; border-radius: 8px; box-shadow: 0 2px 6px rgba(0,0,0,0.05); overflow: hidden;">
        <div style="background: linear-gradient(135deg, #4a90e2, #357ab8); padding: 16px 20px; color: #fff;">
          <h2 style="margin: 0; font-size: 20px; font-weight: 600;">{service_name} - 관심 공고 안내</h2>
          <p style="margin: 4px 0 0; font-size: 13px; opacity: 0.9;">설정하신 관심 공고 정보를 보내드립니다.</p>
        </div>

        <table style="border-collapse: collapse; width: 100%; font-size:14px;">
          <tbody>
            <tr>
              <td style="padding:12px; background:#f9fafb; font-weight:600; width:120px; border-bottom:1px solid #eee;">정책명</td>
              <td style="padding:12px; border-bottom:1px solid #eee;">{title or '(정보 없음)'}</td>
            </tr>
            <tr>
              <td style="padding:12px; background:#f9fafb; font-weight:600; border-bottom:1px solid #eee;">지역</td>
              <td style="padding:12px; border-bottom:1px solid #eee;">{region or '(정보 없음)'}</td>
            </tr>
            <tr>
              <td style="padding:12px; background:#f9fafb; font-weight:600; border-bottom:1px solid #eee;">주관</td>
              <td style="padding:12px; border-bottom:1px solid #eee;">{agency or '(정보 없음)'}</td>
            </tr>
            <tr>
              <td style="padding:12px; background:#f9fafb; font-weight:600; border-bottom:1px solid #eee;">마감일자</td>
              <td style="padding:12px; border-bottom:1px solid #eee;">{deadline_raw or '(정보 없음)'}</td>
            </tr>
            <tr>
              <td style="padding:12px; background:#f9fafb; font-weight:600; border-bottom:1px solid #eee;">사업개요</td>
              <td style="padding:12px; border-bottom:1px solid #eee;">{overview or '(정보 없음)'}</td>
            </tr>
            <tr>
              <td style="padding:12px; background:#f9fafb; font-weight:600; border-bottom:1px solid #eee;">지원내용</td>
              <td style="padding:12px; border-bottom:1px solid #eee;">{elig or '(정보 없음)'}</td>
            </tr>
            <tr>
              <td style="padding:12px; background:#f9fafb; font-weight:600; border-bottom:1px solid #eee;">문의처</td>
              <td style="padding:12px; border-bottom:1px solid #eee;">{ ' / '.join([x for x in [org_dept, phone, url] if x]) or '(정보 없음)'}</td>
            </tr>
            <tr>
              <td style="padding:12px; background:#f9fafb; font-weight:600;">상세링크</td>
              <td style="padding:12px;">
                <a href="{link}" target="_blank" style="color:#4a90e2; text-decoration:none;">{link or '(정보 없음)'}</a>
              </td>
            </tr>
          </tbody>
        </table>

        <div style="padding: 16px 20px; font-size: 12px; color: #888; text-align: center; background: #f9fafb; border-top: 1px solid #eee;">
          이 메일은 {service_name}에서 발송되었습니다.
        </div>
      </div>
    </div>
    """
    return subject, html

# ---- 실제 발송 함수 ----
def send_html_email(
    to_addr: str,
    subject: str,
    html: str,
    sender_email: Optional[str] = None,
    app_password: Optional[str] = None,
    smtp_host: Optional[str] = None,
    smtp_port: Optional[int] = None,
    charset: str = "utf-8",
) -> None:
    """
    HTML 이메일 발송 (TLS)
    - 인자 값이 비어 있으면 환경변수 기본값 사용
    - 실패 시 예외 발생
    """
    sender_email = (sender_email or SENDER_EMAIL).strip()
    app_password = (app_password or APP_PASSWORD).strip()
    smtp_host = (smtp_host or SMTP_HOST).strip()
    smtp_port = int(smtp_port or SMTP_PORT)

    if not validate_email(to_addr):
        raise ValueError(f"수신자 이메일 형식이 올바르지 않습니다: {to_addr}")
    if not validate_email(sender_email):
        raise ValueError(f"발신자 이메일 형식이 올바르지 않습니다: {sender_email}")
    if not app_password:
        raise ValueError("앱 비밀번호(APP_PASSWORD)가 설정되지 않았습니다.")

    msg = MIMEText(html, "html", charset)
    msg["Subject"] = subject
    msg["From"] = sender_email
    msg["To"] = to_addr

    with smtplib.SMTP(smtp_host, smtp_port) as server:
        server.ehlo()
        server.starttls()
        server.login(sender_email, app_password)
        server.sendmail(sender_email, [to_addr], msg.as_string())

# ---- 외부 호출용 래퍼 (Agent state 연동) ----
def send_policy_email_oneoff_html(
    to_addr: str,
    policy: Mapping[str, Any],
    sender_email: Optional[str] = None,
    app_password: Optional[str] = None,
    state: Optional[Mapping[str, Any]] = None,
) -> None:
    """
    정책/공고 정보를 HTML로 만들어 단건 발송.
    - sender_email / app_password / SMTP_HOST / SMTP_PORT는 우선순위로 덮어씀:
      1) 함수 인자
      2) state["sender_email"], state["app_password"], state["smtp_host"], state["smtp_port"]
      3) 환경변수 기본값
    """
    subject, html = build_interest_email_html(policy)

    # state에서 override
    st_sender = (state or {}).get("sender_email")
    st_passwd = (state or {}).get("app_password")
    st_host   = (state or {}).get("smtp_host")
    st_port   = (state or {}).get("smtp_port")

    send_html_email(
        to_addr=to_addr,
        subject=subject,
        html=html,
        sender_email=sender_email or st_sender,
        app_password=app_password or st_passwd,
        smtp_host=st_host or None,     # None이면 env 기본값 사용
        smtp_port=st_port or None,
    )

# if __name__ == "__main__":
#     # ==== 테스트 실행 ====
#     RECEIVER_EMAIL = "kimchaehyoung@gmail.com"  # 하드코딩

#     # 예시 정책 데이터 (실제는 DB나 CSV에서 읽어오는 dict)
#     test_policy = {'index': 4,
#                    'title': '[경기] 2025년 팹리스 수요연계 양산지원 사업 양산성능평가 신규지원 대상 과제 공고',
#                    'region': '경기도',
#                    'amount': 300000000.0,
#                    'deadline': '2025-08-22',
#                    'link': 'https://mybiz.pay.naver.com/subvention/detail/bizInfo_PBLN_000000000113053',
#                    'agency': '차세대융합기술연구원',
#                    'business_overview': '경기도 시스템반도체 산업의 경쟁력 강화를 위해, 팹리스 기업의 수요처 연계 양산성능평가를 지원함으로써 신속한 국내 공급망 확보를 도모하고자 합니다. 이에 따라 「2025년 팹리스 수요연계 양산성능평가 지원사업」의 모집 계획을 다음과 같이 공고하오니, 사업 참여를 희망하는 도내 기업 및 기관은 절차에 따라 신청하여 주시기 바랍니다.',
#                    'eligibility_content': 'ㅇ 지원예산 : 과제당 3억원/년 내외 ㅇ 지원분야 - 팹리스 양산단계 핵심기술 분야 * 신청 시 기술 명기 및 분야명(소분류 등급) 기재 ㅇ 과제유형 - 양산성능평가 지원 (평가를 통해 최대 9개월 지원 예정) (자유 공모) * 팹리스 상용화 제품의 수요기업 연계를 통한 양산',
#                    'application_method': 'ㅇ 제출방법 : 전자문서(파일 1개로 압축, 제출서류_ㅇㅇㅇ 기관명) * 접수는 마감 시각 내 E-mail 도착해야 함(연구개발계획서 및 구비서류 전체) * 마감 시각 이후에 도착은 접수되지 않음 ㅇ 제 출 처 - 온라인 접수 : sshfish85@snu.ac.kr ㅇ 문 의 처 : 차세대융합기술연구원 경기도반도체혁신센터',
#                    'contact_info': '차세대융합기술연구원\n\n경기도반도체혁신센터\n\n031-888–9110\n\naict.snu.ac.kr',
#                    'source': '기업마당',
#                    'org_dept': '차세대융합기술연구원 경기도반도체혁신센터',
#                    'phone': '031-888–9110',
#                    'url': 'aict.snu.ac.kr'}

#     try:
#         send_policy_email_oneoff_html(
#             to_addr=RECEIVER_EMAIL,
#             policy=test_policy,
#             sender_email=SENDER_EMAIL,
#             app_password=APP_PASSWORD
#         )
#         print(f"✅ 메일 발송 완료 → {RECEIVER_EMAIL}")
#     except Exception as e:
#         print(f"❌ 메일 발송 실패: {e}")
