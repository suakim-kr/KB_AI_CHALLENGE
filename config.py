from pathlib import Path
import os
from dotenv import load_dotenv

load_dotenv()

# 현재 config.py 파일이 있는 디렉토리
BASE_DIR = Path(__file__).resolve().parent

# 데이터 경로
POLICY_CSV = os.getenv("POLICY_CSV", str(BASE_DIR / "data" / "930_preprocessed.csv"))

# 프롬프트 경로
PROMPTS_DIR = BASE_DIR / "prompts"
RAG_PROMPT_FILE = PROMPTS_DIR / "rag_prompt.txt"
T2SQL_PROMPT_FILE = PROMPTS_DIR / "t2sql_prompt.txt"

# LLM 설정
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
GEN_MODEL = os.getenv("GEN_MODEL", "gemini-2.5-flash")

LOCAL_TZ = "Asia/Seoul"
RAG_TOP_K_DEFAULT = int(os.getenv("RAG_TOP_K_DEFAULT", "5"))
