from pathlib import Path
from .config import RAG_PROMPT_FILE, T2SQL_PROMPT_FILE

def read_text_file(path_like: str | Path) -> str:
    p = Path(path_like)
    if not p.exists():
        raise FileNotFoundError(f"파일을 찾을 수 없습니다: {p}")
    return p.read_text(encoding="utf-8").strip()

def load_rag_prompt() -> str:
    return read_text_file(RAG_PROMPT_FILE)

def load_t2sql_prompt() -> str:
    return read_text_file(T2SQL_PROMPT_FILE)
