from pathlib import Path
import os
import re


def env(name: str, default: str = "") -> str:
    value = os.getenv(name, "").strip()
    return default if not value or value == name else value


def env_int(name: str, default: int, low: int, high: int) -> int:
    try:
        return max(low, min(high, int(env(name))))
    except ValueError:
        return default


def env_float(name: str, default: float) -> float:
    try:
        return max(0.0, min(2.0, float(env(name))))
    except ValueError:
        return default


DATA_DIR = Path(env("DATA_DIR", "data"))
API_BASE = (env("LLM_BASE_URL") or env("BASE_URL") or env("OPENAI_BASE_URL")).rstrip("/")
LLM_KEY = env("LLM_API_KEY") or env("OPENAI_API_KEY")
EMBED_KEY = env("EMBEDDER_API_KEY") or LLM_KEY
RERANK_KEY = env("RERANKER_API_KEY") or LLM_KEY

LLM_MODEL = env("LLM_MODEL") or env("GEN_MODEL") or env("OPENAI_MODEL", "deepseek/deepseek-v4.1-flash")
EMB_MODEL = env("EMB_MODEL", "baai/bge-m3")
RERANKER_MODEL = env("RERANKER_MODEL", "cohere/rerank-5-fast")

TOP_K = env_int("TOP_K", 4, 1, 8)
RERANKER_TOP_N = env_int("RERANKER_TOP_N", 2, 1, 5)
MAX_TOKENS = env_int("MAX_TOKENS", 1024, 128, 8192)
MAX_WORKERS = env_int("MAX_WORKERS", 2, 1, 8)
TEMPERATURE = env_float("TEMPERATURE", 0.0)
CHUNKER = env("CHUNKER", "common")

SNIPPET_CHARS = 1200
FALLBACK = "Не удалось получить ответ."
WORD_RE = re.compile(r"[0-9A-Za-zА-Яа-яЁё]{4,}")
