from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path


PACKAGE_DIR = Path(__file__).resolve().parents[1]
ROOT_DIR = PACKAGE_DIR.parents[1]


@dataclass(frozen=True, slots=True)
class Settings:
    app_name: str
    api_prefix: str
    api_key: str
    log_level: str
    llm_mode: str
    job_store_backend: str
    database_url: str
    ollama_url: str
    ollama_model: str
    hf_base_model: str
    hf_adapter_path: Path | None
    hf_device: str
    hf_load_in_4bit: bool
    hf_max_new_tokens: int
    hf_batch_size: int
    hf_retry_max_new_tokens: int
    hf_repair_max_new_tokens: int
    warmup_llm_on_start: bool
    request_timeout_seconds: int
    knowledge_limit: int
    hard_timeout_seconds: int
    max_answer_length: int
    max_session_items: int
    data_dir: Path
    prompt_path: Path
    db_schema_path: Path
    demo_cases_path: Path
    demo_template_path: Path


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings(
        app_name="Interview Coach Analysis Module",
        api_prefix="/assessment/v1",
        api_key=os.getenv("ANALYSIS_API_KEY", "demo-api-key"),
        log_level=os.getenv("ANALYSIS_LOG_LEVEL", "INFO").strip().upper(),
        llm_mode=os.getenv("ANALYSIS_LLM_MODE", "mock").strip().lower(),
        job_store_backend=os.getenv("ANALYSIS_JOB_STORE_BACKEND", "memory").strip().lower(),
        database_url=os.getenv(
            "ANALYSIS_DATABASE_URL",
            "postgresql://analysis:analysis@localhost:5432/analysis_module",
        ),
        ollama_url=os.getenv("ANALYSIS_OLLAMA_URL", "http://localhost:11434/api/generate"),
        ollama_model=os.getenv("ANALYSIS_OLLAMA_MODEL", "qwen2.5:3b"),
        hf_base_model=os.getenv("ANALYSIS_HF_BASE_MODEL", "Qwen/Qwen2.5-3B-Instruct"),
        hf_adapter_path=_optional_path(os.getenv("ANALYSIS_HF_ADAPTER_PATH", "training/artifacts/qwen2.5-3b-interview-full-ru-qlora-v1")),
        hf_device=os.getenv("ANALYSIS_HF_DEVICE", "auto").strip().lower(),
        hf_load_in_4bit=_env_bool("ANALYSIS_HF_LOAD_IN_4BIT", False),
        hf_max_new_tokens=int(os.getenv("ANALYSIS_HF_MAX_NEW_TOKENS", "220")),
        hf_batch_size=int(os.getenv("ANALYSIS_HF_BATCH_SIZE", "3")),
        hf_retry_max_new_tokens=int(os.getenv("ANALYSIS_HF_RETRY_MAX_NEW_TOKENS", "320")),
        hf_repair_max_new_tokens=int(os.getenv("ANALYSIS_HF_REPAIR_MAX_NEW_TOKENS", "220")),
        warmup_llm_on_start=_env_bool("ANALYSIS_WARMUP_LLM_ON_START", False),
        request_timeout_seconds=int(os.getenv("ANALYSIS_REQUEST_TIMEOUT_SECONDS", "300")),
        knowledge_limit=int(os.getenv("ANALYSIS_KNOWLEDGE_LIMIT", "1")),
        hard_timeout_seconds=int(os.getenv("ANALYSIS_HARD_TIMEOUT_SECONDS", "30")),
        max_answer_length=int(os.getenv("ANALYSIS_MAX_ANSWER_LENGTH", "4000")),
        max_session_items=int(os.getenv("ANALYSIS_MAX_SESSION_ITEMS", "20")),
        data_dir=PACKAGE_DIR / "data",
        prompt_path=PACKAGE_DIR / "prompts" / "report_system_prompt.txt",
        db_schema_path=PACKAGE_DIR / "db" / "schema.sql",
        demo_cases_path=PACKAGE_DIR / "demo" / "demo_cases.json",
        demo_template_path=PACKAGE_DIR / "demo" / "demo.html",
    )


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _optional_path(value: str | None) -> Path | None:
    if value is None or not value.strip():
        return None
    path = Path(value.strip())
    if path.is_absolute():
        return path
    return ROOT_DIR / path
