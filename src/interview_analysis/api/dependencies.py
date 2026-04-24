from __future__ import annotations

from functools import lru_cache

from fastapi import Header

from interview_analysis.core.config import Settings, get_settings
from interview_analysis.exceptions import AuthenticationError, IntegrationError
from interview_analysis.repositories.content_repository import JSONContentRepository
from interview_analysis.repositories.job_store import InMemoryAssessmentJobStore
from interview_analysis.services.analysis_pipeline import AnalysisPipeline
from interview_analysis.services.assessment_service import AssessmentService
from interview_analysis.services.llm.mock_provider import MockLLMProvider
from interview_analysis.services.llm.ollama_provider import OllamaLLMProvider
from interview_analysis.services.metrics import MetricsRegistry
from interview_analysis.services.report_builder import ReportBuilder
from interview_analysis.services.retrieval import SimpleKnowledgeRetriever


@lru_cache(maxsize=1)
def get_service() -> AssessmentService:
    settings = get_settings()
    repository = JSONContentRepository(settings.data_dir)
    retriever = SimpleKnowledgeRetriever(repository, settings.knowledge_limit)
    llm_provider = _build_llm_provider(settings)
    pipeline = AnalysisPipeline(repository, retriever, llm_provider, ReportBuilder())
    return AssessmentService(
        pipeline=pipeline,
        job_store=_build_job_store(settings),
        metrics=MetricsRegistry(),
        settings=settings,
    )


def _build_llm_provider(settings: Settings):
    if settings.llm_mode == "ollama":
        return OllamaLLMProvider(
            url=settings.ollama_url,
            model=settings.ollama_model,
            prompt_path=settings.prompt_path,
            timeout_seconds=settings.request_timeout_seconds,
            fallback_to_grounded=settings.llm_fallback_to_grounded,
        )
    if settings.llm_mode == "hf":
        from interview_analysis.services.llm.hf_provider import HFLLMProvider

        return HFLLMProvider(
            base_model=settings.hf_base_model,
            adapter_path=settings.hf_adapter_path,
            device=settings.hf_device,
            max_new_tokens=settings.hf_max_new_tokens,
            load_in_4bit=settings.hf_load_in_4bit,
            batch_size=settings.hf_batch_size,
            retry_max_new_tokens=settings.hf_retry_max_new_tokens,
            repair_max_new_tokens=settings.hf_repair_max_new_tokens,
            fallback_to_grounded=settings.llm_fallback_to_grounded,
            disable_on_cpu=settings.disable_llm_on_cpu,
        )
    return MockLLMProvider()


def _build_job_store(settings: Settings):
    if settings.job_store_backend == "postgres":
        try:
            from interview_analysis.repositories.postgres_job_store import PostgresAssessmentJobStore
        except ImportError as exc:  # pragma: no cover - optional dependency branch
            raise IntegrationError(
                "PostgreSQL backend was requested, but psycopg is not installed.",
                code="DATABASE_DRIVER_MISSING",
            ) from exc
        return PostgresAssessmentJobStore(
            dsn=settings.database_url,
            schema_path=settings.db_schema_path,
        )
    return InMemoryAssessmentJobStore()


def verify_api_key(x_api_key: str | None = Header(default=None, alias="X-API-Key")) -> None:
    settings = get_settings()
    if settings.api_key and x_api_key != settings.api_key:
        raise AuthenticationError()
