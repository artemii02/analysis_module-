from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

from interview_analysis.core.config import Settings
from interview_analysis.core.serialization import to_canonical_json
from interview_analysis.exceptions import (
    AnalysisError,
    IntegrationError,
    InvalidInputError,
    ReportNotReadyError,
)
from interview_analysis.models import AssessmentJob, AssessmentReport, JobStatus
from interview_analysis.services.analysis_pipeline import AnalysisPipeline
from interview_analysis.services.metrics import MetricsRegistry


@dataclass(slots=True)
class SubmissionResult:
    job: AssessmentJob
    created_new: bool


class AssessmentService:
    def __init__(
        self,
        pipeline: AnalysisPipeline,
        job_store,
        metrics: MetricsRegistry,
        settings: Settings,
    ) -> None:
        self.pipeline = pipeline
        self.job_store = job_store
        self.metrics = metrics
        self.settings = settings

    def register_request(self, request) -> SubmissionResult:
        self._validate_request(request)
        self.metrics.record_submission(request.mode.value)
        fingerprint = to_canonical_json(request)
        job, created_new = self.job_store.register(
            request_id=request.request_id,
            session_id=request.session_id,
            fingerprint=fingerprint,
        )
        return SubmissionResult(job=job, created_new=created_new)

    def process_sync(self, job_id: str, request) -> AssessmentReport:
        job = self.job_store.get(job_id)
        if job.status == JobStatus.READY and job.report is not None:
            return job.report
        return self._execute(job_id, request)

    def process_async(self, job_id: str, request) -> None:
        try:
            self._execute(job_id, request)
        except AnalysisError:
            return

    def get_job(self, job_id: str) -> AssessmentJob:
        return self.job_store.get(job_id)

    def get_report(self, job_id: str) -> AssessmentReport:
        job = self.job_store.get(job_id)
        if job.status == JobStatus.READY and job.report is not None:
            return job.report
        if job.status == JobStatus.ERROR:
            raise IntegrationError(
                job.error_message or "Не удалось сформировать отчёт.",
                code=job.error_code or "INTERNAL_ERROR",
            )
        raise ReportNotReadyError(job_id)

    def metrics_snapshot(self) -> dict[str, int | float]:
        return self.metrics.snapshot()

    def health_snapshot(self) -> dict[str, str]:
        if hasattr(self.job_store, "healthcheck"):
            return self.job_store.healthcheck()
        return {
            "backend": "unknown",
            "status": "ok",
        }

    def _execute(self, job_id: str, request) -> AssessmentReport:
        started = perf_counter()
        self.job_store.mark_processing(job_id)
        try:
            report = self.pipeline.analyze(request)
        except AnalysisError as exc:
            self.job_store.mark_error(job_id, exc.code, exc.message)
            self.metrics.record_failure(_duration_ms(started))
            raise
        except Exception as exc:
            self.job_store.mark_error(job_id, "INTERNAL_ERROR", str(exc))
            self.metrics.record_failure(_duration_ms(started))
            raise IntegrationError("Внутренняя ошибка при формировании отчёта.") from exc

        self.job_store.mark_ready(job_id, report)
        self.metrics.record_success(_duration_ms(started))
        return report

    def _validate_request(self, request) -> None:
        if not request.request_id or not request.session_id or not request.client_id:
            raise InvalidInputError("Поля request_id, session_id и client_id обязательны.")
        if not request.items:
            raise InvalidInputError("Сессия должна содержать хотя бы один вопрос с ответом.")
        if len(request.items) > self.settings.max_session_items:
            raise InvalidInputError(
                "Превышено максимальное количество вопросов в одной сессии.",
                details={"max_session_items": self.settings.max_session_items},
            )
        for item in request.items:
            if not item.answer_text.strip():
                raise InvalidInputError(
                    "Текст ответа не должен быть пустым.",
                    details={"item_id": item.item_id, "question_id": item.question_id},
                )
            if len(item.answer_text) > self.settings.max_answer_length:
                raise InvalidInputError(
                    "Текст ответа превышает допустимую длину.",
                    details={
                        "item_id": item.item_id,
                        "question_id": item.question_id,
                        "max_answer_length": self.settings.max_answer_length,
                    },
                )


def _duration_ms(started_at: float) -> int:
    return round((perf_counter() - started_at) * 1000)
