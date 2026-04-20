from __future__ import annotations


class AnalysisError(Exception):
    def __init__(
        self,
        code: str,
        message: str,
        status_code: int = 400,
        details: dict | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.status_code = status_code
        self.details = details or {}


class AuthenticationError(AnalysisError):
    def __init__(self, message: str = "Неверный API-ключ.") -> None:
        super().__init__("AUTHENTICATION_ERROR", message, status_code=401)


class InvalidInputError(AnalysisError):
    def __init__(self, message: str, details: dict | None = None) -> None:
        super().__init__("INVALID_INPUT", message, status_code=422, details=details)


class ConflictError(AnalysisError):
    def __init__(self, message: str, details: dict | None = None) -> None:
        super().__init__("REQUEST_CONFLICT", message, status_code=409, details=details)


class UnknownQuestionError(AnalysisError):
    def __init__(self, question_id: str) -> None:
        super().__init__(
            "UNKNOWN_QUESTION",
            f"Вопрос '{question_id}' отсутствует или недоступен для выбранного профиля.",
            status_code=404,
            details={"question_id": question_id},
        )


class IntegrationError(AnalysisError):
    def __init__(self, message: str, code: str = "INTEGRATION_ERROR", details: dict | None = None) -> None:
        super().__init__(code, message, status_code=502, details=details)


class ReportNotReadyError(AnalysisError):
    def __init__(self, job_id: str) -> None:
        super().__init__(
            "REPORT_NOT_READY",
            f"Отчёт для задачи '{job_id}' ещё не готов.",
            status_code=409,
            details={"job_id": job_id},
        )


class JobNotFoundError(AnalysisError):
    def __init__(self, job_id: str) -> None:
        super().__init__(
            "JOB_NOT_FOUND",
            f"Задача '{job_id}' не найдена.",
            status_code=404,
            details={"job_id": job_id},
        )
