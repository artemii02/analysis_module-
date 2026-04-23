from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from interview_analysis.core.serialization import camel_to_snake, snake_to_camel
from interview_analysis.models import AssessmentRequest, ExecutionMode, Grade, ScenarioContext, SessionItem, Specialization


class CamelCaseCompatibleModel(BaseModel):
    model_config = ConfigDict(
        populate_by_name=True,
        alias_generator=snake_to_camel,
    )

    @model_validator(mode="before")
    @classmethod
    def normalize_camel_case_keys(cls, value: Any) -> Any:
        return _normalize_keys(value)


class ScenarioPayload(CamelCaseCompatibleModel):
    scenario_id: str | None = None
    specialization: Literal["backend", "frontend", "devops"]
    grade: Literal["junior", "middle"]
    topics: list[str] = Field(default_factory=list)
    report_language: Literal["ru", "en"] = "ru"


class SessionItemPayload(CamelCaseCompatibleModel):
    item_id: str = Field(min_length=1)
    question_id: str = Field(min_length=1)
    question_text: str = Field(min_length=1)
    answer_text: str = Field(min_length=1, max_length=4000)
    asked_at: str | None = None
    tags: list[str] = Field(default_factory=list)


class AssessmentRequestPayload(CamelCaseCompatibleModel):
    request_id: str = Field(min_length=1)
    session_id: str = Field(min_length=1)
    client_id: str = Field(min_length=1)
    mode: Literal["sync", "async"] = "sync"
    scenario: ScenarioPayload
    items: list[SessionItemPayload] = Field(min_length=1, max_length=20)
    metadata: dict[str, Any] = Field(default_factory=dict)

    def to_domain(self) -> AssessmentRequest:
        return AssessmentRequest(
            request_id=self.request_id,
            session_id=self.session_id,
            client_id=self.client_id,
            mode=ExecutionMode(self.mode),
            scenario=ScenarioContext(
                scenario_id=self.scenario.scenario_id,
                specialization=Specialization(self.scenario.specialization),
                grade=Grade(self.scenario.grade),
                topics=self.scenario.topics,
                report_language=self.scenario.report_language,
            ),
            items=[
                SessionItem(
                    item_id=item.item_id,
                    question_id=item.question_id,
                    question_text=item.question_text,
                    answer_text=item.answer_text,
                    asked_at=item.asked_at,
                    tags=item.tags,
                )
                for item in self.items
            ],
            metadata=self.metadata,
        )


class RetrievedKnowledgeChunkPayload(CamelCaseCompatibleModel):
    chunk_id: str
    source_title: str
    source_url: str
    excerpt: str
    score: float


class QuestionFeedbackPayload(CamelCaseCompatibleModel):
    item_id: str
    question_id: str
    question_text: str
    topic: str
    score: int
    criterion_scores: dict[str, int]
    summary: str
    strengths: list[str]
    issues: list[str]
    covered_keypoints: list[str]
    missing_keypoints: list[str]
    detected_mistakes: list[str]
    recommendations: list[str]
    context_snippets: list[RetrievedKnowledgeChunkPayload]


class TopicSummaryPayload(CamelCaseCompatibleModel):
    topic: str
    average_score: int
    strengths: list[str]
    gaps: list[str]


class VersionInfoPayload(CamelCaseCompatibleModel):
    model_config = ConfigDict(
        populate_by_name=True,
        alias_generator=snake_to_camel,
        protected_namespaces=(),
    )

    model_version: str
    rubric_version: str
    kb_version: str
    questions_version: str
    prompt_version: str


class AssessmentReportPayload(CamelCaseCompatibleModel):
    request_id: str
    session_id: str
    client_id: str
    specialization: str
    grade: str
    overall_score: int
    criterion_scores: dict[str, int]
    summary: str
    questions: list[QuestionFeedbackPayload]
    topics: list[TopicSummaryPayload]
    recommendations: list[str]
    versions: VersionInfoPayload
    generated_at: str


class JobPayload(CamelCaseCompatibleModel):
    status: Literal["created", "processing", "ready", "error"]
    job_id: str
    request_id: str
    session_id: str
    created_at: str
    updated_at: str
    error_code: str | None = None
    error_message: str | None = None


class ReportResponsePayload(CamelCaseCompatibleModel):
    status: Literal["ready"]
    job: JobPayload
    report: AssessmentReportPayload


class QuestionBankItemPayload(CamelCaseCompatibleModel):
    question_id: str
    specialization: str
    grade: str
    topic_code: str
    topic_label: str
    question_text: str
    tags: list[str]
    version: str


class QuestionBankResponsePayload(CamelCaseCompatibleModel):
    status: Literal["ok"] = "ok"
    specialization: str
    grade: str
    count: int
    items: list[QuestionBankItemPayload]


class HealthResponsePayload(CamelCaseCompatibleModel):
    status: Literal["ok"]
    service: str
    llm_mode: str
    llm_model: str
    api_prefix: str
    job_store: dict[str, str]


def _normalize_keys(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            _camel_to_snake(key): _normalize_keys(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_normalize_keys(item) for item in value]
    return value


def _camel_to_snake(value: str) -> str:
    return camel_to_snake(value)
