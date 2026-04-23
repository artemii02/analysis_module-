from __future__ import annotations

import json
from dataclasses import fields, is_dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from interview_analysis.models import AssessmentReport, QuestionFeedback, RetrievedKnowledgeChunk, TopicSummary, VersionInfo



def to_primitive(value: Any) -> Any:
    if is_dataclass(value):
        return {
            field.name: to_primitive(getattr(value, field.name))
            for field in fields(value)
        }
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {key: to_primitive(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [to_primitive(item) for item in value]
    return value



def to_canonical_json(value: Any) -> str:
    return json.dumps(
        to_primitive(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )



def to_camel_case_keys(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            snake_to_camel(str(key)): to_camel_case_keys(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [to_camel_case_keys(item) for item in value]
    return value


def camel_to_snake(value: str) -> str:
    if "_" in value:
        return value
    result: list[str] = []
    for index, char in enumerate(value):
        if char.isupper() and index > 0:
            result.append("_")
        result.append(char.lower())
    return "".join(result)


def snake_to_camel(value: str) -> str:
    if "_" not in value:
        return value
    head, *tail = value.split("_")
    return head + "".join(part.capitalize() for part in tail)


def utcnow_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()



def report_from_primitive(payload: dict[str, Any]) -> AssessmentReport:
    return AssessmentReport(
        request_id=payload["request_id"],
        session_id=payload["session_id"],
        client_id=payload["client_id"],
        specialization=payload["specialization"],
        grade=payload["grade"],
        overall_score=int(payload["overall_score"]),
        criterion_scores={key: int(value) for key, value in payload["criterion_scores"].items()},
        summary=payload["summary"],
        questions=[_question_feedback_from_primitive(item) for item in payload["questions"]],
        topics=[TopicSummary(**item) for item in payload["topics"]],
        recommendations=[str(item) for item in payload["recommendations"]],
        versions=VersionInfo(**payload["versions"]),
        generated_at=payload["generated_at"],
    )



def _question_feedback_from_primitive(payload: dict[str, Any]) -> QuestionFeedback:
    return QuestionFeedback(
        item_id=payload["item_id"],
        question_id=payload["question_id"],
        question_text=payload["question_text"],
        topic=payload["topic"],
        score=int(payload["score"]),
        criterion_scores={key: int(value) for key, value in payload["criterion_scores"].items()},
        summary=payload["summary"],
        strengths=[str(item) for item in payload["strengths"]],
        issues=[str(item) for item in payload["issues"]],
        covered_keypoints=[str(item) for item in payload["covered_keypoints"]],
        missing_keypoints=[str(item) for item in payload["missing_keypoints"]],
        detected_mistakes=[str(item) for item in payload["detected_mistakes"]],
        recommendations=[str(item) for item in payload["recommendations"]],
        context_snippets=[RetrievedKnowledgeChunk(**item) for item in payload["context_snippets"]],
    )
