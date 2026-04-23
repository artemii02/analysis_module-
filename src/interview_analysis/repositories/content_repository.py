from __future__ import annotations

import json
from pathlib import Path

from interview_analysis.core.topic_catalog import TOPIC_LABELS, topic_label
from interview_analysis.exceptions import UnknownQuestionError
from interview_analysis.models import (
    Grade,
    KnowledgeChunk,
    MistakePattern,
    QuestionDefinition,
    RubricCriterion,
    RubricDefinition,
    Specialization,
    VersionInfo,
)
from interview_analysis.services.preprocessor import significant_tokens


DEFAULT_CRITERIA = (
    ("correctness", 0.3, "Техническая корректность и отсутствие фактических ошибок."),
    ("completeness", 0.25, "Полнота раскрытия вопроса и охват ключевых аспектов."),
    ("clarity", 0.15, "Структурность, ясность формулировок и связность ответа."),
    ("practicality", 0.15, "Наличие практических примеров, ограничений и trade-offs."),
    ("terminology", 0.15, "Корректность и уместность технической терминологии."),
)
RUNTIME_TOPIC_STOPWORDS = {
    "такое",
    "какой",
    "какая",
    "какие",
    "каково",
    "чем",
    "между",
    "ними",
    "объясни",
    "объяснить",
    "разница",
    "различия",
    "отличия",
    "why",
    "what",
    "when",
    "where",
    "which",
    "difference",
    "between",
    "about",
}


class JSONContentRepository:
    def __init__(self, data_dir: Path) -> None:
        self.data_dir = data_dir
        self._questions, self._questions_version = self._load_questions()
        self._question_text_index = self._build_question_text_index()
        self._rubrics, self._rubrics_version = self._load_rubrics()
        self._knowledge_chunks, self._kb_version = self._load_knowledge()

    def _load_questions(self) -> tuple[dict[str, QuestionDefinition], str]:
        payload = json.loads((self.data_dir / "questions.json").read_text(encoding="utf-8-sig"))
        questions: dict[str, QuestionDefinition] = {}
        for item in payload["items"]:
            questions[item["question_id"]] = QuestionDefinition(
                question_id=item["question_id"],
                specialization=Specialization(item["specialization"]),
                grade=Grade(item["grade"]),
                topic=item["topic"],
                question_text=item["question_text"],
                tags=item.get("tags", []),
                version=item["version"],
            )
        return questions, payload["version"]

    def _load_rubrics(self) -> tuple[dict[str, RubricDefinition], str]:
        payload = json.loads((self.data_dir / "rubrics.json").read_text(encoding="utf-8-sig"))
        rubrics: dict[str, RubricDefinition] = {}
        for item in payload["items"]:
            rubrics[item["question_id"]] = RubricDefinition(
                question_id=item["question_id"],
                specialization=Specialization(item["specialization"]),
                grade=Grade(item["grade"]),
                topic=item["topic"],
                criteria=[
                    RubricCriterion(
                        name=criterion["name"],
                        weight=criterion["weight"],
                        description=criterion["description"],
                    )
                    for criterion in item["criteria"]
                ],
                keypoints=item["keypoints"],
                recommendation_hints=item.get("recommendation_hints", []),
                mistake_patterns=[
                    MistakePattern(
                        trigger_terms=pattern["trigger_terms"],
                        message=pattern["message"],
                    )
                    for pattern in item.get("mistake_patterns", [])
                ],
                version=item["version"],
            )
        return rubrics, payload["version"]

    def _load_knowledge(self) -> tuple[list[KnowledgeChunk], str]:
        payload = json.loads((self.data_dir / "knowledge_base.json").read_text(encoding="utf-8-sig"))
        chunks: list[KnowledgeChunk] = []
        for item in payload["items"]:
            chunks.append(
                KnowledgeChunk(
                    chunk_id=item["chunk_id"],
                    specialization=Specialization(item["specialization"]),
                    topics=item["topics"],
                    tags=item.get("tags", []),
                    content=item["content"],
                    source_title=item["source_title"],
                    source_url=item["source_url"],
                    version=item["version"],
                )
            )
        return chunks, payload["version"]

    def get_question(
        self,
        question_id: str,
        specialization: Specialization,
        grade: Grade,
    ) -> QuestionDefinition:
        question = self._questions.get(question_id)
        if question is None:
            raise UnknownQuestionError(question_id)
        if question.specialization != specialization or question.grade != grade:
            raise UnknownQuestionError(question_id)
        return question

    def resolve_question(
        self,
        question_id: str,
        question_text: str,
        specialization: Specialization,
        grade: Grade,
    ) -> QuestionDefinition:
        question = self._questions.get(question_id)
        if question is not None and question.specialization == specialization and question.grade == grade:
            return question

        normalized_text = _normalize_question_text(question_text)
        resolved_question = self._question_text_index.get((specialization, grade, normalized_text))
        if resolved_question is not None:
            return resolved_question
        raise UnknownQuestionError(question_id)

    def build_runtime_question(
        self,
        question_id: str,
        question_text: str,
        specialization: Specialization,
        grade: Grade,
        scenario_topics: list[str] | None = None,
        tags: list[str] | None = None,
    ) -> QuestionDefinition:
        runtime_tags = _build_runtime_tags(question_text, scenario_topics or [], tags or [])
        topic_code = _infer_runtime_topic_code(question_text, scenario_topics or [], tags or [])
        return QuestionDefinition(
            question_id=question_id,
            specialization=specialization,
            grade=grade,
            topic=topic_code,
            question_text=question_text,
            tags=runtime_tags,
            version="external-backend-question-v1",
        )

    def build_runtime_rubric(
        self,
        question: QuestionDefinition,
        scenario_topics: list[str] | None = None,
    ) -> RubricDefinition:
        keypoints = _build_runtime_keypoints(question.question_text, question.topic)
        recommendation_hints = _build_runtime_recommendation_hints(question.question_text, question.topic)
        return RubricDefinition(
            question_id=question.question_id,
            specialization=question.specialization,
            grade=question.grade,
            topic=question.topic,
            criteria=[
                RubricCriterion(name=name, weight=weight, description=description)
                for name, weight, description in DEFAULT_CRITERIA
            ],
            keypoints=keypoints,
            recommendation_hints=recommendation_hints,
            mistake_patterns=[],
            version="runtime-rubric-v1",
        )

    def get_rubric(
        self,
        question_id: str,
        specialization: Specialization,
        grade: Grade,
    ) -> RubricDefinition:
        rubric = self._rubrics.get(question_id)
        if rubric is None:
            raise UnknownQuestionError(question_id)
        if rubric.specialization != specialization or rubric.grade != grade:
            raise UnknownQuestionError(question_id)
        return rubric

    def list_questions(
        self,
        specialization: Specialization,
        grade: Grade,
        limit: int | None = None,
    ) -> list[QuestionDefinition]:
        questions = [
            question
            for question in self._questions.values()
            if question.specialization == specialization and question.grade == grade
        ]
        questions.sort(key=lambda item: item.question_id)
        if limit is None:
            return questions
        return questions[:limit]

    def list_knowledge_chunks(self, specialization: Specialization) -> list[KnowledgeChunk]:
        return [
            chunk
            for chunk in self._knowledge_chunks
            if chunk.specialization == specialization
        ]

    def build_version_info(
        self,
        model_version: str,
        prompt_version: str,
        *,
        rubric_version_override: str | None = None,
        questions_version_override: str | None = None,
    ) -> VersionInfo:
        return VersionInfo(
            model_version=model_version,
            rubric_version=rubric_version_override or self._rubrics_version,
            kb_version=self._kb_version,
            questions_version=questions_version_override or self._questions_version,
            prompt_version=prompt_version,
        )

    def _build_question_text_index(self) -> dict[tuple[Specialization, Grade, str], QuestionDefinition]:
        index: dict[tuple[Specialization, Grade, str], QuestionDefinition] = {}
        for question in self._questions.values():
            index[
                (
                    question.specialization,
                    question.grade,
                    _normalize_question_text(question.question_text),
                )
            ] = question
        return index


def _normalize_question_text(value: str) -> str:
    return " ".join(value.strip().lower().replace("ё", "е").split())


def _infer_runtime_topic_code(question_text: str, scenario_topics: list[str], tags: list[str]) -> str:
    normalized_question = _normalize_question_text(question_text)
    normalized_candidates = [
        _normalize_topic_code(candidate)
        for candidate in [*tags, *scenario_topics]
        if candidate and candidate.strip()
    ]
    for candidate in normalized_candidates:
        if candidate in normalized_question.replace("-", " "):
            return candidate

    question_tokens = _runtime_topic_tokens(question_text)
    for topic_code, label in TOPIC_LABELS.items():
        label_tokens = set(significant_tokens(label))
        topic_tokens = set(significant_tokens(topic_code.replace("_", " ")))
        if len(set(question_tokens) & (label_tokens | topic_tokens)) >= 2:
            return topic_code

    if question_tokens:
        return "_".join(question_tokens[:3])
    return "external_question"


def _build_runtime_tags(question_text: str, scenario_topics: list[str], tags: list[str]) -> list[str]:
    result: list[str] = []
    for value in [*tags, *scenario_topics, *_runtime_topic_tokens(question_text)[:8]]:
        cleaned = value.strip() if isinstance(value, str) else ""
        if cleaned and cleaned not in result:
            result.append(cleaned)
    return result


def _build_runtime_keypoints(question_text: str, topic_code: str) -> list[str]:
    normalized_question = _normalize_question_text(question_text)
    display_topic = topic_label(topic_code)
    keypoints = ['Дать прямой и технически корректный ответ по существу вопроса.']

    if any(marker in normalized_question for marker in ('разница', 'отлич', 'сравни', 'сравнить', 'versus', 'vs')):
        keypoints.append('Ясно сравнить варианты и объяснить ключевые различия между ними.')
    elif any(marker in normalized_question for marker in ('что такое', 'что это', 'что означает')):
        keypoints.append('Дать определение термину и пояснить его назначение.')
    elif any(marker in normalized_question for marker in ('зачем', 'для чего', 'когда используют', 'когда нужен')):
        keypoints.append('Пояснить, в каких сценариях это применяется и какую задачу решает.')
    elif any(marker in normalized_question for marker in ('как', 'каким образом', 'объясни процесс', 'как работает')):
        keypoints.append('Описать механизм работы и ключевые шаги или компоненты.')
    else:
        keypoints.append(f"Раскрыть базовые аспекты темы '{display_topic}'.")

    keypoints.append('Использовать корректную техническую терминологию и не путать связанные понятия.')
    keypoints.append('Привести практический пример, ограничение или trade-off.')
    return keypoints[:4]


def _build_runtime_recommendation_hints(question_text: str, topic_code: str) -> list[str]:
    display_topic = topic_label(topic_code)
    hints = [
        f"Повторить тему '{display_topic}' и базовые определения по ней.",
        'Добавить больше технической конкретики, а не ограничиваться общими формулировками.',
        'Подкреплять ответ коротким практическим примером или сценарием применения.',
    ]
    normalized_question = _normalize_question_text(question_text)
    if any(marker in normalized_question for marker in ('разница', 'отлич', 'сравни', 'сравнить')):
        hints.insert(1, 'Отдельно проговорить различия, плюсы, минусы и типичные сценарии выбора.')
    return hints[:4]


def _normalize_topic_code(value: str) -> str:
    return "_".join(_normalize_question_text(value).split())


def _runtime_topic_tokens(question_text: str) -> list[str]:
    return [
        token
        for token in significant_tokens(question_text)
        if token not in RUNTIME_TOPIC_STOPWORDS
    ]
