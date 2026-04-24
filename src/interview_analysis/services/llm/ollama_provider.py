from __future__ import annotations

import json
import logging
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from interview_analysis.core.topic_catalog import topic_label
from interview_analysis.exceptions import IntegrationError
from interview_analysis.models import QuestionAnalysisContext, QuestionAssessment, RubricCriterion
from interview_analysis.services.grounded_assessment import build_grounded_assessment, should_skip_llm


logger = logging.getLogger(__name__)


class OllamaLLMProvider:
    prompt_version = "ollama-json-ru-v6"
    batch_size = 3
    single_max_tokens = 420
    single_retry_max_tokens = 260
    repair_max_tokens = 320
    batch_min_tokens = 700
    batch_tokens_per_item = 200

    def __init__(
        self,
        url: str,
        model: str,
        prompt_path: Path,
        timeout_seconds: int,
        fallback_to_grounded: bool = True,
    ) -> None:
        self.url = url
        self.model = model
        self.model_version = model
        self.prompt_template = prompt_path.read_text(encoding="utf-8")
        self.timeout_seconds = timeout_seconds
        self.fallback_to_grounded = fallback_to_grounded

    def assess(self, context: QuestionAnalysisContext) -> QuestionAssessment:
        prompt = _render_prompt(
            self.prompt_template,
            report_language=context.scenario.report_language,
            specialization=context.scenario.specialization.value,
            grade=context.scenario.grade.value,
            topic_code=context.question.topic,
            topic=topic_label(context.question.topic),
            question_text=context.session_item.question_text,
            answer_text=context.session_item.answer_text,
            criteria=json.dumps(
                [
                    {"name": criterion.name, "weight": criterion.weight, "description": criterion.description}
                    for criterion in context.rubric.criteria
                ],
                ensure_ascii=False,
            ),
            keypoints=json.dumps(context.rubric.keypoints, ensure_ascii=False),
            recommendation_hints=json.dumps(context.rubric.recommendation_hints, ensure_ascii=False),
            mistake_patterns=json.dumps(
                [
                    {"trigger_terms": pattern.trigger_terms, "message": pattern.message}
                    for pattern in context.rubric.mistake_patterns
                ],
                ensure_ascii=False,
            ),
            context_snippets=json.dumps([snippet.excerpt for snippet in context.retrieved_chunks], ensure_ascii=False),
        )
        grounded = build_grounded_assessment(context)
        if should_skip_llm(context):
            return grounded
        try:
            schema = _single_assessment_schema()
            content = self._generate_single(prompt, schema)
            parsed = self._parse_or_repair(content, schema)
            assessment = _build_assessment(parsed, context)
            return assessment
        except IntegrationError as exc:
            if not self._should_fallback(exc):
                raise
            logger.warning(
                'ollama.assess.fallback_grounded item_id=%s question_id=%s code=%s',
                context.session_item.item_id,
                context.session_item.question_id,
                exc.code,
            )
            return grounded

    def assess_batch(self, contexts: list[QuestionAnalysisContext]) -> list[QuestionAssessment]:
        if not contexts:
            return []
        if len(contexts) == 1:
            return [self.assess(contexts[0])]
        if len(contexts) > self.batch_size:
            assessments: list[QuestionAssessment] = []
            for start in range(0, len(contexts), self.batch_size):
                assessments.extend(self.assess_batch(contexts[start : start + self.batch_size]))
            return assessments

        prompt = _build_batch_prompt(contexts)
        try:
            content = self._generate(
                prompt,
                max_tokens=_batch_max_tokens(self, len(contexts)),
                response_format=_batch_assessment_schema(),
            )
            parsed = _parse_llm_json(content)
            return _build_batch_assessments(parsed, contexts)
        except IntegrationError as exc:
            if exc.code in {"MODEL_TIMEOUT", "INVALID_MODEL_OUTPUT"} and len(contexts) > 1:
                midpoint = max(1, len(contexts) // 2)
                return self.assess_batch(contexts[:midpoint]) + self.assess_batch(contexts[midpoint:])
            raise

    def _generate_single(self, prompt: str, schema: dict) -> str:
        try:
            return self._generate(
                prompt,
                max_tokens=self.single_max_tokens,
                response_format=schema,
            )
        except IntegrationError as exc:
            if exc.code != "MODEL_TIMEOUT":
                raise
            return self._generate(
                prompt,
                max_tokens=self.single_retry_max_tokens,
                response_format=schema,
            )

    def _parse_or_repair(self, content: str, schema: dict) -> dict:
        try:
            return _parse_llm_json(content)
        except IntegrationError as exc:
            if exc.code != "INVALID_MODEL_OUTPUT":
                raise
            repaired = self._generate(
                _build_repair_prompt(content, schema),
                max_tokens=self.repair_max_tokens,
                response_format=schema,
            )
            return _parse_llm_json(repaired)

    def _generate(self, prompt: str, max_tokens: int, response_format: str | dict = "json") -> str:
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "format": response_format,
            "options": {
                "temperature": 0,
                "num_predict": max_tokens,
            },
        }
        request = Request(
            self.url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:
                raw_payload = json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            raise IntegrationError(
                "Ошибка обращения к Ollama по HTTP.",
                code="MODEL_HTTP_ERROR",
                details={"status_code": exc.code},
            ) from exc
        except URLError as exc:
            raise IntegrationError(
                "Не удалось подключиться к локальному сервису Ollama.",
                code="MODEL_UNAVAILABLE",
                details={"reason": str(exc.reason)},
            ) from exc
        except TimeoutError as exc:
            raise IntegrationError("Модель превысила время ожидания ответа.", code="MODEL_TIMEOUT") from exc

        return str(raw_payload.get("response", ""))

    def _should_fallback(self, exc: IntegrationError) -> bool:
        if not self.fallback_to_grounded:
            return False
        return exc.code in {
            "INVALID_MODEL_OUTPUT",
            "MODEL_TIMEOUT",
            "MODEL_UNAVAILABLE",
            "MODEL_HTTP_ERROR",
        }


def _batch_max_tokens(provider: OllamaLLMProvider, item_count: int) -> int:
    return max(provider.batch_min_tokens, item_count * provider.batch_tokens_per_item)



def _criterion_scores_schema() -> dict:
    return {
        "type": "object",
        "properties": {
            "correctness": {"type": "integer"},
            "completeness": {"type": "integer"},
            "clarity": {"type": "integer"},
            "practicality": {"type": "integer"},
            "terminology": {"type": "integer"},
        },
        "required": ["correctness", "completeness", "clarity", "practicality", "terminology"],
    }



def _single_assessment_schema() -> dict:
    return {
        "type": "object",
        "properties": {
            "criterion_scores": _criterion_scores_schema(),
            "summary": {"type": "string"},
            "strengths": {"type": "array", "items": {"type": "string"}},
            "issues": {"type": "array", "items": {"type": "string"}},
            "covered_keypoints": {"type": "array", "items": {"type": "string"}},
            "missing_keypoints": {"type": "array", "items": {"type": "string"}},
            "detected_mistakes": {"type": "array", "items": {"type": "string"}},
            "recommendations": {"type": "array", "items": {"type": "string"}},
        },
        "required": [
            "criterion_scores",
            "summary",
            "strengths",
            "issues",
            "covered_keypoints",
            "missing_keypoints",
            "detected_mistakes",
            "recommendations",
        ],
    }



def _batch_assessment_schema() -> dict:
    single_schema = _single_assessment_schema()
    item_schema = {
        "type": "object",
        "properties": {
            "item_id": {"type": "string"},
            **single_schema["properties"],
        },
        "required": ["item_id", *single_schema["required"]],
    }
    return {
        "type": "object",
        "properties": {
            "items": {
                "type": "array",
                "items": item_schema,
            }
        },
        "required": ["items"],
    }



def _build_repair_prompt(raw_content: str, schema: dict) -> str:
    instructions = [
        "Return only a valid JSON object that matches the provided schema.",
        "Do not add markdown, explanations or comments.",
        f"Schema: {json.dumps(schema, ensure_ascii=False)}",
        f"Broken content to repair: {raw_content}",
    ]
    return "\n".join(instructions)



def _build_batch_prompt(contexts: list[QuestionAnalysisContext]) -> str:
    report_language = contexts[0].scenario.report_language if contexts else "ru"
    specialization = contexts[0].scenario.specialization.value if contexts else "backend"
    grade = contexts[0].scenario.grade.value if contexts else "junior"
    payload = []
    for context in contexts:
        payload.append(
            {
                "item_id": context.session_item.item_id,
                "question_id": context.session_item.question_id,
                "topic_code": context.question.topic,
                "topic": topic_label(context.question.topic),
                "question_text": context.session_item.question_text,
                "answer_text": context.session_item.answer_text,
                "criteria": [
                    {"name": criterion.name, "weight": criterion.weight, "description": criterion.description}
                    for criterion in context.rubric.criteria
                ],
                "keypoints": context.rubric.keypoints,
                "recommendation_hints": context.rubric.recommendation_hints,
                "mistake_patterns": [
                    {"trigger_terms": pattern.trigger_terms, "message": pattern.message}
                    for pattern in context.rubric.mistake_patterns
                ],
                "context_snippets": [snippet.excerpt for snippet in context.retrieved_chunks],
            }
        )

    instructions = [
        "You assess multiple interview answers and must return only a valid JSON object.",
        "Never return markdown or any text outside JSON.",
        "",
        "Use this exact schema:",
        "{",
        '  "items": [',
        "    {",
        '      "item_id": "string",',
        '      "criterion_scores": {"correctness": 0, "completeness": 0, "clarity": 0, "practicality": 0, "terminology": 0},',
        '      "summary": "short summary",',
        '      "strengths": ["..."],',
        '      "issues": ["..."],',
        '      "covered_keypoints": ["..."],',
        '      "missing_keypoints": ["..."],',
        '      "detected_mistakes": ["..."],',
        '      "recommendations": ["..."]',
        "    }",
        "  ]",
        "}",
        "",
        f"Requested report language: {report_language}. If it is ru, every textual JSON field must be in Russian.",
        f"Specialization: {specialization}. Grade: {grade}.",
        "For each item, fill criterion_scores strictly for correctness, completeness, clarity, practicality, terminology.",
        "Use criteria weights and descriptions provided inside each item.",
        "Be concrete and technically strict. Do not praise the candidate.",
        "Keep output concise: summary up to 2 sentences, strengths/issues/recommendations up to 3 items, covered_keypoints and missing_keypoints up to 3 items.",
        f"Items to assess: {json.dumps(payload, ensure_ascii=False)}",
    ]
    return "\n".join(instructions)



def _build_batch_assessments(parsed: dict, contexts: list[QuestionAnalysisContext]) -> list[QuestionAssessment]:
    items = parsed.get("items")
    if not isinstance(items, list):
        raise IntegrationError(
            "Ответ модели не содержит список items.",
            code="INVALID_MODEL_OUTPUT",
        )
    items_by_id = {}
    for item in items:
        if isinstance(item, dict) and "item_id" in item:
            items_by_id[str(item["item_id"])] = item

    assessments: list[QuestionAssessment] = []
    for context in contexts:
        item_id = context.session_item.item_id
        if item_id not in items_by_id:
            raise IntegrationError(
                "Ответ модели не содержит оценку для одного из вопросов.",
                code="INVALID_MODEL_OUTPUT",
                details={"item_id": item_id},
            )
        assessments.append(_build_assessment(items_by_id[item_id], context))
    return assessments



def _build_assessment(parsed: dict, context: QuestionAnalysisContext) -> QuestionAssessment:
    criterion_scores = _coerce_criterion_scores(parsed.get("criterion_scores", {}), context.rubric.criteria)
    return QuestionAssessment(
        score=_weighted_score(criterion_scores, context.rubric.criteria),
        criterion_scores=criterion_scores,
        summary=_coerce_text(parsed.get("summary")),
        strengths=_coerce_text_list(parsed.get("strengths"), limit=3),
        issues=_coerce_text_list(parsed.get("issues"), limit=3),
        covered_keypoints=_coerce_text_list(parsed.get("covered_keypoints"), limit=3),
        missing_keypoints=_coerce_text_list(parsed.get("missing_keypoints"), limit=3),
        detected_mistakes=_coerce_text_list(parsed.get("detected_mistakes"), limit=3),
        recommendations=_coerce_text_list(parsed.get("recommendations"), limit=3),
    )



def _coerce_text(value: object) -> str:
    return str(value or "").strip()



def _coerce_text_list(value: object, limit: int) -> list[str]:
    if not isinstance(value, list):
        return []
    items = [str(item).strip() for item in value if str(item).strip()]
    return items[:limit]



def _render_prompt(template: str, **values: str) -> str:
    rendered = template
    for key, value in values.items():
        rendered = rendered.replace(f"{{{key}}}", value)
    return rendered



def _coerce_criterion_scores(payload: dict, criteria: list[RubricCriterion]) -> dict[str, int]:
    scores: dict[str, int] = {}
    for criterion in criteria:
        if criterion.name not in payload:
            raise IntegrationError(
                "Ответ модели не содержит одну из обязательных оценок.",
                code="INVALID_MODEL_OUTPUT",
                details={"missing_criterion": criterion.name},
            )
        try:
            value = int(payload[criterion.name])
        except (TypeError, ValueError) as exc:
            raise IntegrationError(
                "Ответ модели содержит некорректное значение критерия.",
                code="INVALID_MODEL_OUTPUT",
                details={"criterion": criterion.name, "value": payload[criterion.name]},
            ) from exc
        scores[criterion.name] = max(0, min(100, value))
    return scores



def _weighted_score(criterion_scores: dict[str, int], criteria: list[RubricCriterion]) -> int:
    if not criteria:
        return 0
    weighted_total = 0.0
    total_weight = 0.0
    for criterion in criteria:
        weighted_total += criterion_scores.get(criterion.name, 0) * criterion.weight
        total_weight += criterion.weight
    if total_weight <= 0:
        return 0
    return round(weighted_total / total_weight)



def _parse_llm_json(content: str) -> dict:
    stripped = content.strip()
    if stripped.startswith("```"):
        stripped = stripped.strip("`")
        stripped = stripped.replace("json", "", 1).strip()
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start == -1 or end == -1:
        raise IntegrationError(
            "Ответ модели не содержит JSON-объект.",
            code="INVALID_MODEL_OUTPUT",
        )
    try:
        return json.loads(stripped[start : end + 1])
    except json.JSONDecodeError as exc:
        raise IntegrationError(
            "Не удалось разобрать JSON из ответа модели.",
            code="INVALID_MODEL_OUTPUT",
            details={"snippet": stripped[start : end + 1][:300]},
        ) from exc
