from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from interview_analysis.core.topic_catalog import topic_label
from interview_analysis.exceptions import IntegrationError
from interview_analysis.models import QuestionAnalysisContext, QuestionAssessment
from interview_analysis.services.grounded_assessment import build_grounded_assessment, should_skip_llm
from interview_analysis.services.llm.base import BaseLLMProvider
from interview_analysis.services.llm.ollama_provider import (
    _build_assessment,
    _build_repair_prompt,
    _parse_llm_json,
    _single_assessment_schema,
)


SFT_SYSTEM_PROMPT = """
Ты оцениваешь технический ответ кандидата на русском языке.
Верни только валидный JSON без markdown, комментариев и текста вне JSON.
Обязательные поля: criterion_scores, summary, strengths, issues, covered_keypoints, missing_keypoints, detected_mistakes, recommendations.
Поле оценок должно называться строго criterion_scores, не criteria_scores.
criterion_scores должен содержать ключи correctness, completeness, clarity, practicality, terminology со значениями 0..100.
Все текстовые поля должны быть на русском языке.
Пиши компактно: summary до 2 предложений, каждый список до 3 пунктов.
""".strip()


class HFLLMProvider(BaseLLMProvider):
    prompt_version = "hf-lora-json-ru-v2"

    def __init__(
        self,
        base_model: str,
        adapter_path: str | Path | None,
        device: str = "auto",
        max_new_tokens: int = 900,
        load_in_4bit: bool = False,
    ) -> None:
        self.base_model = base_model
        self.adapter_path = Path(adapter_path) if adapter_path else None
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.load_in_4bit = load_in_4bit
        self.model_version = self._build_model_version()
        self._tokenizer = None
        self._model = None
        self._torch = None

    def assess(self, context: QuestionAnalysisContext) -> QuestionAssessment:
        grounded = build_grounded_assessment(context)
        if should_skip_llm(context):
            return grounded

        schema = _single_assessment_schema()
        prompt = self._build_chat_prompt(context)
        parsed = self._generate_and_parse(prompt, schema)
        return _build_assessment(_normalize_payload(parsed), context)

    def _generate_and_parse(self, prompt: str, schema: dict[str, Any]) -> dict[str, Any]:
        content = self._generate(prompt, self.max_new_tokens)
        try:
            return _parse_llm_json(content)
        except IntegrationError as exc:
            if exc.code != "INVALID_MODEL_OUTPUT":
                raise

        expanded_tokens = min(max(self.max_new_tokens * 2, 1200), 1800)
        content = self._generate(prompt, expanded_tokens)
        try:
            return _parse_llm_json(content)
        except IntegrationError as exc:
            if exc.code != "INVALID_MODEL_OUTPUT":
                raise

        repair_prompt = self._build_repair_chat_prompt(content, schema)
        repaired = self._generate(repair_prompt, min(900, expanded_tokens))
        return _parse_llm_json(repaired)

    def _build_model_version(self) -> str:
        if self.adapter_path:
            return f"{self.base_model}+{self.adapter_path.as_posix()}"
        return self.base_model

    def _build_chat_prompt(self, context: QuestionAnalysisContext) -> str:
        user_payload = {
            "specialization": context.scenario.specialization.value,
            "grade": context.scenario.grade.value,
            "topic": context.question.topic,
            "topic_label": topic_label(context.question.topic),
            "question_id": context.question.question_id,
            "question_text": context.session_item.question_text,
            "answer_text": context.session_item.answer_text,
            "criteria": [
                {"name": criterion.name, "weight": criterion.weight, "description": criterion.description}
                for criterion in context.rubric.criteria
            ],
            "expected_keypoints": context.rubric.keypoints,
            "recommendation_hints": context.rubric.recommendation_hints,
            "mistake_patterns": [
                {"trigger_terms": pattern.trigger_terms, "message": pattern.message}
                for pattern in context.rubric.mistake_patterns
            ],
            "context_snippets": [snippet.excerpt for snippet in context.retrieved_chunks],
            "output_schema": {
                "criterion_scores": {
                    "correctness": 0,
                    "completeness": 0,
                    "clarity": 0,
                    "practicality": 0,
                    "terminology": 0,
                },
                "summary": "краткий вывод",
                "strengths": ["до 3 пунктов"],
                "issues": ["до 3 пунктов"],
                "covered_keypoints": ["до 3 пунктов"],
                "missing_keypoints": ["до 3 пунктов"],
                "detected_mistakes": ["до 3 пунктов"],
                "recommendations": ["до 3 пунктов"],
            },
        }
        return self._apply_chat_template(
            [
                {"role": "system", "content": SFT_SYSTEM_PROMPT},
                {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
            ]
        )

    def _build_repair_chat_prompt(self, raw_content: str, schema: dict[str, Any]) -> str:
        return self._apply_chat_template(
            [
                {"role": "system", "content": "Ты исправляешь ответ модели и возвращаешь только валидный JSON."},
                {"role": "user", "content": _build_repair_prompt(raw_content, schema)},
            ]
        )

    def _apply_chat_template(self, messages: list[dict[str, str]]) -> str:
        tokenizer, _ = self._load()
        if hasattr(tokenizer, "apply_chat_template"):
            return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        rendered = []
        for message in messages:
            rendered.append(f"{message['role'].upper()}: {message['content']}")
        rendered.append("ASSISTANT:")
        return "\n".join(rendered)

    def _generate(self, prompt: str, max_new_tokens: int) -> str:
        tokenizer, model = self._load()
        torch = self._torch
        try:
            inputs = tokenizer(prompt, return_tensors="pt")
            input_device = self._input_device(model)
            inputs = {key: value.to(input_device) for key, value in inputs.items()}
            with torch.inference_mode():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            generated = output_ids[0, inputs["input_ids"].shape[-1] :]
            return tokenizer.decode(generated, skip_special_tokens=True).strip()
        except RuntimeError as exc:
            message = str(exc)
            code = "MODEL_RUNTIME_ERROR"
            if "out of memory" in message.lower():
                code = "MODEL_OUT_OF_MEMORY"
            raise IntegrationError(
                "Ошибка локальной HF/LoRA модели при генерации ответа.",
                code=code,
                details={"reason": message[:500]},
            ) from exc

    def _load(self):
        if self._tokenizer is not None and self._model is not None:
            return self._tokenizer, self._model

        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise IntegrationError(
                "Для режима hf нужны зависимости transformers, torch и peft. Установи: pip install -e .[training]",
                code="MODEL_DEPENDENCIES_MISSING",
            ) from exc

        self._torch = torch
        tokenizer = AutoTokenizer.from_pretrained(self.base_model, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model_kwargs: dict[str, Any] = {"trust_remote_code": True}
        if self.load_in_4bit:
            try:
                from transformers import BitsAndBytesConfig
            except ImportError as exc:
                raise IntegrationError(
                    "Для 4-bit загрузки нужен bitsandbytes. В Windows обычно надежнее запускать QLoRA через WSL2/Linux.",
                    code="MODEL_DEPENDENCIES_MISSING",
                ) from exc
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
            model_kwargs["device_map"] = "auto"
        elif self.device == "auto" and torch.cuda.is_available():
            model_kwargs["torch_dtype"] = torch.float16
            model_kwargs["device_map"] = "auto"
        elif self.device.startswith("cuda"):
            model_kwargs["torch_dtype"] = torch.float16
        else:
            model_kwargs["torch_dtype"] = torch.float32

        try:
            model = AutoModelForCausalLM.from_pretrained(self.base_model, **model_kwargs)
            if self.adapter_path:
                if not self.adapter_path.exists():
                    raise IntegrationError(
                        "LoRA-адаптер не найден.",
                        code="MODEL_ADAPTER_NOT_FOUND",
                        details={"adapter_path": self.adapter_path.as_posix()},
                    )
                try:
                    from peft import PeftModel
                except ImportError as exc:
                    raise IntegrationError(
                        "Для загрузки LoRA-адаптера нужна зависимость peft. Установи: pip install -e .[training]",
                        code="MODEL_DEPENDENCIES_MISSING",
                    ) from exc
                model = PeftModel.from_pretrained(model, self.adapter_path.as_posix())
            if "device_map" not in model_kwargs and self.device != "auto":
                model = model.to(self.device)
            model.eval()
        except IntegrationError:
            raise
        except Exception as exc:
            raise IntegrationError(
                "Не удалось загрузить локальную HF/LoRA модель.",
                code="MODEL_LOAD_FAILED",
                details={"reason": str(exc)[:500]},
            ) from exc

        self._tokenizer = tokenizer
        self._model = model
        return tokenizer, model

    def _input_device(self, model):
        try:
            return next(model.parameters()).device
        except StopIteration:
            return "cpu"


def _normalize_payload(parsed: dict[str, Any]) -> dict[str, Any]:
    if "criterion_scores" not in parsed and "criteria_scores" in parsed:
        parsed["criterion_scores"] = parsed["criteria_scores"]
    return parsed
