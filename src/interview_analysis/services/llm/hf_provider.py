from __future__ import annotations

import json
import logging
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


logger = logging.getLogger(__name__)


SFT_SYSTEM_PROMPT = """
Ты оцениваешь технический ответ кандидата на русском языке.
Верни только валидный JSON без markdown, комментариев и текста вне JSON.
Обязательные поля: criterion_scores, summary, strengths, issues, covered_keypoints, missing_keypoints, detected_mistakes, recommendations.
Поле оценок должно называться строго criterion_scores, не criteria_scores.
criterion_scores должен содержать ключи correctness, completeness, clarity, practicality, terminology со значениями 0..100.
Все текстовые поля должны быть на русском языке.
Пиши очень компактно: summary в 1 коротком предложении до 12 слов, каждый список строго до 1 пункта.
""".strip()


class HFLLMProvider(BaseLLMProvider):
    prompt_version = "hf-lora-json-ru-v3"

    def __init__(
        self,
        base_model: str,
        adapter_path: str | Path | None,
        device: str = "auto",
        max_new_tokens: int = 220,
        load_in_4bit: bool = False,
        batch_size: int = 3,
        retry_max_new_tokens: int = 320,
        repair_max_new_tokens: int = 220,
        fallback_to_grounded: bool = True,
        disable_on_cpu: bool = False,
    ) -> None:
        self.base_model = base_model
        self.adapter_path = Path(adapter_path) if adapter_path else None
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.load_in_4bit = load_in_4bit
        self.batch_size = max(1, batch_size)
        self.retry_max_new_tokens = max(max_new_tokens, retry_max_new_tokens)
        self.repair_max_new_tokens = repair_max_new_tokens
        self.fallback_to_grounded = fallback_to_grounded
        self.disable_on_cpu = disable_on_cpu
        self.model_version = self._build_model_version()
        self._tokenizer = None
        self._model = None
        self._torch = None

    def assess(self, context: QuestionAnalysisContext) -> QuestionAssessment:
        grounded = build_grounded_assessment(context)
        if should_skip_llm(context):
            return grounded
        if self._should_bypass_llm():
            logger.info(
                'hf.assess.bypassed reason=cpu_only item_id=%s question_id=%s',
                context.session_item.item_id,
                context.session_item.question_id,
            )
            return grounded

        try:
            schema = _single_assessment_schema()
            prompt = self._build_chat_prompt(context)
            parsed = self._generate_and_parse(prompt, schema)
            return _build_assessment(_normalize_payload(parsed), context)
        except IntegrationError as exc:
            if not self._should_fallback(exc):
                raise
            logger.warning(
                'hf.assess.fallback_grounded item_id=%s question_id=%s code=%s',
                context.session_item.item_id,
                context.session_item.question_id,
                exc.code,
            )
            return grounded

    def assess_batch(self, contexts: list[QuestionAnalysisContext]) -> list[QuestionAssessment]:
        if not contexts:
            return []
        if self._should_bypass_llm():
            logger.info(
                'hf.assess_batch.bypassed reason=cpu_only items=%s',
                len(contexts),
            )
            return [build_grounded_assessment(context) for context in contexts]

        assessments: list[QuestionAssessment | None] = [None] * len(contexts)
        llm_contexts: list[tuple[int, QuestionAnalysisContext]] = []
        for index, context in enumerate(contexts):
            grounded = build_grounded_assessment(context)
            if should_skip_llm(context):
                assessments[index] = grounded
                continue
            llm_contexts.append((index, context))

        if not llm_contexts:
            return [item for item in assessments if item is not None]

        for start in range(0, len(llm_contexts), self.batch_size):
            chunk = llm_contexts[start : start + self.batch_size]
            chunk_assessments = self._assess_chunk(chunk)
            for index, assessment in chunk_assessments:
                assessments[index] = assessment

        return [item for item in assessments if item is not None]

    def _generate_and_parse(self, prompt: str, schema: dict[str, Any]) -> dict[str, Any]:
        content = self._generate(prompt, self.max_new_tokens)
        try:
            return _parse_llm_json(content)
        except IntegrationError as exc:
            if exc.code != "INVALID_MODEL_OUTPUT":
                raise
            logger.warning('hf.generate.invalid_json_first_pass')

        repair_prompt = self._build_repair_chat_prompt(content, schema)
        repaired = self._generate(repair_prompt, self.repair_max_new_tokens)
        try:
            return _parse_llm_json(repaired)
        except IntegrationError as exc:
            if exc.code != "INVALID_MODEL_OUTPUT":
                raise
            logger.warning('hf.generate.invalid_json_repair_pass')

        content = self._generate(prompt, self.retry_max_new_tokens)
        try:
            return _parse_llm_json(content)
        except IntegrationError as exc:
            if exc.code != "INVALID_MODEL_OUTPUT":
                raise
            logger.warning('hf.generate.invalid_json_retry_pass')
            raise

    def _assess_chunk(
        self,
        chunk: list[tuple[int, QuestionAnalysisContext]],
    ) -> list[tuple[int, QuestionAssessment]]:
        if not chunk:
            return []
        if len(chunk) == 1:
            index, context = chunk[0]
            return [(index, self.assess(context))]

        prompts = [self._build_chat_prompt(context) for _, context in chunk]
        try:
            contents = self._generate_batch(prompts, self.max_new_tokens)
        except IntegrationError as exc:
            if exc.code in {"MODEL_TIMEOUT", "MODEL_RUNTIME_ERROR", "MODEL_OUT_OF_MEMORY"} and len(chunk) > 1:
                midpoint = max(1, len(chunk) // 2)
                return self._assess_chunk(chunk[:midpoint]) + self._assess_chunk(chunk[midpoint:])
            raise

        results: list[tuple[int, QuestionAssessment]] = []
        for (index, context), prompt, content in zip(chunk, prompts, contents, strict=True):
            try:
                parsed = _parse_llm_json(content)
            except IntegrationError as exc:
                if exc.code != "INVALID_MODEL_OUTPUT":
                    raise
                parsed = self._generate_and_parse(prompt, _single_assessment_schema())
            results.append((index, _build_assessment(_normalize_payload(parsed), context)))
        return results

    def _build_model_version(self) -> str:
        if self.adapter_path:
            return f"{self.base_model}+{self.adapter_path.as_posix()}"
        return self.base_model

    def _build_chat_prompt(self, context: QuestionAnalysisContext) -> str:
        user_payload = self._build_user_payload(context)
        return self._apply_chat_template(
            [
                {"role": "system", "content": SFT_SYSTEM_PROMPT},
                {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
            ]
        )

    def _build_user_payload(self, context: QuestionAnalysisContext) -> dict[str, Any]:
        return {
            "specialization": context.scenario.specialization.value,
            "grade": context.scenario.grade.value,
            "topic_code": context.question.topic,
            "topic_label": topic_label(context.question.topic),
            "question_id": context.question.question_id,
            "question_text": context.session_item.question_text,
            "answer_text": context.session_item.answer_text,
            "criterion_weights": {
                criterion.name: criterion.weight
                for criterion in context.rubric.criteria
            },
            "expected_keypoints": context.rubric.keypoints[:3],
            "recommendation_hints": context.rubric.recommendation_hints[:1],
            "mistake_hints": [
                pattern.message
                for pattern in context.rubric.mistake_patterns[:1]
            ],
            "context_snippets": [snippet.excerpt for snippet in context.retrieved_chunks[:1]],
            "required_json_keys": [
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

    def _generate_batch(self, prompts: list[str], max_new_tokens: int) -> list[str]:
        tokenizer, model = self._load()
        torch = self._torch
        try:
            inputs = tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
            )
            input_device = self._input_device(model)
            inputs = {key: value.to(input_device) for key, value in inputs.items()}
            prompt_token_count = inputs["input_ids"].shape[-1]
            with torch.inference_mode():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            generated = output_ids[:, prompt_token_count:]
            return [item.strip() for item in tokenizer.batch_decode(generated, skip_special_tokens=True)]
        except RuntimeError as exc:
            message = str(exc)
            code = "MODEL_RUNTIME_ERROR"
            if "out of memory" in message.lower():
                code = "MODEL_OUT_OF_MEMORY"
            raise IntegrationError(
                "Ошибка локальной HF/LoRA модели при пакетной генерации ответа.",
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
                "Для режима hf нужны зависимости transformers, torch и peft. Установи: pip install -e .[hf_runtime]",
                code="MODEL_DEPENDENCIES_MISSING",
            ) from exc

        self._torch = torch
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        tokenizer = AutoTokenizer.from_pretrained(self.base_model, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"

        runtime_device = self._resolve_runtime_device(torch)
        use_auto_offload = self.device == "auto" and torch.cuda.is_available() and not self.load_in_4bit
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
            model_kwargs["dtype"] = torch.float16
        elif use_auto_offload:
            model_kwargs["dtype"] = torch.float16
            model_kwargs["device_map"] = "auto"
        elif runtime_device.startswith("cuda"):
            model_kwargs["dtype"] = torch.float16
        else:
            model_kwargs["dtype"] = torch.float32

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
                        "Для загрузки LoRA-адаптера нужна зависимость peft. Установи: pip install -e .[hf_runtime]",
                        code="MODEL_DEPENDENCIES_MISSING",
                    ) from exc
                adapter_kwargs: dict[str, Any] = {}
                if "device_map" in model_kwargs:
                    adapter_kwargs["torch_device"] = runtime_device
                    if runtime_device.startswith("cuda"):
                        adapter_kwargs["ephemeral_gpu_offload"] = True
                model = PeftModel.from_pretrained(
                    model,
                    self.adapter_path.as_posix(),
                    **adapter_kwargs,
                )
            if "device_map" not in model_kwargs:
                model = model.to(runtime_device)
            model.eval()
            if hasattr(model, "generation_config") and model.generation_config is not None:
                model.generation_config.use_cache = True
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

    def _resolve_runtime_device(self, torch) -> str:
        if self.device != "auto":
            return self.device
        cuda = getattr(torch, "cuda", None)
        if cuda is not None and callable(getattr(cuda, "is_available", None)) and cuda.is_available():
            return "cuda:0"
        return "cpu"

    def _input_device(self, model):
        try:
            return next(model.parameters()).device
        except StopIteration:
            return "cpu"

    def _should_bypass_llm(self) -> bool:
        if not self.disable_on_cpu:
            return False
        try:
            import torch
        except ImportError:
            return False
        try:
            return self._resolve_runtime_device(torch).startswith("cpu")
        except AttributeError:
            return False

    def _should_fallback(self, exc: IntegrationError) -> bool:
        if not self.fallback_to_grounded:
            return False
        return exc.code in {
            "INVALID_MODEL_OUTPUT",
            "MODEL_TIMEOUT",
            "MODEL_RUNTIME_ERROR",
            "MODEL_OUT_OF_MEMORY",
            "MODEL_LOAD_FAILED",
        }


def _normalize_payload(parsed: dict[str, Any]) -> dict[str, Any]:
    if "criterion_scores" not in parsed and "criteria_scores" in parsed:
        parsed["criterion_scores"] = parsed["criteria_scores"]
    return parsed
