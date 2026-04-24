from __future__ import annotations

import json

from interview_analysis.exceptions import IntegrationError
from interview_analysis.models import (
    Grade,
    QuestionAnalysisContext,
    QuestionDefinition,
    RubricCriterion,
    RubricDefinition,
    ScenarioContext,
    SessionItem,
    Specialization,
)
from interview_analysis.services.grounded_assessment import build_grounded_assessment
from interview_analysis.services.llm.hf_provider import HFLLMProvider


def test_assess_batch_keeps_order_and_skips_low_signal_answer() -> None:
    provider = HFLLMProvider(
        base_model='Qwen/Qwen2.5-3B-Instruct',
        adapter_path=None,
        batch_size=3,
        load_in_4bit=False,
    )
    contexts = [
        _build_context('item-1'),
        _build_context('item-2', answer_text='хз'),
        _build_context('item-3'),
    ]
    batch_prompts: list[str] = []

    provider._build_chat_prompt = lambda context: f"prompt::{context.session_item.item_id}"

    def fake_generate_batch(prompts: list[str], max_new_tokens: int) -> list[str]:
        batch_prompts.extend(prompts)
        return [
            _single_result(f'{prompt} summary')
            for prompt in prompts
        ]

    provider._generate_batch = fake_generate_batch
    assessments = provider.assess_batch(contexts)

    assert len(assessments) == 3
    assert assessments[0].summary == 'prompt::item-1 summary'
    assert assessments[1].score <= 5
    assert assessments[2].summary == 'prompt::item-3 summary'
    assert batch_prompts == ['prompt::item-1', 'prompt::item-3']


def test_assess_batch_falls_back_to_single_generation_for_invalid_batch_item() -> None:
    provider = HFLLMProvider(
        base_model='Qwen/Qwen2.5-3B-Instruct',
        adapter_path=None,
        batch_size=2,
        load_in_4bit=False,
    )
    contexts = [_build_context('item-1'), _build_context('item-2')]

    provider._build_chat_prompt = lambda context: f"prompt::{context.session_item.item_id}"

    def fake_generate_batch(prompts: list[str], max_new_tokens: int) -> list[str]:
        return [
            '{"criterion_scores": {"correctness": 80}',
            _single_result('batch item-2 summary'),
        ]

    def fake_generate(prompt: str, max_new_tokens: int) -> str:
        return _single_result(f'{prompt} repaired')

    provider._generate_batch = fake_generate_batch
    provider._generate = fake_generate
    assessments = provider.assess_batch(contexts)

    assert assessments[0].summary == 'prompt::item-1 repaired'
    assert assessments[1].summary == 'batch item-2 summary'


def test_assess_batch_splits_when_batch_generation_fails() -> None:
    provider = HFLLMProvider(
        base_model='Qwen/Qwen2.5-3B-Instruct',
        adapter_path=None,
        batch_size=2,
        load_in_4bit=False,
    )
    contexts = [_build_context('item-1'), _build_context('item-2')]

    provider._build_chat_prompt = lambda context: f"prompt::{context.session_item.item_id}"

    def fake_generate_batch(prompts: list[str], max_new_tokens: int) -> list[str]:
        if len(prompts) > 1:
            raise IntegrationError('oom', code='MODEL_OUT_OF_MEMORY')
        return [_single_result(f'{prompts[0]} split')]

    def fake_generate(prompt: str, max_new_tokens: int) -> str:
        return _single_result(f'{prompt} split')

    provider._generate_batch = fake_generate_batch
    provider._generate = fake_generate
    assessments = provider.assess_batch(contexts)

    assert [item.summary for item in assessments] == [
        'prompt::item-1 split',
        'prompt::item-2 split',
    ]


def test_assess_falls_back_to_grounded_on_invalid_output() -> None:
    provider = HFLLMProvider(
        base_model='Qwen/Qwen2.5-3B-Instruct',
        adapter_path=None,
        batch_size=1,
        load_in_4bit=False,
        fallback_to_grounded=True,
        disable_on_cpu=False,
    )
    context = _build_context('item-1')
    expected = build_grounded_assessment(context)

    provider._build_chat_prompt = lambda context: f"prompt::{context.session_item.item_id}"
    provider._build_repair_chat_prompt = lambda raw_content, schema: 'repair'
    provider._generate = lambda prompt, max_new_tokens: '{"criterion_scores": {"correctness": 80}'

    assessment = provider.assess(context)

    assert assessment.summary == expected.summary
    assert assessment.score == expected.score


def test_assess_batch_bypasses_llm_on_cpu_when_enabled() -> None:
    provider = HFLLMProvider(
        base_model='Qwen/Qwen2.5-3B-Instruct',
        adapter_path=None,
        batch_size=2,
        load_in_4bit=False,
        fallback_to_grounded=True,
        disable_on_cpu=True,
        device='cpu',
    )
    contexts = [_build_context('item-1'), _build_context('item-2')]

    def fail_generate_batch(prompts: list[str], max_new_tokens: int) -> list[str]:
        raise AssertionError('LLM batch generation should be bypassed on CPU')

    provider._generate_batch = fail_generate_batch
    assessments = provider.assess_batch(contexts)

    assert len(assessments) == 2
    assert all(item.summary for item in assessments)


def _build_context(item_id: str, answer_text: str | None = None) -> QuestionAnalysisContext:
    scenario = ScenarioContext(
        specialization=Specialization.BACKEND,
        grade=Grade.JUNIOR,
        report_language='ru',
    )
    session_item = SessionItem(
        item_id=item_id,
        question_id=f'question-{item_id}',
        question_text=f'Question text for {item_id}',
        answer_text=answer_text or f'Answer text for {item_id}',
        tags=['http', 'rest'],
    )
    question = QuestionDefinition(
        question_id=session_item.question_id,
        specialization=Specialization.BACKEND,
        grade=Grade.JUNIOR,
        topic='http_api',
        question_text=session_item.question_text,
        tags=['http', 'rest'],
        version='questions-test',
    )
    rubric = RubricDefinition(
        question_id=session_item.question_id,
        specialization=Specialization.BACKEND,
        grade=Grade.JUNIOR,
        topic='http_api',
        criteria=[
            RubricCriterion(name='correctness', weight=0.35, description=''),
            RubricCriterion(name='completeness', weight=0.3, description=''),
            RubricCriterion(name='clarity', weight=0.15, description=''),
            RubricCriterion(name='practicality', weight=0.1, description=''),
            RubricCriterion(name='terminology', weight=0.1, description=''),
        ],
        keypoints=['GET reads a resource'],
        recommendation_hints=['Add more detail'],
        mistake_patterns=[],
        version='rubrics-test',
    )
    return QuestionAnalysisContext(
        scenario=scenario,
        session_item=session_item,
        question=question,
        rubric=rubric,
        retrieved_chunks=[],
        normalized_answer=session_item.answer_text.casefold(),
    )


def _single_result(summary: str) -> str:
    return json.dumps(
        {
            'criterion_scores': {
                'correctness': 80,
                'completeness': 70,
                'clarity': 75,
                'practicality': 65,
                'terminology': 70,
            },
            'summary': summary,
            'strengths': ['good'],
            'issues': ['gap'],
            'covered_keypoints': ['GET reads a resource'],
            'missing_keypoints': [],
            'detected_mistakes': [],
            'recommendations': ['Add more detail'],
        },
        ensure_ascii=False,
    )
