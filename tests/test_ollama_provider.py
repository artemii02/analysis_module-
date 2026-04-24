from __future__ import annotations

import json
from pathlib import Path

from interview_analysis.core.topic_catalog import topic_label
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
from interview_analysis.services.llm.ollama_provider import OllamaLLMProvider, _render_prompt, _weighted_score


ROOT = Path(__file__).resolve().parents[1]
PROMPT_PATH = ROOT / 'src' / 'interview_analysis' / 'prompts' / 'report_system_prompt.txt'


def test_render_prompt_keeps_json_braces_and_replaces_known_tokens() -> None:
    template = '''Return JSON:
{
  "score": 0,
  "summary": "text"
}
Topic: {topic}
Answer: {answer_text}
'''

    rendered = _render_prompt(
        template,
        topic=topic_label('http_api'),
        answer_text='GET reads the resource.',
    )

    assert '"score": 0' in rendered
    assert f"Topic: {topic_label('http_api')}" in rendered
    assert 'Answer: GET reads the resource.' in rendered
    assert '{topic}' not in rendered
    assert '{answer_text}' not in rendered


def test_report_prompt_mentions_criteria_weights_and_russian_output() -> None:
    template = PROMPT_PATH.read_text(encoding='utf-8')
    rendered = _render_prompt(
        template,
        report_language='ru',
        specialization='backend',
        grade='junior',
        topic_code='http_api',
        topic=topic_label('http_api'),
        question_text='How do GET and POST differ?',
        answer_text='GET reads, POST creates.',
        criteria='[]',
        keypoints='[]',
        recommendation_hints='[]',
        mistake_patterns='[]',
        context_snippets='[]',
    )

    assert 'Requested report language: ru' in rendered
    assert 'If report_language = ru, every textual JSON field must be in Russian.' in rendered
    assert 'The final score must equal the weighted average of criterion_scores' in rendered
    assert 'Criteria with weights: []' in rendered


def test_weighted_score_uses_rubric_weights() -> None:
    criteria = [
        RubricCriterion(name='correctness', weight=0.5, description=''),
        RubricCriterion(name='completeness', weight=0.3, description=''),
        RubricCriterion(name='clarity', weight=0.2, description=''),
    ]
    score = _weighted_score(
        {'correctness': 80, 'completeness': 60, 'clarity': 40},
        criteria,
    )

    assert score == 66


def test_assess_batch_splits_on_timeout() -> None:
    provider = OllamaLLMProvider(
        url='http://localhost:11434/api/generate',
        model='qwen2.5:3b',
        prompt_path=PROMPT_PATH,
        timeout_seconds=1,
    )
    contexts = [_build_context('item-1'), _build_context('item-2')]

    def fake_generate(prompt: str, max_tokens: int, response_format='json') -> str:
        if '"item_id": "item-1"' in prompt and '"item_id": "item-2"' in prompt:
            raise IntegrationError('timeout', code='MODEL_TIMEOUT')
        if 'Question text for item-1' in prompt:
            return _single_result('item-1 summary')
        if 'Question text for item-2' in prompt:
            return _single_result('item-2 summary')
        raise AssertionError('unexpected prompt passed to fake_generate')

    provider._generate = fake_generate
    assessments = provider.assess_batch(contexts)

    assert len(assessments) == 2
    assert assessments[0].summary == 'item-1 summary'
    assert assessments[1].summary == 'item-2 summary'
    assert all(item.score == 74 for item in assessments)


def test_assess_repairs_invalid_json() -> None:
    provider = OllamaLLMProvider(
        url='http://localhost:11434/api/generate',
        model='qwen2.5:3b',
        prompt_path=PROMPT_PATH,
        timeout_seconds=1,
    )
    context = _build_context('item-1')
    calls: list[str] = []

    def fake_generate(prompt: str, max_tokens: int, response_format='json') -> str:
        calls.append(prompt)
        if prompt.startswith('Return only a valid JSON object'):
            return _single_result('repaired summary')
        return '{"criterion_scores": {"correctness": 80}'

    provider._generate = fake_generate
    assessment = provider.assess(context)

    assert assessment.summary == 'repaired summary'
    assert len(calls) == 2


def test_assess_skips_model_for_low_signal_answer() -> None:
    provider = OllamaLLMProvider(
        url='http://localhost:11434/api/generate',
        model='qwen2.5:3b',
        prompt_path=PROMPT_PATH,
        timeout_seconds=1,
    )
    context = _build_context('item-1', answer_text='??')

    def fake_generate(prompt: str, max_tokens: int, response_format='json') -> str:
        raise AssertionError('model should not be called for low-signal answer')

    provider._generate = fake_generate
    assessment = provider.assess(context)

    assert assessment.score <= 5
    assert assessment.strengths == []
    assert assessment.covered_keypoints == []
    assert 'заглушк' in assessment.issues[0]


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


def test_assess_falls_back_to_grounded_on_invalid_output() -> None:
    provider = OllamaLLMProvider(
        url='http://localhost:11434/api/generate',
        model='qwen2.5:3b',
        prompt_path=PROMPT_PATH,
        timeout_seconds=1,
        fallback_to_grounded=True,
    )
    context = _build_context('item-1')
    expected = build_grounded_assessment(context)

    provider._generate = lambda prompt, max_tokens, response_format='json': '{"criterion_scores": {"correctness": 80}'
    assessment = provider.assess(context)

    assert assessment.summary == expected.summary
    assert assessment.score == expected.score
