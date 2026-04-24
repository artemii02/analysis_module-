from __future__ import annotations

from pathlib import Path

from interview_analysis.core.config import Settings
from interview_analysis.models import AssessmentRequest, ExecutionMode, Grade, ScenarioContext, SessionItem, Specialization
from interview_analysis.repositories.content_repository import JSONContentRepository
from interview_analysis.repositories.job_store import InMemoryAssessmentJobStore
from interview_analysis.services.analysis_pipeline import AnalysisPipeline
from interview_analysis.services.assessment_service import AssessmentService
from interview_analysis.services.llm.mock_provider import MockLLMProvider
from interview_analysis.services.metrics import MetricsRegistry
from interview_analysis.services.report_builder import ReportBuilder
from interview_analysis.services.retrieval import SimpleKnowledgeRetriever


ROOT = Path(__file__).resolve().parents[1]
PACKAGE_DIR = ROOT / 'src' / 'interview_analysis'


def build_service() -> AssessmentService:
    settings = Settings(
        app_name='Test Analysis Module',
        api_prefix='/assessment/v1',
        api_key='demo-api-key',
        log_level='INFO',
        llm_mode='mock',
        job_store_backend='memory',
        database_url='postgresql://analysis:analysis@localhost:5432/analysis_module',
        ollama_url='http://localhost:11434/api/generate',
        ollama_model='qwen2.5:3b',
        hf_base_model='Qwen/Qwen2.5-3B-Instruct',
        hf_adapter_path=ROOT / 'training' / 'artifacts' / 'qwen2.5-3b-interview-full-ru-qlora-v1',
        hf_device='auto',
        hf_load_in_4bit=True,
        hf_max_new_tokens=900,
        hf_batch_size=3,
        hf_retry_max_new_tokens=320,
        hf_repair_max_new_tokens=220,
        llm_fallback_to_grounded=True,
        disable_llm_on_cpu=True,
        warmup_llm_on_start=False,
        request_timeout_seconds=30,
        knowledge_limit=3,
        hard_timeout_seconds=30,
        max_answer_length=4000,
        max_session_items=20,
        data_dir=PACKAGE_DIR / 'data',
        prompt_path=PACKAGE_DIR / 'prompts' / 'report_system_prompt.txt',
        db_schema_path=PACKAGE_DIR / 'db' / 'schema.sql',
        demo_cases_path=PACKAGE_DIR / 'demo' / 'demo_cases.json',
        demo_template_path=PACKAGE_DIR / 'demo' / 'demo.html',
    )
    repository = JSONContentRepository(settings.data_dir)
    pipeline = AnalysisPipeline(
        repository=repository,
        retriever=SimpleKnowledgeRetriever(repository, limit=3),
        llm_provider=MockLLMProvider(),
        report_builder=ReportBuilder(),
    )
    return AssessmentService(
        pipeline=pipeline,
        job_store=InMemoryAssessmentJobStore(),
        metrics=MetricsRegistry(),
        settings=settings,
    )


def test_pipeline_builds_report_from_mock_provider() -> None:
    service = build_service()
    request = AssessmentRequest(
        request_id='req-001',
        session_id='session-001',
        client_id='backend-service',
        mode=ExecutionMode.SYNC,
        scenario=ScenarioContext(
            specialization=Specialization.BACKEND,
            grade=Grade.JUNIOR,
            topics=['http_rest', 'sql_indexes'],
        ),
        items=[
            SessionItem(
                item_id='item-1',
                question_id='be_junior_http_rest_003',
                question_text='Чем отличаются PUT и PATCH в HTTP?',
                answer_text='PUT обычно заменяет ресурс целиком и чаще считается идемпотентным. PATCH используют для частичных обновлений, а его идемпотентность зависит от конкретной операции. Например, в REST API это важно при повторной отправке запроса.',
            ),
            SessionItem(
                item_id='item-2',
                question_id='be_junior_sql_indexes_002',
                question_text='Что такое индекс в базе данных и зачем он нужен?',
                answer_text='Индекс — это дополнительная структура данных, которая помогает быстрее находить строки. Он полезен для WHERE, JOIN и сортировки, но занимает место и может замедлять INSERT и UPDATE.',
            ),
        ],
    )

    submission = service.register_request(request)
    report = service.process_sync(submission.job.job_id, request)

    assert report.overall_score >= 30
    assert set(report.criterion_scores) == {'correctness', 'completeness', 'clarity', 'practicality', 'terminology'}
    assert len(report.questions) == 2
    assert report.versions.rubric_version == 'rubrics-2026.04-full-ru-v1'
    assert report.versions.questions_version == 'questions-2026.04-full-ru-v1'
    assert any(item.covered_keypoints for item in report.questions)
    assert report.recommendations


def test_pipeline_rejects_too_long_answer() -> None:
    service = build_service()
    request = AssessmentRequest(
        request_id='req-002',
        session_id='session-002',
        client_id='backend-service',
        mode=ExecutionMode.SYNC,
        scenario=ScenarioContext(
            specialization=Specialization.BACKEND,
            grade=Grade.JUNIOR,
        ),
        items=[
            SessionItem(
                item_id='item-1',
                question_id='be_junior_http_rest_003',
                question_text='Q',
                answer_text='a' * 5001,
            )
        ],
    )

    try:
        service.register_request(request)
    except Exception as exc:  # pragma: no cover - simple smoke assertion
        assert exc.__class__.__name__ == 'InvalidInputError'
    else:  # pragma: no cover
        raise AssertionError('Expected InvalidInputError for oversized answer')


def test_pipeline_rejects_more_than_twenty_items() -> None:
    service = build_service()
    items = [
        SessionItem(
            item_id=str(index),
            question_id='be_junior_http_rest_003',
            question_text='Чем отличаются PUT и PATCH в HTTP?',
            answer_text='PUT заменяет ресурс целиком, PATCH применяют для частичного обновления.',
        )
        for index in range(21)
    ]
    request = AssessmentRequest(
        request_id='req-003',
        session_id='session-003',
        client_id='backend-service',
        mode=ExecutionMode.SYNC,
        scenario=ScenarioContext(
            specialization=Specialization.BACKEND,
            grade=Grade.JUNIOR,
        ),
        items=items,
    )

    try:
        service.register_request(request)
    except Exception as exc:  # pragma: no cover - simple smoke assertion
        assert exc.__class__.__name__ == 'InvalidInputError'
    else:  # pragma: no cover
        raise AssertionError('Expected InvalidInputError for more than 20 items')


def test_pipeline_resolves_external_question_id_by_question_text() -> None:
    service = build_service()
    request = AssessmentRequest(
        request_id='req-004',
        session_id='session-004',
        client_id='main-backend',
        mode=ExecutionMode.SYNC,
        scenario=ScenarioContext(
            specialization=Specialization.BACKEND,
            grade=Grade.JUNIOR,
        ),
        items=[
            SessionItem(
                item_id='item-1',
                question_id='21000000-0000-0000-0000-000000000001',
                question_text='Что такое JVM, JRE и JDK и в чём между ними разница?',
                answer_text='JVM выполняет Java-байткод, JRE включает JVM и библиотеки для запуска, а JDK содержит JRE и инструменты разработки.',
            )
        ],
    )

    submission = service.register_request(request)
    report = service.process_sync(submission.job.job_id, request)

    assert len(report.questions) == 1
    assert report.questions[0].question_id == '21000000-0000-0000-0000-000000000001'
    assert report.questions[0].topic == 'Jvm jre jdk'
    assert report.versions.rubric_version == 'runtime-rubric-v1'
    assert report.versions.questions_version == 'external-backend-v1'





