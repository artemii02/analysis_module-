from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import replace
from pathlib import Path
from statistics import mean
from typing import Iterable

from interview_analysis.core.config import get_settings
from interview_analysis.core.serialization import to_primitive
from interview_analysis.core.topic_catalog import topic_label
from interview_analysis.models import AssessmentRequest, ExecutionMode, Grade, ScenarioContext, SessionItem, Specialization
from interview_analysis.repositories.content_repository import JSONContentRepository
from interview_analysis.repositories.job_store import InMemoryAssessmentJobStore
from interview_analysis.services.analysis_pipeline import AnalysisPipeline
from interview_analysis.services.assessment_service import AssessmentService
from interview_analysis.services.llm.hf_provider import HFLLMProvider
from interview_analysis.services.llm.mock_provider import MockLLMProvider
from interview_analysis.services.llm.ollama_provider import OllamaLLMProvider
from interview_analysis.services.metrics import MetricsRegistry
from interview_analysis.services.report_builder import ReportBuilder
from interview_analysis.services.retrieval import SimpleKnowledgeRetriever


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REPORT_PATH = ROOT / 'training' / 'reports' / 'console_report.json'
DEFAULT_EVAL_PATH = ROOT / 'training' / 'data' / 'raw_eval.jsonl'
DEFAULT_EVAL_OUTPUT = ROOT / 'training' / 'reports' / 'console_eval_predictions.jsonl'


CRITERION_LABELS = {
    'correctness': 'Корректность',
    'completeness': 'Полнота',
    'clarity': 'Ясность',
    'practicality': 'Практичность',
    'terminology': 'Терминология',
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog='interview-analysis-console',
        description='Консольная проверка модуля анализа ответов без запуска demo UI.',
    )
    subparsers = parser.add_subparsers(dest='command', required=True)

    questions = subparsers.add_parser('questions', help='Показать банк вопросов.')
    questions.add_argument('--specialization', default='backend', choices=[item.value for item in Specialization])
    questions.add_argument('--grade', default='junior', choices=[item.value for item in Grade])
    questions.add_argument('--limit', type=int, default=10)

    run = subparsers.add_parser('run', help='Пройти сессию в консоли и сформировать отчёт.')
    run.add_argument('--specialization', default='backend', choices=[item.value for item in Specialization])
    run.add_argument('--grade', default='junior', choices=[item.value for item in Grade])
    run.add_argument('--limit', type=int, default=10)
    run.add_argument('--llm', default='mock', choices=['mock', 'ollama', 'hf'])
    run.add_argument('--output', type=Path, default=DEFAULT_REPORT_PATH)

    sample = subparsers.add_parser('sample', help='Собрать отчёт на готовых ответах из датасета.')
    sample.add_argument('--quality', default='good', choices=['empty', 'weak', 'medium', 'good', 'strong'])
    sample.add_argument('--specialization', default='backend', choices=[item.value for item in Specialization])
    sample.add_argument('--grade', default='junior', choices=[item.value for item in Grade])
    sample.add_argument('--limit', type=int, default=10)
    sample.add_argument('--llm', default='mock', choices=['mock', 'ollama', 'hf'])
    sample.add_argument('--output', type=Path, default=DEFAULT_REPORT_PATH)

    evaluate = subparsers.add_parser('evaluate', help='Сравнить предсказания модуля с размеченным датасетом.')
    evaluate.add_argument('--input', type=Path, default=DEFAULT_EVAL_PATH)
    evaluate.add_argument('--limit', type=int, default=24)
    evaluate.add_argument('--llm', default='mock', choices=['mock', 'ollama', 'hf'])
    evaluate.add_argument('--output', type=Path, default=DEFAULT_EVAL_OUTPUT)

    return parser.parse_args()


def build_service(llm_mode: str) -> AssessmentService:
    settings = replace(get_settings(), llm_mode=llm_mode, job_store_backend='memory')
    repository = JSONContentRepository(settings.data_dir)
    retriever = SimpleKnowledgeRetriever(repository, settings.knowledge_limit)
    if llm_mode == 'ollama':
        llm_provider = OllamaLLMProvider(
            url=settings.ollama_url,
            model=settings.ollama_model,
            prompt_path=settings.prompt_path,
            timeout_seconds=settings.request_timeout_seconds,
        )
    elif llm_mode == 'hf':
        llm_provider = HFLLMProvider(
            base_model=settings.hf_base_model,
            adapter_path=settings.hf_adapter_path,
            device=settings.hf_device,
            max_new_tokens=settings.hf_max_new_tokens,
            load_in_4bit=settings.hf_load_in_4bit,
        )
    else:
        llm_provider = MockLLMProvider()
    pipeline = AnalysisPipeline(repository, retriever, llm_provider, ReportBuilder())
    return AssessmentService(
        pipeline=pipeline,
        job_store=InMemoryAssessmentJobStore(),
        metrics=MetricsRegistry(),
        settings=settings,
    )


def load_repository() -> JSONContentRepository:
    return JSONContentRepository(get_settings().data_dir)


def command_questions(args: argparse.Namespace) -> int:
    repository = load_repository()
    questions = repository.list_questions(
        specialization=Specialization(args.specialization),
        grade=Grade(args.grade),
        limit=args.limit,
    )
    print(f'Найдено вопросов: {len(questions)}')
    for index, question in enumerate(questions, start=1):
        print(f'{index}. [{question.question_id}] {topic_label(question.topic)}')
        print(f'   {question.question_text}')
    return 0


def command_run(args: argparse.Namespace) -> int:
    repository = load_repository()
    questions = repository.list_questions(
        specialization=Specialization(args.specialization),
        grade=Grade(args.grade),
        limit=args.limit,
    )
    if not questions:
        print('Для выбранного профиля нет вопросов.')
        return 1

    print('Консольная сессия интервью')
    print('Ответ можно писать в несколько строк. Чтобы завершить ответ, оставь пустую строку и нажми Enter.')
    items: list[SessionItem] = []
    for index, question in enumerate(questions, start=1):
        print('\n' + '=' * 80)
        print(f'Вопрос {index}/{len(questions)}: {question.question_text}')
        print(f'Тема: {topic_label(question.topic)}')
        answer = read_multiline_answer()
        items.append(
            SessionItem(
                item_id=f'item-{index}',
                question_id=question.question_id,
                question_text=question.question_text,
                answer_text=answer,
                tags=question.tags,
            )
        )

    request = build_request(repository, args.specialization, args.grade, items, source='console-manual')
    report = analyze_request(args.llm, request)
    print_report(report)
    write_report(args.output, report)
    print(f'\nJSON-отчёт сохранён: {args.output}')
    return 0


def command_sample(args: argparse.Namespace) -> int:
    records = load_records_by_quality(args.quality, args.limit, args.specialization, args.grade)
    if not records:
        print(f'В датасете нет записей с quality={args.quality}.')
        return 1
    items = [
        SessionItem(
            item_id=f'item-{index}',
            question_id=record['question_id'],
            question_text=record['question_text'],
            answer_text=record['answer_text'],
        )
        for index, record in enumerate(records, start=1)
    ]
    request = build_request(load_repository(), args.specialization, args.grade, items, source=f'console-sample-{args.quality}')
    report = analyze_request(args.llm, request)
    print_report(report)
    write_report(args.output, report)
    print(f'\nJSON-отчёт сохранён: {args.output}')
    return 0


def command_evaluate(args: argparse.Namespace) -> int:
    records = load_jsonl(args.input)[: args.limit]
    if not records:
        print(f'Файл не содержит записей: {args.input}')
        return 1

    output_rows = []
    absolute_errors: list[int] = []
    errors_by_band: dict[str, list[int]] = defaultdict(list)
    repository = load_repository()
    service = build_service(args.llm)
    for index, record in enumerate(records, start=1):
        item = SessionItem(
            item_id=f'eval-{index}',
            question_id=record['question_id'],
            question_text=record['question_text'],
            answer_text=record['answer_text'],
        )
        request = build_request(repository, record['specialization'], record['grade'], [item], source=f'console-eval-{index}')
        report = analyze_request_with_service(service, request)
        predicted = report.questions[0].score
        expected = int(record['expected_feedback']['score'])
        error = abs(predicted - expected)
        absolute_errors.append(error)
        errors_by_band[record.get('answer_quality_band', 'unknown')].append(error)
        output_rows.append(
            {
                'record_id': record['record_id'],
                'question_id': record['question_id'],
                'specialization': record['specialization'],
                'grade': record['grade'],
                'topic': record['topic'],
                'answer_quality_band': record.get('answer_quality_band'),
                'expected_score': expected,
                'predicted_score': predicted,
                'absolute_error': error,
                'summary': report.questions[0].summary,
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        ''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in output_rows),
        encoding='utf-8',
    )

    print('Оценка качества на датасете')
    print(f'Файл: {args.input}')
    print(f'Записей: {len(records)}')
    print(f'LLM режим: {args.llm}')
    print(f'MAE по score: {mean(absolute_errors):.2f}')
    print(f'Max error: {max(absolute_errors)}')
    print('MAE по quality band:')
    for band in sorted(errors_by_band):
        print(f'  {band}: {mean(errors_by_band[band]):.2f} ({len(errors_by_band[band])} записей)')
    print(f'Предсказания сохранены: {args.output}')
    return 0


def read_multiline_answer() -> str:
    lines: list[str] = []
    while True:
        line = input('> ')
        if not line.strip():
            if lines:
                break
            print('Ответ не должен быть пустым. Напиши хотя бы одну строку.')
            continue
        lines.append(line.rstrip())
    return '\n'.join(lines).strip()


def build_request(
    repository: JSONContentRepository,
    specialization: str,
    grade: str,
    items: list[SessionItem],
    source: str,
) -> AssessmentRequest:
    spec = Specialization(specialization)
    current_grade = Grade(grade)
    topics = sorted({repository.get_question(item.question_id, spec, current_grade).topic for item in items})
    return AssessmentRequest(
        request_id=f'{source}-request',
        session_id=f'{source}-session',
        client_id='console',
        mode=ExecutionMode.SYNC,
        scenario=ScenarioContext(
            scenario_id=source,
            specialization=spec,
            grade=current_grade,
            topics=topics,
            report_language='ru',
        ),
        items=items,
        metadata={'source': source},
    )


def analyze_request(llm_mode: str, request: AssessmentRequest):
    service = build_service(llm_mode)
    return analyze_request_with_service(service, request)


def analyze_request_with_service(service: AssessmentService, request: AssessmentRequest):
    submission = service.register_request(request)
    return service.process_sync(submission.job.job_id, request)


def print_report(report) -> None:
    print('\n' + '=' * 80)
    print('ИТОГОВЫЙ ОТЧЁТ')
    print(f'Общий балл: {report.overall_score}/100')
    print(f'Профиль: {report.specialization} / {report.grade}')
    print(f'Краткое резюме: {report.summary}')
    print('\nКритерии:')
    for name, score in report.criterion_scores.items():
        label = CRITERION_LABELS.get(name, name)
        print(f'  {label}: {score}/100')

    print('\nРекомендации:')
    for recommendation in report.recommendations:
        print(f'  - {recommendation}')

    print('\nРазбор по вопросам:')
    for index, item in enumerate(report.questions, start=1):
        print('\n' + '-' * 80)
        print(f'{index}. {item.question_text}')
        print(f'Тема: {item.topic}')
        print(f'Балл: {item.score}/100')
        print(f'Комментарий: {item.summary}')
        print_items('Сильные стороны', item.strengths)
        print_items('Проблемы', item.issues)
        print_items('Что повторить', item.recommendations)


def print_items(title: str, items: Iterable[str]) -> None:
    values = list(items)
    if not values:
        return
    print(f'{title}:')
    for value in values:
        print(f'  - {value}')


def write_report(path: Path, report) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_primitive(report), ensure_ascii=False, indent=2), encoding='utf-8')


def load_records_by_quality(quality: str, limit: int, specialization: str, grade: str) -> list[dict]:
    records: list[dict] = []
    for path in [ROOT / 'training' / 'data' / 'raw_train.jsonl', ROOT / 'training' / 'data' / 'raw_eval.jsonl', ROOT / 'training' / 'data' / 'raw_test.jsonl']:
        for record in load_jsonl(path):
            if record.get('answer_quality_band') != quality:
                continue
            if record.get('specialization') != specialization or record.get('grade') != grade:
                continue
            records.append(record)
            if len(records) >= limit:
                return records
    return records


def load_jsonl(path: Path) -> list[dict]:
    with path.open('r', encoding='utf-8') as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main() -> int:
    args = parse_args()
    if args.command == 'questions':
        return command_questions(args)
    if args.command == 'run':
        return command_run(args)
    if args.command == 'sample':
        return command_sample(args)
    if args.command == 'evaluate':
        return command_evaluate(args)
    raise ValueError(args.command)


if __name__ == '__main__':
    raise SystemExit(main())





