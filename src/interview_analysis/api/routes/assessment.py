from __future__ import annotations

import logging
from typing import Literal

from fastapi import APIRouter, BackgroundTasks, Depends, Query
from fastapi.responses import JSONResponse

from interview_analysis.api.dependencies import get_service, verify_api_key
from interview_analysis.core.serialization import to_camel_case_keys, to_primitive
from interview_analysis.core.topic_catalog import topic_label
from interview_analysis.models import Grade, JobStatus, Specialization
from interview_analysis.schemas.api import (
    AssessmentRequestPayload,
    JobPayload,
    QuestionBankResponsePayload,
    ReportResponsePayload,
)


router = APIRouter()
logger = logging.getLogger(__name__)


@router.get('/questions', response_model=QuestionBankResponsePayload, response_model_by_alias=True)
def list_questions(
    specialization: Literal['backend', 'frontend', 'devops'] = Query(default='backend'),
    grade: Literal['junior', 'middle'] = Query(default='junior'),
    limit: int = Query(default=10, ge=1, le=20),
    _: None = Depends(verify_api_key),
    service=Depends(get_service),
):
    repository = service.pipeline.repository
    items = repository.list_questions(
        Specialization(specialization),
        Grade(grade),
        limit=limit,
    )
    logger.info(
        'assessment.questions.list specialization=%s grade=%s limit=%s count=%s',
        specialization,
        grade,
        limit,
        len(items),
    )
    return {
        'status': 'ok',
        'specialization': specialization,
        'grade': grade,
        'count': len(items),
        'items': [_question_payload(item) for item in items],
    }


@router.post('/report', response_model=ReportResponsePayload | JobPayload, response_model_by_alias=True)
def create_report(
    payload: AssessmentRequestPayload,
    background_tasks: BackgroundTasks,
    _: None = Depends(verify_api_key),
    service=Depends(get_service),
):
    request = payload.to_domain()
    logger.info(
        'assessment.report.received request_id=%s session_id=%s client_id=%s mode=%s specialization=%s grade=%s items=%s',
        request.request_id,
        request.session_id,
        request.client_id,
        request.mode.value,
        request.scenario.specialization.value,
        request.scenario.grade.value,
        len(request.items),
    )
    submission = service.register_request(request)
    job = submission.job

    if request.mode.value == 'async':
        if job.status == JobStatus.READY and job.report is not None:
            logger.info(
                'assessment.report.async_cached job_id=%s request_id=%s status=%s',
                job.job_id,
                job.request_id,
                job.status.value,
            )
            return _report_payload(job)
        if submission.created_new or job.status == JobStatus.CREATED:
            logger.info(
                'assessment.report.async_enqueued job_id=%s request_id=%s created_new=%s',
                job.job_id,
                job.request_id,
                submission.created_new,
            )
            background_tasks.add_task(service.process_async, job.job_id, request)
        else:
            logger.info(
                'assessment.report.async_reused job_id=%s request_id=%s status=%s',
                job.job_id,
                job.request_id,
                job.status.value,
            )
        return JSONResponse(status_code=202, content=to_camel_case_keys(_job_payload(job)))

    logger.info(
        'assessment.report.sync_started job_id=%s request_id=%s',
        job.job_id,
        job.request_id,
    )
    report = service.process_sync(job.job_id, request)
    job = service.get_job(job.job_id)
    logger.info(
        'assessment.report.sync_ready job_id=%s request_id=%s overall_score=%s questions=%s',
        job.job_id,
        job.request_id,
        report.overall_score,
        len(report.questions),
    )
    return _report_payload(job, report)


@router.get('/report/{job_id}/status', response_model=JobPayload, response_model_by_alias=True)
def get_report_status(
    job_id: str,
    _: None = Depends(verify_api_key),
    service=Depends(get_service),
):
    job = service.get_job(job_id)
    logger.info(
        'assessment.report.status job_id=%s status=%s',
        job_id,
        job.status.value,
    )
    return _job_payload(job)


@router.get('/report/{job_id}', response_model=ReportResponsePayload, response_model_by_alias=True)
def get_report(
    job_id: str,
    _: None = Depends(verify_api_key),
    service=Depends(get_service),
):
    report = service.get_report(job_id)
    job = service.get_job(job_id)
    logger.info(
        'assessment.report.fetch job_id=%s request_id=%s overall_score=%s',
        job_id,
        job.request_id,
        report.overall_score,
    )
    return _report_payload(job, report)



def _question_payload(question) -> dict:
    return {
        'question_id': question.question_id,
        'specialization': question.specialization.value,
        'grade': question.grade.value,
        'topic_code': question.topic,
        'topic_label': topic_label(question.topic),
        'question_text': question.question_text,
        'tags': question.tags,
        'version': question.version,
    }



def _job_payload(job) -> dict:
    return {
        'status': job.status.value,
        'job_id': job.job_id,
        'request_id': job.request_id,
        'session_id': job.session_id,
        'created_at': job.created_at,
        'updated_at': job.updated_at,
        'error_code': job.error_code,
        'error_message': job.error_message,
    }



def _report_payload(job, report=None) -> dict:
    resolved_report = report or job.report
    return {
        'status': 'ready',
        'job': _job_payload(job),
        'report': to_primitive(resolved_report),
    }
