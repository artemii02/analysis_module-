from __future__ import annotations

import logging

from fastapi import APIRouter, Depends

from interview_analysis.api.dependencies import get_service, verify_api_key
from interview_analysis.core.config import Settings, get_settings
from interview_analysis.core.serialization import to_camel_case_keys
from interview_analysis.schemas.api import HealthResponsePayload


router = APIRouter()
logger = logging.getLogger(__name__)


@router.get('/health', response_model=HealthResponsePayload, response_model_by_alias=True)
def health(service=Depends(get_service)) -> dict:
    settings = get_settings()
    payload = {
        'status': 'ok',
        'service': settings.app_name,
        'llm_mode': settings.llm_mode,
        'llm_model': _llm_model_name(settings),
        'api_prefix': settings.api_prefix,
        'job_store': service.health_snapshot(),
    }
    logger.info(
        'health.snapshot llm_mode=%s job_store_backend=%s status=%s',
        payload['llm_mode'],
        payload['job_store'].get('backend', 'unknown'),
        payload['job_store'].get('status', 'unknown'),
    )
    return payload


@router.get('/metrics')
def metrics(
    _: None = Depends(verify_api_key),
    service=Depends(get_service),
) -> dict:
    payload = {
        'status': 'ok',
        'metrics': service.metrics_snapshot(),
        'job_store': service.health_snapshot(),
    }
    logger.info(
        'health.metrics requests_total=%s successes=%s failures=%s',
        payload['metrics'].get('requests_total', 0),
        payload['metrics'].get('success_total', 0),
        payload['metrics'].get('failure_total', 0),
    )
    return to_camel_case_keys(payload)


def _llm_model_name(settings: Settings) -> str:
    if settings.llm_mode == 'ollama':
        return settings.ollama_model
    if settings.llm_mode == 'hf':
        if settings.hf_adapter_path:
            return f'{settings.hf_base_model}+{settings.hf_adapter_path.name}'
        return settings.hf_base_model
    return 'mock-heuristic-v1'
