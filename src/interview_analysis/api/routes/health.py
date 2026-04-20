from __future__ import annotations

from fastapi import APIRouter, Depends

from interview_analysis.api.dependencies import get_service, verify_api_key
from interview_analysis.core.config import Settings, get_settings
from interview_analysis.schemas.api import HealthResponsePayload


router = APIRouter()


@router.get('/health', response_model=HealthResponsePayload)
def health(service=Depends(get_service)) -> dict:
    settings = get_settings()
    return {
        'status': 'ok',
        'service': settings.app_name,
        'llm_mode': settings.llm_mode,
        'llm_model': _llm_model_name(settings),
        'api_prefix': settings.api_prefix,
        'job_store': service.health_snapshot(),
    }


@router.get('/metrics')
def metrics(
    _: None = Depends(verify_api_key),
    service=Depends(get_service),
) -> dict:
    return {
        'status': 'ok',
        'metrics': service.metrics_snapshot(),
        'job_store': service.health_snapshot(),
    }


def _llm_model_name(settings: Settings) -> str:
    if settings.llm_mode == 'ollama':
        return settings.ollama_model
    if settings.llm_mode == 'hf':
        if settings.hf_adapter_path:
            return f'{settings.hf_base_model}+{settings.hf_adapter_path.name}'
        return settings.hf_base_model
    return 'mock-heuristic-v1'
