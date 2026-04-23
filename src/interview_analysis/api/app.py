from __future__ import annotations

import logging
from time import perf_counter

from fastapi import FastAPI, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse

from interview_analysis import __version__
from interview_analysis.api.dependencies import get_service
from interview_analysis.api.routes import assessment, demo, health
from interview_analysis.core.config import Settings, get_settings
from interview_analysis.core.serialization import to_camel_case_keys
from interview_analysis.exceptions import AnalysisError


logger = logging.getLogger(__name__)


def create_app() -> FastAPI:
    settings = get_settings()
    _configure_logging(settings)
    app = FastAPI(
        title=settings.app_name,
        version=__version__,
        description='Сервис анализа ответов пользователя на техническом интервью с локальной LLM и JSON-отчётом для интеграции.',
    )
    app.include_router(demo.router, tags=['demo'])
    app.include_router(health.router, prefix=settings.api_prefix, tags=['health'])
    app.include_router(assessment.router, prefix=settings.api_prefix, tags=['assessment'])

    @app.middleware('http')
    async def log_http_requests(request: Request, call_next):
        started = perf_counter()
        logger.info(
            'http.request.started method=%s path=%s client=%s',
            request.method,
            request.url.path,
            request.client.host if request.client else 'unknown',
        )
        try:
            response = await call_next(request)
        except Exception:
            logger.exception(
                'http.request.failed method=%s path=%s duration_ms=%s',
                request.method,
                request.url.path,
                _duration_ms(started),
            )
            raise
        logger.info(
            'http.request.completed method=%s path=%s status_code=%s duration_ms=%s',
            request.method,
            request.url.path,
            response.status_code,
            _duration_ms(started),
        )
        return response

    @app.on_event('startup')
    async def warmup_llm() -> None:
        if not settings.warmup_llm_on_start or settings.llm_mode != 'hf':
            logger.info(
                'startup.warmup.skipped warmup_enabled=%s llm_mode=%s',
                settings.warmup_llm_on_start,
                settings.llm_mode,
            )
            return
        logger.info('startup.warmup.started llm_mode=%s', settings.llm_mode)
        service = get_service()
        loader = getattr(service.pipeline.llm_provider, '_load', None)
        if callable(loader):
            await run_in_threadpool(loader)
        logger.info('startup.warmup.completed llm_mode=%s', settings.llm_mode)

    @app.get('/')
    def root() -> dict:
        return to_camel_case_keys(
            {
                'service': settings.app_name,
                'version': __version__,
                'health_url': f'{settings.api_prefix}/health',
                'demo_url': '/demo',
            }
        )

    @app.exception_handler(AnalysisError)
    async def handle_analysis_error(_: Request, exc: AnalysisError) -> JSONResponse:
        logger.warning(
            'http.analysis_error code=%s message=%s status_code=%s',
            exc.code,
            exc.message,
            exc.status_code,
        )
        return JSONResponse(
            status_code=exc.status_code,
            content=to_camel_case_keys(
                {
                    'status': 'error',
                    'error_code': exc.code,
                    'message': exc.message,
                    'details': exc.details,
                }
            ),
        )

    return app


def _configure_logging(settings: Settings) -> None:
    level = getattr(logging, settings.log_level.upper(), logging.INFO)
    package_logger = logging.getLogger('interview_analysis')
    package_logger.setLevel(level)
    if not logging.getLogger().handlers:
        logging.basicConfig(
            level=level,
            format='%(asctime)s %(levelname)s %(name)s %(message)s',
        )


def _duration_ms(started_at: float) -> int:
    return round((perf_counter() - started_at) * 1000)
