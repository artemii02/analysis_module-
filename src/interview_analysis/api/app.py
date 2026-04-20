from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from interview_analysis import __version__
from interview_analysis.api.routes import assessment, demo, health
from interview_analysis.core.config import get_settings
from interview_analysis.exceptions import AnalysisError


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(
        title=settings.app_name,
        version=__version__,
        description='Сервис анализа ответов пользователя на техническом интервью с локальной LLM и JSON-отчётом для интеграции.',
    )
    app.include_router(demo.router, tags=['demo'])
    app.include_router(health.router, prefix=settings.api_prefix, tags=['health'])
    app.include_router(assessment.router, prefix=settings.api_prefix, tags=['assessment'])

    @app.get('/')
    def root() -> dict:
        return {
            'service': settings.app_name,
            'version': __version__,
            'health_url': f'{settings.api_prefix}/health',
            'demo_url': '/demo',
        }

    @app.exception_handler(AnalysisError)
    async def handle_analysis_error(_: Request, exc: AnalysisError) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content={
                'status': 'error',
                'error_code': exc.code,
                'message': exc.message,
                'details': exc.details,
            },
        )

    return app
