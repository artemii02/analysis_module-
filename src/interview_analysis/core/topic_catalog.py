from __future__ import annotations

import json
from pathlib import Path


DATA_DIR = Path(__file__).resolve().parents[1] / "data"
TOPIC_LABELS_PATH = DATA_DIR / "topic_labels.json"

BUILTIN_TOPIC_LABELS: dict[str, str] = {
    "http_rest": "HTTP и REST",
    "api_design": "Дизайн API и идемпотентность",
    "jwt_auth": "JWT и безопасность",
    "auth_jwt": "Аутентификация и JWT",
    "sql_indexes": "SQL и индексы",
    "transactions": "Транзакции",
    "caching": "Кэширование",
    "caching_redis": "Кэширование и Redis",
    "http_api": "HTTP API и REST-подход",
    "sql_performance": "SQL: индексы и производительность",
    "distributed_systems": "Распределенные системы и надежность",
    "react_basics": "React: основы компонентов",
    "frontend_architecture": "Frontend-архитектура и состояние приложения",
    "containers": "Docker и контейнеризация",
    "delivery_pipeline": "CI/CD и pipeline доставки",
}


def _load_dataset_topic_labels() -> dict[str, str]:
    if not TOPIC_LABELS_PATH.exists():
        return {}
    payload = json.loads(TOPIC_LABELS_PATH.read_text(encoding="utf-8-sig"))
    return {
        str(item["topic"]): str(item["label"])
        for item in payload.get("items", [])
        if item.get("topic") and item.get("label")
    }


TOPIC_LABELS: dict[str, str] = {**BUILTIN_TOPIC_LABELS, **_load_dataset_topic_labels()}


def topic_label(topic_code: str) -> str:
    if topic_code in TOPIC_LABELS:
        return TOPIC_LABELS[topic_code]
    return topic_code.replace("_", " ").strip().capitalize()
