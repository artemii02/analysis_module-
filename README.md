# Модуль анализа ответов пользователя

Автономный модуль для анализа ответов кандидата после технического интервью. Модуль принимает вопросы и ответы пользователя, сопоставляет их с рубриками и локальной базой знаний, формирует оценку по критериям и возвращает итоговый JSON-отчёт. Работает через консоль и через REST API с OpenAPI-описанием.

## Состав модуля

- Банк вопросов: 120 карточек на русском языке.
- Профили: Backend, Frontend, DevOps.
- Грейды: Junior, Middle.
- Размеченный датасет: 600 записей, split `train/eval/test` = 360/120/120.
- Локальная базовая модель: `Qwen/Qwen2.5-3B-Instruct`.
- Дообученный адаптер: LoRA/QLoRA `training/artifacts/qwen2.5-3b-interview-full-ru-qlora-v1`.
- Runtime-провайдеры: `mock`, `ollama`, `hf`.
- Хранилище задач API: in-memory или PostgreSQL.
- Ограничения API: до 20 вопросов в одной сессии, до 4000 символов в одном ответе.

## Режимы работы

`hf` — основной рабочий режим модуля. Базовая модель Qwen 2.5 3B и LoRA-адаптер загружаются локально через Transformers/PEFT. Этот режим используется для демонстрации итогового модуля и проверки качества модели.

`mock` — быстрый режим без загрузки LLM. Используется для проверки консольных команд, API, структуры JSON и общей работоспособности пайплайна.

`ollama` — Docker-режим для запуска API вместе с PostgreSQL и Ollama. Он использует модель `qwen2.5:3b` из Ollama и не подключает LoRA-адаптер из `training/artifacts`.

## Установка для локального HF/LoRA запуска

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -e ".[dev,training,qlora]"
pip install --force-reinstall torch --index-url https://download.pytorch.org/whl/cu128
pip install fsspec==2025.3.0
```

Проверить CUDA:

```powershell
.\.venv\Scripts\python.exe -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no cuda')"
```

При первом запуске `hf` модель скачивается в кеш Hugging Face. Последующие запуски используют локальный кеш.

## Быстрый запуск консоли

Показать вопросы:

```powershell
.\console.bat questions --specialization backend --grade junior --limit 10
```

Быстро сформировать отчёт без загрузки LLM:

```powershell
.\console.bat sample --quality good --specialization backend --grade junior --limit 10 --llm mock --output training\reports\mock_report.json
```

Сформировать отчёт на готовом ответе из датасета через дообученную локальную модель:

```powershell
.\console.bat sample --quality good --specialization backend --grade junior --limit 1 --llm hf --output training\reports\hf_3b_lora_sample_report.json
```

Пройти интервью вручную в консоли:

```powershell
.\console.bat run --specialization backend --grade junior --limit 10 --llm hf --output training\reports\manual_hf_report.json
```

## Как строится оценка

1. Модуль получает `question_id`, текст вопроса и ответ пользователя.
2. По `question_id`, профилю и грейду загружаются карточка вопроса, рубрика, ключевые пункты ответа и фрагмент локальной базы знаний.
3. Если ответ является заглушкой или слишком коротким текстом (`хз`, `не знаю`, пустой ответ и похожие случаи), LLM не вызывается: модуль возвращает низкую детерминированную оценку.
4. Для содержательного ответа `hf` или `ollama` провайдер формирует JSON с оценками по критериям `correctness`, `completeness`, `clarity`, `practicality`, `terminology`.
5. Балл за вопрос считается как взвешенное среднее критериев по рубрике вопроса.
6. Итоговый балл сессии считается как среднее значение баллов по всем вопросам.
7. В отчёт добавляются сильные стороны, проблемы, покрытые и пропущенные ключевые пункты, рекомендации и версии модели/рубрик/базы знаний.

## Проверка качества

Запустить оценку на части eval-датасета:

```powershell
.\console.bat evaluate --limit 5 --llm hf --output training\reports\hf_3b_lora_eval_predictions.jsonl
```

Посчитать метрики по сохранённым предсказаниям:

```powershell
.\.venv\Scripts\python.exe training\scripts\evaluate_predictions.py --input training\reports\hf_3b_lora_eval_predictions.jsonl --output training\reports\hf_3b_lora_quality_metrics.json
```

Основные метрики:

- `MAE` — средняя абсолютная ошибка в баллах.
- `RMSE` — среднеквадратичная ошибка, сильнее штрафует крупные промахи.
- `bias` — среднее смещение прогноза: положительное значение означает завышение оценок, отрицательное — занижение.
- `within_10` — доля ответов, где модель ошиблась не более чем на 10 баллов.
- `within_15` — доля ответов, где модель ошиблась не более чем на 15 баллов.

Метрики дополнительно считаются по quality band, профилям и темам.

## REST API и OpenAPI

Запуск API в локальном HF/LoRA режиме:

```powershell
.\start_hf_api.bat
```

Запуск API в Docker/Ollama режиме:

```powershell
.\start.bat
```

Docker-режим поднимает три сервиса: `analysis-module`, `postgres`, `ollama`.

После запуска доступны:

- Swagger UI: `http://127.0.0.1:8000/docs`
- OpenAPI JSON: `http://127.0.0.1:8000/openapi.json`
- Health: `http://127.0.0.1:8000/assessment/v1/health`
- Метрики: `http://127.0.0.1:8000/assessment/v1/metrics`
- Demo UI: `http://127.0.0.1:8000/demo`

Остановка Docker-режима:

```powershell
.\stop.bat
```

Получить список вопросов:

```http
GET /assessment/v1/questions?specialization=backend&grade=junior&limit=10
X-API-Key: demo-api-key
```

Сформировать отчёт:

```http
POST /assessment/v1/report
X-API-Key: demo-api-key
Content-Type: application/json
```

Проверить статус асинхронной задачи:

```http
GET /assessment/v1/report/{job_id}/status
X-API-Key: demo-api-key
```

Получить готовый отчёт:

```http
GET /assessment/v1/report/{job_id}
X-API-Key: demo-api-key
```

Пример тела запроса:

```json
{
  "request_id": "req-1001",
  "session_id": "session-1001",
  "client_id": "main-backend",
  "mode": "sync",
  "scenario": {
    "scenario_id": "backend_junior_session",
    "specialization": "backend",
    "grade": "junior",
    "topics": ["http_rest", "sql_indexes"],
    "report_language": "ru"
  },
  "items": [
    {
      "item_id": "item-1",
      "question_id": "be_junior_http_rest_003",
      "question_text": "В чём разница между PUT и PATCH?",
      "answer_text": "PUT обычно заменяет ресурс целиком, PATCH применяют для частичного обновления. Идемпотентность PATCH зависит от операции.",
      "tags": ["http_rest"]
    }
  ],
  "metadata": {
    "source": "main-backend"
  }
}
```

Ответ API содержит `job` и `report`. В `report` возвращаются:

- `overall_score` — общий балл сессии.
- `criterion_scores` — агрегированные оценки по критериям.
- `summary` — краткое резюме.
- `questions` — детальный разбор каждого ответа.
- `topics` — агрегированный разбор по темам.
- `recommendations` — итоговые рекомендации.
- `versions` — версии модели, рубрик, базы знаний, банка вопросов и промпта.

## Настройки модели и API

Основные переменные окружения:

```powershell
$env:ANALYSIS_API_KEY="demo-api-key"
$env:ANALYSIS_LLM_MODE="hf"
$env:ANALYSIS_JOB_STORE_BACKEND="memory"
$env:ANALYSIS_HF_BASE_MODEL="Qwen/Qwen2.5-3B-Instruct"
$env:ANALYSIS_HF_ADAPTER_PATH="training/artifacts/qwen2.5-3b-interview-full-ru-qlora-v1"
$env:ANALYSIS_HF_LOAD_IN_4BIT="true"
$env:ANALYSIS_HF_MAX_NEW_TOKENS="900"
$env:ANALYSIS_MAX_SESSION_ITEMS="20"
$env:ANALYSIS_MAX_ANSWER_LENGTH="4000"
```

По умолчанию режим `hf` использует адаптер `training/artifacts/qwen2.5-3b-interview-full-ru-qlora-v1`.

## Датасет

Полный датасет находится в `interviewcoach_dataset_full_ru`.

Структура датасета:

- `question_cards` — карточки вопросов с рубриками и ключевыми пунктами.
- `raw` — размеченные записи для обучения и оценки.
- `export` — SFT-датасет для дообучения.
- `coverage_matrix.csv` — покрытие профилей, грейдов и тем.
- `annotation_guidelines.md` — правила разметки.
- `manifests/sources.csv` — источники и ссылки на материалы.
- `reports/validation_report.json` — отчёт валидации датасета.

Синхронизировать датасет с runtime-файлами и `training/data`:

```powershell
.\.venv\Scripts\python.exe training\scripts\sync_full_dataset.py
```

Проверить структуру датасета:

```powershell
.\.venv\Scripts\python.exe training\scripts\validate_dataset.py
```

Ожидаемый результат:

```text
Dataset validation passed: 600 records checked.
```

## Дообучение модели

Smoke-run 3B QLoRA:

```powershell
.\.venv\Scripts\python.exe training\scripts\finetune_lora.py --config training\configs\qlora_rtx3060_6gb_smoke.json
```

Полное дообучение 3B QLoRA:

```powershell
.\.venv\Scripts\python.exe training\scripts\finetune_lora.py --config training\configs\qlora_rtx3060_6gb.json
```

Проверить инициализацию без запуска обучения:

```powershell
.\.venv\Scripts\python.exe training\scripts\finetune_lora.py --config training\configs\qlora_rtx3060_6gb.json --dry-run
```

Финальный адаптер сохраняется в:

```text
training/artifacts/qwen2.5-3b-interview-full-ru-qlora-v1
```

После дообучения этот путь нужно оставить в `ANALYSIS_HF_ADAPTER_PATH`.
