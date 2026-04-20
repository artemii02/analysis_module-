# Обучение и проверка модели

Эта папка содержит скрипты для подготовки датасета, экспорта SFT-формата, запуска LoRA/QLoRA-дообучения и проверки качества модуля анализа ответов.

## Текущий датасет

Основной датасет первой итерации находится в `interviewcoach_dataset_v1` и синхронизирован в `training/data`:

- `raw_train.jsonl` — обучающая часть.
- `raw_eval.jsonl` — валидационная часть.
- `raw_test.jsonl` — тестовая часть.
- `sft_train.jsonl`, `sft_eval.jsonl`, `sft_test.jsonl` — формат для supervised fine-tuning.

Текущая область: backend, junior, русский язык, 24 вопроса и 120 размеченных ответов.

## Проверка датасета

```powershell
python training/scripts/validate_dataset.py
```

Ожидаемый результат: сообщение `Dataset validation passed`.

## Пересборка SFT-файлов

```powershell
python training/scripts/export_sft_dataset.py
```

Скрипт пересобирает `sft_train.jsonl`, `sft_eval.jsonl` и `sft_test.jsonl` из raw-разметки.

## Дообучение

Установить зависимости:

```powershell
pip install -e ".[dev,training]"
```

Проверить, что модель, датасет и `SFTTrainer` инициализируются без запуска обучения:

```powershell
python training/scripts/finetune_lora.py --config training/configs/lora_config.example.json --dry-run
```

Запустить обучение:

```powershell
python training/scripts/finetune_lora.py --config training/configs/lora_config.example.json
```

Результат сохраняется в `training/artifacts/qwen2.5-3b-interview-v1`.

Перед финальным дообучением нужно вручную проверить часть synthetic-разметки, особенно ответы из групп `medium`, `good` и спорные критерии оценивания.

## Проверка качества работающего сервиса

```powershell
python training/scripts/benchmark_runtime.py --input training/data/raw_eval.jsonl --output training/reports/predictions_eval.jsonl
```

Скрипт отправляет записи в API модуля и считает метрики отклонения от эталонной разметки.
