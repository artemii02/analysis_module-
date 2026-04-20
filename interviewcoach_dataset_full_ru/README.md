# InterviewCoach Full RU Dataset v1

Полный русскоязычный synthetic-curated dataset для evaluator model модуля пост-сессионной оценки ответов кандидата.

## Scope
- specializations: backend, frontend, devops
- grades: junior, middle
- topics per combination: 5
- questions per topic: 4
- quality bands per question: 5 (empty, weak, medium, good, strong)

## Counts
- question cards: 120
- total records: 600
- train: 360
- eval: 120
- test: 120

## Structure
- question_cards/
- raw/raw_train.jsonl, raw/raw_eval.jsonl, raw/raw_test.jsonl
- export/sft_train.jsonl, export/sft_eval.jsonl, export/sft_test.jsonl
- manifests/sources.csv, split_manifest.csv, dedup_groups.csv
- coverage_matrix.csv
- annotation_guidelines.md
- reports/validation_report.json

## Notes
- Вопросы и темы перефразированы на русском по мотивам публичных материалов Habr и Stack Overflow.
- Содержимое записей не является verbatim-копией исходных страниц.
- Все answer_text и expected_feedback собраны синтетически и требуют human review перед финальным production fine-tune.
