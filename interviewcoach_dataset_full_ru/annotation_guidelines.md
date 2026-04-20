# Annotation Guidelines

## Цель
Датасет предназначен для rubric-based evaluator model, а не для обычного QA-бота.

## Выход модели
- score
- criterion_scores
- summary
- strengths
- issues
- covered_keypoints
- missing_keypoints
- detected_mistakes
- recommendations

## Критерии
- correctness — фактологическая корректность
- completeness — полнота покрытия вопроса
- clarity — ясность и структура ответа
- practicality — практические примеры и применимость
- terminology — корректная техническая терминология

## Веса
- correctness: 0.35
- completeness: 0.25
- clarity: 0.15
- practicality: 0.15
- terminology: 0.10

## Шкала общего score
- 0–5 — пустой / отказ / low-signal
- 6–25 — очень слабый
- 26–45 — слабый, есть фрагменты понимания
- 46–65 — средний ответ
- 66–80 — хороший ответ
- 81–100 — сильный ответ

## Инварианты
- covered_keypoints и missing_keypoints — подмножества keypoints
- covered_keypoints ∩ missing_keypoints = ∅
- strengths не заполняются для empty-ответов
- summary должен быть коротким и factual
- detected_mistakes не должен дублировать issues слово в слово

## Важно
Этот пакет собран как synthetic-curated bootstrap dataset на основе публичных тем и вопросов. Перед финальным fine-tune рекомендуется ручной double-check хотя бы 20–30% записей.
