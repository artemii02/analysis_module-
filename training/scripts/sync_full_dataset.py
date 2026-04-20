from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET_ROOT = ROOT / "interviewcoach_dataset_full_ru"
RUNTIME_DATA_DIR = ROOT / "src" / "interview_analysis" / "data"
TRAINING_DATA_DIR = ROOT / "training" / "data"

VERSION_SUFFIX = "2026.04-full-ru-v1"

CRITERION_DESCRIPTIONS = {
    "correctness": "Фактическая корректность ответа и отсутствие ключевых технических ошибок.",
    "completeness": "Покрытие ожидаемых ключевых пунктов вопроса.",
    "clarity": "Структурированность, логичность и понятность объяснения.",
    "practicality": "Связь ответа с практическими инженерными сценариями.",
    "terminology": "Корректное использование технической терминологии.",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Синхронизировать полный RU датасет с runtime-данными и training/data."
    )
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--skip-training-copy", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_root.resolve()
    if not dataset_root.exists():
        raise SystemExit(f"Датасет не найден: {dataset_root}")

    cards = load_cards(dataset_root)
    if not cards:
        raise SystemExit(f"В датасете нет карточек вопросов: {dataset_root / 'question_cards'}")

    RUNTIME_DATA_DIR.mkdir(parents=True, exist_ok=True)
    write_json(RUNTIME_DATA_DIR / "questions.json", build_questions_payload(cards))
    write_json(RUNTIME_DATA_DIR / "rubrics.json", build_rubrics_payload(cards))
    write_json(RUNTIME_DATA_DIR / "knowledge_base.json", build_knowledge_payload(cards, dataset_root))
    write_json(RUNTIME_DATA_DIR / "topic_labels.json", build_topic_labels_payload(cards))

    if not args.skip_training_copy:
        copy_training_files(dataset_root)

    print(f"Runtime questions: {len(cards)}")
    print(f"Runtime topics: {len({card['topic'] for card in cards})}")
    print(f"Runtime data dir: {RUNTIME_DATA_DIR}")
    if not args.skip_training_copy:
        print(f"Training data dir: {TRAINING_DATA_DIR}")


def load_cards(dataset_root: Path) -> list[dict[str, Any]]:
    cards = []
    for path in sorted((dataset_root / "question_cards").rglob("*.json")):
        card = json.loads(path.read_text(encoding="utf-8"))
        card["_source_path"] = path
        validate_card(card, path)
        cards.append(card)
    cards.sort(key=lambda item: (item["specialization"], item["grade"], item["topic"], item["question_id"]))
    return cards


def validate_card(card: dict[str, Any], path: Path) -> None:
    required = {
        "question_id",
        "question_text",
        "specialization",
        "grade",
        "topic",
        "topic_name",
        "rubric",
        "keypoints",
        "recommendation_hints",
    }
    missing = sorted(required - card.keys())
    if missing:
        raise SystemExit(f"{path}: отсутствуют поля {missing}")
    if not card["rubric"].get("criteria") or not card["rubric"].get("weights"):
        raise SystemExit(f"{path}: некорректная rubric")


def build_questions_payload(cards: list[dict[str, Any]]) -> dict[str, Any]:
    version = f"questions-{VERSION_SUFFIX}"
    return {
        "version": version,
        "items": [
            {
                "question_id": card["question_id"],
                "specialization": card["specialization"],
                "grade": card["grade"],
                "topic": card["topic"],
                "question_text": card["question_text"],
                "tags": build_tags(card),
                "version": version,
            }
            for card in cards
        ],
    }


def build_rubrics_payload(cards: list[dict[str, Any]]) -> dict[str, Any]:
    version = f"rubrics-{VERSION_SUFFIX}"
    return {
        "version": version,
        "items": [
            {
                "question_id": card["question_id"],
                "specialization": card["specialization"],
                "grade": card["grade"],
                "topic": card["topic"],
                "criteria": build_criteria(card),
                "keypoints": card["keypoints"],
                "recommendation_hints": card.get("recommendation_hints", []),
                "mistake_patterns": [
                    {"trigger_terms": [], "message": mistake}
                    for mistake in card.get("canonical_mistakes", [])
                ],
                "version": version,
            }
            for card in cards
        ],
    }


def build_criteria(card: dict[str, Any]) -> list[dict[str, Any]]:
    criteria = []
    weights = card["rubric"]["weights"]
    for name in card["rubric"]["criteria"]:
        criteria.append(
            {
                "name": name,
                "weight": float(weights[name]),
                "description": CRITERION_DESCRIPTIONS.get(name, name),
            }
        )
    return criteria


def build_knowledge_payload(cards: list[dict[str, Any]], dataset_root: Path) -> dict[str, Any]:
    version = f"kb-{VERSION_SUFFIX}"
    return {
        "version": version,
        "items": [
            {
                "chunk_id": f"kb_{card['question_id']}",
                "specialization": card["specialization"],
                "topics": [card["topic"]],
                "tags": build_tags(card),
                "content": build_knowledge_content(card),
                "source_title": f"Карточка вопроса {card['question_id']}",
                "source_url": f"local://{card['_source_path'].relative_to(dataset_root).as_posix()}",
                "version": version,
            }
            for card in cards
        ],
    }


def build_knowledge_content(card: dict[str, Any]) -> str:
    keypoints = " ".join(f"{index}. {value}" for index, value in enumerate(card["keypoints"], start=1))
    mistakes = " ".join(
        f"{index}. {value}" for index, value in enumerate(card.get("canonical_mistakes", []), start=1)
    )
    recommendations = " ".join(
        f"{index}. {value}" for index, value in enumerate(card.get("recommendation_hints", []), start=1)
    )
    parts = [
        f"Тема: {card['topic_name']}.",
        f"Вопрос: {card['question_text']}",
        f"Ключевые пункты хорошего ответа: {keypoints}",
    ]
    if mistakes:
        parts.append(f"Типичные ошибки: {mistakes}")
    if recommendations:
        parts.append(f"Рекомендации для повторения: {recommendations}")
    return " ".join(parts)


def build_topic_labels_payload(cards: list[dict[str, Any]]) -> dict[str, Any]:
    labels = {}
    for card in cards:
        labels[card["topic"]] = card["topic_name"]
    return {
        "version": f"topic-labels-{VERSION_SUFFIX}",
        "items": [{"topic": code, "label": labels[code]} for code in sorted(labels)],
    }


def build_tags(card: dict[str, Any]) -> list[str]:
    tags = [
        card["specialization"],
        card["grade"],
        card["topic"],
        card["question_id"],
    ]
    tags.extend(card.get("source_refs", []))
    return sorted(dict.fromkeys(str(tag) for tag in tags if str(tag).strip()))


def copy_training_files(dataset_root: Path) -> None:
    TRAINING_DATA_DIR.mkdir(parents=True, exist_ok=True)
    for split in ("train", "eval", "test"):
        copy_required(
            dataset_root / "raw" / f"raw_{split}.jsonl",
            TRAINING_DATA_DIR / f"raw_{split}.jsonl",
        )
        copy_required(
            dataset_root / "export" / f"sft_{split}.jsonl",
            TRAINING_DATA_DIR / f"sft_{split}.jsonl",
        )


def copy_required(source: Path, target: Path) -> None:
    if not source.exists():
        raise SystemExit(f"Файл датасета не найден: {source}")
    shutil.copyfile(source, target)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
