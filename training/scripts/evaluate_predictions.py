from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = ROOT / "training" / "reports" / "console_eval_predictions.jsonl"
DEFAULT_OUTPUT = ROOT / "training" / "reports" / "quality_metrics.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Расчет метрик качества по JSONL с предсказаниями модуля.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = load_jsonl(args.input)
    if not rows:
        raise SystemExit(f"Файл не содержит предсказаний: {args.input}")

    metrics = {
        "records": len(rows),
        "overall": build_metrics(rows),
        "by_quality_band": build_grouped_metrics(rows, "answer_quality_band"),
        "by_profile": build_grouped_metrics(rows, "profile"),
        "by_topic": build_grouped_metrics(rows, "topic"),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print("Метрики качества")
    print(f"Файл: {args.input}")
    print(f"Записей: {metrics['records']}")
    print(f"MAE: {metrics['overall']['mae']:.2f}")
    print(f"RMSE: {metrics['overall']['rmse']:.2f}")
    print(f"Within 10 points: {metrics['overall']['within_10']:.3f}")
    print(f"Within 15 points: {metrics['overall']['within_15']:.3f}")
    print(f"Метрики сохранены: {args.output}")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        rows = []
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if "profile" not in row:
                row["profile"] = f"{row.get('specialization', 'unknown')}/{row.get('grade', 'unknown')}"
            rows.append(row)
        return rows


def build_grouped_metrics(rows: list[dict[str, Any]], key: str) -> dict[str, dict[str, float | int]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row.get(key) or "unknown")].append(row)
    return {group: build_metrics(items) for group, items in sorted(groups.items())}


def build_metrics(rows: list[dict[str, Any]]) -> dict[str, float | int]:
    errors = [abs(int(row["predicted_score"]) - int(row["expected_score"])) for row in rows]
    signed_errors = [int(row["predicted_score"]) - int(row["expected_score"]) for row in rows]
    return {
        "count": len(rows),
        "mae": mean(errors),
        "rmse": math.sqrt(mean(error * error for error in errors)),
        "bias": mean(signed_errors),
        "max_error": max(errors),
        "within_10": sum(1 for error in errors if error <= 10) / len(errors),
        "within_15": sum(1 for error in errors if error <= 15) / len(errors),
    }


if __name__ == "__main__":
    main()
