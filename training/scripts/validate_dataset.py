from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FILES = [
    ROOT / "training" / "data" / "raw_train.jsonl",
    ROOT / "training" / "data" / "raw_eval.jsonl",
    ROOT / "training" / "data" / "raw_test.jsonl",
]
REQUIRED_KEYS = {
    "record_id",
    "split",
    "specialization",
    "grade",
    "topic",
    "question_id",
    "question_text",
    "answer_text",
    "keypoints",
    "expected_feedback",
}
EXPECTED_FEEDBACK_KEYS = {
    "score",
    "criterion_scores",
    "summary",
    "strengths",
    "issues",
    "covered_keypoints",
    "missing_keypoints",
    "detected_mistakes",
    "recommendations",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Проверка структуры raw JSONL датасета.")
    parser.add_argument("files", nargs="*", type=Path, default=DEFAULT_FILES)
    return parser.parse_args()


def validate_record(record: dict, source: Path, line_no: int) -> list[str]:
    errors: list[str] = []
    missing_keys = REQUIRED_KEYS - record.keys()
    if missing_keys:
        errors.append(f"{source}:{line_no} missing keys: {sorted(missing_keys)}")
        return errors

    feedback = record["expected_feedback"]
    missing_feedback_keys = EXPECTED_FEEDBACK_KEYS - feedback.keys()
    if missing_feedback_keys:
        errors.append(f"{source}:{line_no} missing feedback keys: {sorted(missing_feedback_keys)}")

    score = feedback.get("score", 0)
    if not isinstance(score, int) or not (0 <= score <= 100):
        errors.append(f"{source}:{line_no} invalid score: {score}")

    for criterion_name, criterion_score in feedback.get("criterion_scores", {}).items():
        if not isinstance(criterion_score, int) or not (0 <= criterion_score <= 100):
            errors.append(f"{source}:{line_no} invalid criterion score for {criterion_name}: {criterion_score}")

    keypoints = set(record["keypoints"])
    covered = set(feedback.get("covered_keypoints", []))
    missing = set(feedback.get("missing_keypoints", []))
    if not covered and not missing:
        errors.append(f"{source}:{line_no} both covered_keypoints and missing_keypoints are empty")
    if covered & missing:
        errors.append(f"{source}:{line_no} covered_keypoints and missing_keypoints overlap")
    if not (covered | missing).issubset(keypoints):
        errors.append(f"{source}:{line_no} covered/missing keypoints contain values outside original keypoints")
    return errors


def main() -> None:
    args = parse_args()
    all_errors: list[str] = []
    total_records = 0
    for path in args.files:
        with path.open("r", encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                total_records += 1
                record = json.loads(line)
                all_errors.extend(validate_record(record, path, line_no))

    if all_errors:
        for error in all_errors:
            print(error)
        raise SystemExit(1)

    print(f"Dataset validation passed: {total_records} records checked.")


if __name__ == "__main__":
    main()
