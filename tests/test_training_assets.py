from __future__ import annotations

import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TRAIN_PATH = ROOT / 'training' / 'data' / 'raw_train.jsonl'
EVAL_PATH = ROOT / 'training' / 'data' / 'raw_eval.jsonl'
TEST_PATH = ROOT / 'training' / 'data' / 'raw_test.jsonl'

EXPECTED_PAIRS = {
    ('backend', 'junior'),
    ('backend', 'middle'),
    ('frontend', 'junior'),
    ('frontend', 'middle'),
    ('devops', 'junior'),
    ('devops', 'middle'),
}


def _load(path: Path) -> list[dict]:
    with path.open('r', encoding='utf-8') as handle:
        return [json.loads(line) for line in handle if line.strip()]


def test_training_dataset_is_full_ru_v1() -> None:
    records = _load(TRAIN_PATH) + _load(EVAL_PATH) + _load(TEST_PATH)
    pairs = {(record['specialization'], record['grade']) for record in records}
    topics = {record['topic'] for record in records}

    assert pairs == EXPECTED_PAIRS
    assert len(records) == 600
    assert len(topics) == 30
    assert all(record['question_text'] and record['answer_text'] for record in records)
    assert all(record['expected_feedback']['summary'] for record in records)


def test_training_dataset_keeps_balanced_splits() -> None:
    assert len(_load(TRAIN_PATH)) == 360
    assert len(_load(EVAL_PATH)) == 120
    assert len(_load(TEST_PATH)) == 120


def test_training_dataset_covers_each_profile_in_each_split() -> None:
    for path, expected_count in [(TRAIN_PATH, 60), (EVAL_PATH, 20), (TEST_PATH, 20)]:
        counts = Counter((record['specialization'], record['grade']) for record in _load(path))
        assert set(counts) == EXPECTED_PAIRS
        assert all(count == expected_count for count in counts.values())
