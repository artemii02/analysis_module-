from __future__ import annotations

from interview_analysis.core.topic_catalog import topic_label


def test_topic_label_returns_human_readable_russian_titles() -> None:
    assert topic_label('http_rest') == 'HTTP и REST'
    assert topic_label('sql_indexes') == 'SQL и индексы'
    assert topic_label('react_architecture') == 'Архитектура React-приложения'
    assert topic_label('terraform_iac') == 'Terraform и IaC'
    assert topic_label('unknown_topic') == 'Unknown topic'
