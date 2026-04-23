from __future__ import annotations

from fastapi.testclient import TestClient

from interview_analysis.main import app


API_HEADERS = {'X-API-Key': 'demo-api-key'}


def test_demo_page_is_available() -> None:
    client = TestClient(app)
    response = client.get('/demo')

    assert response.status_code == 200
    assert 'id="runButton"' in response.text
    assert 'window.__DEMO_CONFIG__ = {' in response.text
    assert '__DEMO_CONFIG_JSON__' not in response.text
    assert 'function resetReportView()' in response.text
    assert '[hidden] { display: none !important; }' in response.text
    assert 'model_version ||' in response.text
    assert 'questionsUrl' in response.text
    assert 'questionProfiles' in response.text
    assert 'Raw JSON' not in response.text
    assert 'feedback' not in response.text
    assert '<h2>JSON-' in response.text
    assert 'job ${escapeHtml' not in response.text


def test_demo_cases_endpoint_returns_seed_cases() -> None:
    client = TestClient(app)
    response = client.get('/demo/cases')

    payload = response.json()
    assert response.status_code == 200
    assert len(payload) >= 3
    assert any(item['case_id'] == 'backend_junior_good' for item in payload)


def test_question_bank_endpoint_returns_backend_junior_questions() -> None:
    client = TestClient(app)
    response = client.get(
        '/assessment/v1/questions?specialization=backend&grade=junior&limit=10',
        headers=API_HEADERS,
    )

    payload = response.json()
    assert response.status_code == 200
    assert payload['status'] == 'ok'
    assert payload['specialization'] == 'backend'
    assert payload['grade'] == 'junior'
    assert payload['count'] == 10
    assert len(payload['items']) == 10
    assert payload['items'][0]['topicLabel']
    assert 'questionId' in payload['items'][0]
    assert 'question_id' not in payload['items'][0]


def test_health_endpoint_exposes_llm_runtime() -> None:
    client = TestClient(app)
    response = client.get('/assessment/v1/health')

    payload = response.json()
    assert response.status_code == 200
    assert payload['status'] == 'ok'
    assert 'llmMode' in payload
    assert 'llmModel' in payload
    assert 'jobStore' in payload


def test_report_endpoint_accepts_camel_case_payload() -> None:
    client = TestClient(app)
    response = client.post(
        '/assessment/v1/report',
        headers=API_HEADERS,
        json={
            'requestId': 'req-camel-001',
            'sessionId': 'session-camel-001',
            'clientId': 'main-backend',
            'mode': 'sync',
            'scenario': {
                'scenarioId': 'backend_junior_session',
                'specialization': 'backend',
                'grade': 'junior',
                'topics': ['http_rest'],
                'reportLanguage': 'ru',
            },
            'items': [
                {
                    'itemId': 'item-1',
                    'questionId': 'be_junior_http_rest_003',
                    'questionText': 'В чём разница между PUT и PATCH?',
                    'answerText': 'хз',
                    'askedAt': '2026-04-23T22:02:46.739301Z',
                    'tags': [],
                }
            ],
            'metadata': {'source': 'main-backend'},
        },
    )

    payload = response.json()
    assert response.status_code == 200
    assert payload['status'] == 'ready'
    assert payload['job']['requestId'] == 'req-camel-001'
    assert payload['report']['requestId'] == 'req-camel-001'
    assert payload['report']['questions'][0]['questionId'] == 'be_junior_http_rest_003'
    assert 'overallScore' in payload['report']
    assert 'criterionScores' in payload['report']
    assert 'request_id' not in payload['report']
