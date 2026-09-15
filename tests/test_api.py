"""Testes automatizados da API de classificação médica."""

from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def test_root() -> None:
    """Valida o endpoint raiz."""
    response = client.get("/")

    assert response.status_code == 200
    assert response.json()["status"] == "running"


def test_health() -> None:
    """Valida o health check e o carregamento do modelo."""
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "healthy",
        "model": "loaded",
    }


def test_predict() -> None:
    """Valida uma inferência completa."""
    payload = {
        "text": (
            "Patient presents with acute chest pain, shortness of breath "
            "and suspected myocardial infarction."
        )
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 200

    body = response.json()

    assert body["class_id"] in [1, 2, 3, 4, 5]
    assert isinstance(body["class_name"], str)
    assert body["latency_ms"] >= 0


def test_predict_rejects_short_text() -> None:
    """Valida o contrato Pydantic para entradas inválidas."""
    response = client.post(
        "/predict",
        json={"text": "short"},
    )

    assert response.status_code == 422


def test_metrics() -> None:
    """Valida a exposição das métricas Prometheus."""
    response = client.get("/metrics")

    assert response.status_code == 200
    assert "medical_api_requests_total" in response.text
    assert "medical_api_request_latency_seconds" in response.text
    assert "medical_api_errors_total" in response.text