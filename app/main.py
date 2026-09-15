"""API REST para inferência do classificador de textos médicos."""

from pathlib import Path
from time import perf_counter

import joblib
from fastapi import FastAPI, HTTPException, Response
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Histogram,
    generate_latest,
)
from pydantic import BaseModel, Field


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "medical_text_classifier.joblib"

CLASS_NAMES = {
    1: "Neoplasms",
    2: "Digestive system diseases",
    3: "Nervous system diseases",
    4: "Cardiovascular diseases",
    5: "General pathological conditions",
}


class PredictionRequest(BaseModel):
    """Contrato de entrada da API."""

    text: str = Field(
        ...,
        min_length=10,
        description="Texto do abstract ou laudo médico.",
    )


class PredictionResponse(BaseModel):
    """Contrato de saída da API."""

    class_id: int
    class_name: str
    latency_ms: float


REQUEST_COUNT = Counter(
    "medical_api_requests_total",
    "Total de requisicoes de inferencia recebidas.",
)

ERROR_COUNT = Counter(
    "medical_api_errors_total",
    "Total de erros ocorridos durante inferencias.",
)

REQUEST_LATENCY = Histogram(
    "medical_api_request_latency_seconds",
    "Latencia da inferencia em segundos.",
)


def load_model():
    """Carrega o pipeline treinado durante a inicialização."""
    if not MODEL_PATH.exists():
        raise RuntimeError(
            f"Modelo nao encontrado em {MODEL_PATH}. "
            "Execute 'python -m src.train' primeiro."
        )

    return joblib.load(MODEL_PATH)


model = load_model()

app = FastAPI(
    title="Medical Text Classification API",
    description=(
        "API de classificacao de textos medicos desenvolvida "
        "para o Tech Challenge - Fase 3."
    ),
    version="1.0.0",
)


@app.get("/")
def root() -> dict[str, str]:
    """Informações básicas da aplicação."""
    return {
        "service": "Medical Text Classification API",
        "version": "1.0.0",
        "status": "running",
    }


@app.get("/health")
def health() -> dict[str, str]:
    """Health check da aplicação."""
    return {
        "status": "healthy",
        "model": "loaded",
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(payload: PredictionRequest) -> PredictionResponse:
    """Classifica um texto médico e mede a latência da inferência."""
    REQUEST_COUNT.inc()
    start_time = perf_counter()

    try:
        prediction = int(model.predict([payload.text])[0])

        latency_seconds = perf_counter() - start_time
        REQUEST_LATENCY.observe(latency_seconds)

        return PredictionResponse(
            class_id=prediction,
            class_name=CLASS_NAMES.get(prediction, "Unknown"),
            latency_ms=round(latency_seconds * 1000, 3),
        )

    except Exception as exc:
        ERROR_COUNT.inc()
        raise HTTPException(
            status_code=500,
            detail="Erro durante a inferencia.",
        ) from exc


@app.get("/metrics")
def metrics() -> Response:
    """Expõe métricas no formato esperado pelo Prometheus."""
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST,
    )