"""Benchmark de latência e throughput: scikit-learn vs ONNX Runtime."""

from pathlib import Path
from time import perf_counter

import joblib
import numpy as np
import onnxruntime as ort

PROJECT_ROOT = Path(__file__).resolve().parents[1]

BASELINE_PATH = PROJECT_ROOT / "models" / "medical_text_classifier.joblib"
VECTORIZER_PATH = PROJECT_ROOT / "models" / "tfidf_vectorizer.joblib"
ONNX_PATH = PROJECT_ROOT / "models" / "medical_classifier.onnx"

WARMUP_RUNS = 20
BENCHMARK_RUNS = 500

SAMPLE_TEXT = (
    "Patient presents with acute chest pain, shortness of breath, "
    "palpitations and suspected myocardial infarction."
)


def calculate_statistics(latencies_ms: list[float]) -> dict[str, float]:
    """Calcula estatísticas do benchmark."""
    values = np.array(latencies_ms)

    mean_ms = float(np.mean(values))
    total_seconds = float(np.sum(values) / 1000)

    return {
        "mean_ms": mean_ms,
        "p50_ms": float(np.percentile(values, 50)),
        "p95_ms": float(np.percentile(values, 95)),
        "throughput_rps": len(values) / total_seconds,
    }


def benchmark_baseline(model) -> dict[str, float]:
    """Mede inferência ponta a ponta do pipeline scikit-learn."""
    for _ in range(WARMUP_RUNS):
        model.predict([SAMPLE_TEXT])

    latencies = []

    for _ in range(BENCHMARK_RUNS):
        start = perf_counter()
        model.predict([SAMPLE_TEXT])
        elapsed = perf_counter() - start
        latencies.append(elapsed * 1000)

    return calculate_statistics(latencies)


def benchmark_onnx(vectorizer, session) -> dict[str, float]:
    """Mede TF-IDF + classificador executado pelo ONNX Runtime."""
    input_name = session.get_inputs()[0].name

    def predict() -> None:
        features = vectorizer.transform([SAMPLE_TEXT])
        dense_features = features.toarray().astype(np.float32)
        session.run(None, {input_name: dense_features})

    for _ in range(WARMUP_RUNS):
        predict()

    latencies = []

    for _ in range(BENCHMARK_RUNS):
        start = perf_counter()
        predict()
        elapsed = perf_counter() - start
        latencies.append(elapsed * 1000)

    return calculate_statistics(latencies)


def validate_predictions(model, vectorizer, session) -> None:
    """Confirma que baseline e ONNX produzem a mesma classe."""
    baseline_prediction = int(model.predict([SAMPLE_TEXT])[0])

    features = vectorizer.transform([SAMPLE_TEXT])
    dense_features = features.toarray().astype(np.float32)

    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: dense_features})

    onnx_prediction = int(np.asarray(outputs[0]).ravel()[0])

    print("=== VALIDAÇÃO DAS PREDIÇÕES ===")
    print(f"Baseline: {baseline_prediction}")
    print(f"ONNX:     {onnx_prediction}")
    print(f"Concordância: {baseline_prediction == onnx_prediction}")


def print_results(
    baseline: dict[str, float],
    onnx: dict[str, float],
) -> None:
    """Exibe comparação antes e depois."""
    speedup = baseline["mean_ms"] / onnx["mean_ms"]
    latency_reduction = (
        (baseline["mean_ms"] - onnx["mean_ms"])
        / baseline["mean_ms"]
        * 100
    )
    throughput_gain = (
        (onnx["throughput_rps"] - baseline["throughput_rps"])
        / baseline["throughput_rps"]
        * 100
    )

    print("\n=== BENCHMARK ===")
    print(f"Execuções por abordagem: {BENCHMARK_RUNS}\n")

    print(
        f"{'Métrica':<20}"
        f"{'Baseline':>15}"
        f"{'ONNX':>15}"
    )
    print("-" * 50)

    print(
        f"{'Média (ms)':<20}"
        f"{baseline['mean_ms']:>15.4f}"
        f"{onnx['mean_ms']:>15.4f}"
    )
    print(
        f"{'P50 (ms)':<20}"
        f"{baseline['p50_ms']:>15.4f}"
        f"{onnx['p50_ms']:>15.4f}"
    )
    print(
        f"{'P95 (ms)':<20}"
        f"{baseline['p95_ms']:>15.4f}"
        f"{onnx['p95_ms']:>15.4f}"
    )
    print(
        f"{'Throughput (req/s)':<20}"
        f"{baseline['throughput_rps']:>15.2f}"
        f"{onnx['throughput_rps']:>15.2f}"
    )

    print("\n=== IMPACTO DA OTIMIZAÇÃO ===")
    print(f"Speedup: {speedup:.2f}x")
    print(f"Redução média de latência: {latency_reduction:.2f}%")
    print(f"Ganho de throughput: {throughput_gain:.2f}%")


def main() -> None:
    """Executa o benchmark completo."""
    print("Carregando artefatos...")

    baseline_model = joblib.load(BASELINE_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)

    session = ort.InferenceSession(
        str(ONNX_PATH),
        providers=["CPUExecutionProvider"],
    )

    validate_predictions(
        baseline_model,
        vectorizer,
        session,
    )

    print("\nExecutando benchmark baseline...")
    baseline_results = benchmark_baseline(baseline_model)

    print("Executando benchmark ONNX...")
    onnx_results = benchmark_onnx(vectorizer, session)

    print_results(
        baseline_results,
        onnx_results,
    )


if __name__ == "__main__":
    main()