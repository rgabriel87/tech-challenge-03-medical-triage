"""Conversão do classificador treinado para ONNX."""

from pathlib import Path

import joblib
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "medical_text_classifier.joblib"
ONNX_PATH = PROJECT_ROOT / "models" / "medical_classifier.onnx"
VECTORIZER_PATH = PROJECT_ROOT / "models" / "tfidf_vectorizer.joblib"


def load_pipeline():
    """Carrega o pipeline baseline treinado."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Modelo baseline nao encontrado. Execute 'python -m src.train' primeiro."
        )

    return joblib.load(MODEL_PATH)


def export_to_onnx() -> None:
    """Separa TF-IDF e converte o classificador para ONNX."""
    pipeline = load_pipeline()

    vectorizer = pipeline.named_steps["tfidf"]
    classifier = pipeline.named_steps["classifier"]

    joblib.dump(vectorizer, VECTORIZER_PATH)

    initial_type = [
        (
            "float_input",
            FloatTensorType([None, len(vectorizer.get_feature_names_out())]),
        )
    ]

    onnx_model = convert_sklearn(
        classifier,
        initial_types=initial_type,
        target_opset=15,
        options={id(classifier): {"zipmap": False}},
    )

    ONNX_PATH.write_bytes(onnx_model.SerializeToString())

    print("Otimizacao concluida.")
    print(f"TF-IDF salvo em: {VECTORIZER_PATH}")
    print(f"Modelo ONNX salvo em: {ONNX_PATH}")
    print(
        "Features:",
        len(vectorizer.get_feature_names_out()),
    )


if __name__ == "__main__":
    export_to_onnx()