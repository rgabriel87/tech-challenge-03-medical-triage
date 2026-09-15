"""Treinamento do classificador de textos médicos."""

from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "train.dat"
MODEL_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODEL_DIR / "medical_text_classifier.joblib"

RANDOM_STATE = 42


def load_data(path: Path = DATA_PATH) -> pd.DataFrame:
    """Carrega o Medical Abstracts TC Corpus."""
    dataframe = pd.read_csv(
        path,
        sep="\t",
        header=None,
        names=["target", "text"],
        encoding="utf-8",
    )

    dataframe = dataframe.dropna(subset=["target", "text"]).copy()
    dataframe["target"] = dataframe["target"].astype(int)
    dataframe["text"] = dataframe["text"].astype(str)

    return dataframe


def build_pipeline() -> Pipeline:
    """Cria pipeline reutilizável de pré-processamento e classificação."""
    return Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    stop_words="english",
                    max_features=20_000,
                    ngram_range=(1, 2),
                    sublinear_tf=True,
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def evaluate_model(
    model: Pipeline,
    x_test: pd.Series,
    y_test: pd.Series,
) -> dict[str, float]:
    """Avalia o modelo utilizando métricas multiclasses."""
    predictions = model.predict(x_test)

    metrics = {
        "accuracy": accuracy_score(y_test, predictions),
        "precision_macro": precision_score(
            y_test, predictions, average="macro", zero_division=0
        ),
        "recall_macro": recall_score(
            y_test, predictions, average="macro", zero_division=0
        ),
        "f1_macro": f1_score(
            y_test, predictions, average="macro", zero_division=0
        ),
    }

    print("\n=== MÉTRICAS ===")
    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")

    print("\n=== CLASSIFICATION REPORT ===")
    print(classification_report(y_test, predictions, zero_division=0))

    return metrics


def train_model(dataframe: pd.DataFrame) -> tuple[Pipeline, dict[str, float]]:
    """Divide os dados, treina e avalia o pipeline."""
    x_train, x_test, y_train, y_test = train_test_split(
        dataframe["text"],
        dataframe["target"],
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=dataframe["target"],
    )

    print(f"Treino: {len(x_train)} amostras")
    print(f"Teste:  {len(x_test)} amostras")

    model = build_pipeline()

    print("\nTreinando modelo...")
    model.fit(x_train, y_train)

    metrics = evaluate_model(model, x_test, y_test)

    return model, metrics


def save_model(model: Pipeline, path: Path = MODEL_PATH) -> None:
    """Persiste pipeline completo para reutilização na inferência."""
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    print(f"\nModelo salvo em: {path}")


def main() -> None:
    """Executa o pipeline completo de treinamento."""
    print("=== TECH CHALLENGE FASE 3 ===")
    print("Classificador NLP de textos médicos\n")

    dataframe = load_data()

    print(f"Dataset carregado: {len(dataframe)} registros")
    print(f"Classes encontradas: {sorted(dataframe['target'].unique().tolist())}")

    print("\nDistribuição das classes:")
    print(dataframe["target"].value_counts().sort_index())

    model, _ = train_model(dataframe)
    save_model(model)

    print("\nTreinamento concluído com sucesso.")


if __name__ == "__main__":
    main()