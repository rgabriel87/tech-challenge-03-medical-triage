"""DAG de treinamento do classificador de textos médicos."""

from datetime import datetime
from pathlib import Path

import pandas as pd
from airflow.decorators import dag, task


DATA_PATH = Path("/opt/airflow/data/train.dat")
MODEL_PATH = Path("/opt/airflow/models/medical_text_classifier.joblib")


@dag(
    dag_id="medical_model_training",
    description="Pipeline de treinamento do classificador NLP de textos médicos",
    schedule=None,
    start_date=datetime(2026, 1, 1),
    catchup=False,
    tags=["machine-learning", "nlp", "fiap"],
)
def medical_training_pipeline():
    """Orquestra validação dos dados, treinamento e validação do modelo."""

    @task
    def validate_dataset() -> dict:
        """Valida se o dataset de treinamento está disponível e íntegro."""
        if not DATA_PATH.exists():
            raise FileNotFoundError(
                f"Dataset não encontrado em: {DATA_PATH}"
            )

        dataframe = pd.read_csv(
            DATA_PATH,
            sep="\t",
            header=None,
            names=["target", "text"],
            encoding="utf-8",
        )

        dataframe = dataframe.dropna(subset=["target", "text"])

        if dataframe.empty:
            raise ValueError("O dataset está vazio.")

        classes = sorted(
            dataframe["target"].astype(int).unique().tolist()
        )

        print(f"Registros encontrados: {len(dataframe)}")
        print(f"Classes encontradas: {classes}")

        return {
            "records": len(dataframe),
            "classes": classes,
        }

    @task
    def train_model(dataset_info: dict) -> str:
        """Executa o treinamento utilizando o módulo principal do projeto."""
        import sys

        sys.path.insert(0, "/opt/airflow")

        from src.train import load_data, save_model, train_model

        print(
            f"Iniciando treinamento com "
            f"{dataset_info['records']} registros."
        )

        dataframe = load_data(DATA_PATH)
        model, metrics = train_model(dataframe)
        save_model(model, MODEL_PATH)

        print("Métricas obtidas:")
        for name, value in metrics.items():
            print(f"{name}: {value:.4f}")

        return str(MODEL_PATH)

    @task
    def validate_model(model_path: str) -> None:
        """Confirma que o artefato final do modelo foi criado."""
        path = Path(model_path)

        if not path.exists():
            raise FileNotFoundError(
                f"Modelo não encontrado em: {path}"
            )

        size_mb = path.stat().st_size / (1024 * 1024)

        print("Modelo validado com sucesso.")
        print(f"Caminho: {path}")
        print(f"Tamanho: {size_mb:.2f} MB")

    dataset_info = validate_dataset()
    trained_model = train_model(dataset_info)
    validate_model(trained_model)


medical_training_pipeline()