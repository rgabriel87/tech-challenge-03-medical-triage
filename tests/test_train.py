"""Testes unitários do pipeline de treinamento."""

import pandas as pd

from src.train import build_pipeline, load_data


def test_build_pipeline() -> None:
    """Valida a estrutura do pipeline de ML."""
    pipeline = build_pipeline()

    assert "tfidf" in pipeline.named_steps
    assert "classifier" in pipeline.named_steps


def test_load_data(tmp_path) -> None:
    """Valida o carregamento do formato original do dataset."""
    dataset = tmp_path / "train.dat"

    dataset.write_text(
        "1\tPatient with digestive disease symptoms.\n"
        "2\tPatient with cardiovascular disease symptoms.\n",
        encoding="utf-8",
    )

    dataframe = load_data(dataset)

    assert isinstance(dataframe, pd.DataFrame)
    assert list(dataframe.columns) == ["target", "text"]
    assert len(dataframe) == 2
    assert dataframe["target"].tolist() == [1, 2]


def test_pipeline_can_train_and_predict() -> None:
    """Valida treinamento e inferência com um dataset mínimo."""
    texts = [
        "heart cardiac artery disease",
        "myocardial cardiac heart condition",
        "brain neurological nerve disease",
        "neurological brain nervous condition",
        "tumor cancer neoplasm disease",
        "cancer tumor malignant neoplasm",
        "digestive stomach intestinal disease",
        "intestinal digestive stomach condition",
        "general pathological medical condition",
        "general pathology medical disorder",
    ]

    targets = [2, 2, 4, 4, 3, 3, 1, 1, 5, 5]

    pipeline = build_pipeline()
    pipeline.fit(texts, targets)

    prediction = pipeline.predict(
        ["patient with cardiac heart disease"]
    )

    assert len(prediction) == 1
    assert int(prediction[0]) in {1, 2, 3, 4, 5}