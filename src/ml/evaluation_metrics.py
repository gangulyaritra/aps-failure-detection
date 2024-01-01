import sqlite3
from datetime import datetime
from src.entity.artifact_entity import ClassificationMetricArtifact
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    cohen_kappa_score,
    roc_auc_score,
)


def get_classification_score(y_true, y_pred) -> ClassificationMetricArtifact:
    model_timestamp = round(datetime.now().timestamp())
    model_f1_score = f1_score(y_true, y_pred)
    model_recall_score = recall_score(y_true, y_pred)
    model_precision_score = precision_score(y_true, y_pred)
    model_cohen_kappa_score = cohen_kappa_score(y_true, y_pred)
    model_roc_auc_score = roc_auc_score(y_true, y_pred)

    return ClassificationMetricArtifact(
        timestamp=model_timestamp,
        f1_score=model_f1_score,
        precision_score=model_precision_score,
        recall_score=model_recall_score,
        cohen_kappa_score=model_cohen_kappa_score,
        roc_auc_score=model_roc_auc_score,
    )


def save_classification_score(data: object, file_path: str) -> None:
    conn = sqlite3.connect(file_path)
    cursor = conn.cursor()

    cursor.execute(
        """CREATE TABLE IF NOT EXISTS METRICS (
                timestamp integer PRIMARY KEY, 
                f1_score float, 
                precision_score float, 
                recall_score float,
                cohen_kappa_score float, 
                roc_auc_score float
        )"""
    )

    cursor.execute(
        "insert into METRICS values (?,?,?,?,?,?)",
        (
            data.timestamp,
            round(data.f1_score, 3),
            round(data.precision_score, 3),
            round(data.recall_score, 3),
            round(data.cohen_kappa_score, 3),
            round(data.roc_auc_score, 3),
        ),
    )

    conn.commit()
