import sqlite3
from datetime import datetime

import numpy as np
from sklearn.metrics import (
    cohen_kappa_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from aps.entity.artifact_entity import ClassificationMetricArtifact


def get_classification_score(y_true, y_pred) -> ClassificationMetricArtifact:
    """
    Computes several classification metrics and returns a ClassificationMetricArtifact.

    :param y_true: The true labels.
    :param y_pred: The predicted labels.
    """
    model_timestamp = round(datetime.now().timestamp())

    y_true, y_pred = map(lambda arr: np.asarray(arr, dtype=int), (y_true, y_pred))

    metrics = {
        "f1_score": f1_score(y_true, y_pred),
        "recall_score": recall_score(y_true, y_pred),
        "precision_score": precision_score(y_true, y_pred),
        "cohen_kappa_score": cohen_kappa_score(y_true, y_pred),
        "roc_auc_score": roc_auc_score(y_true, y_pred),
    }

    return ClassificationMetricArtifact(
        timestamp=model_timestamp,
        f1_score=metrics["f1_score"],
        precision_score=metrics["precision_score"],
        recall_score=metrics["recall_score"],
        cohen_kappa_score=metrics["cohen_kappa_score"],
        roc_auc_score=metrics["roc_auc_score"],
    )


def save_classification_score(
    data: ClassificationMetricArtifact, file_path: str
) -> None:
    """
    Saves the classification metrics to a local SQLite database.

    :param data: A ClassificationMetricArtifact instance containing the classification metrics.
    :param file_path: The file path to the SQLite database.
    """
    with sqlite3.connect(file_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """CREATE TABLE IF NOT EXISTS METRICS (
                timestamp INTEGER PRIMARY KEY,
                f1_score REAL,
                precision_score REAL,
                recall_score REAL,
                cohen_kappa_score REAL,
                roc_auc_score REAL
            )"""
        )

        insert_query = """
        INSERT INTO METRICS 
            (timestamp, f1_score, precision_score, recall_score, cohen_kappa_score, roc_auc_score)
        VALUES
            (?, ?, ?, ?, ?, ?)
        """
        cursor.execute(
            insert_query,
            (
                data.timestamp,
                round(data.f1_score, 3),
                round(data.precision_score, 3),
                round(data.recall_score, 3),
                round(data.cohen_kappa_score, 3),
                round(data.roc_auc_score, 3),
            ),
        )
