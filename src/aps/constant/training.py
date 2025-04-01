import os
from typing import Union

from aps.constant import COLLECTION_NAME

URL = "https://archive.ics.uci.edu/static/public/421/aps+failure+at+scania+trucks.zip"

SAVED_MODEL_DIR = os.path.join("saved_models")

TARGET_COLUMN = "class"
PIPELINE_NAME: str = "aps-failure-detection"

ARTIFACT_DIR: str = "artifact"
PREDICTION_ARTIFACT_DIR: str = "inference_artifact"
SCHEMA_FILE_PATH = os.path.join("config", "schema.yaml")
SCHEMA_DROP_COLS = "drop_columns"

RAW_CSV_FILE_NAME: str = "aps_failure_training_set.csv"
FILE_NAME: str = "sensor.csv"
TRAIN_FILE_NAME: str = "train.csv"
TEST_FILE_NAME: str = "test.csv"

PREPROCSSING_OBJECT_FILE_NAME: str = "transformer.pkl"
MODEL_FILE_NAME: str = "sensor_model.pkl"
SQLITE_DB_NAME: str = "evaluation_metrics.db"

# Data Ingestion Constants.
DATA_INGESTION_COLLECTION_NAME: str = COLLECTION_NAME
DATA_INGESTION_DIR_NAME: str = "data_ingestion"
DATA_INGESTION_FEATURE_STORE_DIR: str = "feature_store"
DATA_INGESTION_INGESTED_DIR: str = "ingested"
DATA_INGESTION_TRAIN_TEST_SPLIT_RATIO: float = 0.2

# Data Validation Constants.
DATA_VALIDATION_DIR_NAME: str = "data_validation"
DATA_VALIDATION_VALID_DIR: str = "valid"
DATA_VALIDATION_INVALID_DIR: str = "invalid"
DATA_VALIDATION_DRIFT_REPORT_DIR: str = "drift_report"
DATA_VALIDATION_DRIFT_REPORT_FILE_NAME: str = "report.yaml"

# Data Transformation Constants.
DATA_TRANSFORMATION_DIR_NAME: str = "data_transformation"
DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR: str = "transformed_dataset"
DATA_TRANSFORMATION_TRANSFORMED_OBJECT_DIR: str = "transformed_object"

# Model Trainer Constants.
MODEL_TRAINER_DIR_NAME: str = "model_trainer"
MODEL_TRAINER_TRAINED_MODEL_DIR: str = "trained_model"
MODEL_TRAINER_TRAINED_MODEL_NAME: str = MODEL_FILE_NAME
MODEL_TRAINER_EXPECTED_F1_SCORE: float = 0.6
MODEL_TRAINER_EXPECTED_ROC_AUC_SCORE: float = 0.7
MODEL_TRAINER_OVERFITTING_UNDERFITTING_THRESHOLD: float = 0.05
MODEL_TRAINER_PARAM_DISTRIBUTIONS: dict[str, list[Union[int, float]]] = {
    "learning_rate": [0.0136, 0.03],
    "max_depth": [7, 11],
    "n_estimators": [469, 877],
}

# Model Evaluation Constants.
MODEL_EVALUATION_DIR_NAME: str = "model_evaluation"
MODEL_EVALUATION_CHANGED_THRESHOLD_SCORE: float = 0.02
MODEL_EVALUATION_REPORT_NAME: str = "report.yaml"

# Model Pusher Constants.
MODEL_PUSHER_DIR_NAME: str = "model_pusher"
MODEL_PUSHER_SAVED_MODEL_DIR: str = SAVED_MODEL_DIR
