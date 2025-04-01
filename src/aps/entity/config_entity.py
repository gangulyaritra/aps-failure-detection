import os
from datetime import datetime

from aps.constant import TIMESTAMP
from aps.constant.training import *


class TrainingPipelineConfig:
    def __init__(self):
        self.pipeline_name: str = PIPELINE_NAME
        self.artifact_dir: str = os.path.join(ARTIFACT_DIR, TIMESTAMP)
        self.timestamp: str = TIMESTAMP


class DataIngestionConfig:
    def __init__(self, training_pipeline_config: TrainingPipelineConfig):
        self.data_ingestion_dir: str = os.path.join(
            training_pipeline_config.artifact_dir, DATA_INGESTION_DIR_NAME
        )
        self.feature_store_file_path: str = os.path.join(
            self.data_ingestion_dir, DATA_INGESTION_FEATURE_STORE_DIR, FILE_NAME
        )
        self.training_file_path: str = os.path.join(
            self.data_ingestion_dir, DATA_INGESTION_INGESTED_DIR, TRAIN_FILE_NAME
        )
        self.testing_file_path: str = os.path.join(
            self.data_ingestion_dir, DATA_INGESTION_INGESTED_DIR, TEST_FILE_NAME
        )
        self.train_test_split_ratio: float = DATA_INGESTION_TRAIN_TEST_SPLIT_RATIO
        self.collection_name: str = DATA_INGESTION_COLLECTION_NAME


class DataValidationConfig:
    def __init__(self, training_pipeline_config: TrainingPipelineConfig):
        self.data_validation_dir: str = os.path.join(
            training_pipeline_config.artifact_dir, DATA_VALIDATION_DIR_NAME
        )
        self.valid_data_dir: str = os.path.join(
            self.data_validation_dir, DATA_VALIDATION_VALID_DIR
        )
        self.invalid_data_dir: str = os.path.join(
            self.data_validation_dir, DATA_VALIDATION_INVALID_DIR
        )
        self.valid_train_file_path: str = os.path.join(
            self.valid_data_dir, TRAIN_FILE_NAME
        )
        self.valid_test_file_path: str = os.path.join(
            self.valid_data_dir, TEST_FILE_NAME
        )
        self.invalid_train_file_path: str = os.path.join(
            self.invalid_data_dir, TRAIN_FILE_NAME
        )
        self.invalid_test_file_path: str = os.path.join(
            self.invalid_data_dir, TEST_FILE_NAME
        )
        self.drift_report_file_path: str = os.path.join(
            self.data_validation_dir,
            DATA_VALIDATION_DRIFT_REPORT_DIR,
            DATA_VALIDATION_DRIFT_REPORT_FILE_NAME,
        )


class DataTransformationConfig:
    def __init__(self, training_pipeline_config: TrainingPipelineConfig):
        self.data_transformation_dir: str = os.path.join(
            training_pipeline_config.artifact_dir, DATA_TRANSFORMATION_DIR_NAME
        )
        self.transformed_train_file_path: str = os.path.join(
            self.data_transformation_dir,
            DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR,
            TRAIN_FILE_NAME.replace("csv", "npy"),
        )
        self.transformed_test_file_path: str = os.path.join(
            self.data_transformation_dir,
            DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR,
            TEST_FILE_NAME.replace("csv", "npy"),
        )
        self.transformed_object_file_path: str = os.path.join(
            self.data_transformation_dir,
            DATA_TRANSFORMATION_TRANSFORMED_OBJECT_DIR,
            PREPROCSSING_OBJECT_FILE_NAME,
        )


class ModelTrainerConfig:
    def __init__(self, training_pipeline_config: TrainingPipelineConfig):
        self.model_trainer_dir: str = os.path.join(
            training_pipeline_config.artifact_dir, MODEL_TRAINER_DIR_NAME
        )
        self.trained_model_file_path: str = os.path.join(
            self.model_trainer_dir, MODEL_TRAINER_TRAINED_MODEL_DIR, MODEL_FILE_NAME
        )
        self.param_distributions: dict = MODEL_TRAINER_PARAM_DISTRIBUTIONS
        self.expected_f1_score: float = MODEL_TRAINER_EXPECTED_F1_SCORE
        self.expected_roc_auc_score: float = MODEL_TRAINER_EXPECTED_ROC_AUC_SCORE
        self.overfitting_underfitting_threshold: float = (
            MODEL_TRAINER_OVERFITTING_UNDERFITTING_THRESHOLD
        )


class ModelEvaluationConfig:
    def __init__(self, training_pipeline_config: TrainingPipelineConfig):
        self.model_evaluation_dir: str = os.path.join(
            training_pipeline_config.artifact_dir, MODEL_EVALUATION_DIR_NAME
        )
        self.report_file_path: str = os.path.join(
            self.model_evaluation_dir, MODEL_EVALUATION_REPORT_NAME
        )
        self.change_threshold: float = MODEL_EVALUATION_CHANGED_THRESHOLD_SCORE


class ModelPusherConfig:
    def __init__(self, training_pipeline_config: TrainingPipelineConfig):
        self.model_evaluation_dir: str = os.path.join(
            training_pipeline_config.artifact_dir, MODEL_PUSHER_DIR_NAME
        )
        self.model_file_path: str = os.path.join(
            self.model_evaluation_dir, MODEL_FILE_NAME
        )
        self.saved_model_path: str = os.path.join(
            SAVED_MODEL_DIR, f"{round(datetime.now().timestamp())}", MODEL_FILE_NAME
        )
