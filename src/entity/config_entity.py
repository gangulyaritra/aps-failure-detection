import os
from datetime import datetime
from src.constant import TIMESTAMP
from src.constant import training


class TrainingPipelineConfig:
    def __init__(self):
        self.pipeline_name: str = training.PIPELINE_NAME
        self.artifact_dir: str = os.path.join(training.ARTIFACT_DIR, TIMESTAMP)
        self.timestamp: str = TIMESTAMP


class DataIngestionConfig:
    def __init__(self, training_pipeline_config: TrainingPipelineConfig):
        self.data_ingestion_dir: str = os.path.join(
            training_pipeline_config.artifact_dir, training.DATA_INGESTION_DIR_NAME
        )
        self.feature_store_file_path: str = os.path.join(
            self.data_ingestion_dir,
            training.DATA_INGESTION_FEATURE_STORE_DIR,
            training.FILE_NAME,
        )
        self.training_file_path: str = os.path.join(
            self.data_ingestion_dir,
            training.DATA_INGESTION_INGESTED_DIR,
            training.TRAIN_FILE_NAME,
        )
        self.testing_file_path: str = os.path.join(
            self.data_ingestion_dir,
            training.DATA_INGESTION_INGESTED_DIR,
            training.TEST_FILE_NAME,
        )
        self.train_test_split_ratio: float = (
            training.DATA_INGESTION_TRAIN_TEST_SPLIT_RATIO
        )
        self.collection_name: str = training.DATA_INGESTION_COLLECTION_NAME


class DataValidationConfig:
    def __init__(self, training_pipeline_config: TrainingPipelineConfig):
        self.data_validation_dir: str = os.path.join(
            training_pipeline_config.artifact_dir, training.DATA_VALIDATION_DIR_NAME
        )
        self.valid_data_dir: str = os.path.join(
            self.data_validation_dir, training.DATA_VALIDATION_VALID_DIR
        )
        self.invalid_data_dir: str = os.path.join(
            self.data_validation_dir, training.DATA_VALIDATION_INVALID_DIR
        )
        self.valid_train_file_path: str = os.path.join(
            self.valid_data_dir, training.TRAIN_FILE_NAME
        )
        self.valid_test_file_path: str = os.path.join(
            self.valid_data_dir, training.TEST_FILE_NAME
        )
        self.invalid_train_file_path: str = os.path.join(
            self.invalid_data_dir, training.TRAIN_FILE_NAME
        )
        self.invalid_test_file_path: str = os.path.join(
            self.invalid_data_dir, training.TEST_FILE_NAME
        )
        self.drift_report_file_path: str = os.path.join(
            self.data_validation_dir,
            training.DATA_VALIDATION_DRIFT_REPORT_DIR,
            training.DATA_VALIDATION_DRIFT_REPORT_FILE_NAME,
        )


class DataTransformationConfig:
    def __init__(self, training_pipeline_config: TrainingPipelineConfig):
        self.data_transformation_dir: str = os.path.join(
            training_pipeline_config.artifact_dir, training.DATA_TRANSFORMATION_DIR_NAME
        )
        self.transformed_train_file_path: str = os.path.join(
            self.data_transformation_dir,
            training.DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR,
            training.TRAIN_FILE_NAME.replace("csv", "npy"),
        )
        self.transformed_test_file_path: str = os.path.join(
            self.data_transformation_dir,
            training.DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR,
            training.TEST_FILE_NAME.replace("csv", "npy"),
        )
        self.transformed_object_file_path: str = os.path.join(
            self.data_transformation_dir,
            training.DATA_TRANSFORMATION_TRANSFORMED_OBJECT_DIR,
            training.PREPROCSSING_OBJECT_FILE_NAME,
        )


class ModelTrainerConfig:
    def __init__(self, training_pipeline_config: TrainingPipelineConfig):
        self.model_trainer_dir: str = os.path.join(
            training_pipeline_config.artifact_dir, training.MODEL_TRAINER_DIR_NAME
        )
        self.trained_model_file_path: str = os.path.join(
            self.model_trainer_dir,
            training.MODEL_TRAINER_TRAINED_MODEL_DIR,
            training.MODEL_FILE_NAME,
        )
        self.expected_f1_score: float = training.MODEL_TRAINER_EXPECTED_F1_SCORE
        self.expected_roc_auc_score: float = (
            training.MODEL_TRAINER_EXPECTED_ROC_AUC_SCORE
        )
        self.overfitting_underfitting_threshold: float = (
            training.MODEL_TRAINER_OVERFITTING_UNDERFITTING_THRESHOLD
        )


class ModelEvaluationConfig:
    def __init__(self, training_pipeline_config: TrainingPipelineConfig):
        self.model_evaluation_dir: str = os.path.join(
            training_pipeline_config.artifact_dir, training.MODEL_EVALUATION_DIR_NAME
        )
        self.report_file_path: str = os.path.join(
            self.model_evaluation_dir, training.MODEL_EVALUATION_REPORT_NAME
        )
        self.change_threshold: float = training.MODEL_EVALUATION_CHANGED_THRESHOLD_SCORE


class ModelPusherConfig:
    def __init__(self, training_pipeline_config: TrainingPipelineConfig):
        self.model_evaluation_dir: str = os.path.join(
            training_pipeline_config.artifact_dir, training.MODEL_PUSHER_DIR_NAME
        )
        self.model_file_path: str = os.path.join(
            self.model_evaluation_dir, training.MODEL_FILE_NAME
        )
        timestamp = round(datetime.now().timestamp())
        self.saved_model_path: str = os.path.join(
            training.SAVED_MODEL_DIR, f"{timestamp}", training.MODEL_FILE_NAME
        )
