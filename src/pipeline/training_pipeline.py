import sys
from src.cloud_storage.s3_syncer import S3Sync
from src.components.data_ingestion import DataIngestion
from src.components.data_validation import DataValidation
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.components.model_evaluation import ModelEvaluation
from src.components.model_pusher import ModelPusher
from src.constant import TRAINING_BUCKET_NAME
from src.constant.training import SAVED_MODEL_DIR, ARTIFACT_DIR
from src.entity.artifact_entity import (
    DataIngestionArtifact,
    DataValidationArtifact,
    DataTransformationArtifact,
    ModelTrainerArtifact,
    ModelEvaluationArtifact,
    ModelPusherArtifact,
)
from src.entity.config_entity import (
    TrainingPipelineConfig,
    DataIngestionConfig,
    DataValidationConfig,
    DataTransformationConfig,
    ModelTrainerConfig,
    ModelEvaluationConfig,
    ModelPusherConfig,
)
from src.logger import logs_path
from src.exception import SensorException


class TrainPipeline:
    is_pipeline_running = False

    def __init__(self):
        self.training_pipeline_config = TrainingPipelineConfig()
        self.s3_sync = S3Sync()

    def start_data_ingestion(self) -> DataIngestionArtifact:
        try:
            self.data_ingestion_config = DataIngestionConfig(
                training_pipeline_config=self.training_pipeline_config
            )
            data_ingestion = DataIngestion(
                data_ingestion_config=self.data_ingestion_config
            )
            return data_ingestion.initiate_data_ingestion()

        except Exception as e:
            raise SensorException(e, sys) from e

    def start_data_validation(
        self, data_ingestion_artifact: DataIngestionArtifact
    ) -> DataValidationArtifact:
        try:
            data_validation_config = DataValidationConfig(
                training_pipeline_config=self.training_pipeline_config
            )
            data_validation = DataValidation(
                data_ingestion_artifact=data_ingestion_artifact,
                data_validation_config=data_validation_config,
            )
            return data_validation.initiate_data_validation()

        except Exception as e:
            raise SensorException(e, sys) from e

    def start_data_transformation(
        self, data_validation_artifact: DataValidationArtifact
    ) -> DataTransformationArtifact:
        try:
            data_transformation_config = DataTransformationConfig(
                training_pipeline_config=self.training_pipeline_config
            )
            data_transformation = DataTransformation(
                data_validation_artifact=data_validation_artifact,
                data_transformation_config=data_transformation_config,
            )
            return data_transformation.initiate_data_transformation()

        except Exception as e:
            raise SensorException(e, sys) from e

    def start_model_trainer(
        self, data_transformation_artifact: DataTransformationArtifact
    ) -> ModelTrainerArtifact:
        try:
            model_trainer_config = ModelTrainerConfig(
                training_pipeline_config=self.training_pipeline_config
            )
            model_trainer = ModelTrainer(
                model_trainer_config, data_transformation_artifact
            )
            return model_trainer.initiate_model_trainer()

        except Exception as e:
            raise SensorException(e, sys) from e

    def start_model_evaluation(
        self,
        data_validation_artifact: DataValidationArtifact,
        model_trainer_artifact: ModelTrainerArtifact,
    ) -> ModelEvaluationArtifact:
        try:
            model_eval_config = ModelEvaluationConfig(self.training_pipeline_config)
            model_eval = ModelEvaluation(
                model_eval_config, data_validation_artifact, model_trainer_artifact
            )
            return model_eval.initiate_model_evaluation()

        except Exception as e:
            raise SensorException(e, sys) from e

    def start_model_pusher(
        self, model_eval_artifact: ModelEvaluationArtifact
    ) -> ModelPusherArtifact:
        try:
            model_pusher_config = ModelPusherConfig(
                training_pipeline_config=self.training_pipeline_config
            )
            model_pusher = ModelPusher(model_pusher_config, model_eval_artifact)
            return model_pusher.initiate_model_pusher()

        except Exception as e:
            raise SensorException(e, sys) from e

    def sync_artifact_dir_to_s3(self):
        try:
            aws_bucket_url = f"s3://{TRAINING_BUCKET_NAME}/{ARTIFACT_DIR}/{self.training_pipeline_config.timestamp}"
            self.s3_sync.sync_folder_to_s3(
                folder=self.training_pipeline_config.artifact_dir,
                aws_bucket_url=aws_bucket_url,
            )

        except Exception as e:
            raise SensorException(e, sys) from e

    def sync_logs_dir_to_s3(self):
        try:
            aws_bucket_url = f"s3://{TRAINING_BUCKET_NAME}/logs/{self.training_pipeline_config.timestamp}"
            self.s3_sync.sync_folder_to_s3(
                folder=logs_path,
                aws_bucket_url=aws_bucket_url,
            )

        except Exception as e:
            raise SensorException(e, sys) from e

    def sync_saved_model_dir_to_s3(self):
        try:
            aws_bucket_url = f"s3://{TRAINING_BUCKET_NAME}/{SAVED_MODEL_DIR}"
            self.s3_sync.sync_folder_to_s3(
                folder=SAVED_MODEL_DIR, aws_bucket_url=aws_bucket_url
            )

        except Exception as e:
            raise SensorException(e, sys) from e

    def run_pipeline(self):
        try:
            TrainPipeline.is_pipeline_running = True

            data_ingestion_artifact: DataIngestionArtifact = self.start_data_ingestion()
            data_validation_artifact = self.start_data_validation(
                data_ingestion_artifact=data_ingestion_artifact
            )
            data_transformation_artifact = self.start_data_transformation(
                data_validation_artifact=data_validation_artifact
            )
            model_trainer_artifact = self.start_model_trainer(
                data_transformation_artifact
            )
            model_eval_artifact = self.start_model_evaluation(
                data_validation_artifact, model_trainer_artifact
            )

            if not model_eval_artifact.is_model_accepted:
                raise Exception("Trained model is not better than the Best model.")

            model_pusher_artifact = self.start_model_pusher(model_eval_artifact)

            TrainPipeline.is_pipeline_running = False
            self.sync_artifact_dir_to_s3()
            self.sync_logs_dir_to_s3()
            self.sync_saved_model_dir_to_s3()

        except Exception as e:
            self.sync_artifact_dir_to_s3()
            self.sync_logs_dir_to_s3()
            TrainPipeline.is_pipeline_running = False
            raise SensorException(e, sys) from e
