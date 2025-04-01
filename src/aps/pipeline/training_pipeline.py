import sys

from aps.cloud_storage.s3_syncer import S3Sync
from aps.components.data_ingestion import DataIngestion
from aps.components.data_transformation import DataTransformation
from aps.components.data_validation import DataValidation
from aps.components.model_evaluation import ModelEvaluation
from aps.components.model_pusher import ModelPusher
from aps.components.model_trainer import ModelTrainer
from aps.constant import TRAINING_BUCKET_NAME
from aps.constant.training import ARTIFACT_DIR, SAVED_MODEL_DIR
from aps.entity.artifact_entity import (
    DataIngestionArtifact,
    DataTransformationArtifact,
    DataValidationArtifact,
    ModelEvaluationArtifact,
    ModelPusherArtifact,
    ModelTrainerArtifact,
)
from aps.entity.config_entity import (
    DataIngestionConfig,
    DataTransformationConfig,
    DataValidationConfig,
    ModelEvaluationConfig,
    ModelPusherConfig,
    ModelTrainerConfig,
    TrainingPipelineConfig,
)
from aps.exception import SensorException
from aps.logger import logs_path


def pipeline_step(func):
    """
    Decorator to handle repetitive try/except logic in pipeline steps.
    Wraps any raised exception into a SensorException.
    """

    def wrapper(self, *args, **kwargs):
        try:
            return func(self, *args, **kwargs)
        except Exception as e:
            raise SensorException(e, sys) from e

    return wrapper


class TrainPipeline:
    is_pipeline_running = False

    def __init__(self):
        self.training_pipeline_config = TrainingPipelineConfig()
        self.s3_sync = S3Sync()

    @pipeline_step
    def start_data_ingestion(self) -> DataIngestionArtifact:
        """
        Triggers the data ingestion component and returns its artifact.
        """
        data_ingestion_config = DataIngestionConfig(
            training_pipeline_config=self.training_pipeline_config
        )
        data_ingestion = DataIngestion(data_ingestion_config=data_ingestion_config)
        return data_ingestion.initiate_data_ingestion()

    @pipeline_step
    def start_data_validation(
        self, data_ingestion_artifact: DataIngestionArtifact
    ) -> DataValidationArtifact:
        """
        Triggers the data validation component based on the data ingestion artifact.
        """
        data_validation_config = DataValidationConfig(
            training_pipeline_config=self.training_pipeline_config
        )
        data_validation = DataValidation(
            data_ingestion_artifact=data_ingestion_artifact,
            data_validation_config=data_validation_config,
        )
        return data_validation.initiate_data_validation()

    @pipeline_step
    def start_data_transformation(
        self, data_validation_artifact: DataValidationArtifact
    ) -> DataTransformationArtifact:
        """
        Triggers the data transformation component based on the data validation artifact.
        """
        data_transformation_config = DataTransformationConfig(
            training_pipeline_config=self.training_pipeline_config
        )
        data_transformation = DataTransformation(
            data_validation_artifact=data_validation_artifact,
            data_transformation_config=data_transformation_config,
        )
        return data_transformation.initiate_data_transformation()

    @pipeline_step
    def start_model_trainer(
        self, data_transformation_artifact: DataTransformationArtifact
    ) -> ModelTrainerArtifact:
        """
        Triggers the model training component based on the data transformation artifact.
        """
        model_trainer_config = ModelTrainerConfig(
            training_pipeline_config=self.training_pipeline_config
        )
        model_trainer = ModelTrainer(model_trainer_config, data_transformation_artifact)
        return model_trainer.initiate_model_trainer()

    @pipeline_step
    def start_model_evaluation(
        self,
        data_validation_artifact: DataValidationArtifact,
        model_trainer_artifact: ModelTrainerArtifact,
    ) -> ModelEvaluationArtifact:
        """
        Evaluate the trained model and compare it with the existing best model.
        """
        model_eval_config = ModelEvaluationConfig(self.training_pipeline_config)
        model_eval = ModelEvaluation(
            model_eval_config, data_validation_artifact, model_trainer_artifact
        )
        return model_eval.initiate_model_evaluation()

    @pipeline_step
    def start_model_pusher(
        self, model_eval_artifact: ModelEvaluationArtifact
    ) -> ModelPusherArtifact:
        """
        Pushes the new model to the model registry if accepted.
        """
        model_pusher_config = ModelPusherConfig(
            training_pipeline_config=self.training_pipeline_config
        )
        model_pusher = ModelPusher(model_pusher_config, model_eval_artifact)
        return model_pusher.initiate_model_pusher()

    @pipeline_step
    def sync_artifact_dir_to_s3(self) -> None:
        """
        Syncs the artifact directory to S3 for persistent storage.
        """
        aws_bucket_url = f"s3://{TRAINING_BUCKET_NAME}/{ARTIFACT_DIR}/{self.training_pipeline_config.timestamp}"
        self.s3_sync.sync_folder_to_s3(
            folder=self.training_pipeline_config.artifact_dir,
            aws_bucket_url=aws_bucket_url,
        )

    @pipeline_step
    def sync_logs_dir_to_s3(self) -> None:
        """
        Syncs the log directory to S3 for persistent storage.
        """
        aws_bucket_url = f"s3://{TRAINING_BUCKET_NAME}/logs/{self.training_pipeline_config.timestamp}"
        self.s3_sync.sync_folder_to_s3(folder=logs_path, aws_bucket_url=aws_bucket_url)

    @pipeline_step
    def sync_saved_model_dir_to_s3(self) -> None:
        """
        Syncs the saved model directory to S3 for persistent storage.
        """
        aws_bucket_url = f"s3://{TRAINING_BUCKET_NAME}/{SAVED_MODEL_DIR}"
        self.s3_sync.sync_folder_to_s3(
            folder=SAVED_MODEL_DIR, aws_bucket_url=aws_bucket_url
        )

    def _sync_all_to_s3(self) -> None:
        """
        Sync all relevant directories to S3.
        """
        self.sync_artifact_dir_to_s3()
        self.sync_logs_dir_to_s3()
        self.sync_saved_model_dir_to_s3()

    def run_pipeline(self):
        """
        Runs the entire training pipeline in sequence.
        """
        try:
            TrainPipeline.is_pipeline_running = True

            data_ingestion_artifact = self.start_data_ingestion()
            data_validation_artifact = self.start_data_validation(
                data_ingestion_artifact
            )
            data_transformation_artifact = self.start_data_transformation(
                data_validation_artifact
            )
            model_trainer_artifact = self.start_model_trainer(
                data_transformation_artifact
            )
            model_eval_artifact = self.start_model_evaluation(
                data_validation_artifact, model_trainer_artifact
            )

            if not model_eval_artifact.is_model_accepted:
                raise Exception(
                    "Trained model is not better than the best existing model."
                )

            self.start_model_pusher(model_eval_artifact)

            TrainPipeline.is_pipeline_running = False
            self._sync_all_to_s3()

        except Exception as e:
            self._sync_all_to_s3()
            TrainPipeline.is_pipeline_running = False
            raise SensorException(e, sys) from e
