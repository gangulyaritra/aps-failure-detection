import os
import shutil
import sys

from aps.entity.artifact_entity import ModelEvaluationArtifact, ModelPusherArtifact
from aps.entity.config_entity import ModelPusherConfig
from aps.exception import SensorException
from aps.logger import logging


class ModelPusher:
    def __init__(
        self,
        model_pusher_config: ModelPusherConfig,
        model_eval_artifact: ModelEvaluationArtifact,
    ):
        self.model_pusher_config = model_pusher_config
        self.model_eval_artifact = model_eval_artifact

    def initiate_model_pusher(self) -> ModelPusherArtifact:
        try:
            logging.info(">>> Model Pusher Component Started.")
            trained_model_path = self.model_eval_artifact.trained_model_path
            model_file_path = self.model_pusher_config.model_file_path

            os.makedirs(os.path.dirname(model_file_path), exist_ok=True)
            shutil.copy(src=trained_model_path, dst=model_file_path)

            saved_model_path = self.model_pusher_config.saved_model_path
            os.makedirs(os.path.dirname(saved_model_path), exist_ok=True)
            shutil.copy(src=trained_model_path, dst=saved_model_path)

            logging.info(">>> Model Pusher Component Ended.")
            return ModelPusherArtifact(
                saved_model_path=saved_model_path, model_file_path=model_file_path
            )

        except Exception as e:
            raise SensorException(e, sys) from e
