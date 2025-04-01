import os
from typing import Any

from aps.constant.training import MODEL_FILE_NAME, SAVED_MODEL_DIR


class TargetValueMapping:
    """
    Maps target values to integers (e.g., negative -> 0, positive -> 1).
    """

    def __init__(self):
        self.neg: int = 0
        self.pos: int = 1

    def to_dict(self) -> dict:
        return self.__dict__

    def reverse_mapping(self) -> dict:
        mapping_response = self.to_dict()
        return dict(zip(mapping_response.values(), mapping_response.keys()))


class SensorModel:
    """
    Wraps a preprocessor and a model to provide a convenient prediction interface.
    """

    def __init__(self, preprocessor: Any, model: Any):
        self.preprocessor = preprocessor
        self.model = model

    def predict(self, x):
        x_transformed = self.preprocessor.transform(x)
        return self.model.predict(x_transformed)


class ModelResolver:
    """
    Identifies and returns the path to the most recent, optimal trained model.
    """

    def __init__(self, model_dir: str = SAVED_MODEL_DIR):
        self.model_dir = model_dir

    def get_best_model_path(self) -> str:
        timestamps = list(map(int, os.listdir(self.model_dir)))
        latest_timestamp = max(timestamps)
        return os.path.join(self.model_dir, f"{latest_timestamp}", MODEL_FILE_NAME)

    def is_model_exists(self) -> bool:
        if not os.path.exists(self.model_dir):
            return False

        timestamps = os.listdir(self.model_dir)
        if len(timestamps) == 0:
            return False

        latest_model_path = self.get_best_model_path()
        return bool(os.path.exists(latest_model_path))
