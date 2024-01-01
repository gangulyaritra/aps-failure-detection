import os
import sys
import glob
import pickle
import pandas as pd
import numpy as np
from src.cloud_storage.s3_syncer import S3Sync
from src.constant import TIMESTAMP, PREDICTION_BUCKET_NAME
from src.constant.training import *
from src.ml.estimator import ModelResolver, TargetValueMapping
from src.utils import read_yaml_file, load_object
from src.exception import SensorException


PREDICTION_ARTIFACT_DIR: str = "inference_artifact"


class InferencePipeline:
    is_pipeline_running = False

    def __init__(self):
        self.s3_sync = S3Sync()
        self._schema_config = read_yaml_file(SCHEMA_FILE_PATH)
        self.timestamp: str = TIMESTAMP

    def create_artifact_dir(self):
        folder_path = os.path.join(PREDICTION_ARTIFACT_DIR, self.timestamp, "")
        os.makedirs(os.path.dirname(folder_path), exist_ok=True)
        return folder_path

    @staticmethod
    def read_csv_data(file_path: str) -> pd.DataFrame:
        return pd.read_csv(file_path)

    def save_csv_data(self, dataframe: pd.DataFrame, file_name: str) -> None:
        folder = self.create_artifact_dir()
        dataframe.to_csv(f"{folder}/{file_name}.csv", index=False, header=True)

    def drop_insignificant_columns(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        dataframe = dataframe.drop(self._schema_config["drop_columns"], axis=1)

        if "_id" in dataframe.columns.to_list():
            dataframe = dataframe.drop(columns=["_id"], axis=1)

        dataframe = dataframe.replace("na", np.nan)
        return dataframe

    def validate_number_of_columns(self, dataframe: pd.DataFrame) -> bool:
        number_of_columns = len(self._schema_config["columns"])
        return len(dataframe.columns) == number_of_columns - 1

    def is_numerical_column_exist(self, dataframe: pd.DataFrame) -> bool:
        numerical_columns = self._schema_config["numerical_columns"]
        dataframe_columns = dataframe.columns

        numerical_column_present = True
        missing_numerical_columns = []

        for num_column in numerical_columns:
            if num_column not in dataframe_columns:
                numerical_column_present = False
                missing_numerical_columns.append(num_column)

        return numerical_column_present

    def load_data_transformer_object(self, dataframe: pd.DataFrame) -> np.ndarray:
        file_path = os.path.join(
            max(glob.glob(f"{ARTIFACT_DIR}/*"), key=os.path.getctime),
            DATA_TRANSFORMATION_DIR_NAME,
            DATA_TRANSFORMATION_TRANSFORMED_OBJECT_DIR,
            PREPROCSSING_OBJECT_FILE_NAME,
        )

        if not os.path.exists(file_path):
            raise Exception("Transformer Object doesn't exist.")

        with open(file_path, "rb") as f:
            preprocessor_object = pickle.load(f)

        return preprocessor_object.transform(dataframe)

    def sync_artifact_dir_to_s3(self) -> None:
        aws_bucket_url = (
            f"s3://{PREDICTION_BUCKET_NAME}/{PREDICTION_ARTIFACT_DIR}/{self.timestamp}"
        )
        self.s3_sync.sync_folder_to_s3(
            folder=os.path.join(PREDICTION_ARTIFACT_DIR, self.timestamp, ""),
            aws_bucket_url=aws_bucket_url,
        )

    def start_batch_prediction(self, file_path):
        try:
            InferencePipeline.is_pipeline_running = True

            sensor_dataframe = self.read_csv_data(file_path)

            self.save_csv_data(sensor_dataframe, file_name="sensor_raw")

            dataframe = self.drop_insignificant_columns(sensor_dataframe)

            cols = dataframe.columns.to_list()
            error_message = ""

            status = self.validate_number_of_columns(dataframe)
            if not status:
                error_message = (
                    f"{error_message} DataFrame doesn't contain all columns.\n"
                )

            status = self.is_numerical_column_exist(dataframe)
            if not status:
                error_message = f"{error_message} DataFrame doesn't contain all numerical columns.\n"

            if len(error_message) > 0:
                raise Exception(error_message)

            transform_data = self.load_data_transformer_object(dataframe)
            transform_data = pd.DataFrame(transform_data, columns=cols)

            model_resolver = ModelResolver(model_dir=SAVED_MODEL_DIR)
            if not model_resolver.is_model_exists():
                raise Exception("Model Not Found.")

            best_model_path = model_resolver.get_best_model_path()
            model = load_object(file_path=best_model_path)

            y_pred = model.predict(transform_data)

            transform_data.insert(loc=0, column="class", value=pd.Series(y_pred))
            transform_data["class"].replace(
                TargetValueMapping().reverse_mapping(), inplace=True
            )
            self.save_csv_data(transform_data, file_name="sensor_output")

            InferencePipeline.is_pipeline_running = False
            self.sync_artifact_dir_to_s3()

        except Exception as e:
            self.sync_artifact_dir_to_s3()
            InferencePipeline.is_pipeline_running = False
            raise SensorException(e, sys) from e
