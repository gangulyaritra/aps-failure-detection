import glob
import os
import pickle
import sys

import numpy as np
import pandas as pd

from aps.cloud_storage.s3_syncer import S3Sync
from aps.constant import PREDICTION_BUCKET_NAME, TIMESTAMP
from aps.constant.training import (
    ARTIFACT_DIR,
    DATA_TRANSFORMATION_DIR_NAME,
    DATA_TRANSFORMATION_TRANSFORMED_OBJECT_DIR,
    PREDICTION_ARTIFACT_DIR,
    PREPROCSSING_OBJECT_FILE_NAME,
    SAVED_MODEL_DIR,
    SCHEMA_FILE_PATH,
)
from aps.exception import SensorException
from aps.logger import logging
from aps.ml.estimator import ModelResolver, TargetValueMapping
from aps.utils import load_object, read_yaml_file


class InferencePipeline:
    """
    InferencePipeline is responsible for running the end-to-end batch prediction pipeline.
    """

    is_pipeline_running = False

    @staticmethod
    def create_artifact_dir():
        """
        Creates a directory for storing artifacts of the prediction process.
        """
        folder_path = os.path.join(PREDICTION_ARTIFACT_DIR, TIMESTAMP, "")
        os.makedirs(os.path.dirname(folder_path), exist_ok=True)
        return folder_path

    @staticmethod
    def read_csv_data(file_path: str) -> pd.DataFrame:
        """
        Reads the data from the file path and returns the pandas DataFrame object.

        :param file_path: The file path to the CSV file.
        :return pd.DataFrame: The Pandas DataFrame containing the CSV data.
        """
        return pd.read_csv(file_path)

    @staticmethod
    def save_csv_data(df: pd.DataFrame, file_name: str) -> None:
        """
        Saves a DataFrame as a CSV file in the artifact directory.

        :param df: The DataFrame to save.
        :param file_name: File name for the CSV file.
        """
        folder = InferencePipeline.create_artifact_dir()
        df.to_csv(os.path.join(folder, f"{file_name}.csv"), index=False, header=True)

    @staticmethod
    def drop_insignificant_columns(df: pd.DataFrame) -> pd.DataFrame:
        """
        Removes unnecessary columns from the DataFrame based on the schema configuration.

        :param df: The input DataFrame.
        :return pd.DataFrame: The cleaned DataFrame.
        """
        drop_cols = read_yaml_file(SCHEMA_FILE_PATH).get("drop_columns", [])
        df = df.drop(columns=drop_cols, errors="ignore")

        if "_id" in df.columns:
            df.drop(columns=["_id"], axis=1, inplace=True)

        df.replace("na", np.nan, inplace=True)
        return df

    @staticmethod
    def validate_number_of_columns(df: pd.DataFrame) -> bool:
        """
        Validates if the DataFrame's column count matches the schema config.

        :param df: The DataFrame to validate.
        :return bool: True if the DataFrame has the expected number of columns; False otherwise.
        """
        expected_columns = read_yaml_file(SCHEMA_FILE_PATH)["columns"]
        expected_count = len(expected_columns) - 1
        return len(df.columns) == expected_count

    @staticmethod
    def is_numerical_column_exist(df: pd.DataFrame) -> bool:
        """
        Check if all numerical columns defined within the schema config exist in the DataFrame.

        :param df: The DataFrame to validate.
        :return bool: True if all required numerical columns exist; False otherwise.
        """
        numerical_columns = read_yaml_file(SCHEMA_FILE_PATH)["numerical_columns"]
        return all(col in df.columns for col in numerical_columns)

    @staticmethod
    def load_data_transformer_object(df: pd.DataFrame) -> np.ndarray:
        """
        Applies the latest data transformation from a file to the DataFrame.

        :param df: The input DataFrame is to be transformed.
        :return np.ndarray: The transformed data as a numpy array.
        """
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

        return preprocessor_object.transform(df)

    @staticmethod
    def sync_artifact_dir_to_s3() -> None:
        """
        Syncs the inference artifact directory to S3 for persistent storage.
        """
        aws_bucket_url = (
            f"s3://{PREDICTION_BUCKET_NAME}/{PREDICTION_ARTIFACT_DIR}/{TIMESTAMP}"
        )
        S3Sync().sync_folder_to_s3(
            folder=os.path.join(PREDICTION_ARTIFACT_DIR, TIMESTAMP, ""),
            aws_bucket_url=aws_bucket_url,
        )

    def start_batch_prediction(self, file_path: str) -> None:
        """
        Executes the end-to-end batch prediction process:
            1. Read raw CSV data.
            2. Save raw data in the artifact directory.
            3. Drop irrelevant columns and handle "na".
            4. Validate column count and numerical column presence.
            5. Transform data using the latest preprocessor object.
            6. Load the best model and make predictions.
            7. Save predictions to artifact directory and sync the directory to S3.

        Raises:
            SensorException: if any failure occurs in the pipeline steps.
        """
        try:
            logging.info(">>> Inference Pipeline Started.")
            InferencePipeline.is_pipeline_running = True

            logging.info("Read the raw CSV data.")
            sensor_df = self.read_csv_data(file_path)

            logging.info("Save raw data in the artifact directory.")
            self.save_csv_data(sensor_df, "sensor_raw")

            logging.info("Drop irrelevant columns and handle 'na'.")
            df_processed = self.drop_insignificant_columns(sensor_df)

            logging.info("Validate column count and numerical column presence.")
            error_message = ""
            if not self.validate_number_of_columns(df_processed):
                error_message += (
                    "DataFrame does not contain the expected number of columns.\n"
                )
            if not self.is_numerical_column_exist(df_processed):
                error_message += "DataFrame is missing one or more numerical columns.\n"
            if error_message:
                raise ValueError(error_message.strip())

            logging.info("Transform data using the latest preprocessor object.")
            transformed_array = self.load_data_transformer_object(df_processed)
            transformed_df = pd.DataFrame(
                transformed_array, columns=df_processed.columns
            )

            logging.info("Load the best model and make predictions.")
            model_resolver = ModelResolver(model_dir=SAVED_MODEL_DIR)
            if not model_resolver.is_model_exists():
                raise FileNotFoundError("No valid model found for inference.")

            model = load_object(file_path=model_resolver.get_best_model_path())

            y_pred = model.predict(transformed_df)

            # Insert predictions as the first column.
            transformed_df.insert(loc=0, column="class", value=y_pred)

            # Convert predicted integer classes to their original labels.
            transformed_df["class"].replace(
                TargetValueMapping().reverse_mapping(), inplace=True
            )

            logging.info("Save predictions to artifact directory.")
            self.save_csv_data(transformed_df, "sensor_output")

            self.sync_artifact_dir_to_s3()
            InferencePipeline.is_pipeline_running = False
            logging.info(">>> Inference Pipeline Ended.")

        except Exception as e:
            self.sync_artifact_dir_to_s3()
            InferencePipeline.is_pipeline_running = False
            raise SensorException(e, sys) from e
