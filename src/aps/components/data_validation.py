import os
import sys

import pandas as pd
from scipy.stats import ks_2samp

from aps.constant.training import SCHEMA_FILE_PATH
from aps.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact
from aps.entity.config_entity import DataValidationConfig
from aps.exception import SensorException
from aps.logger import logging
from aps.utils import read_yaml_file, write_yaml_file


class DataValidation:
    def __init__(
        self,
        data_ingestion_artifact: DataIngestionArtifact,
        data_validation_config: DataValidationConfig,
    ):
        self.data_ingestion_artifact = data_ingestion_artifact
        self.data_validation_config = data_validation_config
        self._schema_config = read_yaml_file(SCHEMA_FILE_PATH)

    def validate_number_of_columns(self, df: pd.DataFrame) -> bool:
        """
        Validates if the DataFrame's column count matches the schema config.

        :param df: The DataFrame to validate.
        :return bool: True if the DataFrame has the expected number of columns; False otherwise.
        """
        expected_columns = len(self._schema_config["columns"])
        actual_columns = len(df.columns)
        logging.info(f"Expected columns: {expected_columns}; Found: {actual_columns}")
        return actual_columns == expected_columns

    def is_numerical_column_exist(self, df: pd.DataFrame) -> bool:
        """
        Check if all numerical columns defined within the schema config exist in the DataFrame.

        :param df: The DataFrame to validate.
        :return bool: True if all required numerical columns exist; False otherwise.
        """
        numerical_columns = self._schema_config["numerical_columns"]
        if missing_columns := [
            col for col in numerical_columns if col not in df.columns
        ]:
            logging.info(f"Missing numerical columns: {missing_columns}")
            return False
        return True

    @staticmethod
    def read_data(file_path: str) -> pd.DataFrame:
        """
        Reads the data from the file path and returns the pandas DataFrame object.

        :param file_path: The file path to the CSV file.
        :return pd.DataFrame: The Pandas DataFrame containing the CSV data.
        """
        return pd.read_csv(file_path)

    def detect_dataset_drift(
        self, base_df: pd.DataFrame, current_df: pd.DataFrame, threshold: float = 0.05
    ) -> bool:
        """
        Detects dataset drift by comparing column distributions between the base and current DataFrames.

        :param base_df: The baseline DataFrame.
        :param current_df: The DataFrame to compare against the baseline.
        :param threshold: The p-value threshold for detecting drift (default is 0.05).
        :return bool: True if no drift is detected (i.e., overall status is good); False otherwise.
        """
        drift_report = {}
        overall_status = True

        for column in base_df.columns:
            stat = ks_2samp(base_df[column], current_df[column])
            # Drift is detected if the p-value is below the threshold.
            drift = stat.pvalue < threshold
            if drift:
                overall_status = False
            drift_report[column] = {
                "p_value": float(stat.pvalue),
                "drift_status": drift,
            }

        drift_report_file_path = self.data_validation_config.drift_report_file_path
        os.makedirs(os.path.dirname(drift_report_file_path), exist_ok=True)
        write_yaml_file(file_path=drift_report_file_path, content=drift_report)
        return overall_status

    def _validate_dataframe(self, df: pd.DataFrame, df_name: str, errors: list) -> None:
        """
        Validates DataFrame by checking the number of columns and existence of numerical columns.

        :param df: The DataFrame to validate.
        :param df_name: A label for the DataFrame used in error messages.
        :param errors: A list to which error messages will be appended if validation fails.
        """
        if not self.validate_number_of_columns(df):
            errors.append(f"{df_name} DataFrame does not contain all required columns.")
        if not self.is_numerical_column_exist(df):
            errors.append(
                f"{df_name} DataFrame does not contain all required numerical columns."
            )

    def initiate_data_validation(self) -> DataValidationArtifact:
        try:
            logging.info(">>> Data Validation Component Started.")
            errors = []

            train_file_path = self.data_ingestion_artifact.train_file_path
            test_file_path = self.data_ingestion_artifact.test_file_path

            logging.info("Load data from provided file paths.")
            train_df = DataValidation.read_data(train_file_path)
            test_df = DataValidation.read_data(test_file_path)

            logging.info("Validate both train and test dataframes.")
            self._validate_dataframe(train_df, "Train", errors)
            self._validate_dataframe(test_df, "Test", errors)

            if errors:
                raise Exception("\n".join(errors))

            logging.info("Checking Data Drift .....")
            drift_status = self.detect_dataset_drift(
                base_df=train_df, current_df=test_df
            )

            data_validation_artifact = DataValidationArtifact(
                validation_status=drift_status,
                valid_train_file_path=train_file_path,
                valid_test_file_path=test_file_path,
                invalid_train_file_path=None,
                invalid_test_file_path=None,
                drift_report_file_path=self.data_validation_config.drift_report_file_path,
            )

            logging.info(f"Data Validation Artifact: [{data_validation_artifact}].")
            logging.info(">>> Data Validation Component Ended.")
            return data_validation_artifact

        except Exception as e:
            raise SensorException(e, sys) from e
