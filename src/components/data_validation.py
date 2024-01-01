import os
import sys
import pandas as pd
from scipy.stats import ks_2samp
from src.constant.training import SCHEMA_FILE_PATH
from src.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact
from src.entity.config_entity import DataValidationConfig
from src.utils import read_yaml_file, write_yaml_file
from src.logger import logging
from src.exception import SensorException


class DataValidation:
    def __init__(
        self,
        data_ingestion_artifact: DataIngestionArtifact,
        data_validation_config: DataValidationConfig,
    ):
        self.data_ingestion_artifact = data_ingestion_artifact
        self.data_validation_config = data_validation_config
        self._schema_config = read_yaml_file(SCHEMA_FILE_PATH)

    def validate_number_of_columns(self, dataframe: pd.DataFrame) -> bool:
        number_of_columns = len(self._schema_config["columns"])
        logging.info(f"Required number of columns: {number_of_columns}")
        logging.info(f"DataFrame has columns: {len(dataframe.columns)}")
        return len(dataframe.columns) == number_of_columns

    def is_numerical_column_exist(self, dataframe: pd.DataFrame) -> bool:
        numerical_columns = self._schema_config["numerical_columns"]
        dataframe_columns = dataframe.columns

        numerical_column_present = True
        missing_numerical_columns = []

        for num_column in numerical_columns:
            if num_column not in dataframe_columns:
                numerical_column_present = False
                missing_numerical_columns.append(num_column)

        logging.info(f"Missing numerical columns: [{missing_numerical_columns}]")
        return numerical_column_present

    @staticmethod
    def read_data(file_path) -> pd.DataFrame:
        return pd.read_csv(file_path)

    def detect_dataset_drift(self, base_df, current_df, threshold=0.05) -> bool:
        try:
            status = True
            report = {}

            for column in base_df.columns:
                d1 = base_df[column]
                d2 = current_df[column]
                is_same_dist = ks_2samp(d1, d2)
                if threshold <= is_same_dist.pvalue:
                    is_found = False
                else:
                    is_found = True
                    status = False
                report[column] = {
                    "p_value": float(is_same_dist.pvalue),
                    "drift_status": is_found,
                }

            drift_report_file_path = self.data_validation_config.drift_report_file_path

            # Create Directory.
            dir_path = os.path.dirname(drift_report_file_path)
            os.makedirs(dir_path, exist_ok=True)
            write_yaml_file(file_path=drift_report_file_path, content=report)
            return status

        except Exception as e:
            raise SensorException(e, sys) from e

    def initiate_data_validation(self) -> DataValidationArtifact:
        try:
            logging.info(">>> Data Validation Component Started.")
            error_message = ""
            train_file_path = self.data_ingestion_artifact.train_file_path
            test_file_path = self.data_ingestion_artifact.test_file_path

            logging.info("Load data from train and test file location.")
            train_dataframe = DataValidation.read_data(train_file_path)
            test_dataframe = DataValidation.read_data(test_file_path)

            logging.info("Validating the number of columns .....")
            status = self.validate_number_of_columns(dataframe=train_dataframe)
            if not status:
                error_message = (
                    f"{error_message} Train DataFrame doesn't contain all columns.\n"
                )

            status = self.validate_number_of_columns(dataframe=test_dataframe)
            if not status:
                error_message = (
                    f"{error_message} Test DataFrame doesn't contain all columns.\n"
                )

            logging.info("Validating the numerical columns .....")
            status = self.is_numerical_column_exist(dataframe=train_dataframe)
            if not status:
                error_message = f"{error_message} Train DataFrame doesn't contain all numerical columns.\n"

            status = self.is_numerical_column_exist(dataframe=test_dataframe)
            if not status:
                error_message = f"{error_message} Test DataFrame doesn't contain all numerical columns.\n"

            if len(error_message) > 0:
                raise Exception(error_message)

            logging.info("Checking Data Drift .....")
            status = self.detect_dataset_drift(
                base_df=train_dataframe, current_df=test_dataframe
            )

            data_validation_artifact = DataValidationArtifact(
                validation_status=status,
                valid_train_file_path=self.data_ingestion_artifact.train_file_path,
                valid_test_file_path=self.data_ingestion_artifact.test_file_path,
                invalid_train_file_path=None,
                invalid_test_file_path=None,
                drift_report_file_path=self.data_validation_config.drift_report_file_path,
            )

            logging.info(f"Data Validation Artifact: [{data_validation_artifact}].")
            logging.info(">>> Data Validation Component Ended.")
            return data_validation_artifact

        except Exception as e:
            raise SensorException(e, sys) from e
