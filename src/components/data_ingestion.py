import os
import sys
from pandas import DataFrame
from sklearn.model_selection import train_test_split
from src.constant.training import SCHEMA_FILE_PATH, TARGET_COLUMN
from src.data_access.sensor_data import SensorData
from src.entity.artifact_entity import DataIngestionArtifact
from src.entity.config_entity import DataIngestionConfig
from src.utils import read_yaml_file
from src.logger import logging
from src.exception import SensorException


class DataIngestion:
    def __init__(self, data_ingestion_config: DataIngestionConfig):
        self.data_ingestion_config = data_ingestion_config
        self._schema_config = read_yaml_file(SCHEMA_FILE_PATH)

    def export_data_into_feature_store(self) -> DataFrame:
        """
        Export MongoDB collection records as DataFrame into Feature Store.
        """
        try:
            logging.info("Export records from MongoDB to Feature Store.")
            sensor_data = SensorData()

            dataframe = sensor_data.export_collection_as_dataframe(
                collection_name=self.data_ingestion_config.collection_name
            )
            feature_store_file_path = self.data_ingestion_config.feature_store_file_path

            # Create Folder.
            dir_path = os.path.dirname(feature_store_file_path)
            os.makedirs(dir_path, exist_ok=True)
            dataframe.to_csv(feature_store_file_path, index=False, header=True)
            return dataframe

        except Exception as e:
            raise SensorException(e, sys) from e

    def split_data_as_train_test(self, dataframe: DataFrame) -> None:
        """
        Feature Store dataset will get split into train and test files.
        """
        try:
            logging.info("Perform train_test_split on the Feature Store dataset.")
            train_set, test_set = train_test_split(
                dataframe,
                test_size=self.data_ingestion_config.train_test_split_ratio,
                stratify=dataframe[TARGET_COLUMN],
                random_state=1,
            )

            dir_path = os.path.dirname(self.data_ingestion_config.training_file_path)
            os.makedirs(dir_path, exist_ok=True)

            logging.info("Exporting train and test file path .....")
            train_set.to_csv(
                self.data_ingestion_config.training_file_path, index=False, header=True
            )
            test_set.to_csv(
                self.data_ingestion_config.testing_file_path, index=False, header=True
            )

        except Exception as e:
            raise SensorException(e, sys) from e

    def initiate_data_ingestion(self) -> DataIngestionArtifact:
        try:
            logging.info(">>> Data Ingestion Component Started.")
            dataframe = self.export_data_into_feature_store()
            dataframe = dataframe.drop(self._schema_config["drop_columns"], axis=1)
            self.split_data_as_train_test(dataframe=dataframe)
            logging.info(">>> Data Ingestion Component Ended.")
            return DataIngestionArtifact(
                train_file_path=self.data_ingestion_config.training_file_path,
                test_file_path=self.data_ingestion_config.testing_file_path,
            )

        except Exception as e:
            raise SensorException(e, sys) from e
