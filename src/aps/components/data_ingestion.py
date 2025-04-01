import os
import sys

from pandas import DataFrame
from sklearn.model_selection import train_test_split

from aps.constant.training import SCHEMA_FILE_PATH, TARGET_COLUMN
from aps.data_access.sensor_data import SensorData
from aps.entity.artifact_entity import DataIngestionArtifact
from aps.entity.config_entity import DataIngestionConfig
from aps.exception import SensorException
from aps.logger import logging
from aps.utils import read_yaml_file


class DataIngestion:
    def __init__(self, data_ingestion_config: DataIngestionConfig) -> None:
        self.data_ingestion_config = data_ingestion_config
        self._schema_config = read_yaml_file(SCHEMA_FILE_PATH)

    def export_data_into_feature_store(self) -> DataFrame:
        """
        Export MongoDB collection records as DataFrame into Feature Store.
        """
        try:
            logging.info("Export records from MongoDB to Feature Store.")
            sensor_data = SensorData()

            df = sensor_data.export_collection_as_dataframe(
                collection_name=self.data_ingestion_config.collection_name
            )
            feature_store_file_path = self.data_ingestion_config.feature_store_file_path

            os.makedirs(os.path.dirname(feature_store_file_path), exist_ok=True)
            df.to_csv(feature_store_file_path, index=False, header=True)
            return df

        except Exception as e:
            raise SensorException(e, sys) from e

    def split_data_as_train_test(self, df: DataFrame) -> None:
        """
        Feature Store dataset will get split into train and test files.

        :param df: The Dataframe to split into training and test sets.
        """
        try:
            logging.info("Split Feature Store dataset using train_test_split.")
            train_set, test_set = train_test_split(
                df,
                test_size=self.data_ingestion_config.train_test_split_ratio,
                stratify=df[TARGET_COLUMN],
                random_state=1,
            )

            os.makedirs(
                os.path.dirname(self.data_ingestion_config.training_file_path),
                exist_ok=True,
            )

            logging.info("Export train and test file path.")
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
            df = self.export_data_into_feature_store()
            df = df.drop(self._schema_config["drop_columns"], axis=1)
            self.split_data_as_train_test(df=df)
            logging.info(">>> Data Ingestion Component Ended.")
            return DataIngestionArtifact(
                train_file_path=self.data_ingestion_config.training_file_path,
                test_file_path=self.data_ingestion_config.testing_file_path,
            )

        except Exception as e:
            raise SensorException(e, sys) from e
