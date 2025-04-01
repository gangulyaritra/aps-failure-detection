import sys

import numpy as np
import pandas as pd
from imblearn.combine import SMOTETomek
from sklearn.impute import KNNImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler

from aps.constant.training import TARGET_COLUMN
from aps.entity.artifact_entity import (
    DataTransformationArtifact,
    DataValidationArtifact,
)
from aps.entity.config_entity import DataTransformationConfig
from aps.exception import SensorException
from aps.logger import logging
from aps.ml.estimator import TargetValueMapping
from aps.utils import save_numpy_array_data, save_object

pd.set_option("future.no_silent_downcasting", True)


class DataTransformation:
    def __init__(
        self,
        data_validation_artifact: DataValidationArtifact,
        data_transformation_config: DataTransformationConfig,
    ):
        self.data_validation_artifact = data_validation_artifact
        self.data_transformation_config = data_transformation_config

    @staticmethod
    def read_data(file_path: str) -> pd.DataFrame:
        """
        Reads the data from the file path and returns the pandas DataFrame object.

        :param file_path: The path to the CSV file.
        :return pd.DataFrame: The loaded data as a DataFrame.
        """
        return pd.read_csv(file_path)

    @classmethod
    def get_data_transformer_object(cls) -> Pipeline:
        """
        Returns a Pipeline object that imputes missing values using KNNImputer and scales the features using RobustScaler.
        The pipeline includes:
            - A KNNImputer to impute missing values using the 5 nearest neighbors.
            - A RobustScaler to scale the features, reducing the influence of outliers.

        :return Pipeline: The data transformation pipeline.
        """
        return Pipeline(
            steps=[("Imputer", KNNImputer(n_neighbors=5)), ("Scaler", RobustScaler())]
        )

    @staticmethod
    def split_features_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
        """
        Splits the features and target columns from the DataFrame and returns them as separate objects.

        :param df: The DataFrame containing both feature and target columns.
        :return tuple[pd.DataFrame, pd.Series]: A tuple containing:
            - The features DataFrame (x).
            - The target Series (y).
        """
        target_mapping = TargetValueMapping().to_dict()
        x = df.drop(columns=[TARGET_COLUMN])
        y = df[TARGET_COLUMN].replace(target_mapping).infer_objects(copy=False)
        return x, y

    def initiate_data_transformation(self) -> DataTransformationArtifact:
        try:
            logging.info(">>> Data Transformation Component Started.")

            logging.info("Read training and test datasets.")
            train_df = self.read_data(
                self.data_validation_artifact.valid_train_file_path
            )
            test_df = self.read_data(self.data_validation_artifact.valid_test_file_path)

            logging.info("Split the dataset into features and target sets.")
            x_train, y_train = self.split_features_target(train_df)
            x_test, y_test = self.split_features_target(test_df)

            logging.info("Perform Data Transformation using the sklearn Pipeline.")
            preprocessor = self.get_data_transformer_object()
            x_train_transformed = preprocessor.fit_transform(x_train)
            x_test_transformed = preprocessor.transform(x_test)

            logging.info("Handle Imbalanced Dataset using SMOTETomek Algorithm.")
            smt = SMOTETomek(sampling_strategy="minority")
            x_train_resampled, y_train_resampled = smt.fit_resample(
                x_train_transformed, y_train
            )
            x_test_resampled, y_test_resampled = smt.fit_resample(
                x_test_transformed, y_test
            )

            logging.info(
                "Combine transformed features and target data into NumPy arrays."
            )
            train_arr = np.c_[x_train_resampled, y_train_resampled]
            test_arr = np.c_[x_test_resampled, y_test_resampled]

            logging.info("Save transformed train and test datasets as NumPy objects.")
            save_numpy_array_data(
                self.data_transformation_config.transformed_train_file_path,
                array=train_arr,
            )
            save_numpy_array_data(
                self.data_transformation_config.transformed_test_file_path,
                array=test_arr,
            )

            logging.info("Save the Transformer Pipeline object into a Pickle file.")
            save_object(
                self.data_transformation_config.transformed_object_file_path,
                preprocessor,
            )

            data_transformation_artifact = DataTransformationArtifact(
                transformed_object_file_path=self.data_transformation_config.transformed_object_file_path,
                transformed_train_file_path=self.data_transformation_config.transformed_train_file_path,
                transformed_test_file_path=self.data_transformation_config.transformed_test_file_path,
            )
            logging.info(
                f"Data Transformation Artifact: [{data_transformation_artifact}]."
            )
            logging.info(">>> Data Transformation Component Ended.")
            return data_transformation_artifact

        except Exception as e:
            raise SensorException(e, sys) from e
