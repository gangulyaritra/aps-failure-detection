import sys
import pandas as pd
import numpy as np
from imblearn.combine import SMOTETomek
from sklearn.impute import KNNImputer
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import Pipeline
from src.constant.training import TARGET_COLUMN
from src.entity.artifact_entity import (
    DataValidationArtifact,
    DataTransformationArtifact,
)
from src.entity.config_entity import DataTransformationConfig
from src.ml.estimator import TargetValueMapping
from src.utils import save_numpy_array_data, save_object
from src.logger import logging
from src.exception import SensorException


class DataTransformation:
    def __init__(
        self,
        data_validation_artifact: DataValidationArtifact,
        data_transformation_config: DataTransformationConfig,
    ):
        self.data_validation_artifact = data_validation_artifact
        self.data_transformation_config = data_transformation_config

    @staticmethod
    def read_data(file_path) -> pd.DataFrame:
        return pd.read_csv(file_path)

    @classmethod
    def get_data_transformer_object(cls) -> Pipeline:
        return Pipeline(
            steps=[
                ("Imputer", KNNImputer(n_neighbors=5)),
                ("Scaler", RobustScaler()),
            ]
        )

    def initiate_data_transformation(self) -> DataTransformationArtifact:
        try:
            logging.info(">>> Data Transformation Component Started.")
            train_df = DataTransformation.read_data(
                self.data_validation_artifact.valid_train_file_path
            )
            test_df = DataTransformation.read_data(
                self.data_validation_artifact.valid_test_file_path
            )

            preprocessor = self.get_data_transformer_object()

            input_feature_train_df = train_df.drop(columns=[TARGET_COLUMN], axis=1)
            target_feature_train_df = train_df[TARGET_COLUMN]
            target_feature_train_df = target_feature_train_df.replace(
                TargetValueMapping().to_dict()
            )

            input_feature_test_df = test_df.drop(columns=[TARGET_COLUMN], axis=1)
            target_feature_test_df = test_df[TARGET_COLUMN]
            target_feature_test_df = target_feature_test_df.replace(
                TargetValueMapping().to_dict()
            )

            logging.info("Perform Data Transformation using the sklearn Pipeline.")
            preprocessor_object = preprocessor.fit(input_feature_train_df)
            transformed_input_train_feature = preprocessor_object.transform(
                input_feature_train_df
            )
            transformed_input_test_feature = preprocessor_object.transform(
                input_feature_test_df
            )

            logging.info("Handle Imbalanced Dataset using SMOTETomek Algorithm.")
            smt = SMOTETomek(sampling_strategy="minority")

            input_feature_train_final, target_feature_train_final = smt.fit_resample(
                transformed_input_train_feature, target_feature_train_df
            )
            input_feature_test_final, target_feature_test_final = smt.fit_resample(
                transformed_input_test_feature, target_feature_test_df
            )

            logging.info("Convert train and test datasets as NumPy array.")
            train_arr = np.c_[
                input_feature_train_final, np.array(target_feature_train_final)
            ]
            test_arr = np.c_[
                input_feature_test_final, np.array(target_feature_test_final)
            ]

            logging.info("Save train and test datasets as NumPy objects.")
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
                preprocessor_object,
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
