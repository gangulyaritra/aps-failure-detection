import os
import sys
from typing import Tuple

import numpy as np
from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBClassifier

from aps.entity.artifact_entity import DataTransformationArtifact, ModelTrainerArtifact
from aps.entity.config_entity import ModelTrainerConfig
from aps.exception import SensorException
from aps.logger import logging
from aps.ml.estimator import SensorModel
from aps.ml.evaluation_metrics import get_classification_score
from aps.utils import load_numpy_array_data, load_object, save_object


class ModelTrainer:
    """
    Model Trainer Component to train the model with the transformed data.
    """

    def __init__(
        self,
        model_trainer_config: ModelTrainerConfig,
        data_transformation_artifact: DataTransformationArtifact,
    ) -> None:
        self.model_trainer_config = model_trainer_config
        self.data_transformation_artifact = data_transformation_artifact

    def perform_hyperparameter_tuning(self, clf: XGBClassifier) -> RandomizedSearchCV:
        """
        Perform Hyperparameter Tuning using RandomizedSearchCV.

        :param clf: XGBoost Classifier.
        :return RandomizedSearchCV: The tuned classifier search object.
        """
        logging.info("Performing Hyperparameter Tuning .....")

        return RandomizedSearchCV(
            estimator=clf,
            param_distributions=self.model_trainer_config.param_distributions,
            n_iter=3,
            cv=3,
            verbose=2,
            random_state=1,
            n_jobs=-1,
        )

    def train_model(
        self, x_train: np.ndarray, y_train: np.ndarray
    ) -> RandomizedSearchCV:
        """
        Train the model using the XGBoost Classifier.

        :param x_train (np.ndarray): Training features.
        :param y_train (np.ndarray): Training targets.
        :return RandomizedSearchCV: The fitted hyperparameter search object.
        """
        search_cv = self.perform_hyperparameter_tuning(XGBClassifier())
        search_cv.fit(x_train, y_train)
        logging.info(f"Best Estimator: {search_cv.best_estimator_}")
        return search_cv

    def _load_datasets(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Loads the feature and target datasets.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: The train and test datasets.
        """
        train_arr = load_numpy_array_data(
            self.data_transformation_artifact.transformed_train_file_path
        )
        test_arr = load_numpy_array_data(
            self.data_transformation_artifact.transformed_test_file_path
        )

        x_train, y_train, x_test, y_test = (
            train_arr[:, :-1],
            train_arr[:, -1],
            test_arr[:, :-1],
            test_arr[:, -1],
        )
        return x_train, y_train, x_test, y_test

    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        try:
            logging.info(">>> Model Trainer Component Started.")

            logging.info("Load NumPy datasets.")
            x_train, y_train, x_test, y_test = self._load_datasets()

            logging.info("Start Model Building .....")
            search_cv_model = self.train_model(x_train, y_train)
            best_model = search_cv_model.best_estimator_

            logging.info("Evaluate the model on the training data.")
            y_train_pred = best_model.predict(x_train)
            train_metrics = get_classification_score(
                y_true=y_train, y_pred=y_train_pred
            )

            if (
                train_metrics.f1_score <= self.model_trainer_config.expected_f1_score
                or train_metrics.roc_auc_score
                <= self.model_trainer_config.expected_roc_auc_score
            ):
                raise Exception(
                    "Trained Model Performance is inadequate to provide expected accuracy."
                )

            logging.info("Evaluate the model on the test data.")
            y_test_pred = best_model.predict(x_test)
            test_metrics = get_classification_score(y_true=y_test, y_pred=y_test_pred)

            # Check for overfitting or underfitting.
            f1_diff = abs(train_metrics.f1_score - test_metrics.f1_score)
            if f1_diff > self.model_trainer_config.overfitting_underfitting_threshold:
                raise Exception(
                    "Model Performance is inadequate due to high overfitting/underfitting."
                )

            logging.info("Load the preprocessor and save the sensor model.")
            preprocessor = load_object(
                file_path=self.data_transformation_artifact.transformed_object_file_path
            )
            model_dir = os.path.dirname(
                self.model_trainer_config.trained_model_file_path
            )
            os.makedirs(model_dir, exist_ok=True)

            sensor_model = SensorModel(preprocessor=preprocessor, model=best_model)
            save_object(self.model_trainer_config.trained_model_file_path, sensor_model)
            logging.info("Saved the model object into a Pickle file.")

            model_trainer_artifact = ModelTrainerArtifact(
                trained_model_file_path=self.model_trainer_config.trained_model_file_path,
                train_metric_artifact=train_metrics,
                test_metric_artifact=test_metrics,
            )

            logging.info(f"Model Trainer Artifact: [{model_trainer_artifact}].")
            logging.info(">>> Model Trainer Component Ended.")
            return model_trainer_artifact

        except Exception as e:
            raise SensorException(e, sys) from e
