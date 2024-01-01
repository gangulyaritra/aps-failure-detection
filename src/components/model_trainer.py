import os
import sys
from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBClassifier
from src.entity.artifact_entity import DataTransformationArtifact, ModelTrainerArtifact
from src.entity.config_entity import ModelTrainerConfig
from src.ml.estimator import SensorModel
from src.ml.evaluation_metrics import get_classification_score
from src.utils import save_object, load_object, load_numpy_array_data
from src.logger import logging
from src.exception import SensorException


class ModelTrainer:
    def __init__(
        self,
        model_trainer_config: ModelTrainerConfig,
        data_transformation_artifact: DataTransformationArtifact,
    ):
        self.model_trainer_config = model_trainer_config
        self.data_transformation_artifact = data_transformation_artifact

    def perform_hyperparamter_tuning(self, clf):
        logging.info("Performing Hyperparameter Tuning .....")
        parameters = {
            "learning_rate": [0.0136, 0.03],
            "max_depth": [7, 11],
            "n_estimators": [469, 877],
        }

        return RandomizedSearchCV(
            estimator=clf,
            param_distributions=parameters,
            n_iter=3,
            cv=3,
            verbose=2,
            random_state=1,
            n_jobs=-1,
        )

    def train_model(self, x_train, y_train):
        xgb_clf = self.perform_hyperparamter_tuning(XGBClassifier())
        xgb_clf.fit(x_train, y_train)
        logging.info(f"Best Estimator: {xgb_clf.best_estimator_}")
        return xgb_clf

    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        try:
            logging.info(">>> Model Trainer Component Started.")
            train_file_path = (
                self.data_transformation_artifact.transformed_train_file_path
            )
            test_file_path = (
                self.data_transformation_artifact.transformed_test_file_path
            )

            logging.info("Load NumPy datasets.")
            train_arr = load_numpy_array_data(train_file_path)
            test_arr = load_numpy_array_data(test_file_path)

            logging.info("Split dataset into train and test sets.")
            x_train, y_train, x_test, y_test = (
                train_arr[:, :-1],
                train_arr[:, -1],
                test_arr[:, :-1],
                test_arr[:, -1],
            )

            logging.info("Start Model Building.")
            model = self.train_model(x_train, y_train)

            # Model Prediction of the train dataset with the Classification Score.
            y_train_pred = model.predict(x_train)
            classification_train_metric = get_classification_score(
                y_true=y_train, y_pred=y_train_pred
            )

            if (
                classification_train_metric.f1_score
                <= self.model_trainer_config.expected_f1_score
            ) or (
                classification_train_metric.roc_auc_score
                <= self.model_trainer_config.expected_roc_auc_score
            ):
                raise Exception(
                    "Trained Model Performance is inadequate to provide expected accuracy."
                )

            # Model Prediction of the test dataset with the Classification Score.
            y_test_pred = model.predict(x_test)
            classification_test_metric = get_classification_score(
                y_true=y_test, y_pred=y_test_pred
            )

            # Overfitting and Underfitting.
            diff = abs(
                classification_train_metric.f1_score
                - classification_test_metric.f1_score
            )

            if diff > self.model_trainer_config.overfitting_underfitting_threshold:
                raise Exception("Model Performance is inadequate.")

            preprocessor = load_object(
                file_path=self.data_transformation_artifact.transformed_object_file_path
            )
            model_dir_path = os.path.dirname(
                self.model_trainer_config.trained_model_file_path
            )

            logging.info("Save the Model object into a Pickle file.")
            os.makedirs(model_dir_path, exist_ok=True)
            sensor_model = SensorModel(preprocessor=preprocessor, model=model)
            save_object(
                self.model_trainer_config.trained_model_file_path, obj=sensor_model
            )

            model_trainer_artifact = ModelTrainerArtifact(
                trained_model_file_path=self.model_trainer_config.trained_model_file_path,
                train_metric_artifact=classification_train_metric,
                test_metric_artifact=classification_test_metric,
            )
            logging.info(f"Model Trainer Artifact: [{model_trainer_artifact}].")
            logging.info(">>> Model Trainer Component Ended.")
            return model_trainer_artifact

        except Exception as e:
            raise SensorException(e, sys) from e
