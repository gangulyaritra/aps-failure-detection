import sys

import pandas as pd

from aps.constant.training import SQLITE_DB_NAME, TARGET_COLUMN
from aps.entity.artifact_entity import (
    DataValidationArtifact,
    ModelEvaluationArtifact,
    ModelTrainerArtifact,
)
from aps.entity.config_entity import ModelEvaluationConfig
from aps.exception import SensorException
from aps.logger import logging
from aps.ml.estimator import ModelResolver, TargetValueMapping
from aps.ml.evaluation_metrics import (
    get_classification_score,
    save_classification_score,
)
from aps.utils import load_object, write_yaml_file


class ModelEvaluation:
    def __init__(
        self,
        model_eval_config: ModelEvaluationConfig,
        data_validation_artifact: DataValidationArtifact,
        model_trainer_artifact: ModelTrainerArtifact,
    ):
        self.model_eval_config = model_eval_config
        self.data_validation_artifact = data_validation_artifact
        self.model_trainer_artifact = model_trainer_artifact

    def initiate_model_evaluation(self) -> ModelEvaluationArtifact:
        try:
            logging.info(">>> Model Evaluation Component Started.")

            logging.info("Load and combine train and test datasets.")
            train_df = pd.read_csv(self.data_validation_artifact.valid_train_file_path)
            test_df = pd.read_csv(self.data_validation_artifact.valid_test_file_path)
            df = pd.concat([train_df, test_df])

            # Extract and map the target column.
            y_true = df[TARGET_COLUMN]
            y_true.replace(TargetValueMapping().to_dict(), inplace=True)
            df.drop(TARGET_COLUMN, axis=1, inplace=True)

            # If no best model exists, accept the trained model by default.
            if not ModelResolver().is_model_exists():
                logging.info(
                    "No best model exists, accepting the trained model by default."
                )
                model_evaluation_artifact = ModelEvaluationArtifact(
                    is_model_accepted=True,
                    improved_f1_score=None,
                    improved_roc_auc_score=None,
                    best_model_path=None,
                    trained_model_path=self.model_trainer_artifact.trained_model_file_path,
                    train_model_metric_artifact=self.model_trainer_artifact.test_metric_artifact,
                    best_model_metric_artifact=None,
                )
                logging.info(
                    f"Model Evaluation Artifact: [{model_evaluation_artifact}]."
                )
                logging.info(">>> Model Evaluation Component Ended.")
                return model_evaluation_artifact

            logging.info("Load the latest model and the trained model.")
            latest_model_path = ModelResolver().get_best_model_path()
            latest_model = load_object(latest_model_path)
            train_model = load_object(
                self.model_trainer_artifact.trained_model_file_path
            )

            logging.info("Generate predictions using both models.")
            y_trained_pred = train_model.predict(df)
            y_latest_pred = latest_model.predict(df)

            logging.info("Evaluate classification scores for both models.")
            trained_metric = get_classification_score(y_true, y_trained_pred)
            latest_metric = get_classification_score(y_true, y_latest_pred)

            logging.info("Calculate model improvements.")
            improved_f1_score = trained_metric.f1_score - latest_metric.f1_score
            improved_roc_auc_score = (
                trained_metric.roc_auc_score - latest_metric.roc_auc_score
            )

            logging.info("Check if the trained model meets the improved threshold.")
            is_model_accepted = (
                improved_f1_score > self.model_eval_config.change_threshold
                or improved_roc_auc_score > self.model_eval_config.change_threshold
            )

            # Create the evaluation artifact.
            model_evaluation_artifact = ModelEvaluationArtifact(
                is_model_accepted=is_model_accepted,
                improved_f1_score=improved_f1_score,
                improved_roc_auc_score=improved_roc_auc_score,
                best_model_path=latest_model_path,
                trained_model_path=self.model_trainer_artifact.trained_model_file_path,
                train_model_metric_artifact=trained_metric,
                best_model_metric_artifact=latest_metric,
            )

            # Save metrics to SQLite and write evaluation report.
            save_classification_score(trained_metric, file_path=f"./{SQLITE_DB_NAME}")
            logging.info("Data Inserted into SQLite Successfully.")
            write_yaml_file(
                self.model_eval_config.report_file_path,
                model_evaluation_artifact.__dict__,
            )

            logging.info(f"Model Evaluation Artifact: [{model_evaluation_artifact}].")
            logging.info(">>> Model Evaluation Component Ended.")
            return model_evaluation_artifact

        except Exception as e:
            raise SensorException(e, sys) from e
