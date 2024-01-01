import sys
import pandas as pd
from src.constant.training import TARGET_COLUMN, SQLITE_DB_NAME
from src.entity.artifact_entity import (
    DataValidationArtifact,
    ModelTrainerArtifact,
    ModelEvaluationArtifact,
)
from src.entity.config_entity import ModelEvaluationConfig
from src.ml.estimator import TargetValueMapping, ModelResolver
from src.ml.evaluation_metrics import (
    get_classification_score,
    save_classification_score,
)
from src.utils import load_object, write_yaml_file
from src.logger import logging
from src.exception import SensorException


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
            valid_train_file_path = self.data_validation_artifact.valid_train_file_path
            valid_test_file_path = self.data_validation_artifact.valid_test_file_path

            train_df = pd.read_csv(valid_train_file_path)
            test_df = pd.read_csv(valid_test_file_path)
            df = pd.concat([train_df, test_df])

            y_true = df[TARGET_COLUMN]
            y_true.replace(TargetValueMapping().to_dict(), inplace=True)
            df.drop(TARGET_COLUMN, axis=1, inplace=True)

            train_model_file_path = self.model_trainer_artifact.trained_model_file_path
            model_resolver = ModelResolver()
            is_model_accepted = True

            if not model_resolver.is_model_exists():
                model_evaluation_artifact = ModelEvaluationArtifact(
                    is_model_accepted=is_model_accepted,
                    improved_f1_score=None,
                    improved_roc_auc_score=None,
                    best_model_path=None,
                    trained_model_path=train_model_file_path,
                    train_model_metric_artifact=self.model_trainer_artifact.test_metric_artifact,
                    best_model_metric_artifact=None,
                )
                logging.info(
                    f"Model Evaluation Artifact: [{model_evaluation_artifact}]."
                )
                logging.info(">>> Model Evaluation Component Ended.")
                return model_evaluation_artifact

            latest_model_path = model_resolver.get_best_model_path()
            latest_model = load_object(file_path=latest_model_path)
            train_model = load_object(file_path=train_model_file_path)

            y_trained_pred = train_model.predict(df)
            y_latest_pred = latest_model.predict(df)

            trained_metric = get_classification_score(y_true, y_trained_pred)
            latest_metric = get_classification_score(y_true, y_latest_pred)

            improved_f1_score = trained_metric.f1_score - latest_metric.f1_score
            improved_roc_auc_score = (
                trained_metric.roc_auc_score - latest_metric.roc_auc_score
            )

            is_model_accepted = (
                self.model_eval_config.change_threshold < improved_f1_score
            ) or (self.model_eval_config.change_threshold < improved_roc_auc_score)

            model_evaluation_artifact = ModelEvaluationArtifact(
                is_model_accepted=is_model_accepted,
                improved_f1_score=improved_f1_score,
                improved_roc_auc_score=improved_roc_auc_score,
                best_model_path=latest_model_path,
                trained_model_path=train_model_file_path,
                train_model_metric_artifact=trained_metric,
                best_model_metric_artifact=latest_metric,
            )

            save_classification_score(trained_metric, file_path=f"./{SQLITE_DB_NAME}")
            logging.info("Data Inserted into SQLite Successfully.")

            model_eval_report = model_evaluation_artifact.__dict__
            write_yaml_file(self.model_eval_config.report_file_path, model_eval_report)

            logging.info(f"Model Evaluation Artifact: [{model_evaluation_artifact}].")
            logging.info(">>> Model Evaluation Component Ended.")
            return model_evaluation_artifact

        except Exception as e:
            raise SensorException(e, sys) from e
