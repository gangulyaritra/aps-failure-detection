from dataclasses import dataclass


@dataclass(frozen=True)
class DataIngestionArtifact:
    train_file_path: str
    test_file_path: str


@dataclass(frozen=True)
class DataValidationArtifact:
    validation_status: bool
    valid_train_file_path: str
    valid_test_file_path: str
    invalid_train_file_path: str
    invalid_test_file_path: str
    drift_report_file_path: str


@dataclass(frozen=True)
class DataTransformationArtifact:
    transformed_object_file_path: str
    transformed_train_file_path: str
    transformed_test_file_path: str


@dataclass(frozen=True)
class ClassificationMetricArtifact:
    timestamp: int
    f1_score: float
    precision_score: float
    recall_score: float
    cohen_kappa_score: float
    roc_auc_score: float


@dataclass(frozen=True)
class ModelTrainerArtifact:
    trained_model_file_path: str
    train_metric_artifact: ClassificationMetricArtifact
    test_metric_artifact: ClassificationMetricArtifact


@dataclass(frozen=True)
class ModelEvaluationArtifact:
    is_model_accepted: bool
    improved_f1_score: float
    improved_roc_auc_score: float
    best_model_path: str
    trained_model_path: str
    train_model_metric_artifact: ClassificationMetricArtifact
    best_model_metric_artifact: ClassificationMetricArtifact


@dataclass(frozen=True)
class ModelPusherArtifact:
    saved_model_path: str
    model_file_path: str
