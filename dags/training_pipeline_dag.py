import pendulum
from datetime import timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from src.pipeline.training_pipeline import TrainPipeline

training_pipeline = None

default_args = {
    "owner": "Aritra Ganguly",
    "depends_on_past": False,
    "start_date": pendulum.datetime(2023, 11, 25, tz="UTC"),
    "email": ["aritraganguly.msc@protonmail.com"],
    "email_on_failure": True,
    "email_on_retry": True,
    "retries": 2,
    "retry_delay": timedelta(minutes=1),
}

with DAG(
    dag_id="aps-failure-detection",
    description="Detection of APS Failure at Scania Trucks with Machine Learning.",
    default_args=default_args,
    schedule_interval="@weekly",
    catchup=False,
    tags=["example"],
) as dag:
    training_pipeline = TrainPipeline()

    def data_ingestion(**kwargs):
        ti = kwargs["ti"]
        data_ingestion_artifact = training_pipeline.start_data_ingestion()
        ti.xcom_push("data_ingestion_artifact", data_ingestion_artifact)

    def data_validation(**kwargs):
        ti = kwargs["ti"]
        data_ingestion_artifact = ti.xcom_pull(
            task_ids="data_ingestion", key="data_ingestion_artifact"
        )
        data_validation_artifact = training_pipeline.start_data_validation(
            data_ingestion_artifact=data_ingestion_artifact
        )
        ti.xcom_push("data_validation_artifact", data_validation_artifact)

    def data_transformation(**kwargs):
        ti = kwargs["ti"]
        data_validation_artifact = ti.xcom_pull(
            task_ids="data_validation", key="data_validation_artifact"
        )
        data_transformation_artifact = training_pipeline.start_data_transformation(
            data_validation_artifact=data_validation_artifact
        )
        ti.xcom_push("data_transformation_artifact", data_transformation_artifact)

    def model_trainer(**kwargs):
        ti = kwargs["ti"]
        data_transformation_artifact = ti.xcom_pull(
            task_ids="data_transformation", key="data_transformation_artifact"
        )
        model_trainer_artifact = training_pipeline.start_model_trainer(
            data_transformation_artifact=data_transformation_artifact
        )
        ti.xcom_push("model_trainer_artifact", model_trainer_artifact)

    def model_evaluation(**kwargs):
        ti = kwargs["ti"]
        data_validation_artifact = ti.xcom_pull(
            task_ids="data_validation", key="data_validation_artifact"
        )
        model_trainer_artifact = ti.xcom_pull(
            task_ids="model_trainer", key="model_trainer_artifact"
        )
        model_evaluation_artifact = training_pipeline.start_model_evaluation(
            data_validation_artifact=data_validation_artifact,
            model_trainer_artifact=model_trainer_artifact,
        )
        ti.xcom_push("model_evaluation_artifact", model_evaluation_artifact)

    def push_model(**kwargs):
        ti = kwargs["ti"]
        model_evaluation_artifact = ti.xcom_pull(
            task_ids="model_evaluation", key="model_evaluation_artifact"
        )

        if model_evaluation_artifact.is_model_accepted:
            model_pusher_artifact = training_pipeline.start_model_pusher(
                model_eval_artifact=model_evaluation_artifact
            )
            print(f"Model Pusher Artifact: {model_pusher_artifact}")
        else:
            print("Trained Model Rejected.")

        print("Training Pipeline Completed.")

    data_ingestion = PythonOperator(
        task_id="data_ingestion", python_callable=data_ingestion
    )

    data_validation = PythonOperator(
        task_id="data_validation", python_callable=data_validation
    )

    data_transformation = PythonOperator(
        task_id="data_transformation", python_callable=data_transformation
    )

    model_trainer = PythonOperator(
        task_id="model_trainer", python_callable=model_trainer
    )

    model_evaluation = PythonOperator(
        task_id="model_evaluation", python_callable=model_evaluation
    )

    push_model = PythonOperator(task_id="push_model", python_callable=push_model)

    (
        data_ingestion
        >> data_validation
        >> data_transformation
        >> model_trainer
        >> model_evaluation
        >> push_model
    )
