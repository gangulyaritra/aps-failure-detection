import os
from src.constant import MONGODB_URL_KEY
from src.constant.training import DATA_INGESTION_COLLECTION_NAME
from src.data_access.sensor_data import SensorData


if __name__ == "__main__":
    """
    Upload Dataset as MongoDB Collections.
    """
    data_file_path = "aps-failure-at-scania-trucks.csv"
    os.environ["MONGO_DB_URL"] = MONGODB_URL_KEY
    print(os.environ["MONGO_DB_URL"])

    sd = SensorData()
    if (
        DATA_INGESTION_COLLECTION_NAME
        in sd.mongo_client.database.list_collection_names()
    ):
        sd.mongo_client.database[DATA_INGESTION_COLLECTION_NAME].drop()

    sd.save_csv_file(
        file_path=data_file_path, collection_name=DATA_INGESTION_COLLECTION_NAME
    )
    print("Dataset Uploaded to MongoDB Successful.")
