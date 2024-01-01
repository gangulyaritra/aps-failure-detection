from dotenv import dotenv_values
from datetime import datetime

config = dotenv_values(".env")
TIMESTAMP = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")

MONGODB_URL_KEY = config["MONGODB_URL_KEY"]
DATABASE_NAME = config["DATABASE_NAME"]
COLLECTION_NAME = config["COLLECTION_NAME"]

AWS_ACCESS_KEY_ID_ENV_KEY = config["AWS_ACCESS_KEY_ID"]
AWS_SECRET_ACCESS_KEY_ENV_KEY = config["AWS_SECRET_ACCESS_KEY"]
REGION_NAME = "ap-south-1"

TRAINING_BUCKET_NAME = "scania-sensor-train"
PREDICTION_BUCKET_NAME = "scania-sensor-inference"
