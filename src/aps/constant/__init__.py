import os
from datetime import datetime

from dopplersdk import DopplerSDK

# Initialize and authenticate the SDK.
doppler = DopplerSDK()
doppler.set_access_token(os.getenv("DOPPLER_SERVICE_TOKEN"))

secrets = doppler.secrets.list(project="aps", config="prd").secrets

TIMESTAMP = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")

MONGODB_URL_KEY = secrets.get("MONGODB_URL_KEY", {}).get("computed")
DATABASE_NAME = secrets.get("DATABASE_NAME", {}).get("computed")
COLLECTION_NAME = secrets.get("COLLECTION_NAME", {}).get("computed")

AWS_ACCESS_KEY_ID_ENV_KEY = secrets.get("AWS_ACCESS_KEY_ID", {}).get("computed")
AWS_SECRET_ACCESS_KEY_ENV_KEY = secrets.get("AWS_SECRET_ACCESS_KEY", {}).get("computed")
REGION_NAME = secrets.get("AWS_REGION_NAME", {}).get("computed")

TRAINING_BUCKET_NAME = secrets.get("TRAINING_BUCKET_NAME", {}).get("computed")
PREDICTION_BUCKET_NAME = secrets.get("PREDICTION_BUCKET_NAME", {}).get("computed")
