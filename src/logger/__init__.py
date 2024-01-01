import os
import logging
from src.constant import TIMESTAMP

LOG_FILE = f"{TIMESTAMP}.log"

logs_path = os.path.join(os.getcwd(), "logs", TIMESTAMP)

os.makedirs(logs_path, exist_ok=True)

logging.basicConfig(
    filename=os.path.join(logs_path, LOG_FILE),
    format="[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
