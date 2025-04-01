import logging
import os

from aps.constant import TIMESTAMP

logs_path = os.path.join(os.getcwd(), "logs", TIMESTAMP)

os.makedirs(logs_path, exist_ok=True)

logging.basicConfig(
    filename=os.path.join(logs_path, f"{TIMESTAMP}.log"),
    format="[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
