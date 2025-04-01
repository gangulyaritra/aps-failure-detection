import io
import logging
import os
import shutil
import zipfile
from pathlib import Path

import pandas as pd
import requests

from aps.constant import MONGODB_URL_KEY
from aps.constant.training import DATA_INGESTION_COLLECTION_NAME, RAW_CSV_FILE_NAME, URL
from aps.data_access.sensor_data import SensorData

# Set the MongoDB URL environment variable.
os.environ["MONGO_DB_URL"] = MONGODB_URL_KEY


class ScaniaETL:

    # Configure logging.
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Creates a logger.
    logger = logging.getLogger("ScaniaETL")

    def __init__(self) -> None:
        """
        Initialize paths for download and file extraction.
        """
        self.download_path: Path = Path(os.getcwd()) / "scania_data"
        self.file_path: Path = self.download_path / RAW_CSV_FILE_NAME

    @staticmethod
    def downsample_dataframe(df: pd.DataFrame, ratio: float) -> pd.DataFrame:
        """
        Downsamples the negative class, returning a new DataFrame with all positives and a fraction of negatives.

        :param df: DataFrame with a 'class' column containing 'neg' and 'pos' values.
        :param ratio: Fraction of negative samples to keep.
        :return pd.DataFrame: Returns a new DataFrame with positives intact and downsampled negatives, shuffled and reset.
        """
        return (
            pd.concat(
                [
                    df.loc[df["class"] == "pos"],
                    df.loc[df["class"] == "neg"].sample(frac=ratio, random_state=42),
                ]
            )
            .sample(frac=1, random_state=42)
            .reset_index(drop=True)
        )

    def extract(self) -> None:
        """
        Download and extract the .zip file from the URL.
        """
        self.logger.info("Initiating the Data Extraction Method.")
        try:
            response = requests.get(URL, timeout=15)
            response.raise_for_status()
        except requests.RequestException as err:
            self.logger.error("Failed to download the dataset: %s", err)
            raise

        self.download_path.mkdir(parents=True, exist_ok=True)

        try:
            with zipfile.ZipFile(io.BytesIO(response.content)) as zip_ref:
                zip_ref.extractall(self.download_path)
        except zipfile.BadZipFile as err:
            self.logger.error("Error extracting zip file: %s", err)
            raise

        self.logger.info("Data Extraction Completed Successfully.")

    def transform(self) -> None:
        """
        Transform the raw CSV file.
        """
        self.logger.info("Initiating the Data Transformation Method.")
        try:
            df = pd.read_csv(self.file_path, header=14)
            df = self.downsample_dataframe(df, 0.3)
            df.to_csv(self.file_path, index=False)
        except Exception as err:
            self.logger.error("Transformation error: %s", err)
            raise

        self.logger.info("Data Transformation Completed Successfully.")

    def load(self) -> None:
        """
        Load the transformed CSV file into MongoDB.
        """
        self.logger.info("Initiating the Data Loading Method.")
        sd = SensorData()
        db = sd.mongo_client.database

        # Drop the collection if it already exists.
        if DATA_INGESTION_COLLECTION_NAME in db.list_collection_names():
            db[DATA_INGESTION_COLLECTION_NAME].drop()

        sd.save_csv_file(
            file_path=self.file_path, collection_name=DATA_INGESTION_COLLECTION_NAME
        )
        self.logger.info("Dataset Successfully Uploaded to MongoDB.")

    def etl(self):
        try:
            self.extract()
        except Exception as err:
            raise RuntimeError(
                f"Scraper failed at Extraction. Error was {err}"
            ) from err
        try:
            self.transform()
        except Exception as err:
            raise RuntimeError(
                f"Scraper failed at Transformation. Error was {err}"
            ) from err
        try:
            self.load()
        except Exception as err:
            raise RuntimeError(f"Scraper failed at Upload. Error was {err}") from err
        finally:
            if self.download_path.exists() and self.download_path.is_dir():
                shutil.rmtree(self.download_path)


def main() -> None:
    etl_process = ScaniaETL()
    etl_process.etl()


if __name__ == "__main__":
    main()
