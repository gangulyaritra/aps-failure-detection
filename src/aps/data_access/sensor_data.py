import json
from typing import Optional

import numpy as np
import pandas as pd

from aps.configuration.mongodb_connection import MongoDBClient
from aps.constant import DATABASE_NAME


class SensorData:
    """
    Provides methods to import CSV data into MongoDB and export MongoDB records as a DataFrame.
    """

    def __init__(self):
        self.mongo_client = MongoDBClient(database_name=DATABASE_NAME)

    def _get_collection(
        self, collection_name: str, database_name: Optional[str] = None
    ):
        """
        Helper method to retrieve a specific MongoDB collection.

        :param collection_name: Specify the MongoDB collection to access.
        :param database_name: An optional parameter to specify an alternative database.
        """
        return (
            self.mongo_client.database[collection_name]
            if database_name is None
            else self.mongo_client[database_name][collection_name]
        )

    def save_csv_file(
        self,
        file_path: str,
        collection_name: str,
        database_name: Optional[str] = None,
        batch_size: int = 5000,
    ) -> int:
        """
        Imports the CSV file into the MongoDB collection in batches.

        :param file_path: The file path to the CSV file.
        :param collection_name: Specify the MongoDB collection to import the data.
        :param database_name: An optional parameter to specify an alternative database.
        :param batch_size: The number of records to insert per batch (default is 5000).
        """
        df = pd.read_csv(file_path)
        df.reset_index(drop=True, inplace=True)
        records = list(json.loads(df.T.to_json()).values())

        collection = self._get_collection(collection_name, database_name)

        for i in range(0, len(records), batch_size):
            batch = records[i : i + batch_size]
            collection.insert_many(batch)

        return len(records)

    def export_collection_as_dataframe(
        self, collection_name: str, database_name: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Exports the entire MongoDB records as a pandas DataFrame.

        :param collection_name: Specify the MongoDB collection to export the data.
        :param database_name: An optional parameter to specify an alternative database.
        """
        collection = self._get_collection(collection_name, database_name)
        df = pd.DataFrame(list(collection.find()))

        if "_id" in df.columns:
            df.drop(columns=["_id"], inplace=True)

        df.replace({"na": np.nan}, inplace=True)
        return df
