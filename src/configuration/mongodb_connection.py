import pymongo
from src.constant import MONGODB_URL_KEY, DATABASE_NAME


class MongoDBClient:
    client = None

    def __init__(self, database_name=DATABASE_NAME) -> None:
        if MongoDBClient.client is None:
            MongoDBClient.client = pymongo.MongoClient(MONGODB_URL_KEY)

        self.client = MongoDBClient.client
        self.database = self.client[database_name]
        self.database_name = database_name
