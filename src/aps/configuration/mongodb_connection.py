import pymongo

from aps.constant import DATABASE_NAME, MONGODB_URL_KEY


class MongoDBClient:
    client = None

    def __init__(self, database_name=DATABASE_NAME) -> None:
        if MongoDBClient.client is None:
            MongoDBClient.client = pymongo.MongoClient(MONGODB_URL_KEY)

        self.client = MongoDBClient.client
        self.database = self.client[database_name]
        self.database_name = database_name
