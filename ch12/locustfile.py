from locust import between, task
from locust.contrib.milvus import MilvusUser

import random

from pymilvus import CollectionSchema, DataType, FieldSchema
from pymilvus.milvus_client import IndexParams


class SimpleMilvusUser(MilvusUser):

    wait_time = between(1, 3)

    def on_start(self):
        """Generate test vectors."""
        self.dimension = 128
        self.test_vectors = [[random.random() for _ in range(self.dimension)] for _ in range(10)]

    def __init__(self, environment):
        # Define collection schema
        schema = CollectionSchema(
            fields=[
                FieldSchema(name="id", dtype=DataType.INT64, is_primary=True),
                FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=128),
                FieldSchema(name="name", dtype=DataType.VARCHAR, max_length=50),
            ],
            description="Test collection",
        )

        # Define index parameters
        index_params = IndexParams()
        index_params.add_index(
            field_name="vector",
            index_type="HNSW",
            metric_type="COSINE",
        )

        super().__init__(
            environment,
            uri=environment.host,
            collection_name="perf_test_collection",
            schema=schema,
            index_params=index_params,
        )

    @task(3)
    def insert_data(self):
        """Insert data into Milvus."""
        data = [
            {
                "id": random.randint(1, 10000),
                "vector": random.choice(self.test_vectors),
                "name": f"item_{random.randint(1, 1000)}",
            }
        ]
        self.insert(data)

    @task(5)
    def search_vectors(self):
        """Search for similar vectors."""
        search_vector = random.choice(self.test_vectors)
        self.search(data=[search_vector], anns_field="vector", limit=5)
