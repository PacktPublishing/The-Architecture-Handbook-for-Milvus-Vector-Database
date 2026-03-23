import time
import random
import numpy as np
import logging
from locust import task, between, events, LoadTestShape
from locust.contrib.milvus import MilvusUser
from locust.runners import MasterRunner, LocalRunner
from pymilvus import FieldSchema, DataType, CollectionSchema, MilvusClient
from pymilvus.milvus_client import IndexParams
from faker import Faker

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
fake = Faker()


def setup_collection(environment):
    """Setup e-commerce product collection with larger dataset for soak testing."""
    logger.info("Setting up e-commerce products collection...")
    collection_name = "ecommerce_products"
    client = MilvusClient(uri=environment.host, token="root:Milvus")

    schema = CollectionSchema(
        fields=[
            FieldSchema(
                name="id", dtype=DataType.INT64, is_primary=True, auto_id=False
            ),
            FieldSchema(name="product_id", dtype=DataType.VARCHAR, max_length=100),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=768),
            FieldSchema(name="category", dtype=DataType.VARCHAR, max_length=50),
            FieldSchema(name="price", dtype=DataType.FLOAT),
            FieldSchema(name="rating", dtype=DataType.FLOAT),
            FieldSchema(name="in_stock", dtype=DataType.BOOL),
            FieldSchema(name="brand", dtype=DataType.VARCHAR, max_length=50),
        ],
        description="E-commerce product embeddings for recommendation system",
    )

    index_params = IndexParams()
    index_params.add_index(
        field_name="embedding",
        index_type="HNSW",
        metric_type="COSINE",
        params={"M": 16, "efConstruction": 256},
    )

    if client.has_collection(collection_name):
        client.drop_collection(collection_name)
    client.create_collection(
        collection_name, schema=schema, index_params=index_params, num_shards=3
    )

    categories = ["electronics", "home", "clothing", "books", "sports"]
    brands = ["BrandA", "BrandB", "BrandC", "BrandD", "BrandE"]
    batch_size = 10_000

    for i in range(10):
        logger.info(f"Inserting batch {i + 1}/10 ({batch_size} records)")
        products = [
            {
                "id": int(time.time() * 1000000) + i * batch_size + j,
                "product_id": fake.uuid4()[:8],
                "embedding": np.random.rand(768).astype(np.float32).tolist(),
                "category": random.choice(categories),
                "price": round(random.uniform(10, 1000), 2),
                "rating": round(random.uniform(3.0, 5.0), 1),
                "in_stock": random.choice([True, False]),
                "brand": random.choice(brands),
            }
            for j in range(batch_size)
        ]
        client.insert(collection_name, products)

    logger.info("Collection setup completed successfully")


@events.init.add_listener
def on_init(environment, **_kwargs):
    if isinstance(environment.runner, (MasterRunner, LocalRunner)):
        logger.info("Initializing test environment...")
        setup_collection(environment)


class DQLUser(MilvusUser):
    """DQL User - Search-only workload for soak testing."""

    wait_time = between(0.1, 1.0)

    def __init__(self, environment):
        super().__init__(
            environment=environment,
            uri=environment.host or "http://localhost:19530",
            collection_name="ecommerce_products",
            token="root:Milvus",
            db_name="default",
            timeout=30,
        )
        self.dim = 768

    @task(1)
    def search_products(self):
        """Search for similar products."""
        query_vector = np.random.rand(self.dim).astype(np.float32).tolist()
        search_params = {"metric_type": "COSINE", "params": {"ef": 128}}
        self.search(
            data=[query_vector],
            anns_field="embedding",
            limit=10,
            search_params=search_params,
            output_fields=["product_id", "category", "price"],
        )


class SoakTestLoadShape(LoadTestShape):
    """
    Long running soak test - adjust duration as needed:
    - 24 hours: 60*60*24 (for basic soak testing)
    - 7 days: 60*60*24*7 (for comprehensive testing)
    """

    stages = [
        {"duration": 60 * 60 * 24, "users": 200, "spawn_rate": 10},  # 24 hours
    ]

    def tick(self):
        run_time = self.get_run_time()
        for stage in self.stages:
            if run_time < stage["duration"]:
                return (stage["users"], stage["spawn_rate"])
        return None
