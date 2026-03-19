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
    """Setup e-commerce product collection with initial data."""
    logger.info("Setting up e-commerce products collection...")
    collection_name = "ecommerce_products"
    client = MilvusClient(uri=environment.host, token="root:Milvus")

    schema = CollectionSchema(
        fields=[
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=False),
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
    client.create_collection(collection_name, schema=schema, index_params=index_params)

    total_data = 10_000
    batch_size = 1000
    categories = ["electronics", "home", "clothing", "books", "sports"]
    brands = ["BrandA", "BrandB", "BrandC", "BrandD", "BrandE"]

    for i in range(total_data // batch_size):
        data = [
            {
                "id": i * batch_size + j,
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
        client.insert(collection_name, data)

    logger.info("Collection setup completed successfully")


@events.init.add_listener
def on_init(environment, **_kwargs):
    if isinstance(environment.runner, (MasterRunner, LocalRunner)):
        logger.info("Initializing test environment...")
        setup_collection(environment)


class EcommerceUser(MilvusUser):
    """E-commerce user simulating realistic product recommendation workload.

    Operation mix: 80% searches, 10% inserts, 5% upserts, 5% deletes
    Task weights control frequency: @task(80) = 8x more likely than @task(10)
    """

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
        self.categories = ["electronics", "home", "clothing", "books", "sports"]
        self.brands = ["BrandA", "BrandB", "BrandC", "BrandD", "BrandE"]
        self.existing_ids = []

    def _generate_product_data(self, num_products=1):
        """Generate fake product data."""
        products = []
        for _ in range(num_products):
            product = {
                "id": int(time.time() * 1000000),
                "product_id": fake.uuid4()[:8],
                "embedding": np.random.rand(self.dim).astype(np.float32).tolist(),
                "category": random.choice(self.categories),
                "price": round(random.uniform(10, 1000), 2),
                "rating": round(random.uniform(3.0, 5.0), 1),
                "in_stock": random.choice([True, False]),
                "brand": random.choice(self.brands),
            }
            if len(self.existing_ids) < 1000:
                self.existing_ids.append(product["id"])
            products.append(product)
        return products

    @task(80)
    def search_similar_products(self):
        """Search for similar products with varying complexity."""
        query_vector = np.random.rand(self.dim).astype(np.float32).tolist()
        search_type = random.choices(
            ["simple", "filtered", "complex"],
            weights=[50, 30, 20],
        )[0]

        search_params = {"metric_type": "COSINE", "params": {"ef": 128}}

        if search_type == "simple":
            # Basic similarity search - most common
            self.search(
                data=[query_vector],
                anns_field="embedding",
                limit=10,
                search_params=search_params,
                output_fields=["product_id", "category", "price"],
            )
        elif search_type == "filtered":
            # Price range filter - common in e-commerce
            price_min = random.randint(50, 200)
            price_max = price_min + random.randint(100, 300)
            self.search(
                data=[query_vector],
                anns_field="embedding",
                limit=10,
                filter=f"price >= {price_min} and price <= {price_max}",
                search_params=search_params,
                output_fields=["product_id", "price", "rating"],
            )
        else:
            # Multi-attribute filter - high complexity
            categories = random.sample(self.categories, 2)
            category_str = str(categories).replace("'", '"')
            self.search(
                data=[query_vector],
                anns_field="embedding",
                limit=20,
                filter=f"category in {category_str} and price < 500 and rating > 4.0 and in_stock == true",
                search_params=search_params,
                output_fields=["product_id", "category", "price", "rating", "brand"],
            )

    @task(10)
    def batch_insert_products(self):
        """Simulate batch product catalog updates."""
        batch_size = random.choice([10, 50, 100])
        products = self._generate_product_data(batch_size)
        self.insert(products)

    @task(5)
    def upsert_products(self):
        """Update existing products."""
        batch_size = random.choice([5, 10, 20])
        if not self.existing_ids:
            return
        update_ids = random.sample(self.existing_ids, min(batch_size, len(self.existing_ids)))
        products = [
            {
                "id": uid,
                "product_id": fake.uuid4()[:8],
                "embedding": np.random.rand(self.dim).astype(np.float32).tolist(),
                "category": random.choice(self.categories),
                "price": round(random.uniform(10, 1000), 2),
                "rating": round(random.uniform(3.0, 5.0), 1),
                "in_stock": random.choice([True, False]),
                "brand": random.choice(self.brands),
            }
            for uid in update_ids
        ]
        self.upsert(products)

    @task(5)
    def delete_out_of_stock(self):
        """Delete out of stock products with low ratings."""
        self.delete(filter="in_stock == false and rating < 3.5")


class StagesLoadShape(LoadTestShape):
    """Custom load shape simulating different traffic patterns."""

    stages = [
        # Phase 1: Gradual ramp-up (0-3 min)
        {"duration": 30, "users": 10, "spawn_rate": 2},
        {"duration": 60, "users": 20, "spawn_rate": 2},
        {"duration": 90, "users": 30, "spawn_rate": 2},
        {"duration": 120, "users": 50, "spawn_rate": 5},
        {"duration": 180, "users": 100, "spawn_rate": 5},
        # Phase 2: Sustained load (3-10 min)
        {"duration": 600, "users": 200, "spawn_rate": 10},
        # Phase 3: Burst traffic simulation (10-12 min)
        {"duration": 630, "users": 500, "spawn_rate": 50},
        {"duration": 720, "users": 500, "spawn_rate": 0},
        # Phase 4: Cool down (12-15 min)
        {"duration": 780, "users": 200, "spawn_rate": 20},
        {"duration": 840, "users": 100, "spawn_rate": 10},
        {"duration": 900, "users": 10, "spawn_rate": 5},
    ]

    def tick(self):
        run_time = self.get_run_time()
        for stage in self.stages:
            if run_time < stage["duration"]:
                return (stage["users"], stage["spawn_rate"])
        return None
