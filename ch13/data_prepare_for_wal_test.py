import random
import numpy as np
import logging
import pandas as pd
from pymilvus import FieldSchema, DataType, CollectionSchema, MilvusClient
from pymilvus.milvus_client import IndexParams
from faker import Faker

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
fake = Faker()

MILVUS_URI = "http://localhost:19530"
COLLECTION_NAME = "ecommerce_products"

logger.info("Setting up e-commerce products collection...")
client = MilvusClient(uri=MILVUS_URI, token="root:Milvus")

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

if client.has_collection(COLLECTION_NAME):
    client.drop_collection(COLLECTION_NAME)

client.create_collection(
    COLLECTION_NAME, schema=schema, index_params=index_params, num_shards=3
)

# Initialize DataFrame to track data changes
data_df = pd.DataFrame()
categories = ["electronics", "home", "clothing", "books", "sports"]
brands = ["BrandA", "BrandB", "BrandC", "BrandD", "BrandE"]

# Step 1: Insert 100,000 records in batches
batch_size = 10_000
for i in range(10):
    logger.info(f"Inserting batch {i + 1}/10 ({batch_size} records)")
    products = [
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
    client.insert(COLLECTION_NAME, products)
    batch_df = pd.DataFrame(products)
    data_df = pd.concat([data_df, batch_df], ignore_index=True)

logger.info("Insert completed")

# Step 2: Delete first 10,000 records
logger.info("Deleting records with id 0-9999...")
client.delete(COLLECTION_NAME, filter="id >= 0 and id < 10000")
data_df = data_df[~((data_df["id"] >= 0) & (data_df["id"] < 10000))]
logger.info("Delete completed")

# Step 3: Upsert records 10,000-19,999 with new values
logger.info("Upserting records with id 10000-19999...")
updated_products = [
    {
        "id": idx,
        "product_id": fake.uuid4()[:8],
        "embedding": np.random.rand(768).astype(np.float32).tolist(),
        "category": random.choice(categories),
        "price": round(random.uniform(10, 1000), 2),
        "rating": round(random.uniform(3.0, 5.0), 1),
        "in_stock": random.choice([True, False]),
        "brand": random.choice(brands),
    }
    for idx in range(10000, 20000)
]
client.upsert(COLLECTION_NAME, updated_products)

update_df = pd.DataFrame(updated_products)
data_df = data_df[~data_df["id"].isin(update_df["id"])].copy()
data_df = pd.concat([data_df, update_df], ignore_index=True)
logger.info("Upsert completed")

# Save ground truth to parquet
parquet_filename = "milvus_wal_test_data.parquet"
data_df.to_parquet(parquet_filename, index=False)
logger.info(f"Ground truth saved to {parquet_filename} ({len(data_df)} records)")
