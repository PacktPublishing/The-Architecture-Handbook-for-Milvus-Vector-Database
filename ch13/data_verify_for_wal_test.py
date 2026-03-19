import logging
import numpy as np
import pandas as pd
from pymilvus import MilvusClient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MILVUS_URI = "http://localhost:19530"
COLLECTION_NAME = "ecommerce_products"

client = MilvusClient(uri=MILVUS_URI, token="root:Milvus")

# Load ground truth
df = pd.read_parquet("milvus_wal_test_data.parquet")

# Verify record count
res = client.query(COLLECTION_NAME, filter="", output_fields=["count(*)"])
actual_count = res[0]["count(*)"]
expected_count = len(df)
assert actual_count == expected_count, (
    f"Record count mismatch: expected {expected_count}, got {actual_count}"
)
logger.info(f"Record count verified: {actual_count}")

# Verify data integrity by sampling
for idx in range(10000, 20000, 1000):
    id_list = list(range(idx, idx + 1000))
    res = client.query(
        COLLECTION_NAME,
        filter=f"id in {id_list}",
        output_fields=[
            "id", "product_id", "embedding", "category",
            "price", "rating", "in_stock", "brand",
        ],
    )
    for r in res:
        product = df.loc[df["id"] == r["id"]].iloc[0]

        assert r["product_id"] == product["product_id"]
        assert r["category"] == product["category"]
        assert round(r["price"], 2) == round(product["price"], 2)
        assert round(r["rating"], 1) == round(product["rating"], 1)
        assert r["in_stock"] == product["in_stock"]
        assert r["brand"] == product["brand"]

        cosine_similarity = np.dot(r["embedding"], product["embedding"]) / (
            np.linalg.norm(r["embedding"]) * np.linalg.norm(product["embedding"])
        )
        assert cosine_similarity > 0.9, (
            f"Cosine similarity {cosine_similarity:.3f} too low for id {r['id']}"
        )

    logger.info(f"Verified batch {idx}-{idx + 999} ({len(res)} records)")

logger.info("WAL durability test passed!")
