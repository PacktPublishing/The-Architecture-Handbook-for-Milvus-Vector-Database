"""
Chapter 14: Generate synthetic test data and import into Milvus.

Creates a collection with 1M 128-dimensional vectors, scalar fields,
and builds an HNSW index for scalability testing.

Usage:
    python generate_test_data.py [--uri localhost:19530] [--dim 128] [--count 1000000]
"""

import argparse
import time

import numpy as np
from pymilvus import MilvusClient, DataType

COLLECTION_NAME = "test_collection"
BATCH_SIZE = 50000


def create_collection(client: MilvusClient, dim: int) -> None:
    if client.has_collection(COLLECTION_NAME):
        print(f"Collection '{COLLECTION_NAME}' already exists, dropping...")
        client.drop_collection(COLLECTION_NAME)

    schema = client.create_schema(auto_id=True, enable_dynamic_field=False)
    schema.add_field("id", DataType.INT64, is_primary=True)
    schema.add_field("vector", DataType.FLOAT_VECTOR, dim=dim)
    schema.add_field("category", DataType.INT64)
    schema.add_field("timestamp", DataType.INT64)

    index_params = client.prepare_index_params()
    index_params.add_index(
        field_name="vector",
        index_type="HNSW",
        metric_type="L2",
        params={"M": 16, "efConstruction": 200},
    )

    client.create_collection(
        collection_name=COLLECTION_NAME,
        schema=schema,
        index_params=index_params,
    )
    print(f"Collection '{COLLECTION_NAME}' created with HNSW index (dim={dim})")


def insert_data(client: MilvusClient, dim: int, total: int) -> None:
    inserted = 0
    start = time.time()

    while inserted < total:
        batch = min(BATCH_SIZE, total - inserted)

        vectors = np.random.randn(batch, dim).astype("float32")
        categories = np.random.randint(0, 100, batch).tolist()
        timestamps = np.random.randint(1600000000, 1700000000, batch).tolist()

        client.insert(
            collection_name=COLLECTION_NAME,
            data=[
                {
                    "vector": vectors[i].tolist(),
                    "category": categories[i],
                    "timestamp": timestamps[i],
                }
                for i in range(batch)
            ],
        )
        inserted += batch
        elapsed = time.time() - start
        rate = inserted / elapsed
        print(f"  Inserted {inserted}/{total} ({rate:.0f} vectors/sec)")

    total_time = time.time() - start
    print(
        f"\nInsert complete: {total} vectors in {total_time:.1f}s ({total / total_time:.0f} vectors/sec)"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Generate test data for Milvus scalability testing"
    )
    parser.add_argument(
        "--uri", default="http://localhost:19530", help="Milvus server URI"
    )
    parser.add_argument("--token", default="", help="Milvus authentication token")
    parser.add_argument("--dim", type=int, default=128, help="Vector dimension")
    parser.add_argument(
        "--count", type=int, default=1_000_000, help="Number of vectors to generate"
    )
    args = parser.parse_args()

    client = MilvusClient(uri=args.uri, token=args.token)
    create_collection(client, args.dim)
    insert_data(client, args.dim, args.count)

    client.load_collection(COLLECTION_NAME)
    print(f"Collection '{COLLECTION_NAME}' loaded into memory")


if __name__ == "__main__":
    main()
