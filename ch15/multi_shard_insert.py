"""
Chapter 15: Multi-Shard Parallelization

This example demonstrates how multiple shards enable parallel
ingestion processing for higher write throughput using Wikipedia embeddings.
"""

import os
import time
from datasets import load_dataset
from pymilvus import MilvusClient, DataType


def load_wikipedia_data(limit=50000, batch_size=5000):
    """Load Wikipedia embeddings from Hugging Face dataset."""
    print(f"Loading Wikipedia dataset (limit={limit})...")
    docs = load_dataset(
        "Cohere/wikipedia-22-12-simple-embeddings", split="train", streaming=True
    )

    batch = []
    count = 0

    for doc in docs:
        if count >= limit:
            break

        batch.append(
            {
                "emb": doc["emb"],
                "title": doc["title"][:100],
                "text": doc["text"][:5000],
                "wiki_id": doc["wiki_id"],
                "views": float(doc["views"]),
            }
        )

        count += 1

        if len(batch) >= batch_size:
            yield batch
            batch = []

    if batch:
        yield batch


def create_collection_with_shards(client, collection_name, num_shards):
    """Create a collection with specified number of shards."""
    dim = 768

    if client.has_collection(collection_name):
        client.drop_collection(collection_name)

    # Create collection
    schema = client.create_schema(auto_id=True, enable_dynamic_field=False)
    schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True)
    schema.add_field(field_name="emb", datatype=DataType.FLOAT_VECTOR, dim=dim)
    schema.add_field(field_name="title", datatype=DataType.VARCHAR, max_length=100)
    schema.add_field(field_name="text", datatype=DataType.VARCHAR, max_length=5000)
    schema.add_field(field_name="wiki_id", datatype=DataType.INT64)
    schema.add_field(field_name="views", datatype=DataType.FLOAT)

    client.create_collection(
        collection_name=collection_name, schema=schema, num_shards=num_shards
    )

    # Create index
    index_params = client.prepare_index_params()
    index_params.add_index(
        field_name="emb",
        index_type="HNSW",
        metric_type="COSINE",
        params={"M": 16, "efConstruction": 200},
    )
    client.create_index(collection_name=collection_name, index_params=index_params)

    print(f"Created collection '{collection_name}' with {num_shards} shard(s)")


def insert_wikipedia_data(client, collection_name, num_vectors=50000, batch_size=5000):
    """Insert Wikipedia data into collection."""
    batch_num = 0
    total_inserted = 0

    for batch in load_wikipedia_data(limit=num_vectors, batch_size=batch_size):
        client.insert(collection_name=collection_name, data=batch)
        batch_num += 1
        total_inserted += len(batch)
        print(
            f"  Inserted batch {batch_num}: {len(batch)} records (total: {total_inserted})"
        )

    print(f"Total inserted: {total_inserted} records")


def main():
    # Connect to Milvus
    client = MilvusClient(uri=os.getenv("MILVUS_URI", "http://localhost:19530"))

    num_vectors = 50000
    batch_size = 5000

    # ========================================================================
    # Example 1: Single Shard Collection
    # ========================================================================
    print("\n" + "=" * 70)
    print("Example 1: Single Shard Collection")
    print("=" * 70)

    collection_name = "wiki_single_shard"
    create_collection_with_shards(client, collection_name, num_shards=1)

    start_time = time.time()
    insert_wikipedia_data(client, collection_name, num_vectors, batch_size)
    insert_time = time.time() - start_time

    print(f"\nInsert time (1 shard): {insert_time:.2f}s")
    print(f"Throughput: {num_vectors / insert_time:.2f} records/sec")

    # ========================================================================
    # Example 2: Multi-Shard Collection (2 Shards)
    # ========================================================================
    print("\n" + "=" * 70)
    print("Example 2: Multi-Shard Collection (2 Shards)")
    print("=" * 70)

    collection_name = "wiki_2_shards"
    create_collection_with_shards(client, collection_name, num_shards=2)

    start_time = time.time()
    insert_wikipedia_data(client, collection_name, num_vectors, batch_size)
    insert_time = time.time() - start_time

    print(f"\nInsert time (2 shards): {insert_time:.2f}s")
    print(f"Throughput: {num_vectors / insert_time:.2f} records/sec")

    # ========================================================================
    # Example 3: Multi-Shard Collection (4 Shards)
    # ========================================================================
    print("\n" + "=" * 70)
    print("Example 3: Multi-Shard Collection (4 Shards)")
    print("=" * 70)

    collection_name = "wiki_4_shards"
    create_collection_with_shards(client, collection_name, num_shards=4)

    start_time = time.time()
    insert_wikipedia_data(client, collection_name, num_vectors, batch_size)
    insert_time = time.time() - start_time

    print(f"\nInsert time (4 shards): {insert_time:.2f}s")
    print(f"Throughput: {num_vectors / insert_time:.2f} records/sec")

    # ========================================================================
    # Cleanup
    # ========================================================================
    print("\nCleaning up...")
    client.drop_collection("wiki_single_shard")
    client.drop_collection("wiki_2_shards")
    client.drop_collection("wiki_4_shards")
    print("Done!")


if __name__ == "__main__":
    main()
