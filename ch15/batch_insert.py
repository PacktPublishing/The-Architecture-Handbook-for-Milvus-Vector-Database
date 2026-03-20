"""
Chapter 15: Batch Insertion Optimization

This example demonstrates how batch insertion dramatically improves
write throughput compared to single-record inserts using Wikipedia embeddings.
"""

import os
import time
from datasets import load_dataset
from pymilvus import MilvusClient, DataType


def get_wikipedia_records(limit=5000):
    """Get Wikipedia records as a list."""
    print(f"Loading {limit} Wikipedia records...")
    docs = load_dataset(
        "Cohere/wikipedia-22-12-simple-embeddings", split="train", streaming=True
    )

    records = []
    for i, doc in enumerate(docs):
        if i >= limit:
            break

        records.append(
            {
                "emb": doc["emb"],
                "title": doc["title"][:100],
                "text": doc["text"][:5000],
                "wiki_id": doc["wiki_id"],
                "views": float(doc["views"]),
            }
        )

    return records


def main():
    # Connect to Milvus
    client = MilvusClient(uri=os.getenv("MILVUS_URI", "http://localhost:19530"))

    collection_name = "wiki_insert_test"
    dim = 768

    # ========================================================================
    # Setup: Create Collection
    # ========================================================================
    print("=" * 70)
    print("Setup: Creating collection")
    print("=" * 70)

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

    client.create_collection(collection_name=collection_name, schema=schema)

    # Create index
    index_params = client.prepare_index_params()
    index_params.add_index(
        field_name="emb",
        index_type="HNSW",
        metric_type="COSINE",
        params={"M": 16, "efConstruction": 200},
    )
    client.create_index(collection_name=collection_name, index_params=index_params)

    # ========================================================================
    # Example 1: Single Record Insert (One at a Time)
    # ========================================================================
    print("\n" + "=" * 70)
    print("Example 1: Single Record Insert (One at a Time)")
    print("=" * 70)

    num_records = 1000
    records = get_wikipedia_records(limit=num_records)

    print(f"Inserting {num_records} records one at a time...")
    start_time = time.time()

    for i, record in enumerate(records):
        client.insert(collection_name=collection_name, data=[record])
        if (i + 1) % 100 == 0:
            print(f"  Inserted {i + 1}/{num_records} records...")

    total_time = time.time() - start_time

    print(f"\nTotal time: {total_time:.2f}s")
    print(f"Average time per record: {total_time / num_records * 1000:.2f}ms")
    print(f"Throughput: {num_records / total_time:.2f} records/sec")

    # Clear the collection for next example
    client.drop_collection(collection_name)
    client.create_collection(collection_name=collection_name, schema=schema)
    index_params = client.prepare_index_params()
    index_params.add_index(
        field_name="emb",
        index_type="HNSW",
        metric_type="COSINE",
        params={"M": 16, "efConstruction": 200},
    )
    client.create_index(collection_name=collection_name, index_params=index_params)

    # ========================================================================
    # Example 2: Batch Insert (All at Once)
    # ========================================================================
    print("\n" + "=" * 70)
    print("Example 2: Batch Insert (All at Once)")
    print("=" * 70)

    records = get_wikipedia_records(limit=num_records)

    print(f"Inserting {num_records} records in a single batch...")
    start_time = time.time()

    client.insert(collection_name=collection_name, data=records)

    total_time = time.time() - start_time

    print(f"\nTotal time: {total_time:.2f}s")
    print(f"Average time per record: {total_time / num_records * 1000:.2f}ms")
    print(f"Throughput: {num_records / total_time:.2f} records/sec")

    # Clear the collection for next example
    client.drop_collection(collection_name)
    client.create_collection(collection_name=collection_name, schema=schema)
    index_params = client.prepare_index_params()
    index_params.add_index(
        field_name="emb",
        index_type="HNSW",
        metric_type="COSINE",
        params={"M": 16, "efConstruction": 200},
    )
    client.create_index(collection_name=collection_name, index_params=index_params)

    # ========================================================================
    # Example 3: Streaming Micro-batch Pattern
    # ========================================================================
    print("\n" + "=" * 70)
    print("Example 3: Streaming Micro-batch Pattern")
    print("=" * 70)

    total_records = 5000
    micro_batch_size = 500

    print(
        f"Inserting {total_records} records in micro-batches of {micro_batch_size}..."
    )

    docs = load_dataset(
        "Cohere/wikipedia-22-12-simple-embeddings", split="train", streaming=True
    )

    batch_buffer = []
    batch_count = 0
    record_count = 0

    start_time = time.time()

    for i, doc in enumerate(docs):
        if i >= total_records:
            break

        record = {
            "emb": doc["emb"],
            "title": doc["title"][:100],
            "text": doc["text"][:5000],
            "wiki_id": doc["wiki_id"],
            "views": float(doc["views"]),
        }
        batch_buffer.append(record)
        record_count += 1

        # Flush batch when full or at end of stream
        if len(batch_buffer) >= micro_batch_size or i == total_records - 1:
            client.insert(collection_name=collection_name, data=batch_buffer)
            batch_count += 1
            print(
                f"  Inserted batch {batch_count}: {len(batch_buffer)} records (total: {record_count})"
            )
            batch_buffer = []

    total_time = time.time() - start_time

    print(f"\nTotal time: {total_time:.2f}s")
    print(f"Number of batches: {batch_count}")
    print(f"Average time per record: {total_time / total_records * 1000:.2f}ms")
    print(f"Throughput: {total_records / total_time:.2f} records/sec")

    # ========================================================================
    # Cleanup
    # ========================================================================
    print("\nCleaning up...")
    client.drop_collection(collection_name)
    print("Done!")


if __name__ == "__main__":
    main()
