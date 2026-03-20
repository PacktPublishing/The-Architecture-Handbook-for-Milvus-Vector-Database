"""
Chapter 15: Batch Search Optimization

This example demonstrates how batch processing dramatically improves
search throughput by amortizing overhead costs using real Wikipedia embeddings.
"""

import os
import time
from datasets import load_dataset
from pymilvus import MilvusClient, DataType


def load_wikipedia_data(limit=50000, batch_size=1000):
    """
    Load Wikipedia embeddings from Hugging Face dataset.

    Args:
        limit: Number of records to load
        batch_size: Records per batch

    Yields:
        Batch of records formatted for Milvus
    """
    print(f"Loading Wikipedia dataset (total_records={limit})...")
    docs = load_dataset("Cohere/wikipedia-22-12-simple-embeddings", split="train")

    # Use slice to get data efficiently
    for start_idx in range(0, limit, batch_size):
        end_idx = min(start_idx + batch_size, limit)
        batch_docs = docs[start_idx:end_idx]

        # Format batch for Milvus
        batch = []
        for i in range(len(batch_docs["emb"])):
            batch.append(
                {
                    "emb": batch_docs["emb"][i],
                    "title": batch_docs["title"][i][:500],  # Truncate to max length
                    "text": batch_docs["text"][i][:2000],  # Truncate to max length
                    "wiki_id": batch_docs["wiki_id"][i],
                    "views": float(batch_docs["views"][i]),
                }
            )

        yield batch


def get_search_vectors(num_searches=100):
    """Get real search vectors from Wikipedia dataset."""
    print(f"Loading {num_searches} search vectors from dataset...")
    docs = load_dataset("Cohere/wikipedia-22-12-simple-embeddings", split="train")

    search_vectors = []

    for i, doc in enumerate(docs):
        if i >= num_searches:
            break
        search_vectors.append(doc["emb"])

    return search_vectors


def main():
    # Connect to Milvus
    client = MilvusClient(uri=os.getenv("MILVUS_URI", "http://localhost:19530"))

    collection_name = "wiki_articles"
    num_vectors = 50000
    dim = 768

    # ========================================================================
    # Setup: Create Collection and Insert Data
    # ========================================================================
    print("=" * 70)
    print("Setup: Creating collection and inserting data")
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

    # Create HNSW index
    index_params = client.prepare_index_params()
    index_params.add_index(
        field_name="emb",
        index_type="HNSW",
        metric_type="COSINE",
        params={"M": 16, "efConstruction": 200},
    )
    client.create_index(collection_name=collection_name, index_params=index_params)

    # Insert data in batches
    batch_num = 0
    total_inserted = 0
    for batch in load_wikipedia_data(limit=num_vectors, batch_size=1000):
        client.insert(collection_name=collection_name, data=batch)
        batch_num += 1
        total_inserted += len(batch)
        if batch_num % 10 == 0:
            print(f"Inserted {total_inserted} records...")

    print(f"Total inserted: {total_inserted} records")

    # Load collection
    client.load_collection(collection_name)

    search_params = {"metric_type": "COSINE", "params": {"ef": 128}}

    # ========================================================================
    # Example 1: Single Search
    # ========================================================================
    print("\n" + "=" * 70)
    print("Example 1: Single Search (Sequential)")
    print("=" * 70)

    search_vectors = get_search_vectors(num_searches=100)
    print("Running 100 sequential single searches...")

    start_time = time.time()
    for i, search_vector in enumerate(search_vectors):
        client.search(
            collection_name=collection_name,
            data=[search_vector],
            anns_field="emb",
            search_params=search_params,
            limit=10,
        )
    total_time = time.time() - start_time

    print(f"Total time: {total_time:.2f}s")
    print(f"Average time per search: {total_time / len(search_vectors) * 1000:.2f}ms")
    print(f"Throughput: {len(search_vectors) / total_time:.2f} searches/sec")

    # ========================================================================
    # Example 2: Batch Search
    # ========================================================================
    print("\n" + "=" * 70)
    print("Example 2: Batch Search (All at Once)")
    print("=" * 70)

    print("Running single batch search with 100 searches...")

    start_time = time.time()
    results = client.search(
        collection_name=collection_name,
        data=search_vectors,
        anns_field="emb",
        search_params=search_params,
        limit=10,
    )
    total_time = time.time() - start_time

    print(f"Total time: {total_time:.2f}s")
    print(f"Average time per search: {total_time / len(search_vectors) * 1000:.2f}ms")
    print(f"Throughput: {len(search_vectors) / total_time:.2f} searches/sec")

    print("\nSample results from first search:")
    for i, hit in enumerate(results[0][:5]):
        print(f"  {i + 1}. ID: {hit['id']}, Distance: {hit['distance']:.4f}")

    # ========================================================================
    # Cleanup
    # ========================================================================
    print("\nCleaning up...")
    client.release_collection(collection_name)
    client.drop_collection(collection_name)
    print("Done!")


if __name__ == "__main__":
    main()
