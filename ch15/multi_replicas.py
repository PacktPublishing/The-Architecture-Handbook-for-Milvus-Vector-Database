"""
Chapter 15: Multi-Replica Concurrency Optimization

This example demonstrates how multiple replicas improve search throughput
under concurrent load by distributing search requests across replicas.
"""

import os
import time
from multiprocessing import Pool
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
                    "title": batch_docs["title"][i][:100],
                    "text": batch_docs["text"][i][:5000],
                    "wiki_id": batch_docs["wiki_id"][i],
                    "views": float(batch_docs["views"][i]),
                }
            )

        yield batch


def create_collection_with_index(client, collection_name, index_type, index_params):
    """Create a collection with specified index type."""
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

    client.create_collection(collection_name=collection_name, schema=schema)

    # Create index
    index_params_obj = client.prepare_index_params()
    index_params_obj.add_index(
        field_name="emb",
        index_type=index_type,
        metric_type="COSINE",
        params=index_params,
    )

    client.create_index(collection_name=collection_name, index_params=index_params_obj)

    print(f"Created collection '{collection_name}' with {index_type} index")


def insert_wikipedia_data(client, collection_name, num_vectors=50000):
    """Insert Wikipedia data into collection."""
    batch_num = 0
    total_inserted = 0

    for batch in load_wikipedia_data(limit=num_vectors, batch_size=1000):
        client.insert(collection_name=collection_name, data=batch)
        batch_num += 1
        total_inserted += len(batch)
        if batch_num % 10 == 0:
            print(f"Inserted {total_inserted} records...")

    print(f"Total inserted: {total_inserted} records")


def search_worker(args):
    """
    Worker function for concurrent search.
    Each worker performs a single search operation.
    """
    collection_name, search_vector, milvus_uri = args

    # Each worker creates its own client connection
    client = MilvusClient(uri=milvus_uri)

    search_params = {"metric_type": "COSINE", "params": {"ef": 128}}

    start_time = time.time()
    client.search(
        collection_name=collection_name,
        data=[search_vector],
        anns_field="emb",
        search_params=search_params,
        limit=10,
    )
    search_time = time.time() - start_time

    return search_time


def run_concurrent_searches(collection_name, search_vectors, milvus_uri, num_workers):
    """
    Run concurrent searches using multiprocessing.

    Args:
        collection_name: Name of the collection to search
        search_vectors: List of search vectors
        milvus_uri: Milvus connection URI
        num_workers: Number of concurrent workers

    Returns:
        Total time and list of individual search times
    """
    # Prepare arguments for each worker
    args_list = [
        (collection_name, search_vector, milvus_uri) for search_vector in search_vectors
    ]

    start_time = time.time()

    # Use multiprocessing pool to execute concurrent searches
    with Pool(processes=num_workers) as pool:
        search_times = pool.map(search_worker, args_list)

    total_time = time.time() - start_time

    return total_time, search_times


def main():
    # Connect to Milvus
    milvus_uri = os.getenv("MILVUS_URI", "http://localhost:19530")
    client = MilvusClient(uri=milvus_uri)

    num_vectors = 50000
    collection_name = "wiki_multi_replica"

    # ========================================================================
    # Setup: Create Collection and Insert Data
    # ========================================================================
    print("=" * 70)
    print("Setup: Creating collection and inserting data")
    print("=" * 70)

    create_collection_with_index(
        client,
        collection_name,
        index_type="HNSW",
        index_params={"M": 16, "efConstruction": 200},
    )

    # Insert data
    start_time = time.time()
    insert_wikipedia_data(client, collection_name, num_vectors)
    insert_time = time.time() - start_time
    print(f"Insert time: {insert_time:.2f}s")

    # Get search vectors from dataset
    print("\nLoading search vectors from dataset...")
    docs = load_dataset("Cohere/wikipedia-22-12-simple-embeddings", split="train")
    num_searches = 100
    search_vectors = [docs[i]["emb"] for i in range(num_searches)]
    print(f"Loaded {len(search_vectors)} search vectors")

    # ========================================================================
    # Example 1: Single Replica with Concurrent Searches
    # ========================================================================
    print("\n" + "=" * 70)
    print("Example 1: Single Replica (replica_number=1)")
    print("=" * 70)

    # Load collection with 1 replica
    client.load_collection(collection_name=collection_name, replica_number=1)
    time.sleep(2)  # Wait for collection to be fully loaded

    print(f"\nRunning {num_searches} concurrent searches with 8 workers...")
    total_time, search_times = run_concurrent_searches(
        collection_name, search_vectors, milvus_uri, num_workers=8
    )

    avg_search_time = sum(search_times) / len(search_times)
    print(f"Total time: {total_time:.2f}s")
    print(f"Average search time: {avg_search_time * 1000:.2f}ms")
    print(f"Throughput: {num_searches / total_time:.2f} searches/sec")

    # Release collection
    client.release_collection(collection_name)
    time.sleep(2)

    # ========================================================================
    # Example 2: Multi-Replica with Concurrent Searches
    # ========================================================================
    print("\n" + "=" * 70)
    print("Example 2: Multi-Replica (replica_number=2)")
    print("=" * 70)

    # Load collection with 2 replicas
    client.load_collection(collection_name=collection_name, replica_number=2)
    time.sleep(2)  # Wait for collection to be fully loaded

    print(f"\nRunning {num_searches} concurrent searches with 8 workers...")
    total_time, search_times = run_concurrent_searches(
        collection_name, search_vectors, milvus_uri, num_workers=8
    )

    avg_search_time = sum(search_times) / len(search_times)
    print(f"Total time: {total_time:.2f}s")
    print(f"Average search time: {avg_search_time * 1000:.2f}ms")
    print(f"Throughput: {num_searches / total_time:.2f} searches/sec")

    # Release collection
    client.release_collection(collection_name)
    time.sleep(2)

    # ========================================================================
    # Example 3: Scaling with Different Worker Counts
    # ========================================================================
    print("\n" + "=" * 70)
    print("Example 3: Scaling with Different Worker Counts (2 replicas)")
    print("=" * 70)

    # Load collection with 2 replicas
    client.load_collection(collection_name=collection_name, replica_number=2)
    time.sleep(2)

    for num_workers in [2, 4, 8, 16]:
        print(f"\nTesting with {num_workers} concurrent workers...")
        total_time, search_times = run_concurrent_searches(
            collection_name, search_vectors[:50], milvus_uri, num_workers=num_workers
        )

        print(f"  Total time: {total_time:.2f}s")
        print(f"  Throughput: {50 / total_time:.2f} searches/sec")

    # ========================================================================
    # Cleanup
    # ========================================================================
    print("\nCleaning up...")
    client.release_collection(collection_name)
    client.drop_collection(collection_name)
    print("Done!")


if __name__ == "__main__":
    main()
