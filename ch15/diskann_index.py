"""
Chapter 15: DiskANN for Billion-Scale Datasets

This example demonstrates how DiskANN enables searching billion-scale
datasets with minimal RAM by storing data on high-performance SSDs using Wikipedia embeddings.
"""

import os
import time
from datasets import load_dataset
from pymilvus import MilvusClient, DataType


def load_wikipedia_data(limit=100000, batch_size=1000):
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


def main():
    # Connect to Milvus
    client = MilvusClient(uri=os.getenv("MILVUS_URI", "http://localhost:19530"))

    dim = 768
    num_vectors = 100000

    # ========================================================================
    # Example 1: HNSW Index (Memory-based)
    # ========================================================================
    print("=" * 70)
    print("Example 1: HNSW Index (Memory-based)")
    print("=" * 70)

    collection_name = "wiki_hnsw"

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

    # Get search vector
    docs = load_dataset(
        "Cohere/wikipedia-22-12-simple-embeddings", split="train", streaming=True
    )
    search_doc = next(iter(docs))
    search_vector = search_doc["emb"]

    # Search
    search_params = {"metric_type": "COSINE", "params": {"ef": 128}}

    start_time = time.time()
    results = client.search(
        collection_name=collection_name,
        data=[search_vector],
        anns_field="emb",
        search_params=search_params,
        limit=10,
    )
    search_time = time.time() - start_time

    print(f"\nSearch time: {search_time * 1000:.2f}ms")
    print("Top 5 results:")
    for i, hit in enumerate(results[0][:5]):
        print(f"  {i + 1}. ID: {hit['id']}, Distance: {hit['distance']:.4f}")

    client.release_collection(collection_name)

    # ========================================================================
    # Example 2: DiskANN Index (Disk-based)
    # ========================================================================
    print("\n" + "=" * 70)
    print("Example 2: DiskANN Index (Disk-based)")
    print("=" * 70)

    collection_name = "wiki_diskann"

    if client.has_collection(collection_name):
        client.drop_collection(collection_name)

    # Create collection
    client.create_collection(collection_name=collection_name, schema=schema)

    # Create DiskANN index
    try:
        index_params = client.prepare_index_params()
        index_params.add_index(
            field_name="emb",
            index_type="DISKANN",
            metric_type="COSINE",
            params={
                "search_list": 100,
                "pq_code_budget_gb": 0.125,
                "build_dram_budget_gb": 2.0,
            },
        )

        client.create_index(collection_name=collection_name, index_params=index_params)

        diskann_supported = True
        print("DiskANN index created successfully")

    except Exception as e:
        print(f"DiskANN creation failed: {e}")
        print("Note: DiskANN may not be supported in your Milvus version")
        diskann_supported = False

    if diskann_supported:
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

        # Search with DiskANN
        search_params = {"metric_type": "COSINE", "params": {"search_list": 100}}

        start_time = time.time()
        results = client.search(
            collection_name=collection_name,
            data=[search_vector],
            anns_field="emb",
            search_params=search_params,
            limit=10,
        )
        search_time = time.time() - start_time

        print(f"\nSearch time: {search_time * 1000:.2f}ms")
        print("Top 5 results:")
        for i, hit in enumerate(results[0][:5]):
            print(f"  {i + 1}. ID: {hit['id']}, Distance: {hit['distance']:.4f}")

        # ========================================================================
        # Example 3: Tuning DiskANN search_list Parameter
        # ========================================================================
        print("\n" + "=" * 70)
        print("Example 3: Tuning DiskANN search_list Parameter")
        print("=" * 70)

        for search_list in [50, 100, 200, 300]:
            search_params = {
                "metric_type": "COSINE",
                "params": {"search_list": search_list},
            }

            start_time = time.time()
            results = client.search(
                collection_name=collection_name,
                data=[search_vector],
                anns_field="emb",
                search_params=search_params,
                limit=10,
            )
            search_time = time.time() - start_time

            print(
                f"search_list={search_list}: search time = {search_time * 1000:.2f}ms"
            )

        # Cleanup
        client.release_collection(collection_name)

    # ========================================================================
    # Cleanup
    # ========================================================================
    print("\nCleaning up...")
    client.drop_collection("wiki_hnsw")
    if diskann_supported:
        client.drop_collection("wiki_diskann")
    print("Done!")


if __name__ == "__main__":
    main()
