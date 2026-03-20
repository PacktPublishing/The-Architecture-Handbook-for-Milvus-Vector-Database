"""
Chapter 15: IVF_SQ8 Quantization for Memory Efficiency

This example demonstrates how IVF_SQ8 quantization reduces memory usage
by 4x while maintaining reasonable search accuracy using Wikipedia embeddings.
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
    # Example 1: IVF_FLAT Index (No Quantization)
    # ========================================================================
    print("=" * 70)
    print("Example 1: IVF_FLAT Index (No Quantization)")
    print("=" * 70)

    collection_name = "wiki_ivf_flat"

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

    # Create IVF_FLAT index
    index_params = client.prepare_index_params()
    index_params.add_index(
        field_name="emb",
        index_type="IVF_FLAT",
        metric_type="COSINE",
        params={"nlist": 4096},
    )

    client.create_index(collection_name=collection_name, index_params=index_params)

    # Insert data
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
    search_params = {"metric_type": "COSINE", "params": {"nprobe": 32}}

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
    # Example 2: IVF_SQ8 Index (With Quantization)
    # ========================================================================
    print("\n" + "=" * 70)
    print("Example 2: IVF_SQ8 Index (With Quantization)")
    print("=" * 70)

    collection_name = "wiki_ivf_sq8"

    if client.has_collection(collection_name):
        client.drop_collection(collection_name)

    # Create collection
    client.create_collection(collection_name=collection_name, schema=schema)

    # Create IVF_SQ8 index
    index_params = client.prepare_index_params()
    index_params.add_index(
        field_name="emb", index_type="IVF_SQ8", metric_type="L2", params={"nlist": 4096}
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

    # Search
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
    # Example 3: Tuning nprobe Parameter
    # ========================================================================
    print("\n" + "=" * 70)
    print("Example 3: Tuning nprobe Parameter")
    print("=" * 70)

    for nprobe in [8, 16, 32, 64, 128]:
        search_params = {"metric_type": "COSINE", "params": {"nprobe": nprobe}}

        start_time = time.time()
        results = client.search(
            collection_name=collection_name,
            data=[search_vector],
            anns_field="emb",
            search_params=search_params,
            limit=10,
        )
        search_time = time.time() - start_time

        print(f"nprobe={nprobe}: search time = {search_time * 1000:.2f}ms")

    # ========================================================================
    # Cleanup
    # ========================================================================
    print("\nCleaning up...")
    client.release_collection("wiki_ivf_flat")
    client.drop_collection("wiki_ivf_flat")
    client.release_collection("wiki_ivf_sq8")
    client.drop_collection("wiki_ivf_sq8")
    print("Done!")


if __name__ == "__main__":
    main()
