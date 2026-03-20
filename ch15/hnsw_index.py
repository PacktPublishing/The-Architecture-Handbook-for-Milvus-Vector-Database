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


def main():
    # Connect to Milvus
    client = MilvusClient(uri=os.getenv("MILVUS_URI", "http://localhost:19530"))

    num_vectors = 50000

    collection_name = "wiki_hnsw"
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

    # Load collection
    client.load_collection(collection_name)

    # Get a real search vector from the dataset
    print("Loading search vector from dataset...")
    docs = load_dataset("Cohere/wikipedia-22-12-simple-embeddings", split="train")
    search_vector = docs[0]["emb"]
    # Search with HNSW
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

    print(f"Search time: {search_time * 1000:.2f}ms")

    client.release_collection(collection_name)
    client.drop_collection("wiki_hnsw")


if __name__ == "__main__":
    main()
