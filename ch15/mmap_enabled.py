"""
Chapter 15: Memory Mapping (mmap) for Capacity Extension

This example demonstrates how memory mapping enables loading more data
than available RAM by intelligently using disk storage with Wikipedia embeddings.
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
    # Example 1: Collection WITHOUT mmap
    # ========================================================================
    print("=" * 70)
    print("Example 1: Collection WITHOUT mmap")
    print("=" * 70)

    collection_name = "wiki_no_mmap"

    if client.has_collection(collection_name):
        client.drop_collection(collection_name)

    # Create collection without mmap
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
        index_type="IVF_FLAT",
        metric_type="COSINE",
        params={"nlist": 2048},
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
    # Example 2: Collection WITH mmap Enabled
    # ========================================================================
    print("\n" + "=" * 70)
    print("Example 2: Collection WITH mmap Enabled")
    print("=" * 70)

    collection_name = "wiki_with_mmap"

    if client.has_collection(collection_name):
        client.drop_collection(collection_name)

    # Create collection with mmap enabled
    client.create_collection(
        collection_name=collection_name,
        schema=schema,
        properties={"mmap.enabled": "true"},
    )

    # Create index
    index_params_mmap = client.prepare_index_params()
    index_params_mmap.add_index(
        field_name="emb",
        index_type="IVF_FLAT",
        metric_type="COSINE",
        params={"nlist": 2048},
    )
    client.create_index(collection_name=collection_name, index_params=index_params_mmap)

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
    # Example 3: Combining mmap + IVF_SQ8
    # ========================================================================
    print("\n" + "=" * 70)
    print("Example 3: Combining mmap + IVF_SQ8 for Maximum Efficiency")
    print("=" * 70)

    collection_name = "wiki_mmap_sq8"

    if client.has_collection(collection_name):
        client.drop_collection(collection_name)

    # Create collection with mmap
    client.create_collection(
        collection_name=collection_name,
        schema=schema,
        properties={"mmap.enabled": "true"},
    )

    # Create IVF_SQ8 index
    index_params_sq8 = client.prepare_index_params()
    index_params_sq8.add_index(
        field_name="emb", index_type="IVF_SQ8", metric_type="L2", params={"nlist": 2048}
    )

    client.create_index(collection_name=collection_name, index_params=index_params_sq8)

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
        search_params={"metric_type": "COSINE", "params": {"nprobe": 32}},
        limit=10,
    )
    search_time = time.time() - start_time

    print(f"\nSearch time: {search_time * 1000:.2f}ms")
    print("Top 5 results:")
    for i, hit in enumerate(results[0][:5]):
        print(f"  {i + 1}. ID: {hit['id']}, Distance: {hit['distance']:.4f}")

    # ========================================================================
    # Example 4: Field-Level mmap Settings
    # ========================================================================
    print("\n" + "=" * 70)
    print("Example 4: Field-Level mmap Settings")
    print("=" * 70)

    collection_name = "wiki_field_mmap"

    if client.has_collection(collection_name):
        client.drop_collection(collection_name)

    # Create schema with field-level mmap settings
    schema_field = client.create_schema(auto_id=True, enable_dynamic_field=False)
    schema_field.add_field(field_name="id", datatype=DataType.INT64, is_primary=True)

    # Enable mmap for vector field (large data)
    schema_field.add_field(
        field_name="emb",
        datatype=DataType.FLOAT_VECTOR,
        dim=dim,
        mmap_enabled=True,  # Enable mmap for vector field
    )

    # Disable mmap for frequently accessed fields
    schema_field.add_field(
        field_name="title",
        datatype=DataType.VARCHAR,
        max_length=100,
        mmap_enabled=False,
    )

    schema_field.add_field(
        field_name="text",
        datatype=DataType.VARCHAR,
        max_length=5000,
        mmap_enabled=True,  # Enable mmap for large text field
    )

    schema_field.add_field(field_name="wiki_id", datatype=DataType.INT64)
    schema_field.add_field(field_name="views", datatype=DataType.FLOAT)

    client.create_collection(collection_name=collection_name, schema=schema_field)

    # Create index
    index_params_field = client.prepare_index_params()
    index_params_field.add_index(
        field_name="emb",
        index_type="IVF_FLAT",
        metric_type="COSINE",
        params={"nlist": 2048},
    )

    client.create_index(
        collection_name=collection_name, index_params=index_params_field
    )

    # Insert sample data
    batch_num = 0
    total_inserted = 0
    for batch in load_wikipedia_data(limit=10000, batch_size=1000):
        client.insert(collection_name=collection_name, data=batch)
        batch_num += 1
        total_inserted += len(batch)
        print(f"Inserted {total_inserted} records...")

    print(f"Field-level mmap collection created with {total_inserted} records")

    # ========================================================================
    # Example 5: Index-Level mmap Settings
    # ========================================================================
    print("\n" + "=" * 70)
    print("Example 5: Index-Level mmap Settings")
    print("=" * 70)

    collection_name = "wiki_index_mmap"

    if client.has_collection(collection_name):
        client.drop_collection(collection_name)

    # Create schema
    schema_index = client.create_schema(auto_id=True, enable_dynamic_field=False)
    schema_index.add_field(field_name="id", datatype=DataType.INT64, is_primary=True)
    schema_index.add_field(field_name="emb", datatype=DataType.FLOAT_VECTOR, dim=dim)
    schema_index.add_field(
        field_name="title", datatype=DataType.VARCHAR, max_length=100
    )
    schema_index.add_field(
        field_name="text", datatype=DataType.VARCHAR, max_length=5000
    )
    schema_index.add_field(field_name="wiki_id", datatype=DataType.INT64)
    schema_index.add_field(field_name="views", datatype=DataType.FLOAT)

    client.create_collection(collection_name=collection_name, schema=schema_index)

    # Create index with index-level mmap settings
    index_params_index = client.prepare_index_params()

    # Vector index with mmap enabled
    index_params_index.add_index(
        field_name="emb",
        index_type="IVF_FLAT",
        metric_type="COSINE",
        params={"nlist": 2048, "mmap.enabled": True},  # Index-level mmap
    )

    # Scalar index with mmap disabled
    index_params_index.add_index(
        field_name="title",
        index_type="AUTOINDEX",
        params={"mmap.enabled": False},
    )

    client.create_index(
        collection_name=collection_name, index_params=index_params_index
    )

    # Insert sample data
    batch_num = 0
    total_inserted = 0
    for batch in load_wikipedia_data(limit=10000, batch_size=1000):
        client.insert(collection_name=collection_name, data=batch)
        batch_num += 1
        total_inserted += len(batch)
        print(f"Inserted {total_inserted} records...")

    print(f"Index-level mmap collection created with {total_inserted} records")

    # ========================================================================
    # Example 6: Modifying mmap Settings Dynamically
    # ========================================================================
    print("\n" + "=" * 70)
    print("Example 6: Modifying mmap Settings Dynamically")
    print("=" * 70)

    collection_name = "wiki_dynamic_mmap"

    if client.has_collection(collection_name):
        client.drop_collection(collection_name)

    # Create collection without mmap initially
    client.create_collection(
        collection_name=collection_name,
        schema=schema,
        properties={"mmap.enabled": "false"},
    )

    # Create index
    index_params_dynamic = client.prepare_index_params()
    index_params_dynamic.add_index(
        field_name="emb",
        index_type="IVF_FLAT",
        metric_type="COSINE",
        params={"nlist": 2048},
    )
    index_params_dynamic.add_index(
        field_name="title",
        index_type="AUTOINDEX",
    )

    client.create_index(
        collection_name=collection_name, index_params=index_params_dynamic
    )

    # Insert sample data
    batch_num = 0
    total_inserted = 0
    for batch in load_wikipedia_data(limit=10000, batch_size=1000):
        client.insert(collection_name=collection_name, data=batch)
        batch_num += 1
        total_inserted += len(batch)
        print(f"Inserted {total_inserted} records...")

    print(f"\nInitial collection created with mmap DISABLED ({total_inserted} records)")

    # Load collection
    client.load_collection(collection_name)

    # Search before enabling mmap
    start_time = time.time()
    results = client.search(
        collection_name=collection_name,
        data=[search_vector],
        anns_field="emb",
        search_params=search_params,
        limit=10,
    )
    search_time_before = time.time() - start_time
    print(f"Search time (before mmap): {search_time_before * 1000:.2f}ms")

    # Release collection to modify settings
    client.release_collection(collection_name)

    # Modify collection-level mmap setting
    print("\nEnabling mmap for the entire collection...")
    client.alter_collection_properties(
        collection_name=collection_name, properties={"mmap.enabled": True}
    )

    # Modify field-level mmap setting
    print("Enabling mmap for 'text' field...")
    client.alter_collection_field(
        collection_name=collection_name,
        field_name="text",
        field_params={"mmap.enabled": True},
    )

    # Modify index-level mmap setting
    print("Enabling mmap for 'title' index...")
    client.alter_index_properties(
        collection_name=collection_name,
        index_name="title",
        properties={"mmap.enabled": True},
    )

    # Reload collection for changes to take effect
    client.load_collection(collection_name)

    # Search after enabling mmap
    start_time = time.time()
    results = client.search(
        collection_name=collection_name,
        data=[search_vector],
        anns_field="emb",
        search_params=search_params,
        limit=10,
    )
    search_time_after = time.time() - start_time
    print(f"Search time (after mmap):  {search_time_after * 1000:.2f}ms")

    print("\nTop 5 results after enabling mmap:")
    for i, hit in enumerate(results[0][:5]):
        print(f"  {i + 1}. ID: {hit['id']}, Distance: {hit['distance']:.4f}")

    # ========================================================================
    # Example 7: Hierarchical mmap Settings Priority
    # ========================================================================
    print("\n" + "=" * 70)
    print("Example 7: Hierarchical mmap Settings Priority")
    print("(Index/Field Level > Collection Level > Global Level)")
    print("=" * 70)

    collection_name = "wiki_hierarchy_mmap"

    if client.has_collection(collection_name):
        client.drop_collection(collection_name)

    # Collection-level: mmap enabled
    schema_hierarchy = client.create_schema(auto_id=True, enable_dynamic_field=False)
    schema_hierarchy.add_field(
        field_name="id", datatype=DataType.INT64, is_primary=True
    )

    # Field-level override: disable mmap for vector (higher priority)
    schema_hierarchy.add_field(
        field_name="emb",
        datatype=DataType.FLOAT_VECTOR,
        dim=dim,
        mmap_enabled=False,  # Field-level: DISABLED
    )

    schema_hierarchy.add_field(
        field_name="title",
        datatype=DataType.VARCHAR,
        max_length=100,
    )

    # Field-level override: enable mmap for text
    schema_hierarchy.add_field(
        field_name="text",
        datatype=DataType.VARCHAR,
        max_length=5000,
        mmap_enabled=True,  # Field-level: ENABLED
    )

    schema_hierarchy.add_field(field_name="wiki_id", datatype=DataType.INT64)
    schema_hierarchy.add_field(field_name="views", datatype=DataType.FLOAT)

    # Collection-level: mmap enabled
    client.create_collection(
        collection_name=collection_name,
        schema=schema_hierarchy,
        properties={"mmap.enabled": "true"},  # Collection-level: ENABLED
    )

    # Create index with index-level override
    index_params_hierarchy = client.prepare_index_params()

    # Index-level override: enable mmap (overrides field-level)
    index_params_hierarchy.add_index(
        field_name="emb",
        index_type="IVF_FLAT",
        metric_type="COSINE",
        params={
            "nlist": 2048,
            "mmap.enabled": True,
        },  # Index-level: ENABLED (highest priority)
    )

    client.create_index(
        collection_name=collection_name, index_params=index_params_hierarchy
    )

    # Insert sample data
    batch_num = 0
    total_inserted = 0
    for batch in load_wikipedia_data(limit=10000, batch_size=1000):
        client.insert(collection_name=collection_name, data=batch)
        batch_num += 1
        total_inserted += len(batch)
        print(f"Inserted {total_inserted} records...")

    print("\nHierarchy demonstration:")
    print("  Collection level: mmap ENABLED")
    print("  Field 'emb':      mmap DISABLED (field-level)")
    print("  Index 'emb':      mmap ENABLED (index-level, highest priority)")
    print("  Field 'text':     mmap ENABLED (field-level)")
    print("  Result: Index-level setting wins for 'emb' field")

    # ========================================================================
    # Cleanup
    # ========================================================================
    print("\n" + "=" * 70)
    print("Cleaning up...")
    print("=" * 70)

    collections_to_clean = [
        "wiki_no_mmap",
        "wiki_with_mmap",
        "wiki_mmap_sq8",
        "wiki_field_mmap",
        "wiki_index_mmap",
        "wiki_dynamic_mmap",
        "wiki_hierarchy_mmap",
    ]

    for col in collections_to_clean:
        if client.has_collection(col):
            client.release_collection(col)
            client.drop_collection(col)
            print(f"Dropped collection: {col}")

    print("\nDone!")


if __name__ == "__main__":
    main()
