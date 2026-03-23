"""
Chapter 15: Bulk Import for Massive Datasets

This example demonstrates how bulk import bypasses the entire online
ingestion pipeline for dramatic throughput improvements using Wikipedia embeddings.
"""

import os
import time
from datasets import load_dataset
from pymilvus import MilvusClient, DataType
from pymilvus.bulk_writer import (
    RemoteBulkWriter,
    BulkFileType,
    bulk_import,
    get_import_progress,
    list_import_jobs,
)


def main():
    # Connect to Milvus
    milvus_uri = os.getenv("MILVUS_URI", "http://localhost:19530")
    client = MilvusClient(uri=milvus_uri)

    # MinIO configuration (adjust according to your setup)
    MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "localhost:9000")
    ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
    SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "minioadmin")
    BUCKET_NAME = os.getenv("MINIO_BUCKET", "a-bucket")

    dim = 768
    num_vectors = 100000

    # ========================================================================
    # Example 1: Prepare Data Files with RemoteBulkWriter
    # ========================================================================
    print("=" * 70)
    print("Example 1: Prepare Data Files with RemoteBulkWriter")
    print("=" * 70)

    collection_name = "wiki_bulk_import"

    if client.has_collection(collection_name):
        client.drop_collection(collection_name)

    # Create collection schema
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

    # Prepare data using RemoteBulkWriter
    try:
        # Configure MinIO connection
        connect_param = RemoteBulkWriter.S3ConnectParam(
            endpoint=MINIO_ENDPOINT,
            access_key=ACCESS_KEY,
            secret_key=SECRET_KEY,
            bucket_name=BUCKET_NAME,
            secure=False,
        )

        # Create RemoteBulkWriter
        writer = RemoteBulkWriter(
            schema=schema,
            remote_path="/wiki_bulk/",
            connect_param=connect_param,
            file_type=BulkFileType.PARQUET,
        )

        print("Loading Wikipedia dataset and writing to Parquet files...")

        # Load Wikipedia dataset
        docs = load_dataset(
            "Cohere/wikipedia-22-12-simple-embeddings", split="train", streaming=True
        )

        # Generate and write data in batches
        batch_size = 10000
        count = 0

        for doc in docs:
            if count >= num_vectors:
                break

            row = {
                "emb": doc["emb"],
                "title": doc["title"][:100],
                "text": doc["text"][:5000],
                "wiki_id": doc["wiki_id"],
                "views": float(doc["views"]),
            }
            writer.append_row(row)
            count += 1

            # Commit every batch_size records
            if (count) % batch_size == 0:
                writer.commit()
                print(f"Committed {count} records to Parquet files...")

        # Final commit for remaining records
        if num_vectors % batch_size != 0:
            writer.commit()
            print(f"Final commit: {count} total records")

        # Get the list of generated files
        batch_files = writer.batch_files
        print(f"\nGenerated {len(batch_files)} Parquet file(s)")

    except Exception as e:
        print(f"Failed to prepare bulk import files: {e}")
        print("Note: Make sure MinIO is running and configured correctly")
        print("You can skip the bulk import examples if MinIO is not available")
        return

    # ========================================================================
    # Example 2: Execute Bulk Import Job
    # ========================================================================
    print("\n" + "=" * 70)
    print("Example 2: Execute Bulk Import Job")
    print("=" * 70)

    try:
        # Start bulk import
        print("Starting bulk import job...")
        resp = bulk_import(
            url=milvus_uri, collection_name=collection_name, files=batch_files
        )

        result = resp.json()
        if result.get("code") != 200:
            print(f"Bulk import failed: {result}")
            return

        job_id = result["data"]["jobId"]
        print(f"Bulk import job started with ID: {job_id}")

    except Exception as e:
        print(f"Failed to start bulk import: {e}")
        return

    # ========================================================================
    # Example 3: Monitor Import Progress
    # ========================================================================
    print("\n" + "=" * 70)
    print("Example 3: Monitor Import Progress")
    print("=" * 70)

    while True:
        try:
            # Get import progress
            progress_resp = get_import_progress(url=milvus_uri, job_id=job_id)

            progress_data = progress_resp.json()
            if progress_data.get("code") != 200:
                print(f"Failed to get progress: {progress_data}")
                break

            data = progress_data["data"]
            state = data["state"]
            progress = data.get("progress", 0)

            print(f"Import progress: {progress}% (state: {state})")

            if state == "Completed":
                print("Bulk import completed successfully!")
                break
            elif state == "Failed":
                print("Bulk import failed!")
                print(f"Reason: {data.get('reason', 'Unknown')}")
                break
            elif state in ["Pending", "Importing"]:
                time.sleep(2)
            else:
                print(f"Unknown state: {state}")
                break

        except Exception as e:
            print(f"Error monitoring progress: {e}")
            break

    # ========================================================================
    # Example 4: List All Import Jobs
    # ========================================================================
    print("\n" + "=" * 70)
    print("Example 4: List All Import Jobs")
    print("=" * 70)

    try:
        jobs_resp = list_import_jobs(url=milvus_uri, collection_name=collection_name)

        jobs_data = jobs_resp.json()
        if jobs_data.get("code") == 200:
            jobs = jobs_data.get("data", {}).get("records", [])
            print(f"Found {len(jobs)} import job(s):")
            for job in jobs:
                print(f"  - Job ID: {job.get('jobId')}, State: {job.get('state')}")

    except Exception as e:
        print(f"Error listing import jobs: {e}")

    # ========================================================================
    # Example 5: Load and Query Imported Data
    # ========================================================================
    print("\n" + "=" * 70)
    print("Example 5: Load and Query Imported Data")
    print("=" * 70)

    # Load collection
    client.load_collection(collection_name)
    print("Collection loaded")

    # Get search vector from dataset
    docs = load_dataset(
        "Cohere/wikipedia-22-12-simple-embeddings", split="train", streaming=True
    )
    search_doc = next(iter(docs))
    search_vector = search_doc["emb"]

    # Sample search
    search_results = client.search(
        collection_name=collection_name,
        data=[search_vector],
        anns_field="emb",
        search_params={"metric_type": "COSINE", "params": {"ef": 128}},
        limit=5,
    )

    print("\nTop 5 search results:")
    for i, hit in enumerate(search_results[0]):
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
