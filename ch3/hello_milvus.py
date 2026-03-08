# Before running the script, you need to install the pymilvus python third-party library in advance.
# Execute: pip install pymilvus

import numpy as np
from pymilvus import MilvusClient, DataType

milvus_client = MilvusClient("http://localhost:19530")
fmt = "\n=== {:30} ===\n"
dim = 8
collection_name = "hello_milvus"
has_collection = milvus_client.has_collection(collection_name, timeout=5)
if has_collection:
    milvus_client.drop_collection(collection_name)

schema = milvus_client.create_schema()
schema.add_field("id", DataType.INT64, is_primary=True)
schema.add_field("vector", DataType.FLOAT_VECTOR, dim=dim)
schema.add_field("score", DataType.INT64)

index_params = milvus_client.prepare_index_params()
index_params.add_index(field_name = "vector", index_type="HNSW", metric_type="L2")
milvus_client.create_collection(
    collection_name,
    schema=schema,
    index_params=index_params,
    consistency_level="Strong"
)

print(fmt.format(" all collections "))
print(milvus_client.list_collections())
print(fmt.format(f"schema of collection {collection_name}"))
print(milvus_client.describe_collection(collection_name))

rng = np.random.default_rng(seed=42)
rows = [
        {"id": 1, "vector": rng.random((1, dim))[0], "score": 100},
        {"id": 2, "vector": rng.random((1, dim))[0], "score": 200},
        {"id": 3, "vector": rng.random((1, dim))[0], "score": 300},
        {"id": 4, "vector": rng.random((1, dim))[0], "score": 400},
        {"id": 5, "vector": rng.random((1, dim))[0], "score": 500},
        {"id": 6, "vector": rng.random((1, dim))[0], "score": 600},
]
print(fmt.format("Start inserting entities"))
insert_result = milvus_client.insert(collection_name, rows, progress_bar=True)
print(fmt.format("Inserting entities done"))
print(insert_result)

print(fmt.format("Start query by specifying primary keys"))
query_results = milvus_client.query(collection_name, ids=[2])
print(query_results[0])

upsert_ret = milvus_client.upsert(
    collection_name,
    {"id": 2 , "vector": rng.random((1, dim))[0], "score": 220}
)
print(upsert_ret)
print(fmt.format("Start query by specifying primary keys"))
query_results = milvus_client.query(collection_name, ids=[2])
print(query_results[0])

print(fmt.format("Start query by specifying filtering expression"))
query_results = milvus_client.query(collection_name, filter= "score == 600")
for ret in query_results:
    print(ret)

print(f"start to delete by specifying filter in collection {collection_name}")
delete_result = milvus_client.delete(collection_name, ids=[6])
print(delete_result)
print(fmt.format("Start query by specifying filtering expression"))
query_results = milvus_client.query(collection_name, filter= "score == 600")
assert len(query_results) == 0

rng = np.random.default_rng(seed=43)
vectors_to_search = rng.random((1, dim))
print("Start search with retrieve several fields.")
result = milvus_client.search(
    collection_name,
    vectors_to_search,
    limit=3,
    output_fields=["score"]
)
for hits in result:
    for hit in hits:
        print(f"hit: {hit}")

# field_index_names = milvus_client.list_indexes(collection_name, field_name = "vector")
# print(f"index names for {collection_name}`s field embeddings:", field_index_names)

# milvus_client.drop_index(collection_name, "vector")
# milvus_client.release_collection(collection_name)
milvus_client.drop_collection(collection_name)
