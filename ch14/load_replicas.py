"""
Chapter 14: Configure in-memory replicas for horizontal scaling.

Loads a collection with a specified number of in-memory replicas to distribute
search load across multiple QueryNode instances.

Usage:
    python load_replicas.py [--uri localhost:19530] [--replicas 3]
"""

import argparse

from pymilvus import MilvusClient

COLLECTION_NAME = "test_collection"


def main():
    parser = argparse.ArgumentParser(description="Configure Milvus in-memory replicas")
    parser.add_argument(
        "--uri", default="http://localhost:19530", help="Milvus server URI"
    )
    parser.add_argument("--token", default="", help="Milvus authentication token")
    parser.add_argument("--collection", default=COLLECTION_NAME, help="Collection name")
    parser.add_argument(
        "--replicas", type=int, default=3, help="Number of in-memory replicas"
    )
    args = parser.parse_args()

    client = MilvusClient(uri=args.uri, token=args.token)

    # Release collection first if already loaded
    try:
        client.release_collection(args.collection)
        print(f"Released collection '{args.collection}'")
    except Exception:
        pass

    # Load with specified replica count
    client.load_collection(
        collection_name=args.collection,
        replica_number=args.replicas,
    )
    print(
        f"Collection '{args.collection}' loaded with {args.replicas} in-memory replicas"
    )
    print(
        f"Search requests will be distributed across {args.replicas} QueryNode instances"
    )


if __name__ == "__main__":
    main()
