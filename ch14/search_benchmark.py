"""
Chapter 14: Search performance benchmark script.

Runs concurrent vector searches against Milvus and reports QPS and latency
metrics (P50, P95, P99). Supports varying search parameters (topK, nprobe,
ef) and concurrency levels to evaluate scalability.

Usage:
    python search_benchmark.py [--uri localhost:19530] [--duration 300] [--concurrency 16]
"""

import argparse
import time
import threading
import statistics
from collections import defaultdict

import numpy as np
from pymilvus import MilvusClient

COLLECTION_NAME = "test_collection"


def percentile(data: list[float], p: float) -> float:
    if not data:
        return 0.0
    sorted_data = sorted(data)
    k = (len(sorted_data) - 1) * (p / 100)
    f = int(k)
    c = f + 1
    if c >= len(sorted_data):
        return sorted_data[f]
    return sorted_data[f] + (k - f) * (sorted_data[c] - sorted_data[f])


def search_worker(
    client: MilvusClient,
    dim: int,
    top_k: int,
    search_params: dict,
    duration: float,
    results: dict,
    stop_event: threading.Event,
):
    latencies = []
    errors = 0
    count = 0
    start = time.time()

    while not stop_event.is_set() and (time.time() - start) < duration:
        query_vector = np.random.randn(1, dim).astype("float32").tolist()
        t0 = time.time()
        try:
            client.search(
                collection_name=COLLECTION_NAME,
                data=query_vector,
                limit=top_k,
                search_params=search_params,
                output_fields=["category"],
            )
            latency_ms = (time.time() - t0) * 1000
            latencies.append(latency_ms)
            count += 1
        except Exception:
            errors += 1

    results["latencies"].extend(latencies)
    results["errors"] += errors
    results["count"] += count


def run_benchmark(
    client: MilvusClient,
    dim: int,
    top_k: int,
    search_params: dict,
    concurrency: int,
    duration: float,
) -> dict:
    results = defaultdict(lambda: 0, latencies=[])
    stop_event = threading.Event()

    threads = []
    for _ in range(concurrency):
        t = threading.Thread(
            target=search_worker,
            args=(client, dim, top_k, search_params, duration, results, stop_event),
        )
        threads.append(t)

    start = time.time()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    elapsed = time.time() - start

    latencies = results["latencies"]
    total = results["count"]
    qps = total / elapsed if elapsed > 0 else 0

    return {
        "qps": qps,
        "total_queries": total,
        "errors": results["errors"],
        "duration_s": elapsed,
        "p50_ms": percentile(latencies, 50),
        "p95_ms": percentile(latencies, 95),
        "p99_ms": percentile(latencies, 99),
        "avg_ms": statistics.mean(latencies) if latencies else 0,
        "max_ms": max(latencies) if latencies else 0,
    }


def main():
    parser = argparse.ArgumentParser(description="Milvus search performance benchmark")
    parser.add_argument(
        "--uri", default="http://localhost:19530", help="Milvus server URI"
    )
    parser.add_argument("--token", default="", help="Milvus authentication token")
    parser.add_argument("--dim", type=int, default=128, help="Vector dimension")
    parser.add_argument(
        "--duration", type=int, default=300, help="Test duration in seconds"
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=16,
        help="Number of concurrent search threads",
    )
    parser.add_argument(
        "--top-k", type=int, default=10, help="Number of results to return per search"
    )
    parser.add_argument("--metric-type", default="L2", help="Distance metric type")
    parser.add_argument("--ef", type=int, default=64, help="HNSW ef search parameter")
    args = parser.parse_args()

    search_params = {"metric_type": args.metric_type, "params": {"ef": args.ef}}

    client = MilvusClient(uri=args.uri, token=args.token)

    print("Search Benchmark Configuration:")
    print(f"  Collection:  {COLLECTION_NAME}")
    print(f"  Dimension:   {args.dim}")
    print(f"  Top-K:       {args.top_k}")
    print(f"  ef:          {args.ef}")
    print(f"  Concurrency: {args.concurrency}")
    print(f"  Duration:    {args.duration}s")
    print()

    print("Running benchmark...")
    result = run_benchmark(
        client=client,
        dim=args.dim,
        top_k=args.top_k,
        search_params=search_params,
        concurrency=args.concurrency,
        duration=args.duration,
    )

    print("\nResults:")
    print(f"  QPS:           {result['qps']:.1f}")
    print(f"  Total queries: {result['total_queries']}")
    print(f"  Errors:        {result['errors']}")
    print(f"  Avg latency:   {result['avg_ms']:.2f} ms")
    print(f"  P50 latency:   {result['p50_ms']:.2f} ms")
    print(f"  P95 latency:   {result['p95_ms']:.2f} ms")
    print(f"  P99 latency:   {result['p99_ms']:.2f} ms")
    print(f"  Max latency:   {result['max_ms']:.2f} ms")


if __name__ == "__main__":
    main()
