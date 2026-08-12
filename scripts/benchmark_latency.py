import time
import numpy as np
import pandas as pd
from fastapi.testclient import TestClient
from src.api.main import app

client = TestClient(app)

def run_latency_benchmarks(num_single_requests=500, num_batch_requests=50, batch_size=100):
    print("=" * 65)
    print("FINANCIAL SENTINEL INFERENCE LATENCY BENCHMARK SUITE")
    print("=" * 65)
    
    single_payload = {
        "Time": 4462.0,
        "Amount": 239.93,
        "v_features": [0.1] * 28
    }

    batch_payload = [
        {
            "Time": float(i),
            "Amount": float(100 + i),
            "v_features": [0.05 * (i % 10)] * 28
        }
        for i in range(batch_size)
    ]

    # Warmup
    print("[1/3] Warming up model inference engine...")
    for _ in range(20):
        client.post("/api/v1/predict", json=single_payload)
        client.post("/api/v1/predict-batch", json=batch_payload)

    # 1. Benchmark Single-Request Latency (/predict)
    print(f"[2/3] Benchmarking {num_single_requests} Single-Request (/predict) calls...")
    single_latencies = []
    for _ in range(num_single_requests):
        t0 = time.perf_counter()
        res = client.post("/api/v1/predict", json=single_payload)
        t1 = time.perf_counter()
        assert res.status_code == 200
        single_latencies.append((t1 - t0) * 1000)  # ms

    # 2. Benchmark Batch-Request Latency (/predict-batch)
    print(f"[3/3] Benchmarking {num_batch_requests} Batch-Request (/predict-batch, {batch_size} items) calls...")
    batch_latencies = []
    per_item_latencies = []
    for _ in range(num_batch_requests):
        t0 = time.perf_counter()
        res = client.post("/api/v1/predict-batch", json=batch_payload)
        t1 = time.perf_counter()
        assert res.status_code == 200
        dur_ms = (t1 - t0) * 1000
        batch_latencies.append(dur_ms)
        per_item_latencies.append(dur_ms / batch_size)

    # Calculate Percentiles
    single_p50 = float(np.percentile(single_latencies, 50))
    single_p95 = float(np.percentile(single_latencies, 95))
    single_p99 = float(np.percentile(single_latencies, 99))
    single_mean = float(np.mean(single_latencies))

    batch_p50 = float(np.percentile(batch_latencies, 50))
    batch_p95 = float(np.percentile(batch_latencies, 95))
    batch_p99 = float(np.percentile(batch_latencies, 99))
    batch_mean = float(np.mean(batch_latencies))

    item_p50 = float(np.percentile(per_item_latencies, 50))
    item_p95 = float(np.percentile(per_item_latencies, 95))
    item_p99 = float(np.percentile(per_item_latencies, 99))

    print("\n" + "=" * 65)
    print("EMPIRICAL LATENCY BENCHMARK RESULTS")
    print("=" * 65)
    
    summary_data = {
        "Metric": ["Single Request (/predict)", f"Batch Request (/predict-batch, {batch_size} items)", "Batch Latency Per Item"],
        "Mean (ms)": [f"{single_mean:.2f}", f"{batch_mean:.2f}", f"{np.mean(per_item_latencies):.3f}"],
        "p50 / Median (ms)": [f"{single_p50:.2f}", f"{batch_p50:.2f}", f"{item_p50:.3f}"],
        "p95 (ms)": [f"{single_p95:.2f}", f"{batch_p95:.2f}", f"{item_p95:.3f}"],
        "p99 (ms)": [f"{single_p99:.2f}", f"{batch_p99:.2f}", f"{item_p99:.3f}"]
    }
    
    df_res = pd.DataFrame(summary_data)
    print(df_res.to_string(index=False))
    print("=" * 65)

    return {
        "single": {"p50": single_p50, "p95": single_p95, "p99": single_p99, "mean": single_mean},
        "batch": {"p50": batch_p50, "p95": batch_p95, "p99": batch_p99, "mean": batch_mean},
        "item": {"p50": item_p50, "p95": item_p95, "p99": item_p99}
    }

if __name__ == "__main__":
    run_latency_benchmarks()
