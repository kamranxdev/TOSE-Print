import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
"""
Experiment 4: Empirical Large-Scale ANN Indexing & Retrieval Benchmark (FAISS HNSW).
Validates sub-linear search time over fixed-length TOSE-Print vectors.
Scales up from 1K to 100K fingerprints, measuring Recall@1, Latency, and QPS.
"""

import os
import time
import numpy as np
import matplotlib.pyplot as plt
from tose_print.core.config import SystemConfig
from tose_print.indexing.faiss_ann import FAISSIndexer


def run_scalability_benchmark(scales=[1000, 5000, 10000, 50000, 100000], dim=325, n_queries=100):
    print("=" * 75)
    print("⚡ EXPERIMENT 4: EMPIRICAL FAISS ANN SCALABILITY BENCHMARK (TIFS Standard)")
    print("=" * 75)

    rng = np.random.RandomState(42)
    config = SystemConfig()

    os.makedirs("results", exist_ok=True)
    results = []

    print(f"{'Database Size (N)':<18} | {'Exact (ms)':<12} | {'HNSW (ms)':<12} | {'Speedup':<10} | {'Recall@1 (%)':<14} | {'QPS':<10}")
    print("-" * 85)

    latex_rows = []

    for n in scales:
        indexer = FAISSIndexer(dimension=dim, config=config.indexing)

        # Generate realistic gallery vectors on unit sphere
        gallery_vecs = rng.randn(n, dim).astype(np.float32)
        gallery_vecs /= np.linalg.norm(gallery_vecs, axis=1, keepdims=True)
        ids = [f"id_{i}" for i in range(n)]

        t_build_0 = time.perf_counter()
        indexer.add(gallery_vecs, ids)
        t_build = time.perf_counter() - t_build_0

        # Query vectors (take first n_queries from gallery + small perturbation)
        q_base = gallery_vecs[:n_queries].copy()
        noise = rng.randn(n_queries, dim).astype(np.float32) * 0.05
        query_vecs = q_base + noise
        query_vecs /= np.linalg.norm(query_vecs, axis=1, keepdims=True)

        bench = indexer.benchmark_recall_and_speed(query_vecs, top_k=10)

        exact_ms = bench["exact_latency_ms"]
        hnsw_ms = bench["hnsw_latency_ms"]
        speedup = bench["speedup_factor"]
        rec1 = bench["recall_at_1"] * 100.0
        qps = bench["hnsw_qps"]

        results.append({
            "n": n,
            "exact_ms": exact_ms,
            "hnsw_ms": hnsw_ms,
            "speedup": speedup,
            "recall_1": rec1,
            "qps": qps
        })

        print(f"{n:<18,d} | {exact_ms:>10.3f}ms | {hnsw_ms:>10.3f}ms | {speedup:>8.1f}x | {rec1:>12.2f}% | {qps:>8.0f}")
        latex_rows.append(f"{n:,} & {exact_ms:.3f} & {hnsw_ms:.3f} & {speedup:.1f}$\\times$ & {rec1:.1f}\\% & {qps:,.0f} \\\\")

    print("=" * 85)

    # Save LaTeX Table
    with open("results/table_ann_scalability.tex", "w") as f:
        f.write("\\begin{table}[t]\n\\centering\n\\caption{Sub-Linear Nearest Neighbor Retrieval Scalability (FAISS HNSW)}\n\\label{tab:ann_scale}\n\\begin{tabular}{rccccc}\n\\hline\n")
        f.write("Gallery Size ($N$) & Exact (ms) & HNSW (ms) & Speedup & Recall@1 (\\%) & QPS \\\\\n\\hline\n")
        f.write("\n".join(latex_rows) + "\n")
        f.write("\\hline\n\\end{tabular}\n\\end{table}\n")

    # Plot Scaling Curve
    ns = [r["n"] for r in results]
    exact_times = [r["exact_ms"] for r in results]
    hnsw_times = [r["hnsw_ms"] for r in results]
    speedups = [r["speedup"] for r in results]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Latency Plot
    ax1.plot(ns, exact_times, 'r-o', linewidth=2, label='Exhaustive Linear Scan $O(N)$')
    ax1.plot(ns, hnsw_times, 'b-s', linewidth=2, label='FAISS HNSW Sub-linear $O(\\log N)$')
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_xlabel('Enrolled Gallery Size ($N$)', fontsize=12)
    ax1.set_ylabel('Query Latency (ms/query)', fontsize=12)
    ax1.set_title('Query Latency vs. Database Size', fontsize=13)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3, which='both')

    # Speedup Plot
    ax2.plot(ns, speedups, 'g-^', linewidth=2)
    ax2.set_xscale('log')
    ax2.set_xlabel('Enrolled Gallery Size ($N$)', fontsize=12)
    ax2.set_ylabel('Speedup Factor ($T_{\\text{exact}} / T_{\\text{HNSW}}$)', fontsize=12)
    ax2.set_title('Sub-linear Acceleration Factor', fontsize=13)
    ax2.grid(True, alpha=0.3, which='both')

    plt.tight_layout()
    plt.savefig("results/ann_scalability.pdf")
    plt.close()

    print("✅ Scalability benchmark completed! LaTeX table and scaling figures saved to results/.")


if __name__ == "__main__":
    run_scalability_benchmark()
