"""
Scalable Approximate Nearest Neighbor (ANN) Indexing via FAISS.
Empirical validation of sub-linear fingerprint retrieval over fixed-length TOSE-Print vectors.
"""

import time
import numpy as np
import faiss
from typing import List, Tuple, Dict, Optional
from tose_print.core.config import IndexingConfig


class FAISSIndexer:
    """Provides exact and approximate nearest neighbor search over TOSE-Print vectors."""

    def __init__(self, dimension: int = 325, config: IndexingConfig = None):
        self.dimension = dimension
        self.config = config or IndexingConfig()

        # Exact brute-force index for ground truth
        self.exact_index = faiss.IndexFlatIP(self.dimension)

        # HNSW sub-linear graph index
        self.hnsw_index = faiss.IndexHNSWFlat(self.dimension, self.config.hnsw_m, faiss.METRIC_INNER_PRODUCT)
        self.hnsw_index.hnsw.efConstruction = self.config.hnsw_ef_construction
        self.hnsw_index.hnsw.efSearch = self.config.hnsw_ef_search

        self.ids: List[str] = []
        self.is_built = False

    def add(self, vectors: np.ndarray, ids: List[str]):
        """
        Adds vectors to both exact and HNSW indexes.
        vectors: shape (N, dimension), assumed unit L2-normalized.
        """
        vecs = np.ascontiguousarray(vectors, dtype=np.float32)
        # Ensure L2 normalized
        faiss.normalize_L2(vecs)

        self.exact_index.add(vecs)
        self.hnsw_index.add(vecs)
        self.ids.extend(ids)
        self.is_built = True

    def search_exact(self, query_vectors: np.ndarray, top_k: int = 10) -> Tuple[np.ndarray, np.ndarray, float]:
        """Performs exact brute-force search. Returns: distances, indices, search_time_ms."""
        q = np.ascontiguousarray(query_vectors, dtype=np.float32)
        faiss.normalize_L2(q)

        start = time.perf_counter()
        distances, indices = self.exact_index.search(q, top_k)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        return distances, indices, elapsed_ms

    def search_hnsw(self, query_vectors: np.ndarray, top_k: int = 10) -> Tuple[np.ndarray, np.ndarray, float]:
        """Performs sub-linear HNSW graph search. Returns: distances, indices, search_time_ms."""
        q = np.ascontiguousarray(query_vectors, dtype=np.float32)
        faiss.normalize_L2(q)

        start = time.perf_counter()
        distances, indices = self.hnsw_index.search(q, top_k)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        return distances, indices, elapsed_ms

    def benchmark_recall_and_speed(
        self,
        query_vectors: np.ndarray,
        top_k: int = 10
    ) -> Dict[str, float]:
        """
        Benchmarks HNSW against exact linear scan to measure Recall@k and Speedup factor.
        """
        exact_dist, exact_idx, exact_time = self.search_exact(query_vectors, top_k)
        hnsw_dist, hnsw_idx, hnsw_time = self.search_hnsw(query_vectors, top_k)

        # Calculate Recall@1 and Recall@top_k
        n_queries = query_vectors.shape[0]
        recall_1 = sum(1 for i in range(n_queries) if hnsw_idx[i, 0] == exact_idx[i, 0]) / n_queries

        recalls_at_k = []
        for i in range(n_queries):
            gt_set = set(exact_idx[i])
            hnsw_set = set(hnsw_idx[i])
            recalls_at_k.append(len(gt_set.intersection(hnsw_set)) / top_k)
        recall_k = float(np.mean(recalls_at_k))

        speedup = (exact_time / hnsw_time) if hnsw_time > 0 else 1.0

        return {
            "num_queries": n_queries,
            "database_size": len(self.ids),
            "recall_at_1": float(recall_1),
            f"recall_at_{top_k}": recall_k,
            "exact_latency_ms": exact_time / n_queries,
            "hnsw_latency_ms": hnsw_time / n_queries,
            "speedup_factor": float(speedup),
            "hnsw_qps": float(n_queries / (hnsw_time / 1000.0)) if hnsw_time > 0 else 0.0
        }
