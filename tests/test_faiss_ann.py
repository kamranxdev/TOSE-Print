"""Unit tests for FAISS sub-linear indexer."""

import numpy as np
import pytest
from tose_print.indexing.faiss_ann import FAISSIndexer


def test_faiss_indexing_and_recall():
    dim = 325
    indexer = FAISSIndexer(dimension=dim)

    rng = np.random.RandomState(42)
    num_gallery = 1000
    gallery_vecs = rng.randn(num_gallery, dim).astype(np.float32)
    ids = [f"finger_{i}" for i in range(num_gallery)]

    indexer.add(gallery_vecs, ids)
    assert indexer.is_built

    # Search with subset of gallery vectors (should achieve 100% Recall@1)
    query_vecs = gallery_vecs[:50]
    bench = indexer.benchmark_recall_and_speed(query_vecs, top_k=5)

    assert bench["recall_at_1"] >= 0.99
    assert bench["speedup_factor"] > 0.0
