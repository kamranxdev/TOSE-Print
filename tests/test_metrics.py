"""Unit tests for ISO biometric verification and identification metrics."""

import numpy as np
import pytest
from tose_print.evaluation.metrics import BiometricMetrics


def test_verification_metrics():
    rng = np.random.RandomState(42)
    # Synthetic genuine scores ~ N(0.85, 0.05^2)
    genuine = rng.normal(0.85, 0.05, 1000).clip(0.0, 1.0).tolist()
    # Synthetic impostor scores ~ N(0.35, 0.08^2)
    impostor = rng.normal(0.35, 0.08, 10000).clip(0.0, 1.0).tolist()

    metrics = BiometricMetrics.compute_verification_metrics(genuine, impostor)

    assert metrics.eer < 0.01
    assert metrics.auc > 0.99
    assert metrics.fmr100 < 0.01


def test_identification_metrics():
    # 10 queries, top-5 rankings
    ground_truth = [f"id_{i}" for i in range(10)]
    ranking_matrix = [
        [f"id_{i}", "other_1", "other_2"] if i < 8 else ["other_1", f"id_{i}", "other_2"]
        for i in range(10)
    ]

    metrics = BiometricMetrics.compute_identification_metrics(ranking_matrix, ground_truth)
    assert metrics.rank_1 == 0.8
    assert metrics.rank_5 == 1.0
    assert metrics.mrr > 0.8
