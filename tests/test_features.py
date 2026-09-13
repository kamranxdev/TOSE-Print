"""Unit tests for TOSE feature extraction."""

import numpy as np
import pytest
from tose_print.core.config import SystemConfig
from tose_print.features.tose import TOSEExtractor


def create_synthetic_fingerprint_pattern(h=300, w=300):
    """Creates a deterministic synthetic fingerprint pattern for unit testing."""
    y, x = np.mgrid[0:h, 0:w]
    cx, cy = w // 2, h // 2
    r = np.sqrt((x - cx)**2 + (y - cy)**2)
    theta = np.arctan2(y - cy, x - cx)
    # Ridge frequency pattern
    pattern = 127.0 + 127.0 * np.sin(r * 0.15 + 2.0 * theta)
    return pattern.astype(np.uint8)


def test_tose_extraction():
    config = SystemConfig()
    extractor = TOSEExtractor(config)
    img = create_synthetic_fingerprint_pattern(300, 300)

    tose_vec = extractor.extract(img)

    # Dimensionality check: 32 (ori) + 32 (loop) + 5 (topo) + 256 (coherence patch) = 325
    assert tose_vec.dimension == 261
    assert len(tose_vec.vector) == 261

    # L2 normalization check
    l2_norm = np.linalg.norm(tose_vec.vector)
    assert pytest.approx(l2_norm, abs=1e-5) == 1.0

    # Self match cosine score should be 1.0
    self_score = extractor.match(tose_vec, tose_vec)
    assert pytest.approx(self_score, abs=1e-5) == 1.0
