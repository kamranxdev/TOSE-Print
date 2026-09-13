"""Unit tests for Cancelable BioHashing."""

import numpy as np
import pytest
from tose_print.core.config import PrivacyConfig
from tose_print.privacy.biohashing import BioHasher


def test_biohashing_properties():
    cfg = PrivacyConfig(projection_dim=512, threshold=0.0)
    hasher = BioHasher(cfg)

    rng = np.random.RandomState(42)
    # Unit vectors
    v1 = rng.randn(325).astype(np.float32)
    v1 /= np.linalg.norm(v1)

    # Genuine intra-class sample: correlated with cosine ~ 0.98
    noise = rng.randn(325).astype(np.float32)
    noise /= np.linalg.norm(noise)
    v1_perturbed = 0.98 * v1 + 0.20 * noise
    v1_perturbed /= np.linalg.norm(v1_perturbed)

    # Impostor: independent random unit vector
    v2 = rng.randn(325).astype(np.float32)
    v2 /= np.linalg.norm(v2)

    user_token_a = "user_secret_token_123"
    user_token_b = "user_secret_token_revoked_456"

    # Same user token: intra-class should match with high similarity (> 0.85)
    t1 = hasher.generate_template(v1, user_token_a)
    t1_pert = hasher.generate_template(v1_perturbed, user_token_a)
    genuine_sim = hasher.match_templates(t1, t1_pert)
    assert genuine_sim > 0.85, f"Expected genuine similarity > 0.85, got {genuine_sim}"

    # Same user token: inter-class should have similarity near ~0.50
    t2 = hasher.generate_template(v2, user_token_a)
    impostor_sim = hasher.match_templates(t1, t2)
    assert 0.40 <= impostor_sim <= 0.60, f"Expected impostor similarity ~0.50, got {impostor_sim}"

    # Revocability: same fingerprint under DIFFERENT keys should be uncorrelated (~0.50)
    t1_new_key = hasher.generate_template(v1, user_token_b)
    revoked_sim = hasher.match_templates(t1, t1_new_key)
    assert 0.40 <= revoked_sim <= 0.60, f"Expected revoked similarity ~0.50, got {revoked_sim}"
