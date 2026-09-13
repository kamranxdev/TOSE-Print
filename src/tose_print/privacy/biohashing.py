"""
Cancelable Biometric Template Generation: Random Projection BioHashing.
Compliant with ISO/IEC 24745 (Revocability, Unlinkability, Irreversibility).
"""

import numpy as np
from typing import Tuple, Union
from tose_print.core.config import PrivacyConfig
from tose_print.core.types import TOSEVector


class BioHasher:
    """
    Transforms continuous TOSE-Print embeddings into cancelable binary representations
    via user-token-seeded random projection and dynamic thresholding.
    """

    def __init__(self, config: PrivacyConfig = None):
        self.config = config or PrivacyConfig()

    def generate_projection_matrix(self, user_key: Union[int, str], input_dim: int) -> np.ndarray:
        """
        Generates projection matrix R in R^{m x d} from user key/token.
        Constructs m independently sampled, normalized random Gaussian hyperplanes
        satisfying the random hyperplane angular distance preservation property.
        """
        if isinstance(user_key, str):
            import hashlib
            seed = int(hashlib.sha256(user_key.encode()).hexdigest()[:8], 16)
        else:
            seed = int(user_key)

        rng = np.random.RandomState(seed)
        m = self.config.projection_dim

        # m independent isotropic Gaussian vectors normalized to the unit sphere S^{d-1}
        R = rng.randn(m, input_dim).astype(np.float32)
        row_norms = np.linalg.norm(R, axis=1, keepdims=True) + 1e-8
        return (R / row_norms).astype(np.float32)

    def generate_template(self, tose_vector: Union[TOSEVector, np.ndarray], user_key: Union[int, str]) -> np.ndarray:
        """
        Generates binary cancelable template:
        1. Project vector: y = R * x
        2. Binarize: b_i = 1 if y_i > threshold else 0
        Returns: 1D uint8 array of length m.
        """
        if isinstance(tose_vector, TOSEVector):
            x = tose_vector.vector
        else:
            x = np.asarray(tose_vector, dtype=np.float32)

        d = x.shape[0]
        R = self.generate_projection_matrix(user_key, d)
        y = np.dot(R, x)

        tau = self.config.threshold
        template = (y > tau).astype(np.uint8)
        return template

    @staticmethod
    def match_templates(template1: np.ndarray, template2: np.ndarray) -> float:
        """
        Computes normalized Hamming similarity:
        similarity = 1 - (Hamming_distance / length) in [0, 1].
        """
        if len(template1) != len(template2):
            raise ValueError("Templates must have identical length for Hamming distance.")
        hamming_dist = np.count_nonzero(template1 != template2)
        similarity = 1.0 - (float(hamming_dist) / float(len(template1)))
        return float(similarity)
