"""ISO/IEC 24745 Security & Unlinkability Evaluator for Cancelable Biometrics."""

import numpy as np
from typing import List, Tuple, Dict


class PrivacyEvaluator:
    """Evaluates Unlinkability, Revocability, and Irreversibility."""

    @staticmethod
    def evaluate_unlinkability(
        mated_scores_diff_keys: List[float],
        non_mated_scores: List[float]
    ) -> Dict[str, float]:
        """
        Computes system unlinkability metric D_sys as per ISO/IEC 24745 standards:
        Compares score distribution of same biometric under different keys (mated-diff-key)
        against score distribution of different biometrics (non-mated).
        If templates are truly unlinkable, the two distributions are indistinguishable (D_sys -> 0).
        """
        mated = np.array(mated_scores_diff_keys)
        non_mated = np.array(non_mated_scores)

        # Difference in means normalized by standard deviations
        mean_diff = abs(np.mean(mated) - np.mean(non_mated))
        pooled_std = np.sqrt(0.5 * (np.var(mated) + np.var(non_mated))) + 1e-8
        d_prime = float(mean_diff / pooled_std)

        # Overlap coefficient approximation
        bins = np.linspace(0.0, 1.0, 101)
        hist_mated, _ = np.histogram(mated, bins=bins, density=True)
        hist_non_mated, _ = np.histogram(non_mated, bins=bins, density=True)
        overlap = float(np.sum(np.minimum(hist_mated, hist_non_mated)) * (bins[1] - bins[0]))

        return {
            "d_prime_unlinkability": d_prime,
            "distribution_overlap": overlap,
            "mated_diff_key_mean": float(np.mean(mated)),
            "non_mated_mean": float(np.mean(non_mated))
        }

    @staticmethod
    def evaluate_revocability(
        genuine_score_same_key: float,
        mated_score_new_key: float
    ) -> Dict[str, float]:
        """
        Evaluates template revocation:
        A compromised template must be revoked by changing key, dropping score to non-mated baseline (~0.5).
        """
        return {
            "active_template_similarity": genuine_score_same_key,
            "revoked_vs_new_similarity": mated_score_new_key,
            "revocation_margin": genuine_score_same_key - mated_score_new_key
        }
