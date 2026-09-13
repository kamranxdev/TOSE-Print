"""Statistical significance testing and bootstrap confidence intervals."""

import numpy as np
from typing import List, Tuple, Dict, Callable


class StatisticalAnalysis:
    """Provides publication-grade 95% confidence intervals and hypothesis testing."""

    @staticmethod
    def bootstrap_confidence_interval(
        metric_func: Callable[[np.ndarray, np.ndarray], float],
        sample_a: np.ndarray,
        sample_b: np.ndarray,
        n_bootstraps: int = 1000,
        ci: float = 0.95,
        seed: int = 42
    ) -> Tuple[float, float, float]:
        """
        Computes non-parametric bootstrap confidence interval:
        Returns: (point_estimate, ci_lower, ci_upper)
        """
        rng = np.random.RandomState(seed)
        n = len(sample_a)
        point_est = metric_func(sample_a, sample_b)

        boot_estimates = []
        for _ in range(n_bootstraps):
            indices = rng.randint(0, n, size=n)
            est = metric_func(sample_a[indices], sample_b[indices])
            boot_estimates.append(est)

        boot_arr = np.sort(boot_estimates)
        alpha = (1.0 - ci) / 2.0
        low_idx = int(alpha * n_bootstraps)
        high_idx = int((1.0 - alpha) * n_bootstraps)

        ci_lower = float(boot_arr[low_idx])
        ci_upper = float(boot_arr[high_idx])

        return float(point_est), ci_lower, ci_upper

    @staticmethod
    def mcnemar_test(correct_a: List[bool], correct_b: List[bool]) -> Dict[str, float]:
        """
        McNemar's test for comparing two binary classification models on paired samples.
        b: model A correct, model B incorrect
        c: model A incorrect, model B correct
        statistic = (|b - c| - 1)^2 / (b + c)
        """
        from scipy.stats import chi2

        a_arr = np.array(correct_a, dtype=bool)
        b_arr = np.array(correct_b, dtype=bool)

        b = int(np.sum(a_arr & ~b_arr))  # A better than B
        c = int(np.sum(~a_arr & b_arr))  # B better than A

        if b + c == 0:
            return {"chi2": 0.0, "p_value": 1.0, "b": b, "c": c}

        chi2_stat = ((abs(b - c) - 1.0) ** 2) / (b + c)
        p_val = 1.0 - chi2.cdf(chi2_stat, df=1)

        return {
            "chi2": float(chi2_stat),
            "p_value": float(p_val),
            "a_only_correct": b,
            "b_only_correct": c
        }
