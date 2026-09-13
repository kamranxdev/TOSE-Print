"""Biometric evaluation metrics: EER, FMR100, FMR1000, ZeroFMR, CMC, and ROC."""

import numpy as np
from sklearn.metrics import roc_curve, auc
from typing import List, Tuple, Dict
from tose_print.core.types import VerificationMetrics, IdentificationMetrics


class BiometricMetrics:
    """Computes ISO/IEC 19795 compliant biometric evaluation metrics."""

    @staticmethod
    def compute_verification_metrics(
        genuine_scores: List[float],
        impostor_scores: List[float]
    ) -> VerificationMetrics:
        """Computes ROC, Equal Error Rate (EER), FMR100, FMR1000, and ZeroFMR."""
        gen = np.array(genuine_scores, dtype=np.float64)
        imp = np.array(impostor_scores, dtype=np.float64)

        y_true = np.concatenate([np.ones_like(gen), np.zeros_like(imp)])
        y_scores = np.concatenate([gen, imp])

        fpr, tpr, thresholds = roc_curve(y_true, y_scores)
        fnr = 1.0 - tpr
        roc_auc = auc(fpr, tpr)

        # Equal Error Rate: point where FPR == FNR
        eer_idx = np.nanargmin(np.abs(fpr - fnr))
        eer = float((fpr[eer_idx] + fnr[eer_idx]) / 2.0)
        eer_thresh = float(thresholds[eer_idx])

        # FMR100: FNMR at FMR = 1% (0.01)
        fmr100_idx = np.where(fpr <= 0.01)[0]
        fmr100 = float(fnr[fmr100_idx[-1]]) if len(fmr100_idx) > 0 else 1.0

        # FMR1000: FNMR at FMR = 0.1% (0.001)
        fmr1000_idx = np.where(fpr <= 0.001)[0]
        fmr1000 = float(fnr[fmr1000_idx[-1]]) if len(fmr1000_idx) > 0 else 1.0

        # ZeroFMR: lowest FNMR for FMR = 0%
        zero_fmr_idx = np.where(fpr == 0.0)[0]
        zero_fmr = float(fnr[zero_fmr_idx[-1]]) if len(zero_fmr_idx) > 0 else 1.0

        return VerificationMetrics(
            eer=eer,
            eer_threshold=eer_thresh,
            fmr100=fmr100,
            fmr1000=fmr1000,
            zero_fmr=zero_fmr,
            auc=float(roc_auc),
            roc_fpr=fpr,
            roc_tpr=tpr,
            thresholds=thresholds
        )

    @staticmethod
    def compute_identification_metrics(
        ranking_matrix: List[List[str]],
        ground_truth_ids: List[str]
    ) -> IdentificationMetrics:
        """
        ranking_matrix: for each query, sorted list of retrieved gallery IDs.
        ground_truth_ids: the true matching gallery ID for each query.
        """
        num_queries = len(ground_truth_ids)
        if num_queries == 0:
            return IdentificationMetrics(0, 0, 0, 0, 0)

        ranks_found = []
        for i in range(num_queries):
            gt = ground_truth_ids[i]
            retrieved = ranking_matrix[i]
            if gt in retrieved:
                rank = retrieved.index(gt) + 1
                ranks_found.append(rank)
            else:
                ranks_found.append(float("inf"))

        ranks_arr = np.array(ranks_found)

        rank_1 = float(np.mean(ranks_arr <= 1))
        rank_5 = float(np.mean(ranks_arr <= 5))
        rank_10 = float(np.mean(ranks_arr <= 10))
        rank_20 = float(np.mean(ranks_arr <= 20))

        # Mean Reciprocal Rank (MRR)
        mrr = float(np.mean([1.0 / r if r < float("inf") else 0.0 for r in ranks_found]))

        max_rank = 20
        cmc_ranks = np.arange(1, max_rank + 1)
        cmc_acc = np.array([np.mean(ranks_arr <= r) for r in cmc_ranks])

        return IdentificationMetrics(
            rank_1=rank_1,
            rank_5=rank_5,
            rank_10=rank_10,
            rank_20=rank_20,
            mrr=mrr,
            cmc_ranks=cmc_ranks,
            cmc_accuracies=cmc_acc
        )
