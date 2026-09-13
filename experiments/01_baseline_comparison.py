import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
"""
Experiment 1: Baseline Comparison under Identical Protocol.
Compares TOSE-Print against:
1. Standard Minutiae Matcher
2. Traditional FingerCode (Gabor filterbank)
3. Keypoint Matching (SIFT)
Produces ROC curves, EER, and Rank-1/5/10 identification accuracies.
"""

import os
import argparse
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from tose_print.core.config import SystemConfig
from tose_print.features.tose import TOSEExtractor
from tose_print.baselines.minutiae import MinutiaeMatcher
from tose_print.baselines.fingercode import FingerCodeExtractor
from tose_print.baselines.keypoint_baselines import KeypointMatcher
from tose_print.evaluation.metrics import BiometricMetrics
from experiments.synthetic_generator import generate_synthetic_fingerprint


def run_experiment(num_subjects: int = 60, seed: int = 42):
    print("=" * 75)
    print("📊 EXPERIMENT 1: COMPARATIVE BASELINE EVALUATION (TIFS Standard)")
    print("=" * 75)

    config = SystemConfig()
    tose_extractor = TOSEExtractor(config)
    minutiae_matcher = MinutiaeMatcher()
    fingercode_extractor = FingerCodeExtractor()
    sift_matcher = KeypointMatcher(method="sift")

    # Generate gallery and probe sets
    print(f"Generating benchmark dataset ({num_subjects} subjects, multi-impression)...")
    gallery = []
    probes = []

    for s_id in range(1, num_subjects + 1):
        # Enrolment impression
        g_sample = generate_synthetic_fingerprint(s_id, finger_name="Left_index", seed=s_id * 10)
        gallery.append(g_sample)

        # Genuine re-test impression
        p_sample = generate_synthetic_fingerprint(s_id, finger_name="Left_index", seed=s_id * 10 + 1)
        probes.append(p_sample)

    print(f"Gallery size: {len(gallery)} | Probe queries: {len(probes)}")

    # Extract all representations
    print("Extracting feature representations across all methods...")
    tose_gallery = [tose_extractor.extract(g.image) for g in tqdm(gallery, desc="TOSE Gallery")]
    tose_probes = [tose_extractor.extract(p.image) for p in tqdm(probes, desc="TOSE Probes")]

    fc_gallery = [fingercode_extractor.extract(g.image) for g in gallery]
    fc_probes = [fingercode_extractor.extract(p.image) for p in probes]

    min_gallery = [minutiae_matcher.extract_minutiae(g.image) for g in gallery]
    min_probes = [minutiae_matcher.extract_minutiae(p.image) for p in probes]

    sift_gallery = [sift_matcher.extract(g.image) for g in gallery]
    sift_probes = [sift_matcher.extract(p.image) for p in probes]

    # Evaluation pairs: Genuine vs Impostor
    methods = ["Minutiae", "FingerCode", "SIFT", "TOSE-Print (Ours)"]
    scores = {m: {"genuine": [], "impostor": [], "rankings": []} for m in methods}

    print("\nRunning pairwise verification and 1:N identification matches...")
    for i in range(len(probes)):
        query_subj = probes[i].subject_id
        # Candidate scores for ranking
        q_scores = {m: [] for m in methods}

        for j in range(len(gallery)):
            gal_subj = gallery[j].subject_id
            is_gen = (query_subj == gal_subj)

            # 1. TOSE-Print
            s_tose = tose_extractor.match(tose_probes[i], tose_gallery[j])
            q_scores["TOSE-Print (Ours)"].append((s_tose, gal_subj))
            if is_gen:
                scores["TOSE-Print (Ours)"]["genuine"].append(s_tose)
            else:
                scores["TOSE-Print (Ours)"]["impostor"].append(s_tose)

            # 2. FingerCode
            s_fc = fingercode_extractor.match(fc_probes[i], fc_gallery[j])
            q_scores["FingerCode"].append((s_fc, gal_subj))
            if is_gen:
                scores["FingerCode"]["genuine"].append(s_fc)
            else:
                scores["FingerCode"]["impostor"].append(s_fc)

            # 3. Minutiae
            s_min = minutiae_matcher.match(min_probes[i], min_gallery[j])
            q_scores["Minutiae"].append((s_min, gal_subj))
            if is_gen:
                scores["Minutiae"]["genuine"].append(s_min)
            else:
                scores["Minutiae"]["impostor"].append(s_min)

            # 4. SIFT
            kp_q, desc_q = sift_probes[i]
            kp_g, desc_g = sift_gallery[j]
            s_sift = sift_matcher.match(desc_q, desc_g, kp_q, kp_g)
            q_scores["SIFT"].append((s_sift, gal_subj))
            if is_gen:
                scores["SIFT"]["genuine"].append(s_sift)
            else:
                scores["SIFT"]["impostor"].append(s_sift)

        for m in methods:
            # Sort descending by score
            sorted_candidates = [cand_id for _, cand_id in sorted(q_scores[m], key=lambda x: x[0], reverse=True)]
            scores[m]["rankings"].append(sorted_candidates)

    ground_truth = [p.subject_id for p in probes]

    # Compute metrics
    os.makedirs("results", exist_ok=True)
    print("\n" + "=" * 85)
    print(f"{'Method':<20} | {'EER (%)':<10} | {'FMR100 (%)':<12} | {'AUC':<8} | {'Rank-1 (%)':<12} | {'Rank-5 (%)':<10}")
    print("-" * 85)

    latex_rows = []
    plt.figure(figsize=(9, 7))

    for m in methods:
        verif = BiometricMetrics.compute_verification_metrics(scores[m]["genuine"], scores[m]["impostor"])
        ident = BiometricMetrics.compute_identification_metrics(scores[m]["rankings"], ground_truth)

        print(f"{m:<20} | {verif.eer * 100:>8.2f}% | {verif.fmr100 * 100:>10.2f}% | {verif.auc:>6.4f} | {ident.rank_1 * 100:>10.2f}% | {ident.rank_5 * 100:>8.2f}%")

        latex_rows.append(
            f"{m} & {verif.eer*100:.2f}\\% & {verif.fmr100*100:.2f}\\% & {verif.auc:.4f} & {ident.rank_1*100:.2f}\\% & {ident.rank_5*100:.2f}\\% \\\\"
        )

        plt.plot(verif.roc_fpr, verif.roc_tpr, label=f"{m} (AUC = {verif.auc:.3f}, EER = {verif.eer*100:.1f}%)", linewidth=2)

    print("=" * 85)

    # Save LaTeX Table
    with open("results/table_baseline_comparison.tex", "w") as f:
        f.write("\\begin{table}[t]\n\\centering\n\\caption{Comparative Performance under Identical Subject-Disjoint Protocol}\n\\label{tab:baseline}\n\\begin{tabular}{lccccc}\n\\hline\n")
        f.write("Method & EER (\\%) & FMR100 (\\%) & AUC & Rank-1 (\\%) & Rank-5 (\\%) \\\\\n\\hline\n")
        f.write("\n".join(latex_rows) + "\n")
        f.write("\\hline\n\\end{tabular}\n\\end{table}\n")

    # Plot ROC
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    plt.xscale('log')
    plt.xlim([1e-3, 1.0])
    plt.ylim([0.4, 1.02])
    plt.xlabel("False Match Rate (FMR)")
    plt.ylabel("True Match Rate (1 - FNMR)")
    plt.title("Biometric Verification ROC Curves (TIFS Benchmark)")
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3, which='both')
    plt.tight_layout()
    plt.savefig("results/roc_baseline_comparison.pdf")
    plt.close()

    print("✅ Baseline experiment completed successfully! LaTeX table and ROC curves saved to results/.")


if __name__ == "__main__":
    run_experiment(num_subjects=50)
