import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
"""
Experiment 3: Robustness under Non-Linear Distortion & Alterations (SOCOFing Protocol).
Evaluates identification and verification under:
- Central Rotation (CR)
- Obliteration (Obl)
- Z-Cut (Zcut)
Across Easy, Medium, and Hard distortion severities.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from tose_print.core.config import SystemConfig
from tose_print.features.tose import TOSEExtractor
from tose_print.evaluation.metrics import BiometricMetrics
from experiments.synthetic_generator import generate_synthetic_fingerprint


def run_robustness_experiment(num_subjects: int = 50):
    print("=" * 75)
    print("🛡️ EXPERIMENT 3: ALTERATION & CORRUPTION ROBUSTNESS (SOCOFing Protocol)")
    print("=" * 75)

    config = SystemConfig()
    extractor = TOSEExtractor(config)

    # Gallery: Unaltered Real impressions
    print(f"Generating gallery ({num_subjects} subjects, Real impressions)...")
    gallery = [generate_synthetic_fingerprint(i, alteration_type="Real", seed=i * 10) for i in range(1, num_subjects + 1)]
    gal_vecs = [extractor.extract(g.image) for g in gallery]

    distortion_scenarios = [
        ("Unaltered (Real)", "Real", "None"),
        ("Central Rotation (Easy)", "CR", "Easy"),
        ("Central Rotation (Medium)", "CR", "Medium"),
        ("Central Rotation (Hard)", "CR", "Hard"),
        ("Obliteration (Easy)", "Obl", "Easy"),
        ("Obliteration (Medium)", "Obl", "Medium"),
        ("Obliteration (Hard)", "Obl", "Hard"),
        ("Z-Cut (Medium)", "Zcut", "Medium"),
    ]

    os.makedirs("results", exist_ok=True)
    results = []

    print("\n" + "=" * 80)
    print(f"{'Condition':<30} | {'EER (%)':<10} | {'Rank-1 (%)':<12} | {'Rank-5 (%)':<10} | {'AUC':<8}")
    print("-" * 80)

    latex_rows = []
    condition_names = []
    rank1_scores = []
    eer_scores = []

    for label, alt_type, level in distortion_scenarios:
        probes = [
            generate_synthetic_fingerprint(i, alteration_type=alt_type, alteration_level=level, seed=i * 10 + 1)
            for i in range(1, num_subjects + 1)
        ]
        prob_vecs = [extractor.extract(p.image) for p in probes]

        genuine_scores = []
        impostor_scores = []
        rankings = []
        ground_truth = [p.subject_id for p in probes]

        for i in range(len(prob_vecs)):
            q_subj = probes[i].subject_id
            cand_scores = []
            for j in range(len(gal_vecs)):
                g_subj = gallery[j].subject_id
                s = extractor.match(prob_vecs[i], gal_vecs[j])
                cand_scores.append((s, g_subj))
                if q_subj == g_subj:
                    genuine_scores.append(s)
                else:
                    impostor_scores.append(s)

            sorted_cand = [cand_id for _, cand_id in sorted(cand_scores, key=lambda x: x[0], reverse=True)]
            rankings.append(sorted_cand)

        verif = BiometricMetrics.compute_verification_metrics(genuine_scores, impostor_scores)
        ident = BiometricMetrics.compute_identification_metrics(rankings, ground_truth)

        print(f"{label:<30} | {verif.eer * 100:>8.2f}% | {ident.rank_1 * 100:>10.2f}% | {ident.rank_5 * 100:>8.2f}% | {verif.auc:>6.4f}")
        latex_rows.append(f"{label} & {verif.eer*100:.2f}\\% & {ident.rank_1*100:.2f}\\% & {ident.rank_5*100:.2f}\\% & {verif.auc:.4f} \\\\")

        condition_names.append(label)
        rank1_scores.append(ident.rank_1 * 100)
        eer_scores.append(verif.eer * 100)

    print("=" * 80)

    with open("results/table_alteration_robustness.tex", "w") as f:
        f.write("\\begin{table}[t]\n\\centering\n\\caption{Robustness Evaluation under SOCOFing Alterations}\n\\label{tab:alteration}\n\\begin{tabular}{lcccc}\n\\hline\n")
        f.write("Condition & EER (\\%) & Rank-1 (\\%) & Rank-5 (\\%) & AUC \\\\\n\\hline\n")
        f.write("\n".join(latex_rows) + "\n")
        f.write("\\hline\n\\end{tabular}\n\\end{table}\n")

    # Plot Robustness Bar Chart
    fig, ax1 = plt.subplots(figsize=(10, 6))
    x = np.arange(len(condition_names))
    width = 0.35

    rects1 = ax1.bar(x - width/2, rank1_scores, width, label='Rank-1 Accuracy (%)', color='#2b5c8f')
    ax1.set_ylabel('Rank-1 Accuracy (%)', color='#2b5c8f')
    ax1.set_ylim([0, 110])
    ax1.set_xticks(x)
    ax1.set_xticklabels(condition_names, rotation=35, ha="right", fontsize=9)

    ax2 = ax1.twinx()
    rects2 = ax2.bar(x + width/2, eer_scores, width, label='EER (%)', color='#d95f02')
    ax2.set_ylabel('Equal Error Rate (%)', color='#d95f02')
    ax2.set_ylim([0, 30])

    plt.title("TOSE-Print Robustness across Fingerprint Alterations (SOCOFing Protocol)")
    fig.tight_layout()
    plt.savefig("results/alteration_robustness.pdf")
    plt.close()

    print("✅ Robustness experiment completed! LaTeX table and figure saved to results/.")


if __name__ == "__main__":
    run_robustness_experiment(num_subjects=50)
