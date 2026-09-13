import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
"""
Experiment 5: ISO/IEC 24745 Privacy, Unlinkability & Revocability Evaluation.
Validates the Cancelable BioHashing template transformation:
1. Baseline accuracy retention (Raw TOSE vs. Protected Template)
2. Revocability margin (compromised key vs. new key)
3. Quantitative Unlinkability (D_sys and score distribution overlap)
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from tose_print.core.config import SystemConfig, PrivacyConfig
from tose_print.features.tose import TOSEExtractor
from tose_print.privacy.biohashing import BioHasher
from tose_print.privacy.security_eval import PrivacyEvaluator
from tose_print.evaluation.metrics import BiometricMetrics
from experiments.synthetic_generator import generate_synthetic_fingerprint


def run_privacy_benchmark(num_subjects: int = 50):
    print("=" * 75)
    print("🔒 EXPERIMENT 5: ISO/IEC 24745 CANCELABLE BIOMETRICS & UNLINKABILITY")
    print("=" * 75)

    config = SystemConfig()
    tose_extractor = TOSEExtractor(config)
    hasher = BioHasher(PrivacyConfig(projection_dim=512, threshold=0.0))

    gallery = [generate_synthetic_fingerprint(i, seed=i * 10) for i in range(1, num_subjects + 1)]
    probes = [generate_synthetic_fingerprint(i, seed=i * 10 + 1) for i in range(1, num_subjects + 1)]

    gal_tose = [tose_extractor.extract(g.image) for g in gallery]
    prob_tose = [tose_extractor.extract(p.image) for p in probes]

    # 1. Unprotected Raw TOSE Scores
    raw_gen = []
    raw_imp = []
    for i in range(len(probes)):
        for j in range(len(gallery)):
            s = tose_extractor.match(prob_tose[i], gal_tose[j])
            if probes[i].subject_id == gallery[j].subject_id:
                raw_gen.append(s)
            else:
                raw_imp.append(s)

    raw_metrics = BiometricMetrics.compute_verification_metrics(raw_gen, raw_imp)

    # 2a. Protected BioHash Templates (Normal Operation / Different Keys for Impostors)
    keys = {s_id: f"user_token_secret_{s_id}" for s_id in range(1, num_subjects + 1)}
    gal_bio = [hasher.generate_template(v, keys[gallery[i].subject_id]) for i, v in enumerate(gal_tose)]
    prob_bio = [hasher.generate_template(v, keys[probes[i].subject_id]) for i, v in enumerate(prob_tose)]

    prot_gen = []
    prot_imp_diff_keys = []
    for i in range(len(probes)):
        for j in range(len(gallery)):
            s = hasher.match_templates(prob_bio[i], gal_bio[j])
            if probes[i].subject_id == gallery[j].subject_id:
                prot_gen.append(s)
            else:
                prot_imp_diff_keys.append(s)

    prot_metrics_diff = BiometricMetrics.compute_verification_metrics(prot_gen, prot_imp_diff_keys)

    # 2b. Stolen-Token Scenario (Worst-Case / Same Key for Impostor)
    # Impostor presents un-mated biometric evaluated against victim's key
    prot_imp_same_key = []
    for i in range(len(probes)):
        victim_id = gallery[i].subject_id
        victim_key = keys[victim_id]
        gal_victim_bio = gal_bio[i]
        for j in range(len(probes)):
            if probes[j].subject_id != victim_id:
                # Impostor j projected with victim i's key
                imp_bio = hasher.generate_template(prob_tose[j], victim_key)
                s = hasher.match_templates(imp_bio, gal_victim_bio)
                prot_imp_same_key.append(s)

    stolen_token_metrics = BiometricMetrics.compute_verification_metrics(prot_gen, prot_imp_same_key)

    # 3. Unlinkability & Revocability Simulation
    # Generate mated samples with DIFFERENT keys
    mated_diff_keys = []
    for i in range(len(probes)):
        s_id = probes[i].subject_id
        t_key1 = hasher.generate_template(gal_tose[i], f"key_iteration_1_{s_id}")
        t_key2 = hasher.generate_template(prob_tose[i], f"key_iteration_2_{s_id}")
        mated_diff_keys.append(hasher.match_templates(t_key1, t_key2))

    unlink_eval = PrivacyEvaluator.evaluate_unlinkability(mated_diff_keys, prot_imp_diff_keys)

    # Output comparison
    os.makedirs("results", exist_ok=True)
    print("\n" + "=" * 95)
    print(f"{'Scenario / Operational Condition':<42} | {'EER (%)':<10} | {'FMR100 (%)':<12} | {'AUC':<8} | {'D_sys':<8}")
    print("-" * 95)
    print(f"{'Raw TOSE (Unprotected Baseline)':<42} | {raw_metrics.eer * 100:>8.2f}% | {raw_metrics.fmr100 * 100:>10.2f}% | {raw_metrics.auc:>6.4f} | {'N/A':>6}")
    print(f"{'Stolen-Token (Worst-Case Key Compromise)':<42} | {stolen_token_metrics.eer * 100:>8.2f}% | {stolen_token_metrics.fmr100 * 100:>10.2f}% | {stolen_token_metrics.auc:>6.4f} | {'N/A':>6}")
    print(f"{'Stolen-Biometric (Normal Token Operation)':<42} | {prot_metrics_diff.eer * 100:>8.2f}% | {prot_metrics_diff.fmr100 * 100:>10.2f}% | {prot_metrics_diff.auc:>6.4f} | {unlink_eval['d_prime_unlinkability']:>6.4f}")
    print("=" * 95)

    print("\n🔒 ISO/IEC 24745 UNLINKABILITY & REVOCABILITY METRICS:")
    print(f"   • Mated (Different Keys) Mean Score: {unlink_eval['mated_diff_key_mean']:.4f}")
    print(f"   • Non-Mated (Impostor) Mean Score:   {unlink_eval['non_mated_mean']:.4f}")
    print(f"   • Unlinkability D_sys (ideal -> 0.0): {unlink_eval['d_prime_unlinkability']:.4f}")
    print(f"   • Distribution Overlap Coefficient:  {unlink_eval['distribution_overlap'] * 100:.2f}%")

    # Plot Unlinkability Score Distributions
    plt.figure(figsize=(9, 5))
    bins = np.linspace(0.35, 1.0, 60)
    plt.hist(prot_gen, bins=bins, alpha=0.6, color='green', density=True, label='Genuine (Same Key)')
    plt.hist(prot_imp_diff_keys, bins=bins, alpha=0.6, color='red', density=True, label='Non-Mated (Impostor)')
    plt.hist(mated_diff_keys, bins=bins, alpha=0.6, color='blue', density=True, label='Mated (Different Keys / Revoked)')
    plt.axvline(0.5, color='gray', linestyle='--', alpha=0.7, label='Theoretical Orthogonal (0.50)')

    plt.xlabel('Normalized BioHash Hamming Similarity', fontsize=12)
    plt.ylabel('Probability Density', fontsize=12)
    plt.title('ISO/IEC 24745 Unlinkability & Revocability Score Distributions', fontsize=13)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("results/unlinkability_distributions.pdf")
    plt.close()

    # LaTeX Table
    with open("results/table_privacy_evaluation.tex", "w") as f:
        f.write("\\begin{table}[t]\n\\centering\n\\caption{Verification Accuracy and Privacy Analysis Under Dual Threat Scenarios}\n\\label{tab:privacy}\n\\begin{tabular}{lcccc}\n\\hline\n")
        f.write("Scenario / Operational Condition & EER (\\%) & FMR100 (\\%) & AUC & Unlinkability ($D_{sys}$) \\\\\n\\hline\n")
        f.write(f"Raw TOSE (Unprotected Baseline) & {raw_metrics.eer*100:.2f}\\% & {raw_metrics.fmr100*100:.2f}\\% & {raw_metrics.auc:.4f} & N/A \\\\\n")
        f.write(f"Stolen-Token (Worst-Case Key Compromise) & {stolen_token_metrics.eer*100:.2f}\\% & {stolen_token_metrics.fmr100*100:.2f}\\% & {stolen_token_metrics.auc:.4f} & N/A \\\\\n")
        f.write(f"\\textbf{{Stolen-Biometric (Normal Token Operation)}} & \\textbf{{{prot_metrics_diff.eer*100:.2f}\\%}} & \\textbf{{{prot_metrics_diff.fmr100*100:.2f}\\%}} & \\textbf{{{prot_metrics_diff.auc:.4f}}} & \\textbf{{{unlink_eval['d_prime_unlinkability']:.4f}}} \\\\\n")
        f.write("\\hline\n\\end{tabular}\n\\end{table}\n")

    print("✅ Privacy evaluation completed! LaTeX table and unlinkability distributions saved to results/.")


if __name__ == "__main__":
    run_privacy_benchmark(num_subjects=50)
