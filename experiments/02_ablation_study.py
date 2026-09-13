import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
"""
Experiment 2: Comprehensive Ablation Study (IEEE TIFS Standard).
Dissects individual contributions of each sub-component:
1. Orientation Histogram only (d = 32)
2. Skeleton Topology only (d = 37: 5 summary + 32 loop hist)
3. Spatial Coherence Patch only (d = 256)
4. Full TOSE-Print (d = 325)
5. TOSE-Print + Spectral Phase Harmonics (d = 341)
"""

import os
import time
import numpy as np
from tose_print.core.config import SystemConfig
from tose_print.features.tose import TOSEExtractor
from tose_print.features.spectral import SpectralPhaseHarmonicExtractor
from tose_print.evaluation.metrics import BiometricMetrics
from experiments.synthetic_generator import generate_synthetic_fingerprint


def run_ablation(num_subjects: int = 50):
    print("=" * 75)
    print("🔬 EXPERIMENT 2: ABLATION STUDY OF TOSE-PRINT CONSTITUENTS")
    print("=" * 75)

    config = SystemConfig()
    tose_extractor = TOSEExtractor(config)
    spectral_extractor = SpectralPhaseHarmonicExtractor(num_harmonics=8)

    gallery = [generate_synthetic_fingerprint(i, seed=i * 10) for i in range(1, num_subjects + 1)]
    probes = [generate_synthetic_fingerprint(i, seed=i * 10 + 1) for i in range(1, num_subjects + 1)]

    # Measure extraction times
    t0 = time.perf_counter()
    tose_gal = [tose_extractor.extract(g.image) for g in gallery]
    t_extract_ms = (time.perf_counter() - t0) * 1000.0 / len(gallery)

    tose_prob = [tose_extractor.extract(p.image) for p in probes]
    spec_gal = [spectral_extractor.extract(g.image) for g in gallery]
    spec_prob = [spectral_extractor.extract(p.image) for p in probes]

    # Sub-feature definitions
    ablation_modes = {
        "Orientation Only": {
            "gal": [v.ori_hist / (np.linalg.norm(v.ori_hist) + 1e-8) for v in tose_gal],
            "prob": [v.ori_hist / (np.linalg.norm(v.ori_hist) + 1e-8) for v in tose_prob],
            "dim": 32
        },
        "Topology Only": {
            "gal": [np.concatenate([v.topo_summary, v.loop_hist]) / (np.linalg.norm(np.concatenate([v.topo_summary, v.loop_hist])) + 1e-8) for v in tose_gal],
            "prob": [np.concatenate([v.topo_summary, v.loop_hist]) / (np.linalg.norm(np.concatenate([v.topo_summary, v.loop_hist])) + 1e-8) for v in tose_prob],
            "dim": 37
        },
        "Coherence Patch Only": {
            "gal": [v.coherence_patch / (np.linalg.norm(v.coherence_patch) + 1e-8) for v in tose_gal],
            "prob": [v.coherence_patch / (np.linalg.norm(v.coherence_patch) + 1e-8) for v in tose_prob],
            "dim": 256
        },
        "Full TOSE-Print (Proposed)": {
            "gal": [v.vector for v in tose_gal],
            "prob": [v.vector for v in tose_prob],
            "dim": 325
        },
        "TOSE-Print + Spectral Harmonics": {
            "gal": [np.concatenate([v.vector, s]) / (np.linalg.norm(np.concatenate([v.vector, s])) + 1e-8) for v, s in zip(tose_gal, spec_gal)],
            "prob": [np.concatenate([v.vector, s]) / (np.linalg.norm(np.concatenate([v.vector, s])) + 1e-8) for v, s in zip(tose_prob, spec_prob)],
            "dim": 341
        }
    }

    ground_truth = [p.subject_id for p in probes]
    os.makedirs("results", exist_ok=True)

    print("\n" + "=" * 80)
    print(f"{'Feature Configuration':<32} | {'Dim':<5} | {'EER (%)':<10} | {'FMR100 (%)':<12} | {'Rank-1 (%)':<10}")
    print("-" * 80)

    latex_rows = []
    for name, data in ablation_modes.items():
        dim = data["dim"]
        g_vecs = data["gal"]
        p_vecs = data["prob"]

        genuine_scores = []
        impostor_scores = []
        rankings = []

        for i in range(len(p_vecs)):
            q_subj = probes[i].subject_id
            cand_scores = []
            for j in range(len(g_vecs)):
                g_subj = gallery[j].subject_id
                score = float(np.dot(p_vecs[i], g_vecs[j]))
                cand_scores.append((score, g_subj))
                if q_subj == g_subj:
                    genuine_scores.append(score)
                else:
                    impostor_scores.append(score)

            sorted_cand = [cand_id for _, cand_id in sorted(cand_scores, key=lambda x: x[0], reverse=True)]
            rankings.append(sorted_cand)

        verif = BiometricMetrics.compute_verification_metrics(genuine_scores, impostor_scores)
        ident = BiometricMetrics.compute_identification_metrics(rankings, ground_truth)

        print(f"{name:<32} | {dim:<5} | {verif.eer * 100:>8.2f}% | {verif.fmr100 * 100:>10.2f}% | {ident.rank_1 * 100:>8.2f}%")
        latex_rows.append(f"{name} & {dim} & {verif.eer*100:.2f}\\% & {verif.fmr100*100:.2f}\\% & {ident.rank_1*100:.2f}\\% \\\\")

    print("=" * 80)
    print(f"⚡ Average extraction latency: {t_extract_ms:.2f} ms per fingerprint")

    with open("results/table_ablation_study.tex", "w") as f:
        f.write("\\begin{table}[t]\n\\centering\n\\caption{Ablation Analysis of TOSE-Print Sub-Feature Invariants}\n\\label{tab:ablation}\n\\begin{tabular}{lcccc}\n\\hline\n")
        f.write("Configuration & Dim & EER (\\%) & FMR100 (\\%) & Rank-1 (\\%) \\\\\n\\hline\n")
        f.write("\n".join(latex_rows) + "\n")
        f.write("\\hline\n\\end{tabular}\n\\end{table}\n")

    print("✅ Ablation study completed successfully! LaTeX table saved to results/table_ablation_study.tex.")


if __name__ == "__main__":
    run_ablation(num_subjects=50)
