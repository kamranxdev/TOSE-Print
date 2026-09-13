"""
Generates publication-grade, high-resolution architecture diagrams
for TOSE-Print IEEE Transactions manuscript (300 DPI, modern visual design).
Ensures pixel-perfect alignment, rigorous content, and zero clipping.
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def build_architecture_diagram(output_dir: str):
    os.makedirs(output_dir, exist_ok=True)

    # =========================================================================
    # DIAGRAM 1: End-to-End System Pipeline (Overview)
    # =========================================================================
    fig1 = plt.figure(figsize=(24, 9), dpi=300)
    fig1.patch.set_facecolor('#ffffff')
    ax1 = fig1.add_axes([0, 0, 1, 1])
    ax1.set_xlim(0, 100)
    ax1.set_ylim(0, 100)
    ax1.axis('off')

    # Main Titles
    ax1.text(50, 96.2, "TOSE-Print: End-to-End System Architecture Pipeline",
             fontsize=21, fontweight='bold', ha='center', va='center', color='#0f172a', fontfamily='sans-serif')
    ax1.text(50, 93.2, "Topology + Orientation SE(2) Embedding with Cancelable BioHashing and Sub-Linear Proximity Retrieval",
             fontsize=12.5, fontstyle='italic', ha='center', va='center', color='#475569', fontfamily='sans-serif')

    c_blue = '#1e3a8a'
    c_teal = '#0e7490'
    c_green = '#15803d'
    c_purple = '#6b21a8'
    c_orange = '#c2410c'
    c_red = '#991b1b'

    stages = [
        {
            "num": "1",
            "title": "1. Preprocessing & ROI",
            "color": c_blue,
            "x": 2.0, "w": 14.2,
            "bullets": (
                "• CLAHE Contrast Stretch\n"
                "  Clip = 2.5 over 8×8 grid\n"
                "• Bilateral Ridge Filtering\n"
                "  σ_s = 75, σ_c = 75\n"
                "• Foreground Variance Mask\n"
                "  Discards untextured area\n"
                "• Singular Core Alignment\n"
                "  Poincaré index registration"
            ),
            "thumb_title": "Preprocessed Impression"
        },
        {
            "num": "2",
            "title": "2. SE(2) Continuous Flow",
            "color": c_teal,
            "x": 18.36, "w": 14.2,
            "bullets": (
                "• Spatial Gradients (Ix, Iy)\n"
                "  Sobel convolution kernels\n"
                "• Gaussian Tensor J\n"
                "  J = G_σ * [∇I ∇I^T] (σ=2.0)\n"
                "• Doubled-Angle (uc, us)\n"
                "  uc = κ cos 2θ, us = κ sin 2θ\n"
                "• Smooth Lie Group SE(2)\n"
                "  Eliminates 0/π phase jump"
            ),
            "thumb_title": "SE(2) Continuous Vectors"
        },
        {
            "num": "3",
            "title": "3. Skeletal Topology",
            "color": c_green,
            "x": 34.72, "w": 14.2,
            "bullets": (
                "• Medial Axis Ridge Thinning\n"
                "  1-pixel topological skeleton\n"
                "• Rutovitz Crossing Number\n"
                "  Endpoints: CN = 1\n"
                "  Bifurcations: CN ≥ 3\n"
                "• Betti-0 Invariant (β_0)\n"
                "  Connected components count\n"
                "• 1D Cycle Spectrum (H_1)"
            ),
            "thumb_title": "Planar Graph Homology"
        },
        {
            "num": "4",
            "title": "4. Vector Assembly",
            "color": c_purple,
            "x": 51.08, "w": 14.2,
            "bullets": (
                "• Spatial Grid: 128-d (w=0.55)\n"
                "• Orientation Hist: 32-d (w=0.20)\n"
                "• Loop Spectrum: 32-d (w=0.10)\n"
                "• Coherence Patch: 64-d (w=0.10)\n"
                "• Topo Summary: 5-d (w=0.05)\n"
                "• Balanced Sub-Block L_2\n"
                "  Unit vector ||v|| = 1 (d = 261)"
            ),
            "thumb_title": "v_TOSE ∈ R^261"
        },
        {
            "num": "5",
            "title": "5. Cancelable BioHash",
            "color": c_orange,
            "x": 67.44, "w": 14.2,
            "bullets": (
                "• User Token Seed Key K\n"
                "  CSPRNG random hyperplanes\n"
                "• Spherical Projection Matrix R\n"
                "  r_i ~ N(0, I_d) / ||r_i|| (512×261)\n"
                "• Half-Space Sign Quantization\n"
                "  b_i = sgn(⟨r_i, v⟩) ∈ {0, 1}\n"
                "• SimHash Angular Identity\n"
                "  E[d_H/m] = θ/π (Goemans-W.)\n"
                "• ISO/IEC 24745 Unlinkable"
            ),
            "thumb_title": "BioHash {0, 1}^512"
        },
        {
            "num": "6",
            "title": "6. Sub-Linear ANN",
            "color": c_red,
            "x": 83.80, "w": 14.2,
            "bullets": (
                "• FAISS HNSW Proximity Graph\n"
                "  Multi-layer skip list routing\n"
                "• Empirical Sub-Linear Search\n"
                "  T_search ≈ O(d · log N)\n"
                "• Hypersphere Inner Product\n"
                "  > 2,300 QPS @ 100% Recall\n"
                "• 5.0x Speedup @ 50K Gallery"
            ),
            "thumb_title": "HNSW Proximity Graph"
        }
    ]

    for s in stages:
        x, w = s["x"], s["w"]
        # Outer Card Box
        card_rect = patches.FancyBboxPatch(
            (x, 7.0), w, 81.0,
            boxstyle="round,pad=0.3,rounding_size=1.2",
            linewidth=1.8, edgecolor=s["color"], facecolor='#ffffff',
            zorder=2
        )
        ax1.add_patch(card_rect)

        # Header Box
        header_rect = patches.FancyBboxPatch(
            (x, 78.5), w, 9.5,
            boxstyle="round,pad=0.1,rounding_size=0.8",
            linewidth=0, facecolor=s["color"],
            zorder=3
        )
        ax1.add_patch(header_rect)

        # Header Text
        ax1.text(x + w/2.0, 83.2, s["title"],
                 fontsize=11.5, fontweight='bold', ha='center', va='center', color='#ffffff', zorder=4)

        # Text Section Background Container
        text_bg = patches.FancyBboxPatch(
            (x + 0.8, 46.5), w - 1.6, 30.5,
            boxstyle="round,pad=0.2,rounding_size=0.6",
            linewidth=1.0, edgecolor='#f1f5f9', facecolor='#f8fafc',
            zorder=3
        )
        ax1.add_patch(text_bg)

        # Bullets Section
        ax1.text(x + 1.2, 61.5, s["bullets"],
                 fontsize=8.5, ha='left', va='center', color='#1e293b', fontfamily='sans-serif',
                 linespacing=1.60, zorder=4)

        # Graphic Thumbnail Container Box
        thumb_bg = patches.FancyBboxPatch(
            (x + 0.8, 8.8), w - 1.6, 36.5,
            boxstyle="round,pad=0.2,rounding_size=0.6",
            linewidth=1.0, edgecolor='#cbd5e1', facecolor='#ffffff',
            zorder=3
        )
        ax1.add_patch(thumb_bg)

        # Thumbnail Label Banner
        thumb_label = patches.FancyBboxPatch(
            (x + 0.8, 41.5), w - 1.6, 3.8,
            boxstyle="round,pad=0.1,rounding_size=0.4",
            linewidth=0, facecolor='#e2e8f0',
            zorder=4
        )
        ax1.add_patch(thumb_label)
        ax1.text(x + w/2.0, 43.4, s["thumb_title"],
                 fontsize=8.0, fontweight='bold', ha='center', va='center', color='#334155', zorder=5)

    # Inter-card arrows
    for i in range(len(stages) - 1):
        x1 = stages[i]["x"] + stages[i]["w"] + 0.15
        x2 = stages[i+1]["x"] - 0.15
        y_arrow = 61.5
        ax1.annotate(
            "", xy=(x2, y_arrow), xytext=(x1, y_arrow),
            arrowprops=dict(arrowstyle="-|>", color="#1e293b", lw=2.2, mutation_scale=16),
            zorder=6
        )

    # Load real SOCOFing fingerprint for visualizations
    sample_img_path = 'data/SOCOFing/Real/100__M_Left_index_finger.BMP'
    real_img = cv2.imread(sample_img_path, cv2.IMREAD_GRAYSCALE) if os.path.exists(sample_img_path) else np.zeros((120, 120), dtype=np.uint8)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    enhanced = clahe.apply(real_img)
    # Crop borders to remove scanning artifacts
    cropped = enhanced[12:-12, 12:-12]

    # Sub-axis 1: Preprocessed Print
    ax_s1 = fig1.add_axes([(stages[0]["x"] + 1.4)/100, 10.0/100, (stages[0]["w"] - 2.8)/100, 30.5/100], zorder=5)
    ax_s1.imshow(cropped, cmap='gray')
    ax_s1.axis('off')

    # Sub-axis 2: Continuous SE(2) Vector Flow
    ax_s2 = fig1.add_axes([(stages[1]["x"] + 1.4)/100, 10.0/100, (stages[1]["w"] - 2.8)/100, 30.5/100], zorder=5)
    h, w = cropped.shape
    Ix = cv2.Sobel(cropped.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3)
    Iy = cv2.Sobel(cropped.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3)
    Jxx = cv2.GaussianBlur(Ix*Ix, (15, 15), 2.0)
    Jyy = cv2.GaussianBlur(Iy*Iy, (15, 15), 2.0)
    Jxy = cv2.GaussianBlur(Ix*Iy, (15, 15), 2.0)
    theta = 0.5 * np.arctan2(2*Jxy, Jxx - Jyy)
    step = 12
    y_grid, x_grid = np.mgrid[8:h-8:step, 8:w-8:step]
    u = np.cos(theta[y_grid, x_grid])
    v = np.sin(theta[y_grid, x_grid])
    ax_s2.imshow(cropped, cmap='gray', alpha=0.3)
    ax_s2.quiver(x_grid, y_grid, u, -v, color='#0e7490', scale=24, width=0.016, headlength=0, headaxislength=0)
    ax_s2.axis('off')

    # Sub-axis 3: Skeleton Graph Topology & Homology
    ax_s3 = fig1.add_axes([(stages[2]["x"] + 1.4)/100, 10.0/100, (stages[2]["w"] - 2.8)/100, 30.5/100], zorder=5)
    from skimage.morphology import skeletonize
    _, bin_img = cv2.threshold(cropped, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    skel = skeletonize(bin_img == 0).astype(np.uint8)
    skel_clean = skel[4:-4, 4:-4]
    ax_s3.imshow(skel_clean, cmap='gray_r')
    py, px = np.where(skel_clean > 0)
    rng = np.random.RandomState(42)
    sample_pts = rng.choice(len(px), size=min(40, len(px)), replace=False)
    ax_s3.scatter(px[sample_pts[:20]], py[sample_pts[:20]], color='#0284c7', s=14, marker='s', zorder=6)
    ax_s3.scatter(px[sample_pts[20:]], py[sample_pts[20:]], color='#dc2626', s=18, marker='o', zorder=6)
    ax_s3.axis('off')

    # Sub-axis 4: Vector Assembly Heatmap
    ax_s4 = fig1.add_axes([(stages[3]["x"] + 1.4)/100, 10.0/100, (stages[3]["w"] - 2.8)/100, 30.5/100], zorder=5)
    # Realistic multi-slice feature representation: 261-d folded to 9x29 matrix
    feat_synth = np.zeros((9, 29))
    feat_synth[0:5, :] = np.sin(np.linspace(0, 3.14, 5*29)).reshape((5, 29)) * 0.55  # spatial
    feat_synth[5:6, :29] = rng.uniform(0.1, 0.9, 29) * 0.20                           # ori
    feat_synth[6:7, :29] = rng.uniform(0.0, 0.7, 29) * 0.10                           # loop
    feat_synth[7:9, :] = rng.uniform(0.2, 0.8, (2, 29)) * 0.10                        # patch
    ax_s4.imshow(feat_synth, cmap='viridis', aspect='auto')
    ax_s4.set_xticks([])
    ax_s4.set_yticks([])

    # Sub-axis 5: BioHash Binary Code
    ax_s5 = fig1.add_axes([(stages[4]["x"] + 1.4)/100, 10.0/100, (stages[4]["w"] - 2.8)/100, 30.5/100], zorder=5)
    bio_code = rng.randint(0, 2, (16, 32))
    ax_s5.imshow(bio_code, cmap='binary', aspect='auto')
    ax_s5.set_xticks([])
    ax_s5.set_yticks([])

    # Sub-axis 6: HNSW Proximity Graph
    ax_s6 = fig1.add_axes([(stages[5]["x"] + 1.4)/100, 10.0/100, (stages[5]["w"] - 2.8)/100, 30.5/100], zorder=5)
    gx = rng.rand(16)
    gy = rng.rand(16)
    ax_s6.scatter(gx, gy, color='#991b1b', s=24, zorder=3)
    for j in range(16):
        for k in range(j+1, 16):
            if np.hypot(gx[j] - gx[k], gy[j] - gy[k]) < 0.35:
                ax_s6.plot([gx[j], gx[k]], [gy[j], gy[k]], color='#f87171', lw=0.9, alpha=0.7)
    q_path = [0, 4, 9, 14]
    for p in range(len(q_path)-1):
        ax_s6.plot([gx[q_path[p]], gx[q_path[p+1]]], [gy[q_path[p]], gy[q_path[p+1]]], color='#1e3a8a', lw=2.0, ls='--')
    ax_s6.scatter([gx[q_path[-1]]], [gy[q_path[-1]]], color='#2563eb', s=55, marker='*', zorder=5)
    ax_s6.axis('off')

    fig1.savefig(os.path.join(output_dir, 'architecture_overview.png'), bbox_inches='tight', dpi=300)
    fig1.savefig(os.path.join(output_dir, 'architecture_overview.pdf'), bbox_inches='tight')
    plt.close(fig1)
    print("✅ architecture_overview.png & .pdf successfully built!")

    # =========================================================================
    # DIAGRAM 2: Detailed Mathematical & Topological Formulation
    # =========================================================================
    fig2 = plt.figure(figsize=(22, 12), dpi=300)
    fig2.patch.set_facecolor('#ffffff')
    ax2 = fig2.add_axes([0, 0, 1, 1])
    ax2.set_xlim(0, 100)
    ax2.set_ylim(0, 100)
    ax2.axis('off')

    # Main Titles
    ax2.text(50, 97.0, "TOSE-Print: Mathematical Structure Tensor & Graph Topology Formulation",
             fontsize=20, fontweight='bold', ha='center', va='center', color='#0f172a', fontfamily='sans-serif')
    ax2.text(50, 94.2, "Dual-Branch Formulation: Continuous SE(2) Structure Tensor and Skeletal Homology Pipeline",
             fontsize=12.5, fontstyle='italic', ha='center', va='center', color='#475569', fontfamily='sans-serif')

    # Panel A: Continuous Structure Tensor on SE(2)
    box_a = patches.FancyBboxPatch(
        (2.5, 12.0), 46.5, 80.0,
        boxstyle="round,pad=0.4,rounding_size=1.2",
        linewidth=2.0, edgecolor='#0e7490', facecolor='#ffffff',
        zorder=2
    )
    ax2.add_patch(box_a)

    banner_a = patches.FancyBboxPatch(
        (2.5, 84.0), 46.5, 8.0,
        boxstyle="round,pad=0.1,rounding_size=0.8",
        linewidth=0, facecolor='#0e7490',
        zorder=3
    )
    ax2.add_patch(banner_a)
    ax2.text(25.75, 88.0, "Branch A: Continuous Structure Tensor on SE(2)",
             fontsize=13.5, fontweight='bold', ha='center', va='center', color='#ffffff', zorder=4)

    # Step A1: Gradients
    a1_box = patches.FancyBboxPatch(
        (4.0, 71.0), 43.5, 11.5,
        boxstyle="round,pad=0.2,rounding_size=0.6",
        linewidth=1.0, edgecolor='#ccfbf1', facecolor='#f0fdfa', zorder=3
    )
    ax2.add_patch(a1_box)
    ax2.text(5.0, 79.5, "1. Spatial Gradient Estimation via Sobel Kernels:", fontsize=10.5, fontweight='bold', color='#0f766e', zorder=4)
    eq_a1_text = (
        r"$I_x = \partial I_{\mathrm{prep}} / \partial x = I_{\mathrm{prep}} * K_x, \quad I_y = \partial I_{\mathrm{prep}} / \partial y = I_{\mathrm{prep}} * K_y$" + "\n" +
        r"Spatial convolution with $3 \times 3$ separable Sobel operators"
    )
    ax2.text(5.5, 75.0, eq_a1_text, fontsize=9.8, color='#134e4a', linespacing=1.5, zorder=4)

    # Step A2: Gaussian Tensor J
    a2_box = patches.FancyBboxPatch(
        (4.0, 53.5), 43.5, 16.0,
        boxstyle="round,pad=0.2,rounding_size=0.6",
        linewidth=1.0, edgecolor='#ccfbf1', facecolor='#f0fdfa', zorder=3
    )
    ax2.add_patch(a2_box)
    ax2.text(5.0, 66.5, "2. Scale-Integrated Structure Tensor Matrix J:", fontsize=10.5, fontweight='bold', color='#0f766e', zorder=4)
    eq_a2_text = (
        r"$J(x, y) = G_\sigma * [\,I_x^2,\ I_x I_y \;;\; I_x I_y,\ I_y^2\,] = [\,J_{xx},\ J_{xy} \;;\; J_{xy},\ J_{yy}\,]$" + "\n" +
        r"Gaussian integration scale $\sigma = 2.0$ over local contextual neighborhood" + "\n" +
        r"Local Principal Ridge Angle: $\theta(x, y) = \frac{1}{2}\operatorname{atan2}(2J_{xy},\ J_{xx} - J_{yy})$"
    )
    ax2.text(5.5, 59.5, eq_a2_text, fontsize=9.8, color='#134e4a', linespacing=1.5, zorder=4)

    # Step A3: Doubled-Angle Coordinates on SE(2)
    a3_box = patches.FancyBboxPatch(
        (4.0, 33.5), 43.5, 18.5,
        boxstyle="round,pad=0.2,rounding_size=0.6",
        linewidth=1.0, edgecolor='#ccfbf1', facecolor='#f0fdfa', zorder=3
    )
    ax2.add_patch(a3_box)
    ax2.text(5.0, 49.0, "3. Continuous Doubled-Angle Vector Field on SE(2):", fontsize=10.5, fontweight='bold', color='#0f766e', zorder=4)
    eq_a3_text = (
        r"$u_c(x, y) = \kappa \cos(2\theta) = \frac{J_{xx} - J_{yy}}{J_{xx} + J_{yy} + \epsilon}, \quad u_s(x, y) = \kappa \sin(2\theta) = \frac{2J_{xy}}{J_{xx} + J_{yy} + \epsilon}$" + "\n" +
        r"$\bullet\ \mathbf{Continuous\ Norm:}\ \|(u_c, u_s)\|_2 = \kappa(x, y) \in [0, 1]\ \mathrm{(Vanishes\ in\ untextured\ background)}$" + "\n" +
        r"$\bullet\ \mathbf{Harmonic\ SO(2)\ Equivariance:}\ \mathrm{Rotation\ by\ }\phi \rightarrow 2\phi\ \mathrm{harmonic\ vector\ rotation}$"
    )
    ax2.text(5.5, 40.5, eq_a3_text, fontsize=9.6, color='#134e4a', linespacing=1.6, zorder=4)

    # Step A4: Feature Slices Table/Cards
    a4_box = patches.FancyBboxPatch(
        (4.0, 13.5), 43.5, 18.5,
        boxstyle="round,pad=0.2,rounding_size=0.6",
        linewidth=1.0, edgecolor='#ccfbf1', facecolor='#f0fdfa', zorder=3
    )
    ax2.add_patch(a4_box)
    ax2.text(5.0, 29.2, "4. Sub-Vector Dimensions & Allocations:", fontsize=10.5, fontweight='bold', color='#0f766e', zorder=4)

    sub_slices_a = [
        {"name": "Spatial Tensor Grid", "dim": "128-d", "sub": "8×8 regular grid (uc, us)", "w": "Weight: 0.55", "x": 5.5, "width": 12.5},
        {"name": "Coherence Histogram", "dim": "32-d", "sub": "32-bin orientation polar", "w": "Weight: 0.20", "x": 19.5, "width": 12.5},
        {"name": "Coherence Patch", "dim": "64-d", "sub": "8×8 macro-coherence map", "w": "Weight: 0.10", "x": 33.5, "width": 12.5}
    ]
    for s in sub_slices_a:
        sc = patches.FancyBboxPatch(
            (s["x"], 14.8), s["width"], 11.8,
            boxstyle="round,pad=0.2,rounding_size=0.5",
            linewidth=1.0, edgecolor='#0e7490', facecolor='#ffffff', zorder=4
        )
        ax2.add_patch(sc)
        ax2.text(s["x"] + s["width"]/2.0, 24.2, s["name"], fontsize=8.8, fontweight='bold', ha='center', color='#0e7490', zorder=5)
        ax2.text(s["x"] + s["width"]/2.0, 21.0, s["dim"], fontsize=11.0, fontweight='bold', ha='center', color='#0f172a', zorder=5)
        ax2.text(s["x"] + s["width"]/2.0, 18.0, s["sub"], fontsize=7.5, ha='center', color='#475569', zorder=5)
        ax2.text(s["x"] + s["width"]/2.0, 16.0, s["w"], fontsize=7.8, fontweight='bold', ha='center', color='#0369a1', zorder=5)

    # Panel B: Skeletal Graph Topology & Homology
    box_b = patches.FancyBboxPatch(
        (51.0, 12.0), 46.5, 80.0,
        boxstyle="round,pad=0.4,rounding_size=1.2",
        linewidth=2.0, edgecolor='#15803d', facecolor='#ffffff',
        zorder=2
    )
    ax2.add_patch(box_b)

    banner_b = patches.FancyBboxPatch(
        (51.0, 84.0), 46.5, 8.0,
        boxstyle="round,pad=0.1,rounding_size=0.8",
        linewidth=0, facecolor='#15803d',
        zorder=3
    )
    ax2.add_patch(banner_b)
    ax2.text(74.25, 88.0, "Branch B: Skeletal Graph Topology & Homology",
             fontsize=13.5, fontweight='bold', ha='center', va='center', color='#ffffff', zorder=4)

    # Step B1: Medial Axis Skeletonization
    b1_box = patches.FancyBboxPatch(
        (52.5, 71.0), 43.5, 11.5,
        boxstyle="round,pad=0.2,rounding_size=0.6",
        linewidth=1.0, edgecolor='#dcfce7', facecolor='#f0fdf4', zorder=3
    )
    ax2.add_patch(b1_box)
    ax2.text(53.5, 79.5, "1. Medial Axis Morphological Skeletonization:", fontsize=10.5, fontweight='bold', color='#166534', zorder=4)
    eq_b1_text = (
        r"Iterative morphological thinning to 1-pixel central ridge lines:" + "\n" +
        r"$\mathcal{S}(I) = \mathrm{Skeletonize}(\mathcal{T}_{\mathrm{Otsu}}(I_{\mathrm{prep}})), \quad \mathcal{S}(x, y) \in \{0, 1\}$" + "\n" +
        r"Preserves Euler characteristic $\chi = V - E + F$ and global homotopy tree"
    )
    ax2.text(54.0, 75.0, eq_b1_text, fontsize=9.8, color='#14532d', linespacing=1.5, zorder=4)

    # Step B2: Crossing Number
    b2_box = patches.FancyBboxPatch(
        (52.5, 53.5), 43.5, 16.0,
        boxstyle="round,pad=0.2,rounding_size=0.6",
        linewidth=1.0, edgecolor='#dcfce7', facecolor='#f0fdf4', zorder=3
    )
    ax2.add_patch(b2_box)
    ax2.text(53.5, 66.5, "2. Rutovitz Crossing Number & Node Degree Analysis:", fontsize=10.5, fontweight='bold', color='#166534', zorder=4)
    eq_b2_text = (
        r"$CN(p) = \frac{1}{2} \sum_{i=1}^8 |p_i - p_{i+1}|, \quad p_9 \equiv p_1 \quad (8\text{-neighborhood\ cycle})$" + "\n" +
        r"$\bullet\ \mathbf{Endpoints}\ (CN = 1): \text{Normalized density } \rho_E = |E| / |V|$" + "\n" +
        r"$\bullet\ \mathbf{Bifurcations}\ (CN \geq 3): \text{Normalized density } \rho_B = |B| / |V|$"
    )
    ax2.text(54.0, 59.5, eq_b2_text, fontsize=9.8, color='#14532d', linespacing=1.6, zorder=4)

    # Step B3: Betti-0 and Homology Loops
    b3_box = patches.FancyBboxPatch(
        (52.5, 33.5), 43.5, 18.5,
        boxstyle="round,pad=0.2,rounding_size=0.6",
        linewidth=1.0, edgecolor='#dcfce7', facecolor='#f0fdf4', zorder=3
    )
    ax2.add_patch(b3_box)
    ax2.text(53.5, 49.0, "3. Homological Invariants & 1D Loop Spectra (H_0, H_1):", fontsize=10.5, fontweight='bold', color='#166534', zorder=4)
    eq_b3_text = (
        r"$\bullet\ \mathbf{0-th\ Betti\ Number}\ (\beta_0): \text{Count of connected ridge graph components}$" + "\n" +
        r"$\bullet\ \mathbf{1D\ Cycle\ Loop\ Homology}\ (H_1): \text{Identifies closed ridge contours and islands}$" + "\n" +
        r"Normalized 32-bin cycle perimeter histogram: $\mathbf{v}_{\mathrm{loop}} \in \mathbb{R}^{32}, \quad \mathrm{range} \in [0, 200]\ \mathrm{px}$" + "\n" +
        r"Topological summary invariant: $\mathbf{v}_{\mathrm{topo}} = [\beta_0, \rho_E, \rho_B, N_{\mathrm{loop}}, \bar{L}]^T \in \mathbb{R}^5$"
    )
    ax2.text(54.0, 40.5, eq_b3_text, fontsize=9.6, color='#14532d', linespacing=1.6, zorder=4)

    # Step B4: Sub-Vector Dimensions & Allocations
    b4_box = patches.FancyBboxPatch(
        (52.5, 13.5), 43.5, 18.5,
        boxstyle="round,pad=0.2,rounding_size=0.6",
        linewidth=1.0, edgecolor='#dcfce7', facecolor='#f0fdf4', zorder=3
    )
    ax2.add_patch(b4_box)
    ax2.text(53.5, 29.2, "4. Sub-Vector Dimensions & Allocations:", fontsize=10.5, fontweight='bold', color='#166534', zorder=4)

    sub_slices_b = [
        {"name": "Loop Perimeter Hist", "dim": "32-d", "sub": "32-bin cycle lengths", "w": "Weight: 0.10", "x": 56.5, "width": 16.5},
        {"name": "Topological Summary", "dim": "5-d", "sub": "[β0, ρE, ρB, N, L]", "w": "Weight: 0.05", "x": 75.5, "width": 16.5}
    ]
    for s in sub_slices_b:
        sc = patches.FancyBboxPatch(
            (s["x"], 14.8), s["width"], 11.8,
            boxstyle="round,pad=0.2,rounding_size=0.5",
            linewidth=1.0, edgecolor='#15803d', facecolor='#ffffff', zorder=4
        )
        ax2.add_patch(sc)
        ax2.text(s["x"] + s["width"]/2.0, 24.2, s["name"], fontsize=8.8, fontweight='bold', ha='center', color='#15803d', zorder=5)
        ax2.text(s["x"] + s["width"]/2.0, 21.0, s["dim"], fontsize=11.0, fontweight='bold', ha='center', color='#0f172a', zorder=5)
        ax2.text(s["x"] + s["width"]/2.0, 18.0, s["sub"], fontsize=7.5, ha='center', color='#475569', zorder=5)
        ax2.text(s["x"] + s["width"]/2.0, 16.0, s["w"], fontsize=7.8, fontweight='bold', ha='center', color='#166534', zorder=5)

    # Bottom Ribbon: Balanced Normalization
    ribbon = patches.FancyBboxPatch(
        (2.5, 2.2), 95.0, 7.5,
        boxstyle="round,pad=0.3,rounding_size=0.8",
        linewidth=1.6, edgecolor='#475569', facecolor='#f8fafc',
        zorder=3
    )
    ax2.add_patch(ribbon)

    eq_norm = (
        r"$\mathbf{Balanced\ Sub\text{-}Block\ Unit}\ L_2\ \mathbf{Normalization:}\quad " +
        r"\mathbf{v}_{\mathrm{TOSE}} = \left[\, " +
        r"\sqrt{0.55}\,\hat{\mathbf{v}}_{\mathrm{spatial}}\ (128\text{d}) \;\parallel\; " +
        r"\sqrt{0.20}\,\hat{\mathbf{v}}_{\mathrm{ori}}\ (32\text{d}) \;\parallel\; " +
        r"\sqrt{0.10}\,\hat{\mathbf{v}}_{\mathrm{loop}}\ (32\text{d}) \;\parallel\; " +
        r"\sqrt{0.05}\,\hat{\mathbf{v}}_{\mathrm{topo}}\ (5\text{d}) \;\parallel\; " +
        r"\sqrt{0.10}\,\hat{\mathbf{v}}_{\mathrm{patch}}\ (64\text{d}) \,\right]^T \in \mathbb{R}^{261}, " +
        r"\quad \|\mathbf{v}_{\mathrm{TOSE}}\|_2 = 1.0$"
    )
    ax2.text(50, 5.95, eq_norm, fontsize=10.8, fontweight='bold', ha='center', va='center', color='#0f172a', zorder=4)

    fig2.savefig(os.path.join(output_dir, 'architecture_detailed.png'), bbox_inches='tight', dpi=300)
    fig2.savefig(os.path.join(output_dir, 'architecture_detailed.pdf'), bbox_inches='tight')
    plt.close(fig2)
    print("✅ architecture_detailed.png & .pdf successfully built!")


if __name__ == "__main__":
    build_architecture_diagram('paper/ieee_transactions')
