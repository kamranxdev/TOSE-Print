"""
Unified TOSE-Print fixed-length vector extractor (v2.0).
Combines Spatial Structure Tensor fields, Orientation Histograms,
and Skeleton Graph Topology into an indexable unit-normalized representation.
"""

import numpy as np
import cv2
from tose_print.core.config import SystemConfig
from tose_print.core.types import TOSEVector
from tose_print.preprocessing.enhancement import FingerprintEnhancer
from tose_print.features.orientation import OrientationFieldExtractor
from tose_print.features.topology import TopologicalFeatureExtractor


class TOSEExtractor:
    """
    Extracts the unified Topology + Orientation SE(2) Embedding (TOSE-Print).
    Fixed-length, unit L2-normalized vector representation enabling exact
    and approximate sub-linear nearest-neighbor retrieval.
    """

    def __init__(self, config: SystemConfig = None, spatial_grid: int = 8):
        self.config = config or SystemConfig()
        self.spatial_grid = spatial_grid
        self.enhancer = FingerprintEnhancer(self.config.preprocessing)
        self.ori_extractor = OrientationFieldExtractor(self.config.orientation)
        self.topo_extractor = TopologicalFeatureExtractor(self.config.topology)

    def extract(self, image: np.ndarray) -> TOSEVector:
        """
        End-to-end TOSE vector extraction:
        1. Preprocess image with CLAHE and bilateral filtering
        2. Compute continuous structure tensor: Jxx, Jyy, Jxy
        3. Extract 2D spatial orientation tensor grid (grid x grid x 2 = 128 dims)
        4. Extract 32-bin weighted orientation histogram (32 dims)
        5. Extract skeleton topology summary (5 dims) + loop spectrum (32 dims)
        6. Extract spatial coherence patch (grid x grid = 64 dims)
        7. Sub-block unit L2 normalization with optimal weighting -> Total 261 dims
        """
        enhanced, mask = self.enhancer.enhance(image)

        # Structure Tensor
        img_f = enhanced.astype(np.float32)
        Ix = cv2.Sobel(img_f, cv2.CV_32F, 1, 0, ksize=3)
        Iy = cv2.Sobel(img_f, cv2.CV_32F, 0, 1, ksize=3)

        sigma = self.config.orientation.gaussian_sigma
        ksize = int(6 * sigma + 1)
        if ksize % 2 == 0:
            ksize += 1

        Jxx = cv2.GaussianBlur(Ix * Ix, (ksize, ksize), sigma)
        Jyy = cv2.GaussianBlur(Iy * Iy, (ksize, ksize), sigma)
        Jxy = cv2.GaussianBlur(Ix * Iy, (ksize, ksize), sigma)

        den = Jxx + Jyy + 1e-8
        coherence = np.clip(np.sqrt((Jxx - Jyy) ** 2 + 4.0 * (Jxy ** 2)) / den, 0.0, 1.0)
        cos2theta = ((Jxx - Jyy) / den) * coherence
        sin2theta = ((2.0 * Jxy) / den) * coherence

        # 1. Spatial orientation tensor grid (8x8x2 = 128 dims)
        g = self.spatial_grid
        c_grid = cv2.resize(cos2theta, (g, g), interpolation=cv2.INTER_AREA).flatten()
        s_grid = cv2.resize(sin2theta, (g, g), interpolation=cv2.INTER_AREA).flatten()
        spatial_tensor = np.concatenate([c_grid, s_grid]).astype(np.float32)
        spatial_tensor /= (np.linalg.norm(spatial_tensor) + 1e-8)

        # 2. 32-bin global orientation histogram (32 dims)
        theta = 0.5 * np.arctan2(sin2theta, cos2theta) % np.pi
        ori_hist, _ = np.histogram(theta, bins=self.config.orientation.ori_bins, range=(0.0, np.pi), weights=coherence)
        ori_hist = ori_hist.astype(np.float32)
        ori_hist /= (np.linalg.norm(ori_hist) + 1e-8)

        # 3. Topology features (5 summary + 32 loop hist = 37 dims)
        topo_summary, loop_hist = self.topo_extractor.extract(enhanced, mask)
        topo_summary /= (np.linalg.norm(topo_summary) + 1e-8)
        loop_hist /= (np.linalg.norm(loop_hist) + 1e-8)

        # 4. Coherence patch (64 dims)
        coherence_patch = cv2.resize(coherence, (g, g), interpolation=cv2.INTER_AREA).flatten().astype(np.float32)
        coherence_patch /= (np.linalg.norm(coherence_patch) + 1e-8)

        # Balanced sub-block weighting
        # Weights: spatial_tensor (0.55), ori_hist (0.20), loop_hist (0.10), topo_summary (0.05), coherence_patch (0.10)
        w_spatial = 0.55
        w_ori = 0.20
        w_loop = 0.10
        w_topo = 0.05
        w_patch = 0.10

        full_vector = np.concatenate([
            np.sqrt(w_spatial) * spatial_tensor,       # 128 dims
            np.sqrt(w_ori) * ori_hist,                 # 32 dims
            np.sqrt(w_loop) * loop_hist,               # 32 dims
            np.sqrt(w_topo) * topo_summary,            # 5 dims
            np.sqrt(w_patch) * coherence_patch         # 64 dims
        ]).astype(np.float32)                          # Total = 261 dims

        full_vector /= (np.linalg.norm(full_vector) + 1e-8)

        return TOSEVector(
            vector=full_vector,
            ori_hist=ori_hist,
            loop_hist=loop_hist,
            topo_summary=topo_summary,
            coherence_patch=coherence_patch
        )

    @staticmethod
    def match(vec_a: TOSEVector, vec_b: TOSEVector) -> float:
        """Cosine similarity between unit L2-normalized TOSE vectors."""
        return float(np.dot(vec_a.vector, vec_b.vector))
