"""Topological feature extraction: skeletonization, Betti numbers, singular degree, and loop spectra."""

import cv2
import numpy as np
from skimage.morphology import skeletonize
from typing import Tuple
from tose_print.core.config import TopologyConfig


class TopologicalFeatureExtractor:
    """Extracts topological invariants from fingerprint ridge skeleton."""

    def __init__(self, config: TopologyConfig = None):
        self.config = config or TopologyConfig()

    def extract(self, gray_image: np.ndarray, mask: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns:
            topo_summary: [beta_0, endpoint_density, bifurcation_density, loop_count, loop_mean_length] (5 dims)
            loop_histogram: 32-bin normalized loop length distribution (32 dims)
        """
        # Binarization via Otsu's thresholding
        _, binary = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        binary = (binary == 0).astype(bool)  # ridges as True (black in original)

        if mask is not None:
            binary = binary & (mask > 0)

        # Thinning / skeletonization
        skel = skeletonize(binary).astype(np.uint8)
        skel_u8 = skel * 255

        h, w = skel.shape
        coords = np.column_stack(np.where(skel > 0))

        if len(coords) == 0:
            return np.zeros(5, dtype=np.float32), np.zeros(self.config.loop_bins, dtype=np.float32)

        skel_set = set(map(tuple, coords.tolist()))

        # 8-neighbor connectivity degree analysis
        endpoints = 0
        bifurcations = 0
        for (y, x) in skel_set:
            deg = 0
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if dx == 0 and dy == 0:
                        continue
                    if (y + dy, x + dx) in skel_set:
                        deg += 1
            if deg == 1:
                endpoints += 1
            elif deg >= 3:
                bifurcations += 1

        total_skel = max(1, len(coords))
        ep_density = float(endpoints) / total_skel
        bif_density = float(bifurcations) / total_skel

        # Connected components (Betti-0 proxy)
        num_labels, _ = cv2.connectedComponents(skel, connectivity=8)
        betti_0 = float(num_labels - 1) / 100.0

        # Loop length spectrum via contour hierarchy
        contours, hierarchy = cv2.findContours(skel_u8, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
        loop_lengths = []
        if hierarchy is not None:
            for idx, cnt in enumerate(contours):
                # Inner contours represent 1D cycles (loops)
                if hierarchy[0][idx][3] != -1:
                    perim = cv2.arcLength(cnt, closed=True)
                    if perim > 10:
                        loop_lengths.append(perim)

        if len(loop_lengths) > 0:
            loop_arr = np.array(loop_lengths, dtype=np.float32)
            loop_count = float(len(loop_lengths)) / 50.0
            loop_mean_len = float(np.mean(loop_arr)) / 100.0
        else:
            loop_arr = np.array([0.0], dtype=np.float32)
            loop_count = 0.0
            loop_mean_len = 0.0

        loop_hist, _ = np.histogram(
            loop_arr,
            bins=self.config.loop_bins,
            range=self.config.loop_range
        )
        loop_hist = loop_hist.astype(np.float32)
        norm = np.sum(loop_hist) + 1e-8
        loop_hist = loop_hist / norm

        topo_summary = np.array([
            betti_0,
            ep_density,
            bif_density,
            loop_count,
            loop_mean_len
        ], dtype=np.float32)

        return topo_summary, loop_hist
