"""SE(2) Canonical Alignment: Core point estimation and orientation normalization."""

import cv2
import numpy as np
from typing import Optional, Tuple


class CanonicalSE2Aligner:
    """Estimates singular core points and canonical orientation for SE(2) invariance."""

    @staticmethod
    def estimate_core_point(orientation: np.ndarray, coherence: np.ndarray, mask: np.ndarray = None) -> Tuple[int, int]:
        """
        Locates the focal center of the fingerprint using Poincaré index approximation
        or maximum orientation curvature.
        Returns: (center_x, center_y)
        """
        h, w = orientation.shape
        if mask is None:
            mask = np.ones((h, w), dtype=np.uint8) * 255

        # Compute curvature of orientation field
        cos2theta = np.cos(2.0 * orientation)
        sin2theta = np.sin(2.0 * orientation)

        # Gradient of orientation vectors
        dcos_y, dcos_x = np.gradient(cos2theta)
        dsin_y, dsin_x = np.gradient(sin2theta)

        curvature = np.sqrt(dcos_x**2 + dcos_y**2 + dsin_x**2 + dsin_y**2) * coherence
        curvature = cv2.GaussianBlur(curvature.astype(np.float32), (15, 15), 3.0)

        # Mask out boundary regions
        eroded_mask = cv2.erode(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31)))
        curvature[eroded_mask == 0] = 0.0

        if curvature.max() > 1e-4:
            max_idx = np.argmax(curvature)
            cy, cx = np.unravel_index(max_idx, (h, w))
            return int(cx), int(cy)
        return w // 2, h // 2

    @staticmethod
    def compute_dominant_orientation(orientation: np.ndarray, coherence: np.ndarray, mask: np.ndarray = None) -> float:
        """
        Computes dominant circular mean orientation weighted by coherence within foreground.
        Returns dominant angle in [0, pi).
        """
        if mask is not None:
            weights = coherence * (mask > 0).astype(np.float32)
        else:
            weights = coherence

        sin_sum = np.sum(weights * np.sin(2.0 * orientation))
        cos_sum = np.sum(weights * np.cos(2.0 * orientation))
        dominant = 0.5 * np.arctan2(sin_sum, cos_sum)
        return float(dominant % np.pi)
