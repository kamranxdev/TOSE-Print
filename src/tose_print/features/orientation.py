"""Orientation field and coherence estimation via structure tensor."""

import cv2
import numpy as np
from typing import Tuple
from tose_print.core.config import OrientationConfig


class OrientationFieldExtractor:
    """
    Computes dense ridge orientation theta in [0, pi) and coherence kappa in [0, 1]
    using the continuous structure tensor (gradient squared averaging).
    """

    def __init__(self, config: OrientationConfig = None):
        self.config = config or OrientationConfig()

    def compute(self, gray_image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Structure tensor algorithm:
        1. Compute spatial gradients: Ix = dI/dx, Iy = dI/dy (Sobel ksize=3)
        2. Form tensor components: Jxx = Ix^2, Jyy = Iy^2, Jxy = Ix*Iy
        3. Smooth with Gaussian window (scale integration)
        4. Orientation: theta = 0.5 * atan2(2*Jxy, Jxx - Jyy)
        5. Coherence: kappa = sqrt((Jxx - Jyy)^2 + 4*Jxy^2) / (Jxx + Jyy + eps)
        """
        img_f = gray_image.astype(np.float32)

        # Gradients
        Ix = cv2.Sobel(img_f, cv2.CV_32F, 1, 0, ksize=3)
        Iy = cv2.Sobel(img_f, cv2.CV_32F, 0, 1, ksize=3)

        Jxx = Ix * Ix
        Jyy = Iy * Iy
        Jxy = Ix * Iy

        # Gaussian smoothing
        sigma = self.config.gaussian_sigma
        ksize = int(6 * sigma + 1)
        if ksize % 2 == 0:
            ksize += 1
        Jxx = cv2.GaussianBlur(Jxx, (ksize, ksize), sigma)
        Jyy = cv2.GaussianBlur(Jyy, (ksize, ksize), sigma)
        Jxy = cv2.GaussianBlur(Jxy, (ksize, ksize), sigma)

        # Continuous orientation
        theta = 0.5 * np.arctan2(2.0 * Jxy, (Jxx - Jyy) + 1e-8)
        theta = theta % np.pi

        # Coherence (anisotropy measure in [0, 1])
        num = np.sqrt((Jxx - Jyy) ** 2 + 4.0 * (Jxy ** 2))
        den = Jxx + Jyy + 1e-8
        coherence = np.clip(num / den, 0.0, 1.0).astype(np.float32)

        return theta.astype(np.float32), coherence

    def compute_orientation_histogram(self, theta: np.ndarray, coherence: np.ndarray, mask: np.ndarray = None) -> np.ndarray:
        """Weighted orientation histogram pooled over foreground in [0, pi)."""
        weights = coherence
        if mask is not None:
            weights = weights * (mask > 0).astype(np.float32)

        hist, _ = np.histogram(
            theta.ravel(),
            bins=self.config.ori_bins,
            range=(0.0, np.pi),
            weights=weights.ravel()
        )
        hist = hist.astype(np.float32)
        norm = np.sum(hist) + 1e-8
        return hist / norm

    def compute_coherence_patch(self, coherence: np.ndarray, patch_size: int = None) -> np.ndarray:
        """
        Spatial pooling of coherence map into a fixed grid (e.g. 16x16 = 256 dimensions).
        Captures global spatial distribution of ridge clarity vs noise.
        """
        ps = patch_size or self.config.coherence_patch_size
        h, w = coherence.shape
        kh = max(1, h // ps)
        kw = max(1, w // ps)
        kx = kw if (kw % 2 == 1) else kw + 1
        ky = kh if (kh % 2 == 1) else kh + 1

        blurred = cv2.blur(coherence, (kx, ky))
        small = cv2.resize(blurred, (ps, ps), interpolation=cv2.INTER_AREA)
        return np.clip(small, 0.0, 1.0).flatten().astype(np.float32)
