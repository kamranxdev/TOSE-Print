"""Fingerprint preprocessing: normalization, CLAHE, segmentation, and bilateral filtering."""

import cv2
import numpy as np
from tose_print.core.config import PreprocessingConfig


class FingerprintEnhancer:
    """Enhances fingerprint ridge clarity and attenuates acquisition noise."""

    def __init__(self, config: PreprocessingConfig = None):
        self.config = config or PreprocessingConfig()
        self.clahe = cv2.createCLAHE(
            clipLimit=self.config.clahe_clip_limit,
            tileGridSize=self.config.clahe_grid_size
        )

    def enhance(self, image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Full enhancement pipeline:
        1. Convert to grayscale if RGB
        2. Resize to standard canonical size (preserves aspect ratio with padding)
        3. Segment foreground mask (variance-based ROI)
        4. CLAHE contrast enhancement
        5. Edge-preserving bilateral filter
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        # Target resize
        th, tw = self.config.target_size
        h, w = gray.shape
        if (h, w) != (th, tw):
            scale = min(th / h, tw / w)
            nh, nw = int(h * scale), int(w * scale)
            resized = cv2.resize(gray, (nw, nh), interpolation=cv2.INTER_AREA)
            canvas = np.full((th, tw), 255, dtype=np.uint8)
            pad_y = (th - nh) // 2
            pad_x = (tw - nw) // 2
            canvas[pad_y:pad_y + nh, pad_x:pad_x + nw] = resized
            gray = canvas

        # Segmentation mask based on local variance
        mask = self.segment_roi(gray)

        # CLAHE enhancement
        equalized = self.clahe.apply(gray)

        # Bilateral filtering for edge-preserving ridge smoothing
        denoised = cv2.bilateralFilter(
            equalized,
            d=self.config.bilateral_d,
            sigmaColor=self.config.bilateral_sigma_color,
            sigmaSpace=self.config.bilateral_sigma_space
        )

        return denoised, mask

    def segment_roi(self, gray: np.ndarray, block_size: int = 16, threshold: float = 100.0) -> np.ndarray:
        """Compute foreground segmentation mask using local block variance."""
        h, w = gray.shape
        mask = np.zeros((h, w), dtype=np.uint8)

        # Compute block standard deviation
        for i in range(0, h, block_size):
            for j in range(0, w, block_size):
                block = gray[i:min(i + block_size, h), j:min(j + block_size, w)]
                if block.size > 0 and np.var(block) > threshold:
                    mask[i:min(i + block_size, h), j:min(j + block_size, w)] = 255

        # Morphological closing and opening to eliminate holes
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        return mask
