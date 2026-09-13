"""FingerCode baseline: Gabor filterbank orientation feature representation (Jain et al. 2000)."""

import cv2
import numpy as np


class FingerCodeExtractor:
    """Extracts standard 8-direction Gabor filterbank FingerCode."""

    def __init__(self, num_angles: int = 8, frequency: float = 0.15):
        self.num_angles = num_angles
        self.kernels = []
        for i in range(num_angles):
            theta = np.pi * i / num_angles
            kernel = cv2.getGaborKernel((25, 25), 3.0, theta, 1.0 / frequency, 0.5, 0, ktype=cv2.CV_32F)
            self.kernels.append(kernel)

    def extract(self, gray_image: np.ndarray, grid_cells: int = 8) -> np.ndarray:
        """
        Filters image with 8 Gabor orientations and computes standard deviation
        in a spatial grid, producing a fixed-length FingerCode vector.
        """
        h, w = gray_image.shape
        cell_h = h // grid_cells
        cell_w = w // grid_cells

        code = []
        for kernel in self.kernels:
            filtered = cv2.filter2D(gray_image.astype(np.float32), -1, kernel)
            for r in range(grid_cells):
                for c in range(grid_cells):
                    patch = filtered[r * cell_h:(r + 1) * cell_h, c * cell_w:(c + 1) * cell_w]
                    code.append(float(np.std(patch)))

        vec = np.array(code, dtype=np.float32)
        norm = np.linalg.norm(vec) + 1e-8
        return vec / norm

    @staticmethod
    def match(code1: np.ndarray, code2: np.ndarray) -> float:
        """Euclidean distance converted to similarity in [0, 1]."""
        dist = np.linalg.norm(code1 - code2)
        return float(max(0.0, 1.0 - dist / 2.0))
