"""Standard minutiae extraction and spatial-angular alignment matcher."""

import cv2
import numpy as np
from skimage.morphology import skeletonize
from typing import List, Tuple
from tose_print.core.types import MinutiaPoint


class MinutiaeMatcher:
    """Extracts minutiae points and matches them using normalized spatial-angular matching."""

    def extract_minutiae(self, gray_image: np.ndarray, mask: np.ndarray = None, max_points: int = 60) -> List[MinutiaPoint]:
        """Extracts ridge endings and bifurcations using Rutovitz crossing number."""
        _, binary = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        skel = skeletonize(binary == 0).astype(np.uint8)

        if mask is not None:
            # Erode mask slightly to avoid false minutiae on perimeter
            eroded = cv2.erode(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))
            skel = skel * (eroded > 0).astype(np.uint8)

        h, w = skel.shape
        minutiae = []

        coords = np.column_stack(np.where(skel > 0))
        for (y, x) in coords:
            if y <= 1 or y >= h - 2 or x <= 1 or x >= w - 2:
                continue

            # 3x3 neighborhood circular order: p1 to p8
            p = [
                int(skel[y-1, x]), int(skel[y-1, x+1]), int(skel[y, x+1]), int(skel[y+1, x+1]),
                int(skel[y+1, x]), int(skel[y+1, x-1]), int(skel[y, x-1]), int(skel[y-1, x-1])
            ]
            # Crossing number CN = 0.5 * sum |p_i - p_{i+1}|
            cn = 0.5 * sum(abs(p[i] - p[(i + 1) % 8]) for i in range(8))

            if cn == 1:  # Ridge ending
                angle = float(np.arctan2(y - h // 2, x - w // 2) % (2 * np.pi))
                minutiae.append(MinutiaPoint(x=float(x), y=float(y), orientation=angle, minutia_type="ending"))
            elif cn == 3:  # Ridge bifurcation
                angle = float(np.arctan2(y - h // 2, x - w // 2) % (2 * np.pi))
                minutiae.append(MinutiaPoint(x=float(x), y=float(y), orientation=angle, minutia_type="bifurcation"))

        # Sort by distance from center for consistent salient selection
        cx, cy = w / 2.0, h / 2.0
        minutiae.sort(key=lambda m: (m.x - cx)**2 + (m.y - cy)**2)
        return minutiae[:max_points]

    def match(self, minutiae1: List[MinutiaPoint], minutiae2: List[MinutiaPoint], distance_thresh: float = 20.0) -> float:
        """
        Greedy 1-to-1 bipartite matching between minutiae sets.
        Returns match score in [0, 1].
        """
        if not minutiae1 or not minutiae2:
            return 0.0

        matched = 0
        used = set()

        for m1 in minutiae1:
            best_dist = float("inf")
            best_idx = -1

            for j, m2 in enumerate(minutiae2):
                if j in used:
                    continue
                # Minutiae type agreement
                if m1.minutia_type != m2.minutia_type:
                    continue

                # Euclidean distance
                dist = np.hypot(m1.x - m2.x, m1.y - m2.y)
                if dist < distance_thresh and dist < best_dist:
                    best_dist = dist
                    best_idx = j

            if best_idx != -1:
                used.add(best_idx)
                matched += 1

        denom = max(1, min(len(minutiae1), len(minutiae2)))
        return float(min(1.0, matched / denom))
