"""SIFT and ORB keypoint feature baselines."""

import cv2
import numpy as np
from typing import Tuple


class KeypointMatcher:
    """Keypoint detection and RANSAC geometric homography matching."""

    def __init__(self, method: str = "sift", n_features: int = 500):
        self.method = method.lower()
        if self.method == "sift":
            self.detector = cv2.SIFT_create(nfeatures=n_features)
            self.matcher = cv2.BFMatcher(cv2.NORM_L2)
        else:
            self.detector = cv2.ORB_create(nfeatures=n_features)
            self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING)

    def extract(self, gray_image: np.ndarray) -> Tuple[list, np.ndarray]:
        kp, desc = self.detector.detectAndCompute(gray_image, None)
        return kp, desc

    def match(self, desc1: np.ndarray, desc2: np.ndarray, kp1: list, kp2: list) -> float:
        if desc1 is None or desc2 is None or len(desc1) < 8 or len(desc2) < 8:
            return 0.0

        matches = self.matcher.knnMatch(desc1, desc2, k=2)
        good = []
        for m_tuple in matches:
            if len(m_tuple) == 2:
                m, n = m_tuple
                if m.distance < 0.75 * n.distance:
                    good.append(m)

        if len(good) < 6:
            return 0.0

        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        _, inlier_mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 4.0)
        if inlier_mask is None:
            return 0.0

        inliers = int(inlier_mask.sum())
        denom = max(1, min(len(kp1), len(kp2)))
        return float(min(1.0, inliers / denom))
