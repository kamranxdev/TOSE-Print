"""Synthetic fingerprint generator with realistic ridge flow and perturbations for standalone benchmarking."""

import cv2
import numpy as np
from typing import List, Tuple
from tose_print.core.types import FingerprintSample


def generate_synthetic_fingerprint(
    subject_id: int,
    finger_name: str = "Left_index",
    alteration_type: str = "Real",
    alteration_level: str = "None",
    seed: int = None
) -> FingerprintSample:
    """Generates a realistic synthetic fingerprint impression with parameterized ridge flow."""
    if seed is None:
        seed = subject_id * 100 + hash(finger_name) % 97

    rng = np.random.RandomState(seed)
    h, w = 300, 300
    y, x = np.mgrid[0:h, 0:w].astype(np.float32)

    cx = w / 2.0 + rng.uniform(-15.0, 15.0)
    cy = h / 2.0 + rng.uniform(-15.0, 15.0)

    # Orientation field flow (arch, loop, or whorl based on subject_id % 3)
    pattern_type = subject_id % 3
    dx = x - cx
    dy = y - cy
    r = np.sqrt(dx**2 + dy**2) + 1e-5

    if pattern_type == 0:  # Whorl
        theta = np.arctan2(dy, dx) + np.pi / 2.0
    elif pattern_type == 1:  # Loop
        theta = np.arctan2(dy, dx) * 0.5 + np.pi / 4.0
    else:  # Arch
        theta = (dx / w) * 0.5

    # Base ridge pattern
    freq = 0.12 + rng.uniform(-0.01, 0.01)
    phase = r * freq + np.sin(2.0 * theta) * 2.0
    ridges = 127.0 + 115.0 * np.cos(phase * 2.0 * np.pi)

    # Elliptical foreground mask
    dist_norm = ((x - w/2.0) / 110.0)**2 + ((y - h/2.0) / 130.0)**2
    mask = (dist_norm <= 1.0).astype(np.float32)

    img = ridges * mask + 240.0 * (1.0 - mask)

    # Add alteration if requested
    if alteration_type == "CR":  # Central rotation
        rot_angle = {"Easy": 15, "Medium": 30, "Hard": 60}.get(alteration_level, 30)
        M = cv2.getRotationMatrix2D((cx, cy), rot_angle, 1.0)
        img = cv2.warpAffine(img, M, (w, h), borderValue=255)
    elif alteration_type == "Obl":  # Obliteration
        num_scratches = {"Easy": 2, "Medium": 5, "Hard": 10}.get(alteration_level, 4)
        for _ in range(num_scratches):
            p1 = (rng.randint(50, 250), rng.randint(50, 250))
            p2 = (rng.randint(50, 250), rng.randint(50, 250))
            cv2.line(img, p1, p2, 255, thickness=rng.randint(4, 10))
    elif alteration_type == "Zcut":  # Z-cut flap perturbation
        pts = np.array([[80, 100], [220, 100], [100, 200], [220, 200]], np.int32)
        cv2.polylines(img, [pts], False, 255, thickness=6)

    # Add acquisition sensor noise
    noise = rng.normal(0, 10.0, (h, w))
    img = np.clip(img + noise, 0, 255).astype(np.uint8)

    fname = f"{subject_id}__M_{finger_name}_finger"
    if alteration_type != "Real":
        fname += f"_{alteration_type}"
    fname += ".BMP"

    return FingerprintSample(
        image=img,
        filename=fname,
        subject_id=subject_id,
        finger_name=finger_name,
        gender="M",
        alteration_type=alteration_type,
        alteration_level=alteration_level
    )
