"""Fourier Spectral Phase Harmonic Analysis (grounded alternative to 'quantum' heuristic)."""

import numpy as np


class SpectralPhaseHarmonicExtractor:
    """
    Extracts multi-scale harmonic phase modulation from 2D Fourier Spectrum.
    Rigorous signal processing descriptor of frequency-domain ridge harmonics.
    """

    def __init__(self, num_harmonics: int = 8):
        self.num_harmonics = num_harmonics

    def extract(self, gray_image: np.ndarray) -> np.ndarray:
        """
        Computes Fourier magnitude and phase harmonic projection:
        Returns: 2 * num_harmonics dimensional feature vector (real + imag projections).
        """
        f_transform = np.fft.fft2(gray_image)
        magnitude = np.abs(f_transform)
        phase = np.angle(f_transform)

        features = []
        for k in range(1, self.num_harmonics + 1):
            # Phase harmonic modulation projection
            harm = np.mean(magnitude * np.exp(1j * phase * k))
            features.append(float(np.real(harm)))
            features.append(float(np.imag(harm)))

        vec = np.array(features, dtype=np.float32)
        norm = np.linalg.norm(vec) + 1e-8
        return vec / norm
