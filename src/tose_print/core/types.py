"""Core type definitions and dataclasses for TOSE-Print."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
import numpy as np


@dataclass
class FingerprintSample:
    """Represents a single fingerprint impression."""
    image: np.ndarray
    filename: str
    subject_id: int
    finger_name: str
    gender: str = "Unknown"
    alteration_type: str = "Real"  # Real, CR (Central Rotation), Obl (Obliteration), ZCut
    alteration_level: str = "None"  # None, Easy, Medium, Hard
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MinutiaPoint:
    """Represents a single minutiae point."""
    x: float
    y: float
    orientation: float  # Angle in radians [0, 2*pi)
    minutia_type: str   # 'ending' or 'bifurcation'
    quality: float = 1.0


@dataclass
class TOSEVector:
    """Fixed-length Topology + Orientation SE(2) Embedding vector."""
    vector: np.ndarray             # Full L2-normalized 1D array
    ori_hist: np.ndarray           # Orientation histogram slice
    loop_hist: np.ndarray          # Topological loop spectrum slice
    topo_summary: np.ndarray       # Endpoints, junctions, Betti-0 summary
    coherence_patch: np.ndarray    # Coherence spatial patch slice
    dimension: int = field(init=False)

    def __post_init__(self):
        self.dimension = int(self.vector.size)


@dataclass
class MatchResult:
    """Result of matching two biometric templates."""
    score: float
    is_genuine: bool
    query_id: str
    gallery_id: str
    modality_scores: Dict[str, float] = field(default_factory=dict)


@dataclass
class VerificationMetrics:
    """Standard ISO/IEC 19795 biometric verification metrics."""
    eer: float
    eer_threshold: float
    fmr100: float       # FNMR at FMR = 1.0% (0.01)
    fmr1000: float      # FNMR at FMR = 0.1% (0.001)
    zero_fmr: float     # FNMR at FMR = 0%
    auc: float
    roc_fpr: np.ndarray = field(default_factory=lambda: np.array([]))
    roc_tpr: np.ndarray = field(default_factory=lambda: np.array([]))
    thresholds: np.ndarray = field(default_factory=lambda: np.array([]))


@dataclass
class IdentificationMetrics:
    """Standard 1:N closed-set identification metrics."""
    rank_1: float
    rank_5: float
    rank_10: float
    rank_20: float
    mrr: float          # Mean Reciprocal Rank
    cmc_ranks: np.ndarray = field(default_factory=lambda: np.array([]))
    cmc_accuracies: np.ndarray = field(default_factory=lambda: np.array([]))
