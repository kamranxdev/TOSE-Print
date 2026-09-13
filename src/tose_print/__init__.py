"""
TOSE-Print: Topology + Orientation SE(2) Embedding for Fingerprint Recognition.
A scalable, privacy-preserving, fixed-length fingerprint representation framework.
"""

__version__ = "1.0.0"
__author__ = "Research Team"

from tose_print.core.config import SystemConfig
from tose_print.core.types import FingerprintSample, TOSEVector, MatchResult
from tose_print.features.tose import TOSEExtractor
from tose_print.privacy.biohashing import BioHasher
from tose_print.indexing.faiss_ann import FAISSIndexer

__all__ = [
    "SystemConfig",
    "FingerprintSample",
    "TOSEVector",
    "MatchResult",
    "TOSEExtractor",
    "BioHasher",
    "FAISSIndexer",
]
