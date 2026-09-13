"""Hierarchical system configuration with validation."""

from dataclasses import dataclass, field
from typing import List, Tuple, Optional
import yaml
import os


@dataclass
class PreprocessingConfig:
    target_size: Tuple[int, int] = (300, 300)
    clahe_clip_limit: float = 2.5
    clahe_grid_size: Tuple[int, int] = (8, 8)
    bilateral_d: int = 9
    bilateral_sigma_color: float = 75.0
    bilateral_sigma_space: float = 75.0


@dataclass
class OrientationConfig:
    block_size: int = 16
    gaussian_sigma: float = 2.0
    ori_bins: int = 32
    coherence_patch_size: int = 16


@dataclass
class TopologyConfig:
    skeleton_method: str = "lee"
    loop_bins: int = 32
    loop_range: Tuple[float, float] = (0.0, 200.0)
    degree_features: bool = True
    betti_features: bool = True


@dataclass
class EmbeddingConfig:
    dimension: int = 261
    l2_normalize: bool = True


@dataclass
class PrivacyConfig:
    enabled: bool = True
    method: str = "biohashing"
    projection_dim: int = 512
    threshold: float = 0.0
    secret_key_seed: int = 42


@dataclass
class IndexingConfig:
    ann_backend: str = "faiss"
    metric: str = "inner_product"
    hnsw_m: int = 32
    hnsw_ef_construction: int = 128
    hnsw_ef_search: int = 64


@dataclass
class SystemConfig:
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    orientation: OrientationConfig = field(default_factory=OrientationConfig)
    topology: TopologyConfig = field(default_factory=TopologyConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    privacy: PrivacyConfig = field(default_factory=PrivacyConfig)
    indexing: IndexingConfig = field(default_factory=IndexingConfig)

    @classmethod
    def from_yaml(cls, path: str) -> "SystemConfig":
        if not os.path.exists(path):
            return cls()
        with open(path, "r") as f:
            data = yaml.safe_load(f) or {}

        cfg = cls()
        if "preprocessing" in data:
            p = data["preprocessing"]
            cfg.preprocessing = PreprocessingConfig(
                target_size=tuple(p.get("target_size", (300, 300))),
                clahe_clip_limit=float(p.get("clahe_clip_limit", 2.5)),
                clahe_grid_size=tuple(p.get("clahe_grid_size", (8, 8))),
                bilateral_d=int(p.get("bilateral_d", 9)),
                bilateral_sigma_color=float(p.get("bilateral_sigma_color", 75.0)),
                bilateral_sigma_space=float(p.get("bilateral_sigma_space", 75.0)),
            )
        if "orientation" in data:
            o = data["orientation"]
            cfg.orientation = OrientationConfig(
                block_size=int(o.get("block_size", 16)),
                gaussian_sigma=float(o.get("gaussian_sigma", 2.0)),
                ori_bins=int(o.get("ori_bins", 32)),
                coherence_patch_size=int(o.get("coherence_patch_size", 16)),
            )
        if "topology" in data:
            t = data["topology"]
            cfg.topology = TopologyConfig(
                skeleton_method=str(t.get("skeleton_method", "lee")),
                loop_bins=int(t.get("loop_bins", 32)),
                loop_range=tuple(t.get("loop_range", (0.0, 200.0))),
                degree_features=bool(t.get("degree_features", True)),
                betti_features=bool(t.get("betti_features", True)),
            )
        if "embedding" in data:
            e = data["embedding"]
            cfg.embedding = EmbeddingConfig(
                dimension=int(e.get("dimension", 261)),
                l2_normalize=bool(e.get("l2_normalize", True)),
            )
        if "privacy" in data:
            pr = data["privacy"]
            cfg.privacy = PrivacyConfig(
                enabled=bool(pr.get("enabled", True)),
                method=str(pr.get("method", "biohashing")),
                projection_dim=int(pr.get("projection_dim", 512)),
                threshold=float(pr.get("threshold", 0.0)),
                secret_key_seed=int(pr.get("secret_key_seed", 42)),
            )
        if "indexing" in data:
            idx = data["indexing"]
            cfg.indexing = IndexingConfig(
                ann_backend=str(idx.get("ann_backend", "faiss")),
                metric=str(idx.get("metric", "inner_product")),
                hnsw_m=int(idx.get("hnsw_m", 32)),
                hnsw_ef_construction=int(idx.get("hnsw_ef_construction", 128)),
                hnsw_ef_search=int(idx.get("hnsw_ef_search", 64)),
            )
        return cfg
