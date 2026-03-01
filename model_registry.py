```python
"""
Model Registry for Netflix RLHF Recommender System.
Handles model versioning, storage, retrieval, and lifecycle management.
"""

import os
import json
import time
import uuid
import hashlib
import logging
import shutil
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field, asdict

import torch
import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class ModelStage(str, Enum):
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    ARCHIVED = "archived"
    DEPRECATED = "deprecated"


class ModelType(str, Enum):
    REWARD_MODEL = "reward_model"
    POLICY_MODEL = "policy_model"
    VALUE_MODEL = "value_model"
    REFERENCE_MODEL = "reference_model"
    EMBEDDING_MODEL = "embedding_model"
    RANKING_MODEL = "ranking_model"


class TrainingMethod(str, Enum):
    SUPERVISED = "supervised"
    RLHF = "rlhf"
    PPO = "ppo"
    DPO = "dpo"
    REWARD_MODELING = "reward_modeling"
    FINE_TUNING = "fine_tuning"


# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------

@dataclass
class ModelMetrics:
    """Performance metrics associated with a model version."""
    # Reward / ranking quality
    ndcg_at_10: float = 0.0
    ndcg_at_50: float = 0.0
    map_at_10: float = 0.0
    mrr: float = 0.0

    # RLHF-specific
    reward_mean: float = 0.0
    reward_std: float = 0.0
    kl_divergence: float = 0.0
    policy_loss: float = 0.0
    value_loss: float = 0.0
    entropy_bonus: float = 0.0

    # User engagement proxies (offline eval)
    ctr_proxy: float = 0.0
    watch_time_proxy: float = 0.0
    diversity_score: float = 0.0
    novelty_score: float = 0.0

    # Training stats
    train_loss: float = 0.0
    val_loss: float = 0.0
    epochs_trained: int = 0
    training_steps: int = 0
    gradient_norm: float = 0.0

    # Human preference alignment
    human_preference_accuracy: float = 0.0
    elo_rating: float = 1500.0

    extra: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelMetrics":
        known = {k for k in cls.__dataclass_fields__}
        base = {k: v for k, v in data.items() if k in known}
        extra = {k: v for k, v in data.items() if k not in known}
        obj = cls(**base)
        obj.extra = extra
        return obj


@dataclass
class HyperParameters:
    """Hyper-parameters used during training."""
    learning_rate: float = 3e-4
    batch_size: int = 64
    num_epochs: int = 10
    warmup_steps: int = 1000
    weight_decay: float = 0.01
    dropout: float = 0.1
    hidden_dim: int = 256
    num_layers: int = 4
    num_heads: int = 8
    embedding_dim: int = 128
    max_seq_len: int = 512

    # RLHF / PPO
    ppo_epochs: int = 4
    ppo_clip: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    gae_lambda: float = 0.95
    gamma: float = 0.99
    kl_penalty: float = 0.1
    kl_target: float = 0.02
    reward_scale: float = 1.0
    reward_clip: float = 5.0

    # DPO
    dpo_beta: float = 0.1

    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "HyperParameters":
        known = {k for k in cls.__dataclass_fields__}
        base = {k: v for k, v in data.items() if k in known}
        extra = {k: v for k, v in data.items() if k not in known}
        obj = cls(**base)
        obj.extra = extra
        return obj


@dataclass
class DatasetInfo:
    """Metadata about the dataset used for training."""
    dataset_id: str = ""
    dataset_name: str = ""
    version: str = "1.0.0"
    num_samples: int = 0
    num_preference_pairs: int = 0
    num_users: int = 0
    num_items: int = 0
    date_range_start: str = ""
    date_range_end: str = ""
    features: List[str] = field(default_factory=list)
    splits: Dict[str, int] = field(default_factory=dict)
    checksum: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DatasetInfo":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class ModelCard:
    """Human-readable model card with intended use and limitations."""
    description: str = ""
    intended_use: str = ""
    out_of_scope_use: str = ""
    limitations: str = ""
    ethical_considerations: str = ""
    training_details: str = ""
    evaluation_details: str = ""
    contact: str = "mlplatform@netflix.com"
    tags: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelCard":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class ModelVersion:
    """Complete metadata for a single model version."""
    model_id: str
    version: str
    model_name: str
    model_type: ModelType
    training_method: TrainingMethod
    stage: ModelStage

    # Timestamps
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    promoted_at: Optional[str] = None
    archived_at: Optional[str] = None

    # Lineage
    parent_model_id: Optional[str] = None
    parent_version: Optional[str] = None
    experiment_id: Optional[str] = None
    run_id: Optional[str] = None
    git_commit: Optional[str] = None
    git_branch: Optional[str] = None

    # Artifacts
    artifact_path: str = ""
    artifact_checksum: str = ""
    artifact_size_bytes: int = 0

    