```python
"""
Netflix RLHF Recommender - Data Pipeline
"""

import os
import json
import random
import hashlib
import logging
import asyncio
from datetime import datetime, timedelta
from typing import Any, Optional
from dataclasses import dataclass, field, asdict
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class PipelineConfig:
    # Paths
    raw_data_dir: str = "data/raw"
    processed_data_dir: str = "data/processed"
    cache_dir: str = "data/cache"
    checkpoint_dir: str = "checkpoints"

    # Content metadata
    content_features: list = field(default_factory=lambda: [
        "title", "genre", "release_year", "duration_minutes",
        "language", "maturity_rating", "description", "cast", "director"
    ])

    # User interaction signals
    interaction_types: list = field(default_factory=lambda: [
        "play", "pause", "seek", "complete", "thumbs_up", "thumbs_down",
        "add_to_list", "remove_from_list", "search", "browse", "skip_intro",
        "autoplay_cancel", "share"
    ])

    # Reward signal weights
    reward_weights: dict = field(default_factory=lambda: {
        "completion_rate": 0.35,
        "explicit_rating": 0.25,
        "rewatch": 0.15,
        "share": 0.10,
        "add_to_list": 0.08,
        "thumbs_up": 0.07,
    })

    # Negative signal weights
    penalty_weights: dict = field(default_factory=lambda: {
        "thumbs_down": -0.30,
        "early_exit": -0.20,
        "autoplay_cancel": -0.15,
        "remove_from_list": -0.10,
    })

    # Sequence modeling
    max_history_length: int = 50
    max_sequence_length: int = 512
    context_window: int = 10

    # Preference pairs (Bradley-Terry model)
    min_preference_gap: float = 0.15
    max_pairs_per_user: int = 100
    pair_sampling_strategy: str = "reward_gap"  # or "random", "hard_negative"

    # Training split
    train_ratio: float = 0.80
    val_ratio: float = 0.10
    test_ratio: float = 0.10

    # Processing
    batch_size: int = 256
    num_workers: int = 4
    prefetch_factor: int = 2
    seed: int = 42

    # Tokenizer
    tokenizer_name: str = "bert-base-uncased"
    max_text_length: int = 128

    # Time decay for rewards
    time_decay_hours: float = 168.0  # 1 week half-life

    # Deduplication
    dedup_window_hours: int = 4

    # Minimum interactions per user to include
    min_user_interactions: int = 10
    min_item_interactions: int = 5


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------

@dataclass
class ContentItem:
    content_id: str
    title: str
    genre: list
    release_year: int
    duration_minutes: int
    language: str
    maturity_rating: str
    description: str
    cast: list
    director: str
    tags: list = field(default_factory=list)
    embedding: Optional[np.ndarray] = None

    def to_text(self) -> str:
        genres = ", ".join(self.genre)
        cast_str = ", ".join(self.cast[:5])
        return (
            f"Title: {self.title}. "
            f"Genre: {genres}. "
            f"Year: {self.release_year}. "
            f"Director: {self.director}. "
            f"Cast: {cast_str}. "
            f"Rating: {self.maturity_rating}. "
            f"Duration: {self.duration_minutes} minutes. "
            f"Description: {self.description}"
        )

    def to_feature_dict(self) -> dict:
        return {
            "content_id": self.content_id,
            "title": self.title,
            "genre": self.genre,
            "release_year": self.release_year,
            "duration_minutes": self.duration_minutes,
            "language": self.language,
            "maturity_rating": self.maturity_rating,
        }


@dataclass
class UserInteraction:
    interaction_id: str
    user_id: str
    content_id: str
    interaction_type: str
    timestamp: datetime
    session_id: str
    watch_duration_seconds: Optional[float] = None
    completion_rate: Optional[float] = None
    device_type: Optional[str] = None
    context: Optional[dict] = None

    def is_positive(self) -> bool:
        return self.interaction_type in {
            "complete", "thumbs_up", "add_to_list", "rewatch", "share"
        }

    def is_negative(self) -> bool:
        return self.interaction_type in {
            "thumbs_down", "autoplay_cancel", "remove_from_list"
        }


@dataclass
class UserProfile:
    user_id: str
    age_group: Optional[str] = None
    country: Optional[str] = None
    preferred_languages: list = field(default_factory=list)
    subscription_tier: Optional[str] = None
    account_age_days: int = 0
    interaction_history: list = field(default_factory=list)  # List[UserInteraction]
    watched_content_ids: set = field(default_factory=set)

    def get_recent_history(self, n: int = 10) -> list:
        sorted_history = sorted(
            self.interaction_history,
            key=lambda x: x.timestamp,
            reverse=True
        )
        return sorted_history[:n]


@dataclass
class RewardSignal:
    user_id: str
    content_id: str
    reward: float
    components: dict
    timestamp: datetime
    session_id: str
    confidence: float = 1.0


@dataclass
class PreferencePair:
    user_id: str
    chosen_content_id: str
    rejected_content_id: str
    chosen_reward: float
    rejected_reward: float
    reward_gap: float
    context: dict
    timestamp: datetime
    pair_id: str = ""

    def __post_init__(self):
        if not self.pair_id:
            hash_input = f"{self.user_id}_{self.chosen_content_id}_{self.rejected_content_id}"
            self.pair_id = hashlib.md5(hash_input.encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Synthetic Data Generator (mirrors real Netflix interaction patterns)
# ---------------------------------------------------------------------------

class NetflixDataGenerator:
    """
    Generates realistic synthetic Netflix interaction data for testing.
    In production this would be replaced by the actual data ingestion layer.
    """

    GENRES = [
        "Action", "Comedy", "Drama", "Thriller", "Horror", "Romance",
        "Documentary", "Animation", "Sci-Fi", "Fantasy", "Crime", "Mystery"
    ]
    LANGUAGES = ["en", "es", "fr", "de", "ja", "ko", "pt", "hi", "zh"]
    RATINGS = ["G", "PG", "PG-13", "TV-14", "TV-MA", "R"]
    DEVICES = ["smart_tv",