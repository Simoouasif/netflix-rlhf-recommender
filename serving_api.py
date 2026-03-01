```python
"""
Netflix RLHF Recommender Serving API with A/B Testing
"""

import asyncio
import hashlib
import json
import logging
import random
import time
import uuid
from collections import defaultdict
from contextlib import asynccontextmanager
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import redis.asyncio as redis
from fastapi import FastAPI, HTTPException, Header, Request, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
from pydantic import BaseModel, Field, validator
import uvicorn

# ─── Logging ──────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("netflix_rlhf_api")

# ─── Prometheus Metrics ────────────────────────────────────────────────────────

REQUEST_COUNT = Counter(
    "recommender_requests_total",
    "Total recommendation requests",
    ["endpoint", "experiment", "variant", "status"],
)
LATENCY = Histogram(
    "recommender_latency_seconds",
    "Request latency in seconds",
    ["endpoint", "experiment", "variant"],
    buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0],
)
REWARD_GAUGE = Gauge(
    "recommender_mean_reward",
    "Mean reward signal per variant",
    ["experiment", "variant"],
)
ACTIVE_EXPERIMENTS = Gauge("active_experiments_total", "Number of active experiments")
CACHE_HITS = Counter("recommender_cache_hits_total", "Cache hits", ["cache_type"])
CACHE_MISSES = Counter("recommender_cache_misses_total", "Cache misses", ["cache_type"])

# ─── Enums & Constants ─────────────────────────────────────────────────────────

class RecommendationStrategy(str, Enum):
    RLHF_V1 = "rlhf_v1"
    RLHF_V2 = "rlhf_v2"
    COLLABORATIVE = "collaborative"
    CONTENT_BASED = "content_based"
    HYBRID = "hybrid"
    CONTROL = "control"


class RewardSignal(str, Enum):
    CLICK = "click"
    PLAY = "play"
    COMPLETE = "complete"
    THUMBS_UP = "thumbs_up"
    THUMBS_DOWN = "thumbs_down"
    ADD_TO_LIST = "add_to_list"
    SHARE = "share"
    SKIP = "skip"


REWARD_WEIGHTS = {
    RewardSignal.THUMBS_UP: 2.0,
    RewardSignal.COMPLETE: 1.5,
    RewardSignal.PLAY: 1.0,
    RewardSignal.ADD_TO_LIST: 0.8,
    RewardSignal.SHARE: 0.7,
    RewardSignal.CLICK: 0.3,
    RewardSignal.SKIP: -0.5,
    RewardSignal.THUMBS_DOWN: -2.0,
}

MAX_RECOMMENDATIONS = 50
DEFAULT_RECOMMENDATIONS = 20
CACHE_TTL_SECONDS = 300
EXPERIMENT_CACHE_TTL = 60

# ─── Data Models ───────────────────────────────────────────────────────────────

class UserContext(BaseModel):
    user_id: str
    session_id: Optional[str] = None
    device_type: Optional[str] = "web"
    locale: Optional[str] = "en-US"
    viewing_history: Optional[List[str]] = Field(default_factory=list)
    preferences: Optional[Dict[str, Any]] = Field(default_factory=dict)
    time_of_day: Optional[str] = None

    @validator("user_id")
    def validate_user_id(cls, v):
        if not v or len(v) < 1:
            raise ValueError("user_id cannot be empty")
        return v


class RecommendationRequest(BaseModel):
    user: UserContext
    num_recommendations: int = Field(default=DEFAULT_RECOMMENDATIONS, ge=1, le=MAX_RECOMMENDATIONS)
    surface: str = Field(default="home", description="UI surface (home, search, detail)")
    exclude_ids: Optional[List[str]] = Field(default_factory=list)
    context_item_id: Optional[str] = None
    force_variant: Optional[str] = None
    request_id: Optional[str] = None

    class Config:
        schema_extra = {
            "example": {
                "user": {"user_id": "user_123", "device_type": "tv"},
                "num_recommendations": 20,
                "surface": "home",
            }
        }


class ContentItem(BaseModel):
    item_id: str
    title: str
    score: float
    rank: int
    strategy: str
    metadata: Dict[str, Any] = Field(default_factory=dict)


class RecommendationResponse(BaseModel):
    request_id: str
    user_id: str
    experiment_id: Optional[str]
    variant_id: Optional[str]
    strategy: str
    recommendations: List[ContentItem]
    model_version: str
    served_at: str
    latency_ms: float
    cache_hit: bool


class FeedbackRequest(BaseModel):
    request_id: str
    user_id: str
    item_id: str
    signal: RewardSignal
    timestamp: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)


class FeedbackResponse(BaseModel):
    feedback_id: str
    reward: float
    accepted: bool
    message: str


# ─── A/B Testing Framework ─────────────────────────────────────────────────────

@dataclass
class Variant:
    variant_id: str
    name: str
    strategy: RecommendationStrategy
    traffic_weight: float
    model_version: str
    config: Dict[str, Any] = field(default_factory=dict)
    is_control: bool = False


@dataclass
class Experiment:
    experiment_id: str
    name: str
    description: str
    variants: List[Variant]
    user_segments: List[str]  # empty = all users
    surfaces: List[str]  # empty = all surfaces
    start_time: datetime
    end_time: Optional[datetime]
    is_active: bool = True
    min_sample_size: int = 1000
    stats: Dict[str, Any] = field(default_factory=dict)

    def is_eligible(self, surface: str, user_segment: str = "all") -> bool:
        if not self.is_active:
            return False
        now = datetime.utcnow()
        if now < self.start_time:
            return False
        if self.end_time and now > self.end_time:
            return False
        if self.surfaces and surface not in self.surfaces:
            return False
        if self.user_segments and user_segment not in self.user_segments:
            return False
        return True

    def assign_variant(self, user_id: str) -> Variant:
        """Deterministic variant assignment using consistent hashing."""
        hash_input = f"{self.experiment_id}:{user_id}"
        hash_val = int(hashlib.md5(hash_input.encode()).hexdigest(), 