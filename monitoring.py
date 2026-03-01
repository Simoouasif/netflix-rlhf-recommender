```python
import time
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from collections import defaultdict, deque
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from prometheus_client import Counter, Histogram, Gauge, Summary, start_http_server
import threading
import json
import warnings

warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# Data Classes
# ─────────────────────────────────────────────

@dataclass
class RecommendationEvent:
    user_id: str
    session_id: str
    content_ids: List[str]
    scores: List[float]
    model_version: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    context: Dict[str, Any] = field(default_factory=dict)
    latency_ms: float = 0.0


@dataclass
class FeedbackEvent:
    user_id: str
    session_id: str
    content_id: str
    feedback_type: str  # click, watch, skip, thumbs_up, thumbs_down, add_to_list
    watch_duration_pct: float = 0.0
    timestamp: datetime = field(default_factory=datetime.utcnow
                               )
    reward_signal: float = 0.0


@dataclass
class ModelMetrics:
    model_version: str
    timestamp: datetime
    ndcg_at_10: float
    precision_at_5: float
    recall_at_20: float
    coverage: float
    diversity: float
    novelty: float
    serendipity: float
    avg_reward: float
    reward_variance: float


@dataclass
class AlertConfig:
    metric_name: str
    threshold: float
    comparison: str  # gt, lt, gte, lte
    window_minutes: int
    severity: str  # critical, warning, info
    cooldown_minutes: int = 15


# ─────────────────────────────────────────────
# Prometheus Metrics
# ─────────────────────────────────────────────

class PrometheusMetrics:
    def __init__(self):
        # Counters
        self.recommendations_served = Counter(
            "rlhf_recommendations_served_total",
            "Total recommendations served",
            ["model_version", "surface"]
        )
        self.feedback_received = Counter(
            "rlhf_feedback_received_total",
            "Total feedback events received",
            ["feedback_type", "model_version"]
        )
        self.model_updates = Counter(
            "rlhf_model_updates_total",
            "Total model updates performed",
            ["update_type"]
        )
        self.alerts_fired = Counter(
            "rlhf_alerts_fired_total",
            "Total alerts fired",
            ["severity", "metric_name"]
        )

        # Histograms
        self.recommendation_latency = Histogram(
            "rlhf_recommendation_latency_ms",
            "Recommendation latency in milliseconds",
            ["model_version"],
            buckets=[10, 25, 50, 100, 200, 500, 1000, 2000, 5000]
        )
        self.reward_distribution = Histogram(
            "rlhf_reward_distribution",
            "Distribution of reward signals",
            ["model_version"],
            buckets=[-1.0, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0]
        )
        self.watch_duration_distribution = Histogram(
            "rlhf_watch_duration_pct",
            "Watch duration percentage distribution",
            buckets=[0, 10, 25, 50, 75, 90, 100]
        )

        # Gauges
        self.ctr = Gauge(
            "rlhf_click_through_rate",
            "Click-through rate",
            ["model_version", "surface"]
        )
        self.ndcg_at_10 = Gauge(
            "rlhf_ndcg_at_10",
            "NDCG@10 ranking quality metric",
            ["model_version"]
        )
        self.coverage = Gauge(
            "rlhf_catalog_coverage",
            "Fraction of catalog covered in recommendations",
            ["model_version"]
        )
        self.diversity = Gauge(
            "rlhf_recommendation_diversity",
            "Intra-list diversity score",
            ["model_version"]
        )
        self.avg_reward = Gauge(
            "rlhf_average_reward",
            "Rolling average reward signal",
            ["model_version"]
        )
        self.active_users = Gauge(
            "rlhf_active_users",
            "Number of active users in window"
        )
        self.model_staleness_hours = Gauge(
            "rlhf_model_staleness_hours",
            "Hours since last model update",
            ["model_version"]
        )
        self.data_drift_score = Gauge(
            "rlhf_data_drift_score",
            "Data drift detection score",
            ["feature_name"]
        )
        self.reward_drift_score = Gauge(
            "rlhf_reward_drift_score",
            "Reward distribution drift score",
            ["model_version"]
        )

        # Summary
        self.reward_summary = Summary(
            "rlhf_reward_summary",
            "Summary statistics of reward signals",
            ["model_version"]
        )

    def start_server(self, port: int = 8000):
        start_http_server(port)
        logger.info(f"Prometheus metrics server started on port {port}")


# ─────────────────────────────────────────────
# Reward Calculator
# ─────────────────────────────────────────────

class RewardCalculator:
    """Compute composite reward from multi-signal feedback."""

    REWARD_WEIGHTS = {
        "click": 0.1,
        "watch": 0.5,
        "thumbs_up": 0.3,
        "thumbs_down": -0.4,
        "skip": -0.1,
        "add_to_list": 0.2,
        "share": 0.25,
        "not_interested": -0.3,
    }

    WATCH_DURATION_CURVE = [
        (0.10, -0.05),
        (0.25, 0.10),
        (0.50, 0.30),
        (0.75, 0.50),
        (0.90, 0.70),
        (1.00, 1.00),
    ]

    def __init__(self, decay_factor: float = 0.95):
        self.decay_factor = decay_factor

    def compute_reward(self, feedback: FeedbackEvent) -> float:
        base_reward = self.REWARD_WEIGHTS.get(feedback.feedback_type, 0.0)

        watch_bonus = 0.0
        if feedback.feedback_type == "watch" and feedback.watch_duration_pct > 0:
            watch_bonus = self._interpolate_watch_reward(
                feedback.watch_duration_pct / 100.0
            )
            base_reward = watch_bonus

        time_decay = self._compute_time_decay(feedback.timestamp)
        final_reward = base_reward * time_decay

        return np.clip(final_reward, -1.0, 1.0)

    def _interpolate_watch_reward(self, pct: float) -> float:
        for