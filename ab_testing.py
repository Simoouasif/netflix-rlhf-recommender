```python
import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging
import json
import hashlib
from enum import Enum
import warnings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ExperimentStatus(Enum):
    DRAFT = "draft"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    STOPPED_EARLY = "stopped_early"


class AllocationStrategy(Enum):
    RANDOM = "random"
    HASH_BASED = "hash_based"
    STRATIFIED = "stratified"
    THOMPSON_SAMPLING = "thompson_sampling"


@dataclass
class Variant:
    name: str
    allocation_percentage: float
    model_config: Dict[str, Any]
    description: str = ""
    is_control: bool = False


@dataclass
class MetricResult:
    metric_name: str
    control_mean: float
    treatment_mean: float
    control_std: float
    treatment_std: float
    control_n: int
    treatment_n: int
    p_value: float
    confidence_interval: Tuple[float, float]
    effect_size: float
    relative_lift: float
    is_significant: bool
    statistical_power: float


@dataclass
class ExperimentConfig:
    experiment_id: str
    name: str
    description: str
    variants: List[Variant]
    primary_metric: str
    secondary_metrics: List[str]
    minimum_sample_size: int
    significance_level: float = 0.05
    statistical_power: float = 0.80
    minimum_detectable_effect: float = 0.01
    max_duration_days: int = 30
    allocation_strategy: AllocationStrategy = AllocationStrategy.HASH_BASED
    early_stopping_enabled: bool = True
    guardrail_metrics: Dict[str, float] = field(default_factory=dict)


@dataclass
class UserInteraction:
    user_id: str
    session_id: str
    timestamp: datetime
    variant: str
    recommended_items: List[str]
    clicked_items: List[str]
    watched_items: List[str]
    watch_duration: Dict[str, float]
    ratings: Dict[str, float]
    skip_events: List[str]
    completion_rates: Dict[str, float]
    search_queries: List[str]
    rlhf_feedback: Dict[str, Any] = field(default_factory=dict)


class SampleSizeCalculator:
    @staticmethod
    def calculate_sample_size(
        baseline_rate: float,
        minimum_detectable_effect: float,
        significance_level: float = 0.05,
        power: float = 0.80,
    ) -> int:
        alpha = significance_level
        beta = 1 - power

        z_alpha = stats.norm.ppf(1 - alpha / 2)
        z_beta = stats.norm.ppf(1 - beta)

        p1 = baseline_rate
        p2 = baseline_rate * (1 + minimum_detectable_effect)

        p_bar = (p1 + p2) / 2

        n = (z_alpha * np.sqrt(2 * p_bar * (1 - p_bar)) + z_beta * np.sqrt(
            p1 * (1 - p1) + p2 * (1 - p2)
        )) ** 2 / (p2 - p1) ** 2

        return int(np.ceil(n))

    @staticmethod
    def calculate_statistical_power(
        n: int,
        baseline_rate: float,
        treatment_rate: float,
        significance_level: float = 0.05,
    ) -> float:
        z_alpha = stats.norm.ppf(1 - significance_level / 2)
        p1 = baseline_rate
        p2 = treatment_rate
        p_bar = (p1 + p2) / 2

        se = np.sqrt(2 * p_bar * (1 - p_bar) / n)
        effect = abs(p2 - p1)

        z_beta = (effect / se) - z_alpha
        power = stats.norm.cdf(z_beta)

        return power


class MetricsCalculator:
    def __init__(self, significance_level: float = 0.05):
        self.significance_level = significance_level

    def calculate_click_through_rate(
        self, interactions: List[UserInteraction]
    ) -> float:
        if not interactions:
            return 0.0
        total_recommendations = sum(
            len(i.recommended_items) for i in interactions
        )
        total_clicks = sum(len(i.clicked_items) for i in interactions)
        return total_clicks / total_recommendations if total_recommendations > 0 else 0.0

    def calculate_watch_rate(self, interactions: List[UserInteraction]) -> float:
        if not interactions:
            return 0.0
        total_clicks = sum(len(i.clicked_items) for i in interactions)
        total_watches = sum(len(i.watched_items) for i in interactions)
        return total_watches / total_clicks if total_clicks > 0 else 0.0

    def calculate_average_watch_duration(
        self, interactions: List[UserInteraction]
    ) -> float:
        durations = []
        for interaction in interactions:
            durations.extend(interaction.watch_duration.values())
        return np.mean(durations) if durations else 0.0

    def calculate_completion_rate(
        self, interactions: List[UserInteraction]
    ) -> float:
        completion_rates = []
        for interaction in interactions:
            completion_rates.extend(interaction.completion_rates.values())
        return np.mean(completion_rates) if completion_rates else 0.0

    def calculate_average_rating(
        self, interactions: List[UserInteraction]
    ) -> float:
        ratings = []
        for interaction in interactions:
            ratings.extend(interaction.ratings.values())
        return np.mean(ratings) if ratings else 0.0

    def calculate_diversity_score(
        self, interactions: List[UserInteraction]
    ) -> float:
        all_items = []
        for interaction in interactions:
            all_items.extend(interaction.recommended_items)

        if not all_items:
            return 0.0

        unique_items = len(set(all_items))
        total_items = len(all_items)
        return unique_items / total_items

    def calculate_rlhf_preference_score(
        self, interactions: List[UserInteraction]
    ) -> float:
        scores = []
        for interaction in interactions:
            if "preference_score" in interaction.rlhf_feedback:
                scores.append(interaction.rlhf_feedback["preference_score"])
        return np.mean(scores) if scores else 0.0

    def calculate_novelty_score(
        self, interactions: List[UserInteraction], item_popularity: Dict[str, float]
    ) -> float:
        novelty_scores = []
        for interaction in interactions:
            for item in interaction.recommended_items:
                popularity = item_popularity.get(item, 0.0)
                novelty = 1.0 - popularity
                novelty_scores.append(novelty)
        return np.mean(novelty_scores) if novelty_scores else 0.0

    def calculate_serendipity_score(
        self, interactions: List[UserInteraction]
    ) -> float:
        serendipity_scores = []
        for interaction in interactions:
            if "serendipity" in interaction.rlhf_feedback:
                serendipity_scores.append(interaction.rlhf_