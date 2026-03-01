```python
"""
feedback_collector.py - Netflix RLHF Recommender Feedback Collection System
"""

import uuid
import time
import logging
import json
from datetime import datetime
from enum import Enum
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field, asdict
from collections import defaultdict
import threading
import queue

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeedbackType(Enum):
    EXPLICIT_RATING = "explicit_rating"
    THUMBS_UP = "thumbs_up"
    THUMBS_DOWN = "thumbs_down"
    WATCH_COMPLETION = "watch_completion"
    SKIP = "skip"
    REWATCH = "rewatch"
    ADD_TO_LIST = "add_to_list"
    REMOVE_FROM_LIST = "remove_from_list"
    SEARCH_CLICK = "search_click"
    RECOMMENDATION_CLICK = "recommendation_click"
    RECOMMENDATION_IGNORE = "recommendation_ignore"
    HOVER = "hover"
    TRAILER_WATCH = "trailer_watch"
    SHARE = "share"
    DOWNLOAD = "download"


class FeedbackSignalStrength(Enum):
    STRONG_POSITIVE = 2.0
    MODERATE_POSITIVE = 1.0
    WEAK_POSITIVE = 0.5
    NEUTRAL = 0.0
    WEAK_NEGATIVE = -0.5
    MODERATE_NEGATIVE = -1.0
    STRONG_NEGATIVE = -2.0


FEEDBACK_SIGNAL_MAP = {
    FeedbackType.EXPLICIT_RATING: None,  # Computed from rating value
    FeedbackType.THUMBS_UP: FeedbackSignalStrength.STRONG_POSITIVE,
    FeedbackType.THUMBS_DOWN: FeedbackSignalStrength.STRONG_NEGATIVE,
    FeedbackType.WATCH_COMPLETION: None,  # Computed from completion percentage
    FeedbackType.SKIP: FeedbackSignalStrength.MODERATE_NEGATIVE,
    FeedbackType.REWATCH: FeedbackSignalStrength.STRONG_POSITIVE,
    FeedbackType.ADD_TO_LIST: FeedbackSignalStrength.MODERATE_POSITIVE,
    FeedbackType.REMOVE_FROM_LIST: FeedbackSignalStrength.MODERATE_NEGATIVE,
    FeedbackType.SEARCH_CLICK: FeedbackSignalStrength.WEAK_POSITIVE,
    FeedbackType.RECOMMENDATION_CLICK: FeedbackSignalStrength.MODERATE_POSITIVE,
    FeedbackType.RECOMMENDATION_IGNORE: FeedbackSignalStrength.WEAK_NEGATIVE,
    FeedbackType.HOVER: FeedbackSignalStrength.WEAK_POSITIVE,
    FeedbackType.TRAILER_WATCH: FeedbackSignalStrength.MODERATE_POSITIVE,
    FeedbackType.SHARE: FeedbackSignalStrength.STRONG_POSITIVE,
    FeedbackType.DOWNLOAD: FeedbackSignalStrength.MODERATE_POSITIVE,
}


@dataclass
class FeedbackEvent:
    user_id: str
    content_id: str
    feedback_type: FeedbackType
    timestamp: float = field(default_factory=time.time)
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    session_id: Optional[str] = None
    context: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    signal_strength: Optional[float] = None
    normalized_reward: Optional[float] = None

    def __post_init__(self):
        if isinstance(self.feedback_type, str):
            self.feedback_type = FeedbackType(self.feedback_type)
        self._compute_signal_strength()

    def _compute_signal_strength(self):
        if self.feedback_type == FeedbackType.EXPLICIT_RATING:
            rating = self.metadata.get("rating", 3.0)
            self.signal_strength = (rating - 3.0) / 2.0  # Normalize to [-1, 1]
        elif self.feedback_type == FeedbackType.WATCH_COMPLETION:
            completion_pct = self.metadata.get("completion_percentage", 0.0)
            if completion_pct >= 0.9:
                self.signal_strength = FeedbackSignalStrength.STRONG_POSITIVE.value
            elif completion_pct >= 0.7:
                self.signal_strength = FeedbackSignalStrength.MODERATE_POSITIVE.value
            elif completion_pct >= 0.5:
                self.signal_strength = FeedbackSignalStrength.WEAK_POSITIVE.value
            elif completion_pct >= 0.25:
                self.signal_strength = FeedbackSignalStrength.NEUTRAL.value
            else:
                self.signal_strength = FeedbackSignalStrength.MODERATE_NEGATIVE.value
        else:
            signal = FEEDBACK_SIGNAL_MAP.get(self.feedback_type)
            if signal is not None:
                self.signal_strength = signal.value

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["feedback_type"] = self.feedback_type.value
        d["datetime"] = datetime.fromtimestamp(self.timestamp).isoformat()
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FeedbackEvent":
        data = data.copy()
        data["feedback_type"] = FeedbackType(data["feedback_type"])
        data.pop("datetime", None)
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class UserFeedbackProfile:
    user_id: str
    total_events: int = 0
    positive_events: int = 0
    negative_events: int = 0
    neutral_events: int = 0
    content_ratings: Dict[str, float] = field(default_factory=dict)
    content_interactions: Dict[str, List[str]] = field(default_factory=lambda: defaultdict(list))
    genre_preferences: Dict[str, float] = field(default_factory=dict)
    last_updated: float = field(default_factory=time.time)
    cumulative_reward: float = 0.0
    feedback_history: List[str] = field(default_factory=list)  # List of event_ids

    def update(self, event: FeedbackEvent):
        self.total_events += 1
        self.last_updated = time.time()
        self.feedback_history.append(event.event_id)

        if event.signal_strength is not None:
            if event.signal_strength > 0.1:
                self.positive_events += 1
            elif event.signal_strength < -0.1:
                self.negative_events += 1
            else:
                self.neutral_events += 1

            self.cumulative_reward += event.signal_strength if event.normalized_reward is None else event.normalized_reward

        if event.feedback_type in (FeedbackType.EXPLICIT_RATING, FeedbackType.THUMBS_UP, FeedbackType.THUMBS_DOWN):
            existing_rating = self.content_ratings.get(event.content_id, 0.0)
            weight = 0.3  # Exponential moving average weight
            new_value = event.signal_strength if event.signal_strength is not None else 0.0
            self.content_ratings[event.content_id] = (1 - weight) * existing_rating + weight * new_value

        self.content_interactions[event.content_id].append(event.feedback_type.value)

        genre = event.context.get("genre")
        if