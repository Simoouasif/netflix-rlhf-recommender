```python
"""
Safety filter for Netflix RLHF recommender system.
Filters and validates content recommendations for safety, appropriateness, and quality.
"""

import re
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ContentRating(Enum):
    """Standard content rating classifications."""
    G = "G"
    PG = "PG"
    PG_13 = "PG-13"
    R = "R"
    NC_17 = "NC-17"
    TV_Y = "TV-Y"
    TV_Y7 = "TV-Y7"
    TV_G = "TV-G"
    TV_PG = "TV-PG"
    TV_14 = "TV-14"
    TV_MA = "TV-MA"
    UNRATED = "UNRATED"


class FilterResult(Enum):
    """Result of safety filter evaluation."""
    APPROVED = "approved"
    REJECTED = "rejected"
    FLAGGED_FOR_REVIEW = "flagged_for_review"
    MODIFIED = "modified"


class ViolationType(Enum):
    """Types of safety violations."""
    AGE_INAPPROPRIATE = "age_inappropriate"
    EXPLICIT_CONTENT = "explicit_content"
    HATE_SPEECH = "hate_speech"
    VIOLENCE = "violence"
    SELF_HARM = "self_harm"
    MISINFORMATION = "misinformation"
    SPAM = "spam"
    BIAS = "bias"
    PRIVACY_VIOLATION = "privacy_violation"
    POLICY_VIOLATION = "policy_violation"
    LOW_QUALITY = "low_quality"
    REPETITIVE = "repetitive"


@dataclass
class UserProfile:
    """User profile for context-aware filtering."""
    user_id: str
    age: Optional[int] = None
    is_child_account: bool = False
    parental_controls_enabled: bool = False
    max_content_rating: Optional[ContentRating] = None
    language_preferences: list = field(default_factory=list)
    viewing_history_count: int = 0
    account_region: str = "US"
    sensitive_content_opt_out: list = field(default_factory=list)


@dataclass
class ContentItem:
    """Represents a content item to be recommended."""
    content_id: str
    title: str
    content_rating: ContentRating
    genres: list = field(default_factory=list)
    tags: list = field(default_factory=list)
    description: str = ""
    language: str = "en"
    release_year: int = 2000
    duration_minutes: int = 0
    has_explicit_content: bool = False
    has_violence: bool = False
    has_strong_language: bool = False
    availability_regions: list = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


@dataclass
class RecommendationItem:
    """A recommendation with associated score and reasoning."""
    content_item: ContentItem
    recommendation_score: float
    model_reasoning: str = ""
    rlhf_reward: float = 0.0
    metadata: dict = field(default_factory=dict)


@dataclass
class FilterViolation:
    """Details of a safety filter violation."""
    violation_type: ViolationType
    severity: float  # 0.0 to 1.0
    description: str
    field_affected: str = ""
    suggested_action: str = ""


@dataclass
class FilterOutput:
    """Output from the safety filter."""
    result: FilterResult
    original_recommendation: RecommendationItem
    filtered_recommendation: Optional[RecommendationItem]
    violations: list = field(default_factory=list)
    filter_score: float = 1.0  # 1.0 = fully safe, 0.0 = completely unsafe
    processing_time_ms: float = 0.0
    filter_version: str = "1.0.0"
    timestamp: float = field(default_factory=time.time)
    audit_log: list = field(default_factory=list)


class AgeAppropriatenessFilter:
    """Filters content based on age appropriateness."""

    # Rating hierarchies for movies and TV
    MOVIE_RATING_HIERARCHY = [
        ContentRating.G,
        ContentRating.PG,
        ContentRating.PG_13,
        ContentRating.R,
        ContentRating.NC_17,
        ContentRating.UNRATED,
    ]

    TV_RATING_HIERARCHY = [
        ContentRating.TV_Y,
        ContentRating.TV_Y7,
        ContentRating.TV_G,
        ContentRating.TV_PG,
        ContentRating.TV_14,
        ContentRating.TV_MA,
        ContentRating.UNRATED,
    ]

    AGE_TO_MAX_MOVIE_RATING = {
        (0, 6): ContentRating.G,
        (7, 12): ContentRating.PG,
        (13, 16): ContentRating.PG_13,
        (17, 17): ContentRating.R,
        (18, 150): ContentRating.NC_17,
    }

    AGE_TO_MAX_TV_RATING = {
        (0, 6): ContentRating.TV_Y,
        (7, 12): ContentRating.TV_G,
        (13, 16): ContentRating.TV_14,
        (17, 150): ContentRating.TV_MA,
    }

    def get_max_rating_for_age(
        self, age: int, is_movie: bool = True
    ) -> ContentRating:
        """Get maximum allowed content rating for a given age."""
        rating_map = (
            self.AGE_TO_MAX_MOVIE_RATING if is_movie else self.AGE_TO_MAX_TV_RATING
        )
        for age_range, rating in rating_map.items():
            if age_range[0] <= age <= age_range[1]:
                return rating
        return ContentRating.NC_17 if is_movie else ContentRating.TV_MA

    def is_rating_appropriate(
        self,
        content_rating: ContentRating,
        max_allowed_rating: ContentRating,
        is_movie: bool = True,
    ) -> bool:
        """Check if content rating is within allowed range."""
        hierarchy = (
            self.MOVIE_RATING_HIERARCHY if is_movie else self.TV_RATING_HIERARCHY
        )

        # Handle cross-category ratings
        if content_rating not in hierarchy or max_allowed_rating not in hierarchy:
            # If rating is not in the hierarchy, be conservative
            if content_rating == ContentRating.UNRATED:
                return False
            return True

        content_idx = hierarchy.index(content_rating)
        max_idx = hierarchy.index(max_allowed_rating)
        return content_idx <= max_idx

    def check(
        self, recommendation: RecommendationItem, user: UserProfile
    ) -> list:
        """Check age appropriateness of content."""
        violations = []
        content = recommendation.content_item

        # Skip if no age information
        if user.age is None and not user.is_child_account:
            return violations

        # Child account override
        if user.is_child_account:
            child_max_movie = ContentRating.PG
            child_max_tv = ContentRating.TV_G

            is_movie = content.content_rating in self.MOVIE_RATING_HIERARCHY
            max_rating = child_max_movie if is_movie else child_max_tv

            if not self.is_rating_appropriate(content.content_rating, max_rating, is_movie):
                violations.append(
                    FilterViolation(
                        violation_type=ViolationType.AGE