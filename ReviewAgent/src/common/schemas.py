"""Pydantic data models — the contracts between all pipeline stages."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class AspectSentiment(BaseModel):
    """A single aspect–sentiment pair extracted from a review."""

    aspect: str
    sentiment: Literal["positive", "negative", "neutral"]
    intensity: float = Field(ge=0.0, le=1.0)


class ExtractedEntities(BaseModel):
    """Named entities extracted from one or more reviews."""

    devices: list[str] = Field(default_factory=list)
    os_versions: list[str] = Field(default_factory=list)
    app_versions: list[str] = Field(default_factory=list)
    screens: list[str] = Field(default_factory=list)
    features: list[str] = Field(default_factory=list)

    def merge(self, other: ExtractedEntities) -> ExtractedEntities:
        """Return a new ExtractedEntities with deduplicated union."""
        return ExtractedEntities(
            devices=sorted(set(self.devices + other.devices)),
            os_versions=sorted(set(self.os_versions + other.os_versions)),
            app_versions=sorted(set(self.app_versions + other.app_versions)),
            screens=sorted(set(self.screens + other.screens)),
            features=sorted(set(self.features + other.features)),
        )


class ReviewObject(BaseModel):
    """Stage 1 output — a fully processed review."""

    review_id: str
    text: str
    rating: int = Field(ge=1, le=5)
    app_id: str
    timestamp: datetime
    labels: list[str] = Field(default_factory=list)
    label_confidences: dict[str, float] = Field(default_factory=dict)
    aspects: list[AspectSentiment] = Field(default_factory=list)
    entities: ExtractedEntities = Field(default_factory=ExtractedEntities)
    flagged_for_hitl: bool = False
    hitl_corrections: dict | None = None


class IssueCluster(BaseModel):
    """Stage 2 output — a schema-mapped, prioritized cluster of reviews."""

    cluster_id: str
    issue_type: Literal[
        "bug_report", "feature_request", "performance", "usability", "compatibility"
    ]
    aspect: str
    sub_category: str
    review_ids: list[str]
    review_count: int
    representative_reviews: list[str]  # top-3 review texts
    entities: ExtractedEntities
    sentiment_distribution: dict[str, float] = Field(default_factory=dict)
    temporal_pattern: str | None = None
    priority_score: float = 0.0
    kg_subgraph_ref: str = ""


class RubricScores(BaseModel):
    """5-dimension expert rubric scores for issue specs or responses."""

    completeness: float = Field(ge=1, le=5, default=0)
    accuracy: float = Field(ge=1, le=5, default=0)
    actionability: float = Field(ge=1, le=5, default=0)
    specificity: float = Field(ge=1, le=5, default=0)
    clarity: float = Field(ge=1, le=5, default=0)

    @property
    def mean(self) -> float:
        scores = [
            self.completeness,
            self.accuracy,
            self.actionability,
            self.specificity,
            self.clarity,
        ]
        return sum(scores) / len(scores)


class IssueSpec(BaseModel):
    """Stage 3 output — a structured, taxonomy-grounded issue specification."""

    issue_id: str
    cluster_id: str
    title: str
    issue_type: str
    description: str
    # Bug report fields (Zimmermann template)
    steps_to_reproduce: list[str] | None = None
    expected_behavior: str | None = None
    actual_behavior: str | None = None
    # Feature request fields
    user_story: str | None = None
    acceptance_criteria: list[str] | None = None
    # Performance fields (ISO/IEC 25010)
    nfr_category: str | None = None
    # Usability fields (Nielsen)
    nielsen_heuristic: str | None = None
    # Compatibility fields
    device_os_matrix: dict | None = None
    # Common fields
    environment: ExtractedEntities = Field(default_factory=ExtractedEntities)
    severity: str = "P2"  # P0-P3
    affected_component: str = ""
    priority_score: float = 0.0
    rubric_scores: RubricScores | None = None
    validated: bool = False


class ComplianceFlags(BaseModel):
    """Stage 5 compliance check flags."""

    no_false_promises: bool = True
    no_info_leak: bool = True
    tone_compliant: bool = True
    legally_safe: bool = True

    @property
    def is_compliant(self) -> bool:
        return all(
            [self.no_false_promises, self.no_info_leak, self.tone_compliant, self.legally_safe]
        )


class GeneratedResponse(BaseModel):
    """Stage 4b output — a generated user-facing response."""

    response_id: str
    issue_id: str
    review_id: str
    text: str
    rag_sources_used: list[str] = Field(default_factory=list)
    refinement_iterations: int = 0
    quality_scores: RubricScores | None = None
    compliance_flags: ComplianceFlags | None = None
