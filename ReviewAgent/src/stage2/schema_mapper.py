"""Maps raw clusters to the standardized IssueCluster schema."""

from __future__ import annotations

import uuid
from collections import Counter
from datetime import datetime

from src.common.schemas import ReviewObject, IssueCluster, ExtractedEntities
from .kg_builder import ReviewKnowledgeGraph


class SchemaMapper:
    """Maps each cluster to the standardized IssueCluster schema."""

    def map_cluster(
        self,
        aspect_id: str,
        sub_label: int,
        review_ids: list[str],
        reviews: list[ReviewObject],
        kg: ReviewKnowledgeGraph,
    ) -> IssueCluster:
        """Convert a raw cluster into a standardized IssueCluster."""
        review_map = {r.review_id: r for r in reviews}
        cluster_reviews = [review_map[rid] for rid in review_ids if rid in review_map]

        aspect_name = aspect_id.replace("aspect:", "")
        issue_type = self._infer_issue_type(cluster_reviews)
        representatives = self._select_representatives(cluster_reviews)
        merged_entities = self._merge_entities(cluster_reviews)
        sentiment_dist = self._compute_sentiment_distribution(cluster_reviews, aspect_name)
        temporal = self._detect_temporal_pattern(cluster_reviews)
        sub_cat = self._infer_sub_category(cluster_reviews, aspect_name)

        return IssueCluster(
            cluster_id=f"CLU-{uuid.uuid4().hex[:6].upper()}",
            issue_type=issue_type,
            aspect=aspect_name,
            sub_category=sub_cat,
            review_ids=review_ids,
            review_count=len(review_ids),
            representative_reviews=representatives,
            entities=merged_entities,
            sentiment_distribution=sentiment_dist,
            temporal_pattern=temporal,
            priority_score=0.0,  # Set later by PriorityRanker
            kg_subgraph_ref=aspect_id,
        )

    def _infer_issue_type(self, reviews: list[ReviewObject]) -> str:
        """Majority vote on review labels within the cluster."""
        type_map = {
            "bug_report": "bug_report",
            "feature_request": "feature_request",
            "performance": "performance",
            "usability": "usability",
            "compatibility": "compatibility",
        }
        label_counts = Counter()
        for r in reviews:
            for label in r.labels:
                if label in type_map:
                    label_counts[label] += 1

        if not label_counts:
            return "bug_report"
        return label_counts.most_common(1)[0][0]

    def _select_representatives(self, reviews: list[ReviewObject], n: int = 3) -> list[str]:
        """Select top-n representative reviews by text length (proxy for informativeness)."""
        sorted_reviews = sorted(reviews, key=lambda r: len(r.text), reverse=True)
        return [r.text for r in sorted_reviews[:n]]

    def _merge_entities(self, reviews: list[ReviewObject]) -> ExtractedEntities:
        """Union all entities from reviews in the cluster."""
        merged = ExtractedEntities()
        for r in reviews:
            merged = merged.merge(r.entities)
        return merged

    def _compute_sentiment_distribution(
        self, reviews: list[ReviewObject], aspect_name: str
    ) -> dict[str, float]:
        """Compute sentiment distribution for the cluster's aspect."""
        counts = {"positive": 0, "negative": 0, "neutral": 0}
        for r in reviews:
            for asp in r.aspects:
                if asp.aspect == aspect_name:
                    counts[asp.sentiment] += 1
        total = sum(counts.values()) or 1
        return {k: round(v / total, 3) for k, v in counts.items()}

    def _detect_temporal_pattern(self, reviews: list[ReviewObject]) -> str | None:
        """Detect if reviews spike after a specific app version."""
        version_counts = Counter()
        for r in reviews:
            for v in r.entities.app_versions:
                version_counts[v] += 1
        if version_counts:
            top_version, count = version_counts.most_common(1)[0]
            if count > len(reviews) * 0.4:
                return f"spike_after_{top_version}"
        return None

    def _infer_sub_category(self, reviews: list[ReviewObject], aspect: str) -> str:
        """Infer a sub-category label from common keywords."""
        all_text = " ".join(r.text.lower() for r in reviews)
        keywords = {
            "crash": "crash_on_action",
            "freeze": "freeze_hang",
            "slow": "performance_degradation",
            "not working": "feature_broken",
            "missing": "missing_feature",
            "ugly": "visual_issue",
            "confusing": "usability_issue",
            "battery": "battery_drain",
            "login": "auth_issue",
        }
        for keyword, category in keywords.items():
            if keyword in all_text:
                return category
        return f"{aspect}_general"
