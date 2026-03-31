"""Priority ranking of issue clusters using graph centrality."""

from __future__ import annotations

from src.common.schemas import IssueCluster
from .kg_builder import ReviewKnowledgeGraph


class PriorityRanker:
    """Computes priority scores using weighted combination of signals."""

    def __init__(
        self,
        pagerank_weight: float = 0.3,
        review_count_weight: float = 0.3,
        sentiment_weight: float = 0.2,
        recency_weight: float = 0.2,
    ):
        self.w_pr = pagerank_weight
        self.w_count = review_count_weight
        self.w_sent = sentiment_weight
        self.w_recency = recency_weight

    def rank(
        self, clusters: list[IssueCluster], kg: ReviewKnowledgeGraph
    ) -> list[IssueCluster]:
        """Compute priority scores and return clusters sorted by priority (descending)."""
        if not clusters:
            return []

        pagerank = kg.compute_pagerank()
        max_count = max(c.review_count for c in clusters) or 1

        for cluster in clusters:
            # PageRank of the aspect node
            aspect_node = f"aspect:{cluster.aspect}"
            pr_score = pagerank.get(aspect_node, 0.0)

            # Normalized review count
            count_score = cluster.review_count / max_count

            # Sentiment negativity intensity
            neg_ratio = cluster.sentiment_distribution.get("negative", 0.0)

            # Recency (has temporal pattern = more recent/urgent)
            recency_score = 1.0 if cluster.temporal_pattern else 0.5

            cluster.priority_score = round(
                self.w_pr * pr_score
                + self.w_count * count_score
                + self.w_sent * neg_ratio
                + self.w_recency * recency_score,
                4,
            )

        return sorted(clusters, key=lambda c: c.priority_score, reverse=True)
