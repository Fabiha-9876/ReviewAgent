"""Hierarchical clustering: aspect-level grouping then sub-clustering."""

from __future__ import annotations

import numpy as np
from sentence_transformers import SentenceTransformer
from hdbscan import HDBSCAN

from src.common.schemas import ReviewObject
from .kg_builder import ReviewKnowledgeGraph


class HierarchicalClusterer:
    """Two-level hierarchical clustering of reviews."""

    def __init__(
        self,
        embedding_model: str = "all-MiniLM-L6-v2",
        min_cluster_size: int = 5,
        min_samples: int = 3,
    ):
        self.encoder = SentenceTransformer(embedding_model)
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples

    def cluster(
        self,
        kg: ReviewKnowledgeGraph,
        reviews: list[ReviewObject],
    ) -> dict[str, list[tuple[int, list[str]]]]:
        """Hierarchical clustering.

        Returns:
            dict mapping aspect_id -> list of (sub_cluster_label, [review_ids])
        """
        review_map = {r.review_id: r for r in reviews}
        results = {}

        # Level 1: Group by aspect
        for aspect_id in kg.get_aspect_nodes():
            aspect_review_ids = kg.get_reviews_for_aspect(aspect_id)
            aspect_review_ids = [rid for rid in aspect_review_ids if rid in review_map]

            if len(aspect_review_ids) < self.min_cluster_size:
                # Too small to sub-cluster — treat as one cluster
                results[aspect_id] = [(0, aspect_review_ids)]
                continue

            # Level 2: Sub-cluster within aspect using embeddings
            texts = [review_map[rid].text for rid in aspect_review_ids]
            embeddings = self.encoder.encode(texts, show_progress_bar=False)

            # Add graph features (degree of review node in aspect subgraph)
            subgraph = kg.get_aspect_subgraph(aspect_id)
            graph_features = np.array([
                [subgraph.degree(rid) if subgraph.has_node(rid) else 0]
                for rid in aspect_review_ids
            ], dtype=np.float32)

            # Normalize and combine
            if graph_features.max() > 0:
                graph_features = graph_features / graph_features.max()
            combined = np.hstack([
                embeddings * 0.7,
                graph_features * 0.3,
            ])

            clusterer = HDBSCAN(
                min_cluster_size=self.min_cluster_size,
                min_samples=self.min_samples,
                metric="euclidean",
            )
            labels = clusterer.fit_predict(combined)

            # Group review IDs by cluster label
            cluster_groups: dict[int, list[str]] = {}
            for i, label in enumerate(labels):
                cluster_groups.setdefault(label, []).append(aspect_review_ids[i])

            results[aspect_id] = [(label, rids) for label, rids in cluster_groups.items()]

        return results
