"""Stage 2 Pipeline: KG construction, clustering, schema mapping, priority ranking."""

from __future__ import annotations

from src.common.schemas import ReviewObject, IssueCluster
from .kg_builder import ReviewKnowledgeGraph
from .clustering import HierarchicalClusterer
from .schema_mapper import SchemaMapper
from .priority_ranker import PriorityRanker


class Stage2Pipeline:
    """Orchestrates the full Stage 2 process."""

    def __init__(
        self,
        clusterer: HierarchicalClusterer | None = None,
        schema_mapper: SchemaMapper | None = None,
        priority_ranker: PriorityRanker | None = None,
    ):
        self.kg = ReviewKnowledgeGraph()
        self.clusterer = clusterer or HierarchicalClusterer()
        self.schema_mapper = schema_mapper or SchemaMapper()
        self.priority_ranker = priority_ranker or PriorityRanker()

    def process(self, reviews: list[ReviewObject]) -> list[IssueCluster]:
        """Full Stage 2: KG → Clustering → Schema Mapping → Priority Ranking."""

        # Layer 1: Build knowledge graph
        self.kg.add_reviews(reviews)

        # Layer 2: Hierarchical clustering
        cluster_assignments = self.clusterer.cluster(self.kg, reviews)

        # Layer 3: Schema mapping
        clusters = []
        for aspect_id, sub_clusters in cluster_assignments.items():
            for sub_label, review_ids in sub_clusters:
                if sub_label == -1:
                    continue  # Skip noise cluster from HDBSCAN
                cluster = self.schema_mapper.map_cluster(
                    aspect_id=aspect_id,
                    sub_label=sub_label,
                    review_ids=review_ids,
                    reviews=reviews,
                    kg=self.kg,
                )
                clusters.append(cluster)

        # Priority ranking
        ranked = self.priority_ranker.rank(clusters, self.kg)
        return ranked

    def get_kg(self) -> ReviewKnowledgeGraph:
        return self.kg
