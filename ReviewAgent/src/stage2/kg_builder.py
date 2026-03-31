"""Knowledge Graph construction from structured review objects."""

from __future__ import annotations

import json
from pathlib import Path

import networkx as nx

from src.common.schemas import ReviewObject


class ReviewKnowledgeGraph:
    """Builds and maintains a NetworkX knowledge graph from ReviewObjects."""

    def __init__(self):
        self.graph = nx.DiGraph()

    def add_reviews(self, reviews: list[ReviewObject]) -> None:
        """Create review, aspect, and entity nodes with weighted edges."""
        for review in reviews:
            # Review node
            self.graph.add_node(
                review.review_id,
                node_type="review",
                text=review.text,
                rating=review.rating,
                labels=review.labels,
                timestamp=review.timestamp.isoformat(),
            )

            # Aspect nodes + sentiment edges
            for asp in review.aspects:
                aspect_id = f"aspect:{asp.aspect}"
                if not self.graph.has_node(aspect_id):
                    self.graph.add_node(aspect_id, node_type="aspect", name=asp.aspect)
                self.graph.add_edge(
                    review.review_id,
                    aspect_id,
                    edge_type="sentiment",
                    sentiment=asp.sentiment,
                    intensity=asp.intensity,
                )

            # Entity nodes + mentions edges
            for device in review.entities.devices:
                eid = f"device:{device}"
                if not self.graph.has_node(eid):
                    self.graph.add_node(eid, node_type="entity", entity_type="device", name=device)
                self.graph.add_edge(review.review_id, eid, edge_type="mentions")

            for os_ver in review.entities.os_versions:
                eid = f"os:{os_ver}"
                if not self.graph.has_node(eid):
                    self.graph.add_node(eid, node_type="entity", entity_type="os", name=os_ver)
                self.graph.add_edge(review.review_id, eid, edge_type="mentions")

            for app_ver in review.entities.app_versions:
                eid = f"appver:{app_ver}"
                if not self.graph.has_node(eid):
                    self.graph.add_node(eid, node_type="entity", entity_type="app_version", name=app_ver)
                self.graph.add_edge(review.review_id, eid, edge_type="mentions")

            for screen in review.entities.screens:
                eid = f"screen:{screen}"
                if not self.graph.has_node(eid):
                    self.graph.add_node(eid, node_type="entity", entity_type="screen", name=screen)
                self.graph.add_edge(review.review_id, eid, edge_type="mentions")

    def get_aspect_nodes(self) -> list[str]:
        """Return all aspect node IDs."""
        return [n for n, d in self.graph.nodes(data=True) if d.get("node_type") == "aspect"]

    def get_reviews_for_aspect(self, aspect_id: str) -> list[str]:
        """Return review IDs connected to an aspect node."""
        return [
            n for n in self.graph.predecessors(aspect_id)
            if self.graph.nodes[n].get("node_type") == "review"
        ]

    def get_aspect_subgraph(self, aspect_id: str) -> nx.DiGraph:
        """Return subgraph containing an aspect and all its connected reviews + entities."""
        review_ids = self.get_reviews_for_aspect(aspect_id)
        all_nodes = {aspect_id} | set(review_ids)
        for rid in review_ids:
            all_nodes.update(self.graph.successors(rid))
        return self.graph.subgraph(all_nodes).copy()

    def compute_pagerank(self) -> dict[str, float]:
        """Compute PageRank on the undirected version of the graph."""
        undirected = self.graph.to_undirected()
        return nx.pagerank(undirected)

    def compute_betweenness(self) -> dict[str, float]:
        """Compute betweenness centrality."""
        undirected = self.graph.to_undirected()
        return nx.betweenness_centrality(undirected)

    def get_edge_data(self, source: str, target: str) -> dict | None:
        return self.graph.get_edge_data(source, target)

    def node_count(self) -> int:
        return self.graph.number_of_nodes()

    def edge_count(self) -> int:
        return self.graph.number_of_edges()

    def export(self, path: str) -> None:
        """Save graph to JSON."""
        data = nx.node_link_data(self.graph)
        Path(path).write_text(json.dumps(data, default=str, indent=2))

    def load(self, path: str) -> None:
        """Load graph from JSON."""
        data = json.loads(Path(path).read_text())
        self.graph = nx.node_link_graph(data, directed=True)
