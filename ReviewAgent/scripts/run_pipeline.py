"""End-to-end ReviewAgent pipeline runner."""

import asyncio
import json
from pathlib import Path

from src.common.llm_client import LLMClient
from src.stage1.classifier import ReviewClassifier
from src.stage1.aspect_sentiment import AspectSentimentAnalyzer
from src.stage1.entity_extractor import EntityExtractor
from src.stage1.pipeline import Stage1Pipeline
from src.stage2.pipeline import Stage2Pipeline
from src.stage3.pipeline import Stage3Pipeline
from src.stage4b.rag_retriever import RAGRetriever
from src.stage4b.pipeline import Stage4bPipeline


async def run_pipeline(reviews_path: str, output_dir: str = "data/processed"):
    """Run the full ReviewAgent pipeline end-to-end."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    # Load raw reviews
    raw_reviews = json.loads(Path(reviews_path).read_text())
    print(f"Loaded {len(raw_reviews)} reviews")

    # Initialize LLM client
    llm = LLMClient(provider="openai", model="gpt-4o")

    # ===== STAGE 1: Intake =====
    print("\n--- Stage 1: Intake ---")
    classifier = ReviewClassifier(model_name_or_path="roberta-base")
    aspect_analyzer = AspectSentimentAnalyzer(llm)
    entity_extractor = EntityExtractor(llm)
    stage1 = Stage1Pipeline(classifier, aspect_analyzer, entity_extractor)

    review_objects = await stage1.process(raw_reviews)
    print(f"  Processed {len(review_objects)} reviews")
    print(f"  Flagged for HITL: {sum(1 for r in review_objects if r.flagged_for_hitl)}")

    # Save Stage 1 output
    with open(output / "stage1_reviews.json", "w") as f:
        json.dump([r.model_dump(mode="json") for r in review_objects], f, indent=2, default=str)

    # ===== STAGE 2: KG + Clustering =====
    print("\n--- Stage 2: KG + Clustering ---")
    stage2 = Stage2Pipeline()
    clusters = stage2.process(review_objects)
    print(f"  Created {len(clusters)} issue clusters")
    print(f"  KG nodes: {stage2.get_kg().node_count()}, edges: {stage2.get_kg().edge_count()}")

    # Save Stage 2 output
    with open(output / "stage2_clusters.json", "w") as f:
        json.dump([c.model_dump(mode="json") for c in clusters], f, indent=2)
    stage2.get_kg().export(str(output / "knowledge_graph.json"))

    # ===== STAGE 3: Translation =====
    print("\n--- Stage 3: Review-to-Issue Translation ---")
    stage3 = Stage3Pipeline(llm)
    issue_specs = await stage3.process(clusters)
    print(f"  Generated {len(issue_specs)} issue specifications")

    # Save Stage 3 output
    with open(output / "stage3_issue_specs.json", "w") as f:
        json.dump([s.model_dump(mode="json") for s in issue_specs], f, indent=2)

    # ===== STAGE 4b: Response Generation =====
    print("\n--- Stage 4b: Response Generation ---")
    retriever = RAGRetriever()
    stage4b = Stage4bPipeline(llm, retriever)

    # Match reviews to specs (use first review from each cluster)
    review_map = {r.review_id: r for r in review_objects}
    matched_reviews = []
    matched_specs = []
    for spec, cluster in zip(issue_specs, clusters):
        if cluster.review_ids:
            rid = cluster.review_ids[0]
            if rid in review_map:
                matched_reviews.append(review_map[rid])
                matched_specs.append(spec)

    responses = await stage4b.process(matched_specs, matched_reviews)
    print(f"  Generated {len(responses)} responses")

    # Save Stage 4b output
    with open(output / "stage4b_responses.json", "w") as f:
        json.dump([r.model_dump(mode="json") for r in responses], f, indent=2)

    print(f"\n=== Pipeline complete! Outputs saved to {output} ===")
    return review_objects, clusters, issue_specs, responses


if __name__ == "__main__":
    import sys

    reviews_path = sys.argv[1] if len(sys.argv) > 1 else "data/raw/sample_reviews.json"
    asyncio.run(run_pipeline(reviews_path))
