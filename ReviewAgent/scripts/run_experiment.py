"""Experiment runner CLI — runs experiments 1, 2, 3, or ablations."""

import asyncio
import json
import sys
from pathlib import Path

from src.common.llm_client import LLMClient
from src.stage2.pipeline import Stage2Pipeline
from src.stage3.pipeline import Stage3Pipeline
from src.stage4b.rag_retriever import RAGRetriever


def load_pipeline_outputs():
    """Load previously generated pipeline outputs."""
    base = Path("data/processed")
    reviews = []
    clusters = []
    specs = []

    if (base / "stage1_reviews.json").exists():
        from src.common.schemas import ReviewObject
        data = json.loads((base / "stage1_reviews.json").read_text())
        reviews = [ReviewObject.model_validate(d) for d in data]

    if (base / "stage2_clusters.json").exists():
        from src.common.schemas import IssueCluster
        data = json.loads((base / "stage2_clusters.json").read_text())
        clusters = [IssueCluster.model_validate(d) for d in data]

    if (base / "stage3_issue_specs.json").exists():
        from src.common.schemas import IssueSpec
        data = json.loads((base / "stage3_issue_specs.json").read_text())
        specs = [IssueSpec.model_validate(d) for d in data]

    return reviews, clusters, specs


async def run_experiment1():
    """Run Experiment 1: Translation Quality."""
    from src.evaluation.experiment1 import Experiment1Runner

    reviews, clusters, specs = load_pipeline_outputs()
    if not clusters:
        print("ERROR: Run the main pipeline first to generate clusters.")
        return

    llm = LLMClient(provider="openai", model="gpt-4o")
    runner = Experiment1Runner(llm)
    results = await runner.run(clusters)
    print(runner.report())


async def run_experiment2():
    """Run Experiment 2: Coupled vs Uncoupled Response Generation."""
    from src.evaluation.experiment2 import Experiment2Runner

    reviews, clusters, specs = load_pipeline_outputs()
    if not reviews or not specs:
        print("ERROR: Run the main pipeline first.")
        return

    llm = LLMClient(provider="openai", model="gpt-4o")
    runner = Experiment2Runner(llm)
    results = await runner.run(reviews, specs)
    print(runner.report())


async def run_ablations():
    """Run all ablation studies."""
    from src.evaluation.ablations import AblationRunner

    reviews, clusters, specs = load_pipeline_outputs()
    if not reviews or not clusters or not specs:
        print("ERROR: Run the main pipeline first.")
        return

    llm = LLMClient(provider="openai", model="gpt-4o")
    runner = AblationRunner(llm)
    results = await runner.run_all(reviews, clusters, specs)
    print(runner.report())


if __name__ == "__main__":
    experiment = sys.argv[1] if len(sys.argv) > 1 else "1"

    if experiment == "1":
        asyncio.run(run_experiment1())
    elif experiment == "2":
        asyncio.run(run_experiment2())
    elif experiment == "3":
        print("Experiment 3 requires RLHF training data. Use Stage 5 pipeline to collect feedback first.")
    elif experiment == "ablations":
        asyncio.run(run_ablations())
    else:
        print(f"Usage: python3 scripts/run_experiment.py [1|2|3|ablations]")
