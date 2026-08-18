"""POST /reviews/intake — process raw reviews through Stage 1."""

import os
from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter()

# The pipeline in the paper runs on Gemini/Qwen/Claude, not gpt-4o. Read the provider from
# the environment so the endpoint uses whatever key the deployment actually has, instead of
# hardcoding a provider for which this repository ships no key.
LLM_PROVIDER = os.getenv("ISSUESPEC_LLM_PROVIDER", "anthropic")
LLM_MODEL = os.getenv("ISSUESPEC_LLM_MODEL", "claude-sonnet-4-5")


class ReviewInput(BaseModel):
    text: str
    rating: int = 3
    app_id: str = "default"


class IntakeResponse(BaseModel):
    review_id: str
    labels: list[str]
    flagged_for_hitl: bool
    aspects: list[dict]
    entities: dict


@router.post("/intake", response_model=list[IntakeResponse])
async def intake_reviews(reviews: list[ReviewInput]):
    """Process raw reviews through Stage 1 intake pipeline."""
    from src.common.llm_client import LLMClient
    from src.stage1.classifier import ReviewClassifier
    from src.stage1.aspect_sentiment import AspectSentimentAnalyzer
    from src.stage1.entity_extractor import EntityExtractor
    from src.stage1.pipeline import Stage1Pipeline

    # Use the released V5 checkpoint. Constructing ReviewClassifier() with no argument
    # falls back to an untrained roberta-base head, which returns arbitrary labels
    # (a crash report comes back as "praise") while six trained checkpoints sit unused.
    repo = Path(__file__).resolve().parents[2]
    ckpt = repo / "models/stage1_classifier_v5"
    if not ckpt.exists():
        raise HTTPException(
            status_code=503,
            detail=f"Stage-1 checkpoint not found at {ckpt}. Fetch it from the Hugging Face "
                   "release before calling this endpoint; refusing to classify with an "
                   "untrained head.",
        )

    llm = LLMClient(provider=LLM_PROVIDER, model=LLM_MODEL)
    pipeline = Stage1Pipeline(
        classifier=ReviewClassifier.load(str(ckpt)),
        aspect_analyzer=AspectSentimentAnalyzer(llm),
        entity_extractor=EntityExtractor(llm),
    )

    raw = [{"text": r.text, "rating": r.rating, "app_id": r.app_id} for r in reviews]
    results = await pipeline.process(raw)

    return [
        IntakeResponse(
            review_id=r.review_id,
            labels=r.labels,
            flagged_for_hitl=r.flagged_for_hitl,
            aspects=[a.model_dump() for a in r.aspects],
            entities=r.entities.model_dump(),
        )
        for r in results
    ]
