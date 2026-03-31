"""POST /reviews/intake — process raw reviews through Stage 1."""

from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter()


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

    llm = LLMClient(provider="openai", model="gpt-4o")
    pipeline = Stage1Pipeline(
        classifier=ReviewClassifier(),
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
