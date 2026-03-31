"""POST /responses/generate — generate a response for a review."""

from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter()


class ResponseRequest(BaseModel):
    review_text: str
    issue_title: str | None = None
    issue_type: str | None = None


class ResponseOutput(BaseModel):
    response_text: str
    refinement_iterations: int


@router.post("/generate", response_model=ResponseOutput)
async def generate_response(req: ResponseRequest):
    """Generate a response for a single review."""
    from src.common.llm_client import LLMClient
    from src.common.schemas import ReviewObject, IssueSpec
    from src.stage4b.response_generator import ResponseGenerator
    from src.stage4b.self_refiner import SelfRefiner
    from datetime import datetime
    import uuid

    llm = LLMClient(provider="openai", model="gpt-4o")
    generator = ResponseGenerator(llm)
    refiner = SelfRefiner(llm)

    review = ReviewObject(
        review_id=str(uuid.uuid4()),
        text=req.review_text,
        rating=1,
        app_id="api",
        timestamp=datetime.now(),
    )

    spec = None
    if req.issue_title:
        spec = IssueSpec(
            issue_id="API",
            cluster_id="API",
            title=req.issue_title,
            issue_type=req.issue_type or "bug_report",
            description=req.issue_title,
        )

    response = await generator.generate(review, spec, include_rag=False)
    response = await refiner.refine(response, spec)

    return ResponseOutput(
        response_text=response.text,
        refinement_iterations=response.refinement_iterations,
    )
