"""FastAPI application for ReviewAgent."""

from fastapi import FastAPI
from api.routes import intake, issues, responses, feedback

app = FastAPI(
    title="ReviewAgent API",
    description="An Agentic Pipeline for App Review Triage, Resolution, and Response Generation",
    version="0.1.0",
)

app.include_router(intake.router, prefix="/reviews", tags=["intake"])
app.include_router(issues.router, prefix="/issues", tags=["issues"])
app.include_router(responses.router, prefix="/responses", tags=["responses"])
app.include_router(feedback.router, prefix="/feedback", tags=["feedback"])


@app.get("/health")
def health():
    return {"status": "ok", "service": "ReviewAgent"}
