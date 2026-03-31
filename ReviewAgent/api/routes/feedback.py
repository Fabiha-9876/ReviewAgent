"""POST /feedback — submit quality + compliance feedback."""

from fastapi import APIRouter
from pydantic import BaseModel

from src.stage5.feedback_collector import DualStreamFeedbackCollector

router = APIRouter()
collector = DualStreamFeedbackCollector()


class QualityFeedback(BaseModel):
    response_id: str
    helpfulness: int
    specificity: int
    empathy: int
    accuracy: int
    actionability: int
    rater_id: str = "anonymous"


class ComplianceFeedback(BaseModel):
    response_id: str
    no_false_promises: bool
    no_info_leak: bool
    tone_compliant: bool
    legally_safe: bool
    rater_id: str = "anonymous"


@router.post("/quality")
def submit_quality(fb: QualityFeedback):
    """Submit quality scores (Stream 1)."""
    collector.record_quality(
        response_id=fb.response_id,
        scores={
            "helpfulness": fb.helpfulness,
            "specificity": fb.specificity,
            "empathy": fb.empathy,
            "accuracy": fb.accuracy,
            "actionability": fb.actionability,
        },
        rater_id=fb.rater_id,
    )
    return {"status": "recorded", "stream": "quality"}


@router.post("/compliance")
def submit_compliance(fb: ComplianceFeedback):
    """Submit compliance flags (Stream 2)."""
    collector.record_compliance(
        response_id=fb.response_id,
        flags={
            "no_false_promises": fb.no_false_promises,
            "no_info_leak": fb.no_info_leak,
            "tone_compliant": fb.tone_compliant,
            "legally_safe": fb.legally_safe,
        },
        rater_id=fb.rater_id,
    )
    return {"status": "recorded", "stream": "compliance"}
