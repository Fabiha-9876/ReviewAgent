"""GET /issues — retrieve generated issue specifications."""

import json
from pathlib import Path
from fastapi import APIRouter, HTTPException

router = APIRouter()

SPECS_PATH = Path("data/processed/stage3_issue_specs.json")


@router.get("/")
def list_issues():
    """List all generated issue specifications."""
    if not SPECS_PATH.exists():
        return {"issues": [], "message": "No issues generated yet. Run the pipeline first."}
    specs = json.loads(SPECS_PATH.read_text())
    return {"issues": specs, "count": len(specs)}


@router.get("/{issue_id}")
def get_issue(issue_id: str):
    """Get a specific issue specification by ID."""
    if not SPECS_PATH.exists():
        raise HTTPException(404, "No issues generated yet.")
    specs = json.loads(SPECS_PATH.read_text())
    for spec in specs:
        if spec.get("issue_id") == issue_id:
            return spec
    raise HTTPException(404, f"Issue {issue_id} not found.")
