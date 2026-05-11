"""
Tier 2.6: Mine GitHub issues from a 4th open-source Android repo to widen
the human-vs-LLM IssueSpec comparison.

Why: Current GitHub baseline is 64 closed issues across 3 repos
(AntennaPod podcast, NewPipe content, Thunderbird Android email). A 4th
repo in a different domain hardens the "structural rather than per-project
artifact" claim.

This script reuses the helpers in scripts/mine_github_issues.py, runs them
on a fourth repo of your choice, and writes a SEPARATE output file so the
existing 3-repo dataset is untouched. After mining, append the new specs
to data/processed/issue_specs/specs_human_github.json (or evaluate
separately and add a new row to Table~\\ref{tab:rq1-perrepo}).

Suggested 4th repos (each in a different domain than the existing three;
all have active issue trackers with bug/feature labels):
    - "osmandapp/OsmAnd"      (maps / navigation)
    - "signalapp/Signal-Android" (messaging / security)
    - "tasks/tasks"           (productivity / to-do)
    - "wikimedia/apps-android-wikipedia" (reference / reading)
    - "tachiyomiorg/tachiyomi" (manga reader)

Run:
    cd /Users/fabihajalal/Desktop/Review\\ Agent/ReviewAgent
    python paper/experiments/exp_2_6_mine_4th_repo.py osmandapp/OsmAnd

Optional second arg = how many issues to over-fetch (default 100).
GitHub API is rate-limited unauthenticated to 60/hour; if you need more,
export GH_TOKEN=ghp_... before running.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path("/Users/fabihajalal/Desktop/Review Agent/ReviewAgent")
sys.path.insert(0, str(REPO_ROOT))

# Reuse the existing mining helpers so the schema is identical.
from scripts.mine_github_issues import (  # noqa: E402
    fetch_issues,
    issue_to_spec,
)

OUT_DIR = REPO_ROOT / "data/processed/issue_specs"


def mine(repo: str, n: int = 100) -> Path:
    print(f"Fetching closed issues from {repo} ...")
    raw = fetch_issues(repo=repo, n=n)
    print(f"  got {len(raw)} raw issues")

    specs = []
    for issue in raw:
        spec = issue_to_spec(issue, repo=repo)
        if spec is not None:
            specs.append(spec)

    by_type: dict[str, int] = {}
    for s in specs:
        by_type[s["issue_type"]] = by_type.get(s["issue_type"], 0) + 1
    print(f"  parsed {len(specs)} usable specs:")
    for k, v in sorted(by_type.items()):
        print(f"    {k:20s} {v}")

    repo_slug = repo.replace("/", "_")
    out = OUT_DIR / f"specs_human_github_{repo_slug}.json"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        json.dump(specs, f, indent=2, ensure_ascii=False)
    print(f"  wrote {out}")
    return out


def main() -> None:
    if len(sys.argv) < 2:
        print(
            "usage: python exp_2_6_mine_4th_repo.py <owner/repo> [n_to_fetch]\n"
            "see suggestions in the file docstring."
        )
        sys.exit(2)
    repo = sys.argv[1]
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 100
    out = mine(repo, n=n)

    print(
        "\nNext steps:\n"
        f"  1. Score the specs in {out} with the same rubric used for "
        "the existing 3-repo set (completeness, desc. words, bugs w/ steps).\n"
        "  2. Append a new row to paper/build/main.tex Table~\\ref{tab:rq1-perrepo}.\n"
        "  3. Update the abstract / intro to say 'four open-source Android projects'.\n"
        "  4. Update the Limitations bullet on GitHub sample size."
    )


if __name__ == "__main__":
    main()
