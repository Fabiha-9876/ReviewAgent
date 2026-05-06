"""
Mine real human-written GitHub issues from AntennaPod for the comparison baseline.

The main paper RQ asks: "How accurately can an LLM agent translate noisy app
review clusters into structured issue specifications compared to human-written
GitHub issues?"

This script fetches recent issues from the AntennaPod GitHub issue tracker
(public, no auth needed for read), parses them into our IssueSpec schema, and
saves them as the human_github condition for Experiment 1.

Output: data/processed/issue_specs/specs_human_github.json
"""

import json
import re
import subprocess
from collections import Counter
from pathlib import Path

OUT_DIR = Path("data/processed/issue_specs")
OUT_DIR.mkdir(parents=True, exist_ok=True)

REPOS = [
    "AntennaPod/AntennaPod",
    "TeamNewPipe/NewPipe",
    "thunderbird/thunderbird-android",   # K-9 Mail's current home
]
N_PER_REPO = 100   # over-fetch per repo then filter


def fetch_issues(repo: str = "AntennaPod/AntennaPod", n: int = 100) -> list[dict]:
    """Fetch closed bug/feature issues using curl (system SSL works; Python's doesn't)."""
    issues = []
    page = 1
    while len(issues) < n and page <= 5:
        url = (f"https://api.github.com/repos/{repo}/issues?"
               f"state=closed&per_page=100&page={page}")
        result = subprocess.run(
            ["curl", "-s", "-H", "Accept: application/vnd.github.v3+json",
             "-H", "User-Agent: ReviewAgent-research/1.0", url],
            capture_output=True, text=True, timeout=60,
        )
        if result.returncode != 0:
            print(f"  curl failed on page {page}: {result.stderr}")
            break
        batch = json.loads(result.stdout)
        if not batch:
            break
        for it in batch:
            if "pull_request" in it:
                continue
            if not it.get("body"):
                continue
            issues.append(it)
            if len(issues) >= n:
                break
        page += 1
    return issues


LABEL_TO_TYPE = {
    "bug": "bug_report",
    "type: bug": "bug_report",
    "type:bug": "bug_report",
    "kind/bug": "bug_report",
    "feature": "feature_request",
    "feature request": "feature_request",
    "type: feature request": "feature_request",
    "enhancement": "feature_request",
    "performance": "performance",
    "ux": "usability",
    "ui": "usability",
    "compatibility": "compatibility",
    "device-specific": "compatibility",
}


def infer_issue_type(issue: dict) -> str:
    """Infer the 7-class label from GitHub labels + body keywords."""
    labels = [l["name"].lower() for l in issue.get("labels", [])]
    for lbl in labels:
        if lbl in LABEL_TO_TYPE:
            return LABEL_TO_TYPE[lbl]
        for k, v in LABEL_TO_TYPE.items():
            if k in lbl:
                return v
    body = (issue.get("body") or "").lower()
    title = issue.get("title", "").lower()
    text = f"{title} {body}"
    if any(k in text for k in ["crash", "error", "fail", "broken", "doesn't work", "not work"]):
        return "bug_report"
    if any(k in text for k in ["feature", "add support", "would be nice", "request"]):
        return "feature_request"
    if any(k in text for k in ["slow", "lag", "battery", "memory", "performance"]):
        return "performance"
    if any(k in text for k in ["ui", "ux", "confusing", "hard to use", "navigation"]):
        return "usability"
    if any(k in text for k in ["samsung", "pixel", "android 1", "device", "tablet"]):
        return "compatibility"
    return "other"


def extract_steps_from_body(body: str) -> list[str] | None:
    """Try to pull a numbered/bulleted reproduction list out of the issue body."""
    if not body: return None
    # Look for "Steps to reproduce" / "How to reproduce" sections
    section_match = re.search(
        r"(?:steps?\s+to\s+reproduce|how\s+to\s+reproduce|reproduce|reproduction)\s*[:\n]+(.*?)(?:\n\s*\n|\Z)",
        body, re.IGNORECASE | re.DOTALL,
    )
    if not section_match:
        return None
    section = section_match.group(1)
    # Extract numbered or bulleted items
    items = re.findall(r"^\s*(?:\d+[.)]|[-*•])\s*(.+?)$", section, re.MULTILINE)
    items = [s.strip() for s in items if s.strip() and len(s) > 5]
    return items[:6] if items else None


def extract_field(body: str, field_pattern: str) -> str | None:
    if not body: return None
    m = re.search(rf"{field_pattern}\s*[:\n]+(.*?)(?:\n\s*\n|\n###|\Z)",
                   body, re.IGNORECASE | re.DOTALL)
    if not m: return None
    text = m.group(1).strip()
    # Strip markdown bullets/numbering and excessive whitespace
    text = re.sub(r"^\s*[-*•]\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"\s+", " ", text).strip()
    return text[:500] if text else None


def issue_to_spec(issue: dict, repo: str = "") -> dict | None:
    """Map a GitHub issue into our IssueSpec schema."""
    issue_type = infer_issue_type(issue)
    if issue_type == "other":
        return None  # don't include 'other' in the comparison set

    body = issue.get("body") or ""
    title = issue.get("title", "").strip()
    if len(body) < 50 or len(title) < 5:
        return None  # skip low-content issues

    description = re.sub(r"\s+", " ", body).strip()[:500]

    # Type-specific extraction
    repo_short = repo.split("/")[-1] if repo else "unknown"
    spec = {
        "issue_id": f"is_gh_{repo_short}_{issue['number']}",
        "cluster_id": f"github_{repo_short}_{issue['number']}",
        "repo": repo,
        "title": title[:120],
        "issue_type": issue_type,
        "description": description,
        "severity": "P2",   # GitHub doesn't have severity per se; default
        "affected_component": "",
        "steps_to_reproduce": None,
        "expected_behavior": None,
        "actual_behavior": None,
        "user_story": None,
        "acceptance_criteria": None,
        "nfr_category": None,
        "nielsen_heuristic": None,
        "device_os_matrix": None,
        "condition": "human_github",
        "github_url": issue.get("html_url"),
        "github_labels": [l["name"] for l in issue.get("labels", [])],
    }

    if issue_type == "bug_report":
        spec["steps_to_reproduce"] = extract_steps_from_body(body)
        spec["expected_behavior"] = extract_field(body, r"expected\s+behaviou?r")
        spec["actual_behavior"]   = extract_field(body, r"(?:actual\s+behaviou?r|observed\s+behaviou?r|current\s+behaviou?r)")
    elif issue_type == "feature_request":
        # The whole body acts as a user story
        spec["user_story"] = description[:300]

    # Try to extract device/OS info for compatibility issues
    if issue_type == "compatibility":
        devices = re.findall(r"(?:Samsung\s+\w+|Pixel\s+\d+|OnePlus\s+\d+|Xiaomi\s+\w+)",
                              body, re.IGNORECASE)
        oses = re.findall(r"Android\s+\d+(?:\.\d+)?", body, re.IGNORECASE)
        if devices or oses:
            spec["device_os_matrix"] = {
                "affected_devices": list(set(devices))[:5],
                "affected_os": list(set(oses))[:5],
            }

    return spec


def main():
    all_specs = []
    per_repo_stats = {}

    for repo in REPOS:
        print(f"\n=== {repo} ===")
        print(f"  Fetching up to {N_PER_REPO} closed issues...")
        raw = fetch_issues(repo=repo, n=N_PER_REPO)
        print(f"  Fetched {len(raw)} non-PR issues with non-empty body")

        repo_specs = []
        for issue in raw:
            spec = issue_to_spec(issue, repo=repo)
            if spec:
                repo_specs.append(spec)

        type_counts = Counter(s["issue_type"] for s in repo_specs)
        print(f"  Parsed by type: {dict(type_counts)}")
        per_repo_stats[repo] = {"fetched": len(raw), "parsed": len(repo_specs),
                                 "by_type": dict(type_counts)}
        all_specs.extend(repo_specs)

    # Diversify across types for the gold standard (now across ALL repos)
    by_type = {}
    for s in all_specs:
        by_type.setdefault(s["issue_type"], []).append(s)

    print(f"\n=== Combined across {len(REPOS)} repos ===")
    print(f"Total parsed: {len(all_specs)}")
    print(f"By type: {dict(Counter(s['issue_type'] for s in all_specs))}")

    # Pick balanced sample (with per-type cap balanced across repos where possible)
    target_per_type = {
        "bug_report": 30,
        "feature_request": 25,
        "performance": 8,
        "usability": 5,
        "compatibility": 5,
    }
    selected = []
    for t, n in target_per_type.items():
        pool = by_type.get(t, [])
        # Round-robin across repos for diversity
        by_repo = {}
        for s in pool:
            by_repo.setdefault(s["repo"], []).append(s)
        picked = []
        idx = 0
        while len(picked) < n and any(by_repo.values()):
            for repo in REPOS:
                if by_repo.get(repo):
                    picked.append(by_repo[repo].pop(0))
                    if len(picked) >= n:
                        break
            if not any(by_repo.get(r) for r in REPOS):
                break
        selected.extend(picked)

    print(f"\nSelected {len(selected)} GitHub issues for comparison set:")
    for t, n in target_per_type.items():
        actual = sum(1 for s in selected if s["issue_type"] == t)
        print(f"  {t:20s} {actual} (target {n})")

    print(f"\nSelected by repo:")
    repo_counts = Counter(s["repo"] for s in selected)
    for r, c in repo_counts.items():
        print(f"  {r}: {c}")

    out = OUT_DIR / "specs_human_github.json"
    with open(out, "w") as f:
        json.dump(selected, f, indent=2)
    print(f"\nSaved {out} ({len(selected)} specs)")

    stats_out = OUT_DIR / "specs_human_github_stats.json"
    with open(stats_out, "w") as f:
        json.dump({
            "n_total": len(selected),
            "repos": REPOS,
            "per_repo_fetch_stats": per_repo_stats,
            "selected_by_type": {t: sum(1 for s in selected if s["issue_type"] == t)
                                  for t in target_per_type},
            "selected_by_repo": dict(repo_counts),
        }, f, indent=2)

    print(f"\nSample specs:")
    for s in selected[:5]:
        print(f"  [{s['cluster_id']}] {s['issue_type']:18s} {s['title'][:80]}")


if __name__ == "__main__":
    main()
