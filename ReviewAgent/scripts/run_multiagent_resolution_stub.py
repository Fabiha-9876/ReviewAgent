"""
Aim 2 proof-of-concept: Multi-agent code resolution stub.

Original Aim 2 design:
  Planner → Navigator → Editor → Executor agents that consume IssueSpecs
  and produce simulated/proposed fixes.

Real implementation requires a working dev environment for an actual app
(GitHub repo + build system + test runner) which RRGen does not provide.
This script implements the workflow at the SPEC level — each "agent"
produces its expected output for a given IssueSpec, demonstrating that
the architecture is viable without actually editing code.

We then mark the response generation step (Stage 4b) as "resolution-aware"
by showing the response references the proposed fix rather than just
acknowledging the issue.

Outputs:
    data/processed/multiagent_resolution/
      sample_workflows.json      5 IssueSpecs run through all 4 agents
      summary.txt                human-readable stub-execution log
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def main():
    out_dir = Path("data/processed/multiagent_resolution")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Pick 5 IssueSpecs to demonstrate the workflow
    specs = json.load(open("data/processed/issue_specs/specs_with_taxonomy.json"))
    # Spread across issue types
    chosen_ids = []
    seen_types = set()
    for s in specs:
        if s["issue_type"] not in seen_types:
            chosen_ids.append(s["cluster_id"])
            seen_types.add(s["issue_type"])
        if len(chosen_ids) >= 5:
            break
    sample = [s for s in specs if s["cluster_id"] in chosen_ids]
    print(f"Selected {len(sample)} IssueSpecs across types: "
          f"{[s['issue_type'] for s in sample]}")

    # The 4 agents are described and their expected outputs are produced as
    # spec-level artifacts (not actual code edits). This is honest scope.
    workflows = []
    for spec in sample:
        wf = run_agent_workflow(spec)
        workflows.append(wf)

    with open(out_dir / "sample_workflows.json", "w") as f:
        json.dump(workflows, f, indent=2)

    # Human-readable summary
    lines = [
        "="*78,
        "AIM 2 PROOF-OF-CONCEPT: MULTI-AGENT CODE RESOLUTION (SPEC-LEVEL STUB)",
        "="*78,
        "",
        "This is a proof-of-concept demonstration. The 4 agents (Planner, Navigator,",
        "Editor, Executor) are exercised at the SPECIFICATION level — each produces",
        "the artifact it would produce in a real run (plan, file paths, patches,",
        "test outcomes), but no actual code is edited because RRGen does not provide",
        "a development environment with the source for these apps.",
        "",
        "Real implementation requires: (1) GitHub repo for each app, (2) build/test",
        "infrastructure, (3) sandboxed execution. We document this as future work in",
        "Section 6.2 of the paper.",
        "",
        f"Demonstrating on {len(sample)} IssueSpecs:",
    ]
    for w in workflows:
        lines.append("")
        lines.append("-"*78)
        lines.append(f"Cluster: {w['cluster_id']}  ({w['issue_type']})")
        lines.append(f"Issue: {w['title']}")
        lines.append("")
        lines.append("PLANNER:")
        for step in w["planner_output"]["steps"]:
            lines.append(f"  • {step}")
        lines.append("")
        lines.append("NAVIGATOR:")
        for f in w["navigator_output"]["candidate_files"]:
            lines.append(f"  • {f}")
        lines.append("")
        lines.append("EDITOR (proposed change):")
        lines.append(f"  {w['editor_output']['proposed_change']}")
        lines.append("")
        lines.append("EXECUTOR (simulated outcome):")
        lines.append(f"  status: {w['executor_output']['status']}")
        lines.append(f"  test signal: {w['executor_output']['test_signal']}")
        lines.append("")
        lines.append("RESOLUTION-AWARE RESPONSE (Stage 4b output, what user receives):")
        lines.append(f"  {w['resolution_aware_response']}")

    summary = "\n".join(lines)
    print("\n" + summary)
    with open(out_dir / "summary.txt", "w") as f:
        f.write(summary)
    print(f"\nSaved {out_dir}/sample_workflows.json + summary.txt")


def run_agent_workflow(spec: dict) -> dict:
    """Run the 4 agents at spec-level on one IssueSpec."""
    cid = spec["cluster_id"]
    issue_type = spec["issue_type"]
    title = spec.get("title", "")

    # ============================================================
    # PLANNER: decompose the IssueSpec into actionable subtasks
    # ============================================================
    if issue_type == "bug_report":
        steps = [
            f"Reproduce: {(spec.get('steps_to_reproduce') or ['—'])[0]}",
            f"Inspect component: {spec.get('affected_component', 'unknown')}",
            f"Diagnose root cause via expected vs actual: "
            f"'{(spec.get('expected_behavior') or '')[:60]}' vs "
            f"'{(spec.get('actual_behavior') or '')[:60]}'",
            "Propose fix",
            "Add regression test",
        ]
    elif issue_type == "feature_request":
        steps = [
            f"Translate user story: '{(spec.get('user_story') or '')[:80]}'",
            "Identify integration points",
            "Implement minimal viable feature",
            "Verify against acceptance criteria",
            "Document and ship",
        ]
    elif issue_type == "performance":
        cat = spec.get("nfr_category", "speed")
        steps = [
            f"Profile {cat} bottleneck in {spec.get('affected_component', 'component')}",
            "Identify hot path",
            f"Optimize {cat}",
            "Benchmark before/after",
            "Add performance regression test",
        ]
    elif issue_type == "usability":
        h = spec.get("nielsen_heuristic", "user_control")
        steps = [
            f"Audit UI for {h} violation",
            f"Redesign affected_component flow ({spec.get('affected_component', '')})",
            "Implement UI change",
            "User-test with 5 participants",
            "Iterate on feedback",
        ]
    elif issue_type == "compatibility":
        matrix = spec.get("device_os_matrix") or {}
        steps = [
            f"Set up emulators for {matrix}",
            "Reproduce on each device/OS combo",
            "Identify branch in code path",
            "Add device-specific compatibility shim",
            "Test on full device matrix",
        ]
    else:
        steps = ["Analyze", "Diagnose", "Fix", "Test", "Ship"]

    planner_output = {"steps": steps, "estimated_effort_pts": len(steps) * 2}

    # ============================================================
    # NAVIGATOR: identify candidate files in (hypothetical) repo
    # ============================================================
    component = spec.get("affected_component", "core")
    component_slug = component.lower().split("/")[0].split()[0].replace("(", "").replace(")", "")
    candidate_files = {
        "bug_report":      [f"src/{component_slug}.py",
                            f"src/handlers/{component_slug}_handler.py",
                            f"tests/test_{component_slug}.py"],
        "feature_request": [f"src/{component_slug}.py",
                            f"src/api/v1/{component_slug}.py",
                            f"src/ui/{component_slug}_view.tsx",
                            "docs/CHANGELOG.md"],
        "performance":     [f"src/{component_slug}.py",
                            f"src/cache.py",
                            f"benchmarks/test_{component_slug}_perf.py"],
        "usability":       [f"src/ui/{component_slug}_view.tsx",
                            f"src/ui/styles.css",
                            "src/ui/i18n.json"],
        "compatibility":   [f"src/platform/{component_slug}_compat.py",
                            "build.gradle", "manifest.xml"],
    }.get(issue_type, [f"src/{component_slug}.py"])

    navigator_output = {
        "candidate_files": candidate_files,
        "selection_method": "name-similarity to affected_component (Navigator stub)",
    }

    # ============================================================
    # EDITOR: produce proposed change description
    # ============================================================
    if issue_type == "bug_report":
        proposed = (f"Add input validation in {candidate_files[0]} for the "
                    f"{component} flow; ensure {(spec.get('expected_behavior') or 'expected behavior')[:50]}; "
                    f"add unit test in {candidate_files[-1]}.")
    elif issue_type == "feature_request":
        proposed = (f"New module/function in {candidate_files[0]}: implement user-story "
                    f"'{(spec.get('user_story') or '')[:60]}'; satisfy {len(spec.get('acceptance_criteria') or [])} "
                    f"acceptance criteria.")
    elif issue_type == "performance":
        proposed = (f"Refactor hot path in {candidate_files[0]}: replace O(n^2) lookup with "
                    f"O(n) cache; expected {spec.get('nfr_category', '')} improvement >= 50%.")
    elif issue_type == "usability":
        proposed = (f"Update {candidate_files[0]} to address Nielsen heuristic "
                    f"'{spec.get('nielsen_heuristic')}'; add visible state cue and undo affordance.")
    elif issue_type == "compatibility":
        proposed = (f"Add device-specific branch in {candidate_files[0]} for "
                    f"{spec.get('device_os_matrix')}; gate via runtime check.")
    else:
        proposed = "Generic fix to affected_component."

    editor_output = {"proposed_change": proposed, "diff_size_loc_estimate": 30}

    # ============================================================
    # EXECUTOR: simulated test outcome
    # ============================================================
    # In a real system this runs unit + integration + regression tests.
    # Here we report the anticipated outcome based on issue type.
    severity = spec.get("severity", "P2")
    if severity in ("P0", "P1"):
        status = "PROPOSED-PATCH-READY-FOR-REVIEW"
        test_signal = "All existing tests pass; new regression test added covers the issue path"
    else:
        status = "PROPOSED-PATCH-LOW-RISK"
        test_signal = "Patch is additive; no regression test required for this severity tier"

    executor_output = {"status": status, "test_signal": test_signal,
                       "would_run_tests": ["unit", "integration", "regression"]}

    # ============================================================
    # Resolution-aware response (this is the key Aim 2 deliverable)
    # ============================================================
    # The response REFERENCES the proposed fix instead of being generic
    sev_phrase = {
        "P0": "treating as a top-priority fix",
        "P1": "scheduled for the next release",
        "P2": "added to our backlog for an upcoming update",
        "P3": "noted as a polish improvement for a future release",
    }.get(severity, "noted for follow-up")

    resolution_aware_response = (
        f"Thanks for flagging this — we've reproduced the issue on our side. "
        f"Specifically, we've identified {component} as the affected area and "
        f"{sev_phrase}. Our team has drafted a fix in {candidate_files[0]} that "
        f"addresses the root cause; it's currently going through code review and "
        f"testing. We'll include the fix in the next release. If you'd like, "
        f"reach out to us at <email> with your device model and a screenshot for "
        f"a personalized status update."
    )

    return {
        "cluster_id": cid,
        "issue_type": issue_type,
        "title": title,
        "severity": severity,
        "planner_output": planner_output,
        "navigator_output": navigator_output,
        "editor_output": editor_output,
        "executor_output": executor_output,
        "resolution_aware_response": resolution_aware_response,
        "note": "Spec-level proof-of-concept; no actual code edits performed. "
                "Real execution requires source repository access (future work).",
    }


if __name__ == "__main__":
    main()
