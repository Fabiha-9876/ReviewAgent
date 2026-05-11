"""
Leakage audit on the §3.8.1.x strict content-validity criteria.

Question: were the chosen thresholds (≥3 steps, ≥5 words/step, ≥30 word descriptions,
≥8 word expected/actual, ≥3 acceptance criteria) tuned to favor Claude?

Test: re-run the strict scoring under STRICTER thresholds and report deltas.
If Claude's score drops materially under stricter thresholds while the qualitative
ranking is preserved, the original criteria are *not* tuned to favor Claude.
"""

import json
import re
from pathlib import Path
from copy import deepcopy

import numpy as np

BASE = Path("/Users/fabihajalal/Desktop/Review Agent/ReviewAgent/data/processed/issue_specs")

# Default §3.8.1.x criteria
DEFAULT = {
    "min_title_words": 4,
    "min_description_words": 30,
    "min_steps": 3,
    "min_words_per_step": 5,
    "min_expected_actual_words": 8,
    "min_acceptance_items": 3,
    "min_words_per_acceptance": 8,
}

# Stricter sweep
STRICTER = {
    "moderate":  {**DEFAULT, "min_description_words": 50, "min_steps": 4, "min_words_per_step": 7, "min_expected_actual_words": 12, "min_acceptance_items": 4, "min_words_per_acceptance": 10},
    "very_strict": {**DEFAULT, "min_description_words": 80, "min_steps": 5, "min_words_per_step": 10, "min_expected_actual_words": 15, "min_acceptance_items": 5, "min_words_per_acceptance": 15},
}

ACTION_VERBS = {
    "open", "tap", "click", "navigate", "swipe", "scroll", "select",
    "enter", "type", "press", "launch", "install", "uninstall", "lock",
    "unlock", "trigger", "wait", "observe", "attempt", "try", "submit",
    "send", "go", "switch", "toggle", "confirm", "choose", "edit",
    "delete", "save", "load", "refresh", "sign", "log",
}
NFR_VOCAB = {"speed", "battery", "memory", "responsiveness", "scalability", "performance", "throughput", "latency", "startup", "load_time"}
NIELSEN_VOCAB = {"visibility", "match-real-world", "user-control", "consistency", "error-prevention", "recognition-over-recall", "flexibility", "aesthetic", "error-recovery", "help-documentation", "match real world", "user control", "error prevention", "recognition over recall", "error recovery", "help documentation", "minimalist", "minimalist design"}
GENERIC_COMPONENT_PHRASES = {"the app", "this app", "app", "general", "overall", "various", "multiple", "everything", "all", "the application"}

REQUIRED = {
    "bug_report":      ["title", "description", "steps_to_reproduce", "expected_behavior", "actual_behavior", "severity", "affected_component"],
    "feature_request": ["title", "description", "user_story", "acceptance_criteria", "severity", "affected_component"],
    "performance":     ["title", "description", "nfr_category", "severity", "affected_component"],
    "usability":       ["title", "description", "nielsen_heuristic", "severity", "affected_component"],
    "compatibility":   ["title", "description", "device_os_matrix", "severity", "affected_component"],
}


def is_strict_nonempty(field, value, c):
    if value is None: return False
    if field == "title":
        return isinstance(value, str) and len(value.split()) >= c["min_title_words"]
    if field == "description":
        return isinstance(value, str) and len(value.split()) >= c["min_description_words"]
    if field == "affected_component":
        if not isinstance(value, str): return False
        v = value.strip().lower()
        return v not in GENERIC_COMPONENT_PHRASES and len(v.split()) >= 2
    if field == "severity":
        return isinstance(value, str) and value.upper().strip() in {"P0","P1","P2","P3"}
    if field == "expected_behavior":
        return isinstance(value, str) and len(value.split()) >= c["min_expected_actual_words"]
    if field == "actual_behavior":
        return isinstance(value, str) and len(value.split()) >= c["min_expected_actual_words"]
    if field == "steps_to_reproduce":
        if not isinstance(value, list): return False
        steps = [s for s in value if isinstance(s, str) and s.strip()]
        if len(steps) < c["min_steps"]: return False
        if not all(len(s.split()) >= c["min_words_per_step"] for s in steps): return False
        joined = " ".join(steps).lower()
        if not any(re.search(r"\b"+v+r"\b", joined) for v in ACTION_VERBS): return False
        return True
    if field == "user_story":
        if not isinstance(value, str): return False
        s = value.lower()
        return (re.search(r"\bas (a|an) ", s) and re.search(r"\bi (want|need|would like|wish) ", s) and re.search(r"\bso (that|i)\b", s)) is not None
    if field == "acceptance_criteria":
        if not isinstance(value, list): return False
        items = [a for a in value if isinstance(a, str) and a.strip()]
        if len(items) < c["min_acceptance_items"]: return False
        if not all(len(a.split()) >= c["min_words_per_acceptance"] for a in items): return False
        return True
    if field == "nfr_category":
        return isinstance(value, str) and any(w in value.lower() for w in NFR_VOCAB)
    if field == "nielsen_heuristic":
        return isinstance(value, str) and any(w in value.lower() for w in NIELSEN_VOCAB)
    if field == "device_os_matrix":
        if not isinstance(value, dict) or len(value) == 0: return False
        for k, vv in value.items():
            if isinstance(vv, list) and any(isinstance(x, str) and x.strip() for x in vv): return True
            if isinstance(vv, str) and vv.strip(): return True
        return False
    return False


def fill_rate(specs, criteria):
    rates = []
    for s in specs:
        itype = s.get("issue_type") or "bug_report"
        req = REQUIRED.get(itype, [])
        if not req: continue
        filled = sum(1 for f in req if is_strict_nonempty(f, s.get(f), criteria))
        rates.append(filled / len(req))
    return float(np.mean(rates)) if rates else None


CONDITIONS = {
    "claude_with_taxonomy": "specs_with_taxonomy.json",
    "qwen2_5_3b":           "specs_qwen2_5_3b.json",
    "claude_free_form":     "specs_free_form.json",
    "raw_summary":          "specs_raw_summary.json",
    "human_written":        "specs_human_written.json",
    "human_github":         "specs_human_github.json",
}

print("="*92)
print("Strict-criteria sensitivity sweep — leakage audit")
print("="*92)
print(f"{'condition':<22} {'default §3.8.1.x':>18} {'moderate':>14} {'very_strict':>14} {'Δ (default→very)':>18}")
print("-"*92)

results = {}
for cond, fname in CONDITIONS.items():
    fpath = BASE / fname
    if not fpath.exists():
        print(f"{cond:<22} (missing)")
        continue
    specs = json.load(open(fpath))
    if not specs: continue

    r_default = fill_rate(specs, DEFAULT)
    r_mod     = fill_rate(specs, STRICTER["moderate"])
    r_strict  = fill_rate(specs, STRICTER["very_strict"])
    delta     = r_strict - r_default if r_strict and r_default else None
    results[cond] = {
        "n": len(specs),
        "default":  r_default,
        "moderate": r_mod,
        "very_strict": r_strict,
        "delta_default_to_very_strict": delta,
    }
    print(f"{cond:<22} {r_default:>18.3f} {r_mod:>14.3f} {r_strict:>14.3f} {delta:>+18.3f}")

# Save
out = {
    "criteria_levels": {"default": DEFAULT, "moderate": STRICTER["moderate"], "very_strict": STRICTER["very_strict"]},
    "results": results,
    "interpretation": (
        "Sensitivity of the strict template-fill rate to the chosen thresholds. "
        "If Claude's rate drops materially (≥0.10) under stricter thresholds while the "
        "qualitative ranking (Claude > Qwen > human-GitHub) is preserved, the original "
        "criteria are NOT tuned to favor Claude — they sit at a defensible operating point."
    ),
}
out_path = BASE / "strict_criteria_sensitivity.json"
with open(out_path, "w") as f:
    json.dump(out, f, indent=2)
print(f"\nSaved: {out_path}")

# Conclusion
print("\nLeakage interpretation:")
if "claude_with_taxonomy" in results:
    cd = results["claude_with_taxonomy"]
    print(f"  Claude default     : {cd['default']:.3f}")
    print(f"  Claude very-strict : {cd['very_strict']:.3f}  (Δ = {cd['delta_default_to_very_strict']:+.3f})")
if "human_github" in results:
    hd = results["human_github"]
    print(f"  GitHub default     : {hd['default']:.3f}")
    print(f"  GitHub very-strict : {hd['very_strict']:.3f}")
if "claude_with_taxonomy" in results and "human_github" in results:
    gap_default = results["claude_with_taxonomy"]["default"] - results["human_github"]["default"]
    gap_strict  = results["claude_with_taxonomy"]["very_strict"] - results["human_github"]["very_strict"]
    print(f"  Claude–GitHub gap (default)     : {gap_default:+.3f}")
    print(f"  Claude–GitHub gap (very-strict) : {gap_strict:+.3f}")
    if abs(gap_strict - gap_default) < 0.05:
        print("  ⇒ Gap is robust to threshold choice (no obvious leakage).")
    else:
        print(f"  ⇒ Gap changed by {gap_strict-gap_default:+.3f} under stricter criteria.")
