#!/usr/bin/env python3
"""
Recompute Stage 3 spec metrics under STRICT content-validity criteria.

Original score_specs.py used is_nonempty(v) — passes trivially for any
filled string. This script applies stricter checks per field type so that
the resulting fill-rate is a genuine signal of substantive content, not
prompt-compliance.

STRICT criteria per field:
  title                : >= 4 words
  description          : >= 30 words
  affected_component   : >= 2 words AND not in generic-phrase blocklist
  severity             : in {P0, P1, P2, P3}
  steps_to_reproduce   : >= 3 steps, each >= 5 words, >= 1 action verb across set
  expected_behavior    : >= 8 words
  actual_behavior      : >= 8 words
  user_story           : contains all three of (As-a, I-want/I-need, so-that)
  acceptance_criteria  : >= 3 items, each >= 8 words
  nfr_category         : in ISO/IEC 25010 vocab {speed, battery, memory, ...}
  nielsen_heuristic    : in Nielsen-10 vocab
  device_os_matrix     : >= 1 device key with >= 1 OS-version value
"""

import json
import re
from pathlib import Path
from collections import defaultdict

BASE = Path(__file__).resolve().parent.parent / "data" / "processed" / "issue_specs"

# Required fields per type (matches original score_specs.py)
REQUIRED = {
    "bug_report":      ["title", "description", "steps_to_reproduce",
                        "expected_behavior", "actual_behavior",
                        "severity", "affected_component"],
    "feature_request": ["title", "description", "user_story",
                        "acceptance_criteria", "severity", "affected_component"],
    "performance":     ["title", "description", "nfr_category",
                        "severity", "affected_component"],
    "usability":       ["title", "description", "nielsen_heuristic",
                        "severity", "affected_component"],
    "compatibility":   ["title", "description", "device_os_matrix",
                        "severity", "affected_component"],
}

ACTION_VERBS = {
    "open", "tap", "click", "navigate", "swipe", "scroll", "select",
    "enter", "type", "press", "launch", "install", "uninstall", "lock",
    "unlock", "trigger", "wait", "observe", "attempt", "try", "submit",
    "send", "go", "switch", "toggle", "confirm", "choose", "edit",
    "delete", "save", "load", "refresh", "sign", "log",
}

NFR_VOCAB = {
    "speed", "battery", "memory", "responsiveness", "scalability",
    "performance", "throughput", "latency", "startup", "load_time",
}

NIELSEN_VOCAB = {
    "visibility", "match-real-world", "user-control", "consistency",
    "error-prevention", "recognition-over-recall", "flexibility",
    "aesthetic", "error-recovery", "help-documentation",
    "match real world", "user control", "error prevention",
    "recognition over recall", "error recovery", "help documentation",
    "minimalist", "minimalist design",
}

GENERIC_COMPONENT_PHRASES = {
    "the app", "this app", "app", "general", "overall",
    "various", "multiple", "everything", "all", "the application",
}


def is_strict_nonempty(field_name, value):
    """Apply strict content-validity check per field type."""

    if value is None:
        return False

    # --- string fields with min word counts ---
    if field_name == "title":
        if not isinstance(value, str): return False
        return len(value.split()) >= 4

    if field_name == "description":
        if not isinstance(value, str): return False
        return len(value.split()) >= 30

    if field_name == "affected_component":
        if not isinstance(value, str): return False
        v = value.strip().lower()
        if v in GENERIC_COMPONENT_PHRASES:
            return False
        return len(v.split()) >= 2

    if field_name == "severity":
        return isinstance(value, str) and value.upper().strip() in {"P0", "P1", "P2", "P3"}

    if field_name == "expected_behavior":
        if not isinstance(value, str): return False
        return len(value.split()) >= 8

    if field_name == "actual_behavior":
        if not isinstance(value, str): return False
        return len(value.split()) >= 8

    # --- list field: steps_to_reproduce ---
    if field_name == "steps_to_reproduce":
        if not isinstance(value, list): return False
        # Need >= 3 steps
        steps = [s for s in value if isinstance(s, str) and s.strip()]
        if len(steps) < 3:
            return False
        # Each step >= 5 words
        if not all(len(s.split()) >= 5 for s in steps):
            return False
        # At least one action verb across the step set
        joined = " ".join(steps).lower()
        if not any(re.search(r"\b" + v + r"\b", joined) for v in ACTION_VERBS):
            return False
        return True

    # --- user_story: As-a / I-want / so-that ---
    if field_name == "user_story":
        if not isinstance(value, str): return False
        s = value.lower()
        has_as_a = re.search(r"\bas (a|an) ", s) is not None
        has_iwant = re.search(r"\bi (want|need|would like|wish) ", s) is not None
        has_sothat = re.search(r"\bso (that|i)\b", s) is not None
        return has_as_a and has_iwant and has_sothat

    # --- acceptance_criteria: >= 3 items each >= 8 words ---
    if field_name == "acceptance_criteria":
        if not isinstance(value, list): return False
        items = [a for a in value if isinstance(a, str) and a.strip()]
        if len(items) < 3:
            return False
        if not all(len(a.split()) >= 8 for a in items):
            return False
        return True

    # --- nfr_category: must match ISO 25010 vocab ---
    if field_name == "nfr_category":
        if not isinstance(value, str): return False
        v = value.lower().strip()
        return any(w in v for w in NFR_VOCAB)

    # --- nielsen_heuristic: must match Nielsen-10 ---
    if field_name == "nielsen_heuristic":
        if not isinstance(value, str): return False
        v = value.lower().strip()
        return any(w in v for w in NIELSEN_VOCAB)

    # --- device_os_matrix: dict with >= 1 device key ---
    if field_name == "device_os_matrix":
        if not isinstance(value, dict): return False
        if len(value) == 0: return False
        # at least one entry with non-empty list/string of OS versions
        has_content = False
        for k, vv in value.items():
            if isinstance(vv, list) and any(isinstance(x, str) and x.strip() for x in vv):
                has_content = True
                break
            if isinstance(vv, str) and vv.strip():
                has_content = True
                break
        return has_content

    # default fallback
    return value is not None


def strict_completeness(spec):
    itype = spec.get("issue_type") or "bug_report"
    req = REQUIRED.get(itype, [])
    if not req:
        return None, 0, 0
    filled = sum(1 for f in req if is_strict_nonempty(f, spec.get(f)))
    return filled / len(req), filled, len(req)


def loose_completeness(spec):
    """The original is_nonempty check, for comparison."""
    itype = spec.get("issue_type") or "bug_report"
    req = REQUIRED.get(itype, [])
    if not req:
        return None, 0, 0
    def loose(v):
        if v is None: return False
        if isinstance(v, str): return v.strip() != ""
        if isinstance(v, list): return len(v) > 0 and any(isinstance(x, str) and x.strip() for x in v)
        if isinstance(v, dict): return len(v) > 0
        return True
    filled = sum(1 for f in req if loose(spec.get(f)))
    return filled / len(req), filled, len(req)


# -------- Run on all 4 conditions plus GitHub --------

CONDITIONS = {
    "llm_with_taxonomy":   "specs_with_taxonomy.json",
    "llm_free_form":       "specs_free_form.json",
    "raw_summary":         "specs_raw_summary.json",
    "human_written":       "specs_human_written.json",
    "human_github":        "specs_human_github.json",
    "llama_groq_70b":      "specs_llama_groq_flat.json",
    "qwen2_5_3b":          "specs_qwen2_5_3b.json",
    "qwen2_5_1_5b":        "specs_qwen2_5_1_5b.json",
}

print("=" * 100)
print(f"{'condition':<22} {'n':>4} {'loose_fill':>12} {'STRICT_fill':>14} {'Δ':>8} {'bug_steps_strict_pct':>22} {'feat_userstory_strict_pct':>26}")
print("=" * 100)

results = {}
for cond, fname in CONDITIONS.items():
    fpath = BASE / fname
    if not fpath.exists():
        print(f"{cond:<22} (file missing: {fname})")
        continue
    with open(fpath) as f:
        specs = json.load(f)
    if not specs:
        continue

    loose_ratios = []
    strict_ratios = []
    bug_steps_strict = []
    feat_userstory_strict = []
    bugs_with_loose_steps = []
    feats_with_loose_userstory = []

    for s in specs:
        l, _, _ = loose_completeness(s)
        st, _, _ = strict_completeness(s)
        if l is None:
            continue
        loose_ratios.append(l)
        strict_ratios.append(st)

        itype = s.get("issue_type")
        if itype == "bug_report":
            bug_steps_strict.append(1 if is_strict_nonempty("steps_to_reproduce", s.get("steps_to_reproduce")) else 0)
            v = s.get("steps_to_reproduce")
            bugs_with_loose_steps.append(1 if (isinstance(v, list) and len(v) > 0) else 0)
        if itype == "feature_request":
            feat_userstory_strict.append(1 if is_strict_nonempty("user_story", s.get("user_story")) else 0)
            v = s.get("user_story")
            feats_with_loose_userstory.append(1 if (isinstance(v, str) and v.strip()) else 0)

    n = len(loose_ratios)
    avg_loose = sum(loose_ratios) / n
    avg_strict = sum(strict_ratios) / n
    delta = avg_strict - avg_loose
    bs_strict_pct = (sum(bug_steps_strict) / len(bug_steps_strict) * 100) if bug_steps_strict else 0
    fu_strict_pct = (sum(feat_userstory_strict) / len(feat_userstory_strict) * 100) if feat_userstory_strict else 0

    bs_loose_pct = (sum(bugs_with_loose_steps) / len(bugs_with_loose_steps) * 100) if bugs_with_loose_steps else 0
    fu_loose_pct = (sum(feats_with_loose_userstory) / len(feats_with_loose_userstory) * 100) if feats_with_loose_userstory else 0

    results[cond] = {
        "n": n,
        "loose_fill": avg_loose,
        "strict_fill": avg_strict,
        "delta": delta,
        "bug_steps_loose_pct": bs_loose_pct,
        "bug_steps_strict_pct": bs_strict_pct,
        "feat_userstory_loose_pct": fu_loose_pct,
        "feat_userstory_strict_pct": fu_strict_pct,
        "n_bugs": len(bug_steps_strict),
        "n_feats": len(feat_userstory_strict),
    }

    print(f"{cond:<22} {n:>4} {avg_loose:>12.3f} {avg_strict:>14.3f} {delta:>+8.3f} "
          f"{bs_strict_pct:>22.1f} {fu_strict_pct:>26.1f}")

print()
print("Loose-vs-strict bug-steps and feature-userstory rates:")
print(f"{'condition':<22} {'n_bugs':>8} {'bug_steps_loose%':>20} {'bug_steps_STRICT%':>22} "
      f"{'n_feats':>8} {'feat_us_loose%':>18} {'feat_us_STRICT%':>20}")
for cond, r in results.items():
    print(f"{cond:<22} {r['n_bugs']:>8} {r['bug_steps_loose_pct']:>20.1f} {r['bug_steps_strict_pct']:>22.1f} "
          f"{r['n_feats']:>8} {r['feat_userstory_loose_pct']:>18.1f} {r['feat_userstory_strict_pct']:>20.1f}")

# Save results
out_path = Path(str(Path(__file__).resolve().parent.parent) + "/data/processed/issue_specs_5dim/strict_validity_recomputation.json")
out_path.parent.mkdir(parents=True, exist_ok=True)
with open(out_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved to: {out_path}")
