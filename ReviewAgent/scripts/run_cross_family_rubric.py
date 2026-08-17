"""
Cross-family five-dimension rubric comparison (reviewer-requested).

Table~\\ref{tab:multi-llm} reports template-fill across four generators but not rubric
quality, so cross-family rank preservation was only shown for structural compliance. This
script runs the same Qwen2.5-3B-Instruct judge and the same five-dimension rubric used for
the Claude condition over the other generators' specs.

Two matched panels, because the four spec sets do not cover the same ground:

  Panel A  four-way, on the clusters where all four generators produced a spec.
           These are bug_report clusters only, since the two Qwen runs cover that type.
  Panel B  Claude vs Llama-3.3-70B, stratified across all five issue types, on the
           clusters where both produced a spec.

Caveat recorded in the output: the judge is Qwen2.5-3B-Instruct and one of the judged
generators is also Qwen2.5-3B-Instruct, so that row carries a self-preference risk and is
labelled as such.

Output
    data/processed/ablations/cross_family_rubric.json
    data/processed/ablations/cross_family_rubric.txt

Usage
    python3 scripts/run_cross_family_rubric.py
"""

from __future__ import annotations

import json
import random
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO = Path(__file__).resolve().parent.parent
SPECS = REPO / "data/processed/issue_specs"
OUT = REPO / "data/processed/ablations/cross_family_rubric.json"
OUT_TXT = OUT.with_suffix(".txt")
JUDGE = "Qwen/Qwen2.5-3B-Instruct"
DIMS = ["completeness", "accuracy", "actionability", "specificity", "clarity"]
SEED = 42
PER_TYPE_PANEL_B = 6

GENERATORS = {
    "Claude (Anthropic)": ("specs_with_taxonomy.json", "flat"),
    "Llama-3.3-70B (Meta)": ("specs_llama_groq.json", "nested"),
    "Qwen2.5-3B (Alibaba)": ("specs_qwen2_5_3b.json", "flat"),
    "Qwen2.5-1.5B (Alibaba)": ("specs_qwen2_5_1_5b.json", "flat"),
}

SYS = """You are an expert software-engineering reviewer scoring a generated bug or feature issue specification on five dimensions, each on a 1-5 scale where 5 is best.

Dimensions:
1. Completeness: are all expected fields populated with substantive content?
2. Accuracy: does the spec faithfully reflect the source review evidence?
3. Actionability: can a developer act on this without going back to the user?
4. Specificity: is it specific enough to file in a defect tracker?
5. Clarity: is it clearly written and well-structured?

Output EXACTLY in this format (one line per dimension):
completeness: <1-5>
accuracy: <1-5>
actionability: <1-5>
specificity: <1-5>
clarity: <1-5>"""

FIELDS = ("title", "issue_type", "description", "severity", "affected_component",
          "steps_to_reproduce", "expected_behavior", "actual_behavior",
          "user_story", "acceptance_criteria", "nfr_category", "nielsen_heuristic",
          "device_os_matrix")


def load_specs(filename, shape):
    raw = json.load(open(SPECS / filename))
    out = {}
    for s in raw:
        cid = s.get("cluster_id")
        if not cid:
            continue
        if shape == "nested":
            body = s.get("spec_json")
            if isinstance(body, str):
                try:
                    body = json.loads(body)
                except json.JSONDecodeError:
                    continue
            if not isinstance(body, dict):
                continue
            body = dict(body)
            body.setdefault("issue_type", s.get("issue_type"))
        else:
            body = s
        out[cid] = body
    return out


def spec_to_text(spec):
    parts = []
    for k in FIELDS:
        v = spec.get(k)
        if not v:
            continue
        if isinstance(v, list):
            v = "; ".join(str(x)[:80] for x in v[:5])
        elif isinstance(v, dict):
            v = str(v)[:200]
        parts.append(f"{k}: {v}")
    return "\n".join(parts)


def main():
    print("Loading spec sets", file=sys.stderr)
    sets = {name: load_specs(fn, shape) for name, (fn, shape) in GENERATORS.items()}
    for name, d in sets.items():
        print(f"  {name}: {len(d)} specs", file=sys.stderr)

    claude = sets["Claude (Anthropic)"]
    types = {cid: s.get("issue_type") for cid, s in claude.items()}

    common4 = sorted(set.intersection(*(set(d) for d in sets.values())))
    common2 = sorted(set(claude) & set(sets["Llama-3.3-70B (Meta)"]))

    rng = random.Random(SEED)
    by_type = defaultdict(list)
    for cid in common2:
        by_type[types.get(cid, "?")].append(cid)
    panel_b = []
    for t, lst in sorted(by_type.items()):
        rng.shuffle(lst)
        panel_b.extend(lst[:PER_TYPE_PANEL_B])
    panel_b = sorted(panel_b)

    print(f"  panel A (all four): {len(common4)} clusters", file=sys.stderr)
    print(f"  panel B (Claude vs Llama): {len(panel_b)} clusters", file=sys.stderr)

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Loading judge {JUDGE} on {device}", file=sys.stderr)
    tok = AutoTokenizer.from_pretrained(JUDGE)
    model = AutoModelForCausalLM.from_pretrained(
        JUDGE, torch_dtype=torch.float16 if device != "cpu" else torch.float32
    ).to(device).eval()

    cache = {}

    def judge(text):
        if text in cache:
            return cache[text]
        msgs = [{"role": "system", "content": SYS},
                {"role": "user", "content": f"IssueSpec to score:\n{text[:2000]}\n\nScore on the 5 dimensions:"}]
        chat = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        enc = tok(chat, return_tensors="pt").to(device)
        with torch.inference_mode():
            gen = model.generate(**enc, max_new_tokens=80, do_sample=False,
                                 pad_token_id=tok.eos_token_id)
        decoded = tok.decode(gen[0, enc["input_ids"].shape[1]:], skip_special_tokens=True)
        scores = {}
        for line in decoded.splitlines():
            m = re.match(r"\s*(completeness|accuracy|actionability|specificity|clarity)\s*:\s*([1-5])",
                         line.lower())
            if m:
                scores[m.group(1)] = int(m.group(2))
        cache[text] = scores
        return scores

    def score_panel(clusters, generators, tag):
        result = {}
        t0 = time.time()
        for name in generators:
            per_spec, rows = [], []
            for i, cid in enumerate(clusters, 1):
                spec = sets[name].get(cid)
                text = spec_to_text(spec) if spec else ""
                if len(text) < 50:
                    continue
                sc = judge(text)
                if len(sc) < len(DIMS):
                    continue
                rows.append({"cluster_id": cid, "issue_type": types.get(cid), **sc})
                per_spec.append(sum(sc[d] for d in DIMS) / len(DIMS))
                print(f"  [{tag}] {name} {i}/{len(clusters)}", file=sys.stderr, end="\r")
            if not per_spec:
                continue
            result[name] = {
                "n_scored": len(per_spec),
                "overall_mean": round(sum(per_spec) / len(per_spec), 3),
                "per_dim": {d: round(sum(r[d] for r in rows) / len(rows), 2) for d in DIMS},
                "rows": rows,
            }
            print(f"  [{tag}] {name}: n={len(per_spec)} mean={result[name]['overall_mean']:.3f}"
                  f"   ({time.time() - t0:.0f}s)", file=sys.stderr)
        return result

    panel_a_res = score_panel(common4, list(GENERATORS), "A")
    panel_b_res = score_panel(panel_b, ["Claude (Anthropic)", "Llama-3.3-70B (Meta)"], "B")

    report = {
        "judge": JUDGE,
        "rubric_dimensions": DIMS,
        "panel_a": {"description": "four-way, clusters covered by all four generators "
                                   "(bug_report only)",
                    "n_clusters": len(common4), "results": panel_a_res},
        "panel_b": {"description": "Claude vs Llama-3.3-70B, stratified across five issue types",
                    "n_clusters": len(panel_b), "results": panel_b_res},
        "caveat": "The judge is Qwen2.5-3B-Instruct and one judged generator is also "
                  "Qwen2.5-3B-Instruct; that row carries a self-preference risk.",
    }
    OUT.write_text(json.dumps(report, indent=2))

    L = ["=" * 78, "CROSS-FAMILY 5-DIMENSION RUBRIC (judge: Qwen2.5-3B-Instruct)", "=" * 78, ""]
    for key, title in (("panel_a", f"PANEL A - four-way, {len(common4)} matched bug_report clusters"),
                       ("panel_b", f"PANEL B - Claude vs Llama, {len(panel_b)} clusters, all five types")):
        L += [title, "-" * 78,
              f"  {'generator':26s}{'n':>4s}{'mean':>7s}" + "".join(f"{d[:5]:>7s}" for d in DIMS)]
        for name, v in report[key]["results"].items():
            L.append(f"  {name:26s}{v['n_scored']:>4d}{v['overall_mean']:>7.2f}"
                     + "".join(f"{v['per_dim'][d]:>7.2f}" for d in DIMS))
        L.append("")
    L += ["Caveat: " + report["caveat"], ""]
    OUT_TXT.write_text("\n".join(L) + "\n")
    print("\n" + "\n".join(L))
    print(f"Saved {OUT}")


if __name__ == "__main__":
    main()
