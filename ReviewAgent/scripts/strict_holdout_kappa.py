"""
Strict-held-out Cohen's κ on the gold-standard subset disjoint from V5 training.

Replays the V5 training-corpus build (stratified cap=15000 per class, seed=42)
to identify which 215K-row indices V5 saw at training. Then computes κ vs
expert on the unseen subset only — addressing the train/eval text-overlap
disclosure in §5.5.

Output: data/processed/expert_evaluation/strict_holdout_kappa.json
"""

import json
import random
from collections import defaultdict
from pathlib import Path

from numbers_parser import Document
from sklearn.metrics import cohen_kappa_score, accuracy_score, f1_score

LABELS = ["bug_report", "feature_request", "performance", "usability",
          "compatibility", "praise", "other"]
LABEL_NORMALIZE = {
    "bug report": "bug_report", "bug_report": "bug_report",
    "feature request": "feature_request", "feature_request": "feature_request",
    "performance": "performance", "usability": "usability",
    "compatibility": "compatibility", "praise": "praise", "other": "other",
}


def normalize(lbl):
    if lbl is None:
        return None
    return LABEL_NORMALIZE.get(str(lbl).strip().lower())


def load_expert_labels():
    doc = Document("annotator_materials/annotator_A.numbers")
    for sheet in doc.sheets:
        for tbl in sheet.tables:
            rows = tbl.rows(values_only=True)
            if not rows:
                continue
            header = rows[0]
            if not any(h and "correct_yn" in str(h).lower() for h in header):
                continue
            col = {h: i for i, h in enumerate(header) if h}
            expert = {}
            for r in rows[1:]:
                row_id = r[col["row_id"]]
                if row_id is None:
                    continue
                yn = str(r[col["correct_yn"]] or "").strip().upper()
                pred = r[col["predicted_label"]]
                final = r[col.get("correct_label_if_no")] if col.get("correct_label_if_no") is not None else None
                if yn == "Y":
                    label = normalize(pred)
                elif yn == "N":
                    label = normalize(final)
                else:
                    continue
                if label in LABELS:
                    expert[int(row_id)] = label
            return expert
    return {}


def replay_v5_seen_indices():
    v5 = json.load(open("data/processed/rrgen_v5_training.json"))
    by_class = defaultdict(list)
    for i, r in enumerate(v5):
        by_class[r.get("final_label")].append(i)
    rng = random.Random(42)
    selected = []
    for cls, idxs in by_class.items():
        rng.shuffle(idxs)
        selected.extend(idxs[:15000])
    return {i for i in selected if i < 215583}


def metrics(expert, classifier, indices, name):
    e, c = [], []
    for idx in indices:
        if idx in expert:
            cl = classifier.get(str(idx)) or classifier.get(idx)
            cl = normalize(cl)
            if cl in LABELS:
                e.append(expert[idx]); c.append(cl)
    if not e:
        return {"name": name, "n": 0}
    return {
        "name": name,
        "n": len(e),
        "accuracy": round(accuracy_score(e, c), 4),
        "cohen_kappa": round(cohen_kappa_score(e, c, labels=LABELS), 4),
        "macro_f1": round(f1_score(e, c, labels=LABELS, average="macro", zero_division=0), 4),
    }


def main():
    expert = load_expert_labels()
    print(f"Expert labels loaded: {len(expert)}")

    v5_seen = replay_v5_seen_indices()
    mk = json.load(open("annotator_materials/master_key.json"))
    gold_idx = mk["main_indices"]
    seen = [i for i in gold_idx if i in v5_seen]
    unseen = [i for i in gold_idx if i not in v5_seen]
    print(f"V5 seen during training: {len(seen)} / {len(gold_idx)}")
    print(f"Strict held-out subset:  {len(unseen)} / {len(gold_idx)}")

    out = {
        "n_total_gold": len(gold_idx),
        "n_v5_seen_during_training": len(seen),
        "n_strict_held_out": len(unseen),
    }

    print("\n--- Metrics on FULL 490 (current paper headline) ---")
    for name, lbls in [
        ("V2 LLM", mk["main_v2_labels"]),
        ("cleanlab corrected_v2", mk["main_corrected_v2_labels"]),
        ("V5 production", mk["main_v5_labels"]),
    ]:
        m = metrics(expert, lbls, gold_idx, name)
        print(f"  {m['name']:25s} n={m['n']:3d} acc={m.get('accuracy', 0):.3f} κ={m.get('cohen_kappa', 0):.3f}")
        out.setdefault("full_490", {})[name] = m

    print("\n--- Metrics on STRICT HELD-OUT subset (V5 unseen) ---")
    for name, lbls in [
        ("V2 LLM", mk["main_v2_labels"]),
        ("cleanlab corrected_v2", mk["main_corrected_v2_labels"]),
        ("V5 production", mk["main_v5_labels"]),
    ]:
        m = metrics(expert, lbls, unseen, name)
        print(f"  {m['name']:25s} n={m['n']:3d} acc={m.get('accuracy', 0):.3f} κ={m.get('cohen_kappa', 0):.3f}")
        out.setdefault("strict_held_out", {})[name] = m

    print("\n--- Metrics on V5-SEEN subset (for contrast) ---")
    for name, lbls in [
        ("V2 LLM", mk["main_v2_labels"]),
        ("cleanlab corrected_v2", mk["main_corrected_v2_labels"]),
        ("V5 production", mk["main_v5_labels"]),
    ]:
        m = metrics(expert, lbls, seen, name)
        print(f"  {m['name']:25s} n={m['n']:3d} acc={m.get('accuracy', 0):.3f} κ={m.get('cohen_kappa', 0):.3f}")
        out.setdefault("v5_seen_only", {})[name] = m

    Path("data/processed/expert_evaluation").mkdir(parents=True, exist_ok=True)
    out_path = Path("data/processed/expert_evaluation/strict_holdout_kappa.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
