"""
Benchmark our aspect extraction (heuristic + local-LLM) against the
Guzman & Maalej 2014 gold-standard aspect dataset.

Input:  data/raw/guzman/guzman_reviews.json
        2,062 sentences from 8 apps; 971 sentences carry 1,040 aspect annotations.
        Each gold annotation is {aspect: str, sentiment: str, intensity: float}.

Outputs:
  data/processed/guzman_benchmark/
    heuristic_results.json    per-sentence extraction + match status
    summary.json              precision/recall/F1 (micro + macro) + per-app
    summary.txt               human-readable

Matching policy:
  We accept three match levels (paper-defensible):
    1. exact:      predicted aspect string == gold aspect string (lowercased, stripped)
    2. lemma:      stemmed/lemma forms match (e.g., "install" == "installs" == "installed")
    3. substring:  predicted contains gold or gold contains predicted (length >=3)

  We report precision/recall/F1 at all three match levels separately.
"""

import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean

import spacy

# Reuse the heuristic extractor from the existing pipeline
sys.path.insert(0, str(Path(__file__).parent.parent))
from scripts.extract_aspects_heuristic import extract_aspects


GUZMAN_PATH = Path("data/raw/guzman/guzman_reviews.json")
OUT_DIR     = Path("data/processed/guzman_benchmark")


def normalize(s: str) -> str:
    """Lowercase, strip, remove leading/trailing punctuation."""
    if s is None:
        return ""
    s = s.lower().strip()
    s = re.sub(r"^[^a-z0-9]+|[^a-z0-9]+$", "", s)
    return s


def lemmatize_word(s: str, nlp) -> str:
    """Lemmatize a phrase (just the head noun for short phrases)."""
    s = normalize(s)
    if not s:
        return ""
    doc = nlp(s)
    if not len(doc):
        return s
    # Lemmatize each token, rejoin
    lemmas = [tok.lemma_ for tok in doc]
    return " ".join(lemmas).strip().lower()


def match_aspects(predicted: list[str], gold: list[str], nlp) -> dict:
    """
    Compute TP/FP/FN at three match levels.
    Predicted and gold are lists of aspect strings (already pre-normalized).
    """
    pred_norm = set(normalize(p) for p in predicted if normalize(p))
    gold_norm = set(normalize(g) for g in gold if normalize(g))

    pred_lemma = set(lemmatize_word(p, nlp) for p in predicted if p)
    gold_lemma = set(lemmatize_word(g, nlp) for g in gold if g)

    # Exact (normalized)
    tp_exact = pred_norm & gold_norm
    fp_exact = pred_norm - gold_norm
    fn_exact = gold_norm - pred_norm

    # Lemma
    tp_lemma = pred_lemma & gold_lemma
    fp_lemma = pred_lemma - gold_lemma
    fn_lemma = gold_lemma - pred_lemma

    # Substring (most permissive)
    tp_sub = set()
    matched_gold = set()
    for p in pred_norm:
        for g in gold_norm:
            if g in matched_gold:
                continue
            if len(p) >= 3 and len(g) >= 3 and (p in g or g in p):
                tp_sub.add(p)
                matched_gold.add(g)
                break
    fp_sub = pred_norm - tp_sub
    fn_sub = gold_norm - matched_gold

    return {
        "predicted": sorted(pred_norm),
        "gold": sorted(gold_norm),
        "exact":     {"tp": sorted(tp_exact), "fp": sorted(fp_exact), "fn": sorted(fn_exact)},
        "lemma":     {"tp": sorted(tp_lemma), "fp": sorted(fp_lemma), "fn": sorted(fn_lemma)},
        "substring": {"tp": sorted(tp_sub),   "fp": sorted(fp_sub),   "fn": sorted(fn_sub)},
    }


def metrics(tp: int, fp: int, fn: int) -> dict:
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    return {"precision": round(p, 4), "recall": round(r, 4), "f1": round(f1, 4)}


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading GUZMAN: {GUZMAN_PATH}")
    data = json.load(open(GUZMAN_PATH))
    print(f"  {len(data):,} sentences")
    n_with_aspects = sum(1 for r in data if r.get("aspects"))
    n_total_aspects = sum(len(r.get("aspects", [])) for r in data)
    print(f"  {n_with_aspects:,} sentences with at least one aspect")
    print(f"  {n_total_aspects:,} total gold aspect annotations")

    print("\nLoading spaCy en_core_web_sm")
    nlp = spacy.load("en_core_web_sm", disable=["ner"])

    print("\nRunning heuristic aspect extractor on all 2,062 sentences...")
    per_sentence = []
    for i, r in enumerate(data):
        text = r["text"]
        gold_aspects = [a["aspect"] for a in r.get("aspects", [])]
        predicted = extract_aspects(text, nlp, kw_model=None)
        result = match_aspects(predicted, gold_aspects, nlp)
        per_sentence.append({
            "sentence_id": f"{r['review_id']}_{r.get('sentence_id', 0)}",
            "app_id": r.get("app_id"),
            "text": text,
            "n_gold": len(gold_aspects),
            "n_predicted": len(predicted),
            **result,
        })
        if (i + 1) % 200 == 0:
            print(f"  {i+1:,} / {len(data):,}", flush=True)

    # Aggregate metrics — overall (only sentences that have gold aspects)
    summary = {}
    for level in ["exact", "lemma", "substring"]:
        # Micro: aggregate TP/FP/FN counts across all sentences
        tp = sum(len(s[level]["tp"]) for s in per_sentence)
        fp = sum(len(s[level]["fp"]) for s in per_sentence)
        fn = sum(len(s[level]["fn"]) for s in per_sentence)
        micro = metrics(tp, fp, fn)

        # Macro: average per-sentence F1 over sentences with at least one gold OR pred
        per_sent_f1s = []
        per_sent_p = []
        per_sent_r = []
        for s in per_sentence:
            tp_s = len(s[level]["tp"])
            fp_s = len(s[level]["fp"])
            fn_s = len(s[level]["fn"])
            if tp_s + fp_s + fn_s == 0:
                continue
            m = metrics(tp_s, fp_s, fn_s)
            per_sent_f1s.append(m["f1"])
            per_sent_p.append(m["precision"])
            per_sent_r.append(m["recall"])

        macro_f1 = round(mean(per_sent_f1s), 4) if per_sent_f1s else 0
        macro_p = round(mean(per_sent_p), 4) if per_sent_p else 0
        macro_r = round(mean(per_sent_r), 4) if per_sent_r else 0

        # Recall on the 971 sentences with at least one gold aspect
        sub_p = []
        sub_r = []
        sub_f1 = []
        for s in per_sentence:
            if s["n_gold"] == 0:
                continue
            tp_s = len(s[level]["tp"])
            fp_s = len(s[level]["fp"])
            fn_s = len(s[level]["fn"])
            if tp_s + fp_s + fn_s == 0:
                continue
            m = metrics(tp_s, fp_s, fn_s)
            sub_p.append(m["precision"])
            sub_r.append(m["recall"])
            sub_f1.append(m["f1"])

        summary[level] = {
            "micro": {
                **micro,
                "tp": tp, "fp": fp, "fn": fn,
            },
            "macro_all_sentences": {"precision": macro_p, "recall": macro_r, "f1": macro_f1,
                                     "n_sentences": len(per_sent_f1s)},
            "macro_gold_only_sentences": {
                "precision": round(mean(sub_p), 4) if sub_p else 0,
                "recall":    round(mean(sub_r), 4) if sub_r else 0,
                "f1":        round(mean(sub_f1), 4) if sub_f1 else 0,
                "n_sentences": len(sub_p),
            },
        }

    # Per-app breakdown (substring level)
    by_app = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0, "n": 0})
    for s in per_sentence:
        if s["n_gold"] == 0:
            continue
        a = s["app_id"]
        by_app[a]["tp"] += len(s["substring"]["tp"])
        by_app[a]["fp"] += len(s["substring"]["fp"])
        by_app[a]["fn"] += len(s["substring"]["fn"])
        by_app[a]["n"] += 1
    per_app = {}
    for app, c in by_app.items():
        m = metrics(c["tp"], c["fp"], c["fn"])
        per_app[app] = {"n_sentences": c["n"], **m, "tp": c["tp"], "fp": c["fp"], "fn": c["fn"]}

    out = {
        "input_file": str(GUZMAN_PATH),
        "n_sentences_total": len(data),
        "n_sentences_with_gold_aspects": n_with_aspects,
        "n_total_gold_aspects": n_total_aspects,
        "summary_by_match_level": summary,
        "per_app_substring": per_app,
        "extractor": "heuristic (spaCy NP + patterns + COMMON_ASPECTS vocab)",
    }
    with open(OUT_DIR / "summary.json", "w") as f:
        json.dump(out, f, indent=2)

    with open(OUT_DIR / "heuristic_results.json", "w") as f:
        json.dump(per_sentence, f)

    # Human-readable
    lines = [
        "="*78,
        "GUZMAN ASPECT-EXTRACTION BENCHMARK",
        "="*78,
        f"Source: Guzman & Maalej 2014 (via Dabrowski IS-22 alt corpus)",
        f"Sentences: {len(data):,}  (with gold aspects: {n_with_aspects:,})",
        f"Total gold aspect annotations: {n_total_aspects:,}",
        f"Extractor: heuristic (spaCy NP-chunking + regex patterns + common-aspect vocab)",
        "",
        "Match levels:",
        "  exact:     predicted == gold (case-insensitive, normalized)",
        "  lemma:     spaCy lemmatized forms match",
        "  substring: predicted contains gold OR gold contains predicted (>=3 chars)",
        "",
        "="*78,
        "RESULTS",
        "="*78,
        f"{'level':12s} {'micro_P':>10} {'micro_R':>10} {'micro_F1':>10}  "
        f"{'macro_P':>9} {'macro_R':>9} {'macro_F1':>10}",
        "-"*78,
    ]
    for level in ["exact", "lemma", "substring"]:
        m = summary[level]["micro"]
        macro = summary[level]["macro_gold_only_sentences"]
        lines.append(f"{level:12s} {m['precision']:>10.4f} {m['recall']:>10.4f} {m['f1']:>10.4f}  "
                     f"{macro['precision']:>9.4f} {macro['recall']:>9.4f} {macro['f1']:>10.4f}")
    lines.append("")
    lines.append(f"(macro is averaged over {summary['substring']['macro_gold_only_sentences']['n_sentences']} "
                 f"sentences with at least one gold aspect)")
    lines.append("")
    lines.append("Per-app (substring level):")
    lines.append(f"{'app':35s} {'n_sent':>6} {'P':>8} {'R':>8} {'F1':>8}")
    lines.append("-"*78)
    for app, m in sorted(per_app.items(), key=lambda x: -x[1]["n_sentences"]):
        lines.append(f"{app:35s} {m['n_sentences']:>6,} {m['precision']:>8.4f} "
                     f"{m['recall']:>8.4f} {m['f1']:>8.4f}")

    summary_text = "\n".join(lines)
    print("\n" + summary_text)
    with open(OUT_DIR / "summary.txt", "w") as f:
        f.write(summary_text)

    print(f"\nSaved {OUT_DIR}/summary.json")
    print(f"Saved {OUT_DIR}/summary.txt")
    print(f"Saved {OUT_DIR}/heuristic_results.json (per-sentence, large)")


if __name__ == "__main__":
    main()
