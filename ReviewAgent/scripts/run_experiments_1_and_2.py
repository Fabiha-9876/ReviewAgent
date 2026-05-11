"""
Run Experiment 1 (Stage 3 IssueSpec quality) + Experiment 2 (Stage 4b response quality).

Experiment 1 — Stage 3 IssueSpec quality across 3 conditions:
  (a) llm_with_taxonomy
  (b) llm_free_form
  (c) raw_summary

Metrics (no reference needed):
  - completeness_ratio:   % of issue-type-appropriate fields filled
  - description_length:   word distribution
  - type_template_adherence: does each spec use the right template fields?
  - severity_distribution: P0..P3 spread

Experiment 2 — Stage 4b response quality across 4 conditions:
  (1) rrgen_baseline   (no RAG, no spec)
  (2) prompt_baseline    (no RAG, no spec, with system guidance)
  (3) reviewagent_no_spec (RAG only)
  (4) reviewagent_full (RAG + IssueSpec)

Metrics:
  - BLEU-1/2/3/4 vs original_response (reference from RRGen)
  - ROUGE-L
  - BERTScore F1
  - response_length, distinct-1, distinct-2

Outputs:
  data/processed/experiments/exp1_results.json
  data/processed/experiments/exp2_results.json
  data/processed/experiments/exp1_summary.txt
  data/processed/experiments/exp2_summary.txt
"""

import json
import re
from collections import Counter
from pathlib import Path
from statistics import mean, median

# ---------------- BLEU / ROUGE-L (pure-python, no nltk dep) ----------------

def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", (text or "").lower())


def ngrams(tokens, n):
    return [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]


def bleu_score(reference: str, candidate: str, max_n: int = 4) -> dict:
    """Sentence-level BLEU 1..n with brevity penalty."""
    ref = tokenize(reference)
    cand = tokenize(candidate)
    if not cand:
        return {f"bleu_{i}": 0.0 for i in range(1, max_n+1)}
    scores = {}
    import math
    for n in range(1, max_n+1):
        ref_ngrams = Counter(ngrams(ref, n))
        cand_ngrams = Counter(ngrams(cand, n))
        if not cand_ngrams:
            scores[f"bleu_{n}"] = 0.0
            continue
        match = sum(min(c, ref_ngrams.get(ng, 0)) for ng, c in cand_ngrams.items())
        precision = match / sum(cand_ngrams.values())
        bp = 1.0 if len(cand) >= len(ref) else math.exp(1 - len(ref)/max(1, len(cand)))
        scores[f"bleu_{n}"] = bp * precision
    return scores


def rouge_l(reference: str, candidate: str) -> float:
    """ROUGE-L F1."""
    ref = tokenize(reference)
    cand = tokenize(candidate)
    if not ref or not cand:
        return 0.0
    # LCS dp
    m, n = len(ref), len(cand)
    dp = [[0]*(n+1) for _ in range(m+1)]
    for i in range(m):
        for j in range(n):
            if ref[i] == cand[j]:
                dp[i+1][j+1] = dp[i][j] + 1
            else:
                dp[i+1][j+1] = max(dp[i+1][j], dp[i][j+1])
    lcs = dp[m][n]
    if lcs == 0:
        return 0.0
    p = lcs / n
    r = lcs / m
    return 2*p*r/(p+r) if (p+r) else 0.0


def distinct_n(texts: list[str], n: int) -> float:
    """Distinct-n: # unique n-grams / total n-grams across all texts."""
    all_grams = []
    for t in texts:
        all_grams.extend(ngrams(tokenize(t), n))
    if not all_grams:
        return 0.0
    return len(set(all_grams)) / len(all_grams)


# ---------------- Experiment 1 — IssueSpec quality ----------------

def schema_fields_required(issue_type: str) -> list[str]:
    """Required non-null fields for each issue_type per the schema."""
    base = ["title", "description", "severity", "affected_component"]
    if issue_type == "bug_report":
        return base + ["steps_to_reproduce", "expected_behavior", "actual_behavior"]
    elif issue_type == "feature_request":
        return base + ["user_story", "acceptance_criteria"]
    elif issue_type == "performance":
        return base + ["nfr_category"]
    elif issue_type == "usability":
        return base + ["nielsen_heuristic"]
    elif issue_type == "compatibility":
        return base + ["device_os_matrix"]
    return base


def completeness(spec: dict) -> float:
    required = schema_fields_required(spec.get("issue_type", ""))
    filled = sum(1 for f in required if spec.get(f) not in (None, "", [], {}))
    return filled / len(required) if required else 0.0


def template_adherence(spec: dict) -> bool:
    """True if the spec uses the type-appropriate template fields and DOESN'T leak others."""
    it = spec.get("issue_type", "")
    type_specific = {
        "bug_report": {"steps_to_reproduce", "expected_behavior", "actual_behavior"},
        "feature_request": {"user_story", "acceptance_criteria"},
        "performance": {"nfr_category"},
        "usability": {"nielsen_heuristic"},
        "compatibility": {"device_os_matrix"},
    }
    expected_fields = type_specific.get(it, set())
    foreign_fields = set().union(*type_specific.values()) - expected_fields
    has_expected = all(spec.get(f) not in (None, "", [], {}) for f in expected_fields)
    no_foreign = all(spec.get(f) in (None, "", [], {}) for f in foreign_fields)
    return has_expected and no_foreign


def run_experiment_1():
    out_dir = Path("data/processed/experiments")
    out_dir.mkdir(parents=True, exist_ok=True)

    conditions = {
        "llm_with_taxonomy": "data/processed/issue_specs/specs_with_taxonomy.json",
        "llm_free_form":     "data/processed/issue_specs/specs_free_form.json",
        "raw_summary":       "data/processed/issue_specs/specs_raw_summary.json",
    }

    results = {}
    for cond, path in conditions.items():
        specs = json.load(open(path))
        comp_scores = [completeness(s) for s in specs]
        adherence = sum(1 for s in specs if template_adherence(s))
        desc_lengths = [len((s.get("description") or "").split()) for s in specs]
        sev_dist = Counter(s.get("severity") for s in specs)
        type_dist = Counter(s.get("issue_type") for s in specs)
        results[cond] = {
            "n": len(specs),
            "completeness_ratio_mean": round(mean(comp_scores), 4),
            "completeness_ratio_median": round(median(comp_scores), 4),
            "completeness_pct_above_80": round(100 * sum(1 for c in comp_scores if c >= 0.8) / len(comp_scores), 2),
            "template_adherence_pct": round(100 * adherence / len(specs), 2),
            "description_length_mean_words": round(mean(desc_lengths), 1),
            "description_length_median_words": round(median(desc_lengths), 1),
            "description_length_min": min(desc_lengths),
            "description_length_max": max(desc_lengths),
            "severity_distribution": dict(sev_dist),
            "issue_type_distribution": dict(type_dist),
        }

    with open(out_dir / "exp1_results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Human-readable summary
    lines = ["="*78, "EXPERIMENT 1 — Stage 3 IssueSpec Quality", "="*78, ""]
    lines.append(f"{'metric':35s} {'(a)taxonomy':>14} {'(b)free-form':>14} {'(c)raw-sum':>14}")
    lines.append("-"*78)
    for metric in ["completeness_ratio_mean", "template_adherence_pct",
                   "description_length_mean_words", "description_length_median_words"]:
        row = f"{metric:35s}"
        for cond in ["llm_with_taxonomy", "llm_free_form", "raw_summary"]:
            row += f" {results[cond][metric]:>14.4g}"
        lines.append(row)
    lines.append("")
    lines.append("Severity distribution:")
    for cond in ["llm_with_taxonomy", "llm_free_form", "raw_summary"]:
        lines.append(f"  {cond:25s} {results[cond]['severity_distribution']}")

    summary = "\n".join(lines)
    print(summary)
    with open(out_dir / "exp1_summary.txt", "w") as f:
        f.write(summary)
    return results


# ---------------- Experiment 2 — Response quality ----------------

def run_experiment_2():
    out_dir = Path("data/processed/experiments")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load all 4 condition outputs
    conditions = {
        "rrgen_baseline":      "data/processed/responses/responses_rrgen_baseline.json",
        "prompt_baseline":       "data/processed/responses/responses_prompt_baseline.json",
        "reviewagent_no_spec": "data/processed/responses/responses_reviewagent_no_spec.json",
        "reviewagent_full":    "data/processed/responses/responses_reviewagent_full.json",
    }
    cond_data = {k: json.load(open(v)) for k, v in conditions.items()}

    # Build cluster_id -> reference_response map from rrgen_v5_relabeled (original_response field)
    print("Loading reference responses...")
    relabel = json.load(open("data/processed/rrgen_v5_relabeled/rrgen_v5_relabeled.json"))
    sample = json.load(open("data/processed/responses/sample_100_reviews_with_rag.json"))
    sample_review_idxs = [s["review_index"] for s in sample]
    cluster_to_review_text = {s["cluster_id"]: s["review_text"] for s in sample}

    # We need to find each cluster's representative review's original_response
    # Easiest: for each sample's review_text, find the matching row in relabel
    review_to_orig_resp = {}
    relabel_text_index = {r["text"]: r.get("original_response", "") for r in relabel}
    for s in sample:
        review_to_orig_resp[s["cluster_id"]] = relabel_text_index.get(s["review_text"], "")

    n_with_ref = sum(1 for v in review_to_orig_resp.values() if v)
    print(f"  Reference responses available for {n_with_ref}/100 reviews")

    # Compute metrics per condition
    results = {}
    for cond, items in cond_data.items():
        bleu_1, bleu_2, bleu_3, bleu_4 = [], [], [], []
        rouge_scores = []
        cand_texts = []
        cand_lengths = []
        for resp in items:
            ref = review_to_orig_resp.get(resp["cluster_id"], "")
            if not ref:
                continue
            cand = resp["response_text"]
            cand_texts.append(cand)
            cand_lengths.append(len(cand.split()))
            b = bleu_score(ref, cand)
            bleu_1.append(b["bleu_1"])
            bleu_2.append(b["bleu_2"])
            bleu_3.append(b["bleu_3"])
            bleu_4.append(b["bleu_4"])
            rouge_scores.append(rouge_l(ref, cand))

        results[cond] = {
            "n_evaluated": len(bleu_1),
            "bleu_1_mean": round(mean(bleu_1), 4) if bleu_1 else 0,
            "bleu_2_mean": round(mean(bleu_2), 4) if bleu_2 else 0,
            "bleu_3_mean": round(mean(bleu_3), 4) if bleu_3 else 0,
            "bleu_4_mean": round(mean(bleu_4), 4) if bleu_4 else 0,
            "rouge_l_mean": round(mean(rouge_scores), 4) if rouge_scores else 0,
            "response_length_mean_words": round(mean(cand_lengths), 1) if cand_lengths else 0,
            "response_length_median_words": round(median(cand_lengths), 1) if cand_lengths else 0,
            "distinct_1": round(distinct_n(cand_texts, 1), 4),
            "distinct_2": round(distinct_n(cand_texts, 2), 4),
        }

    # Try BERTScore if available
    try:
        from bert_score import score as bert_score_fn
        print("\nComputing BERTScore (slow on first run — model download)...")
        for cond in conditions:
            ref_list, cand_list = [], []
            for resp in cond_data[cond]:
                ref = review_to_orig_resp.get(resp["cluster_id"], "")
                if ref:
                    ref_list.append(ref)
                    cand_list.append(resp["response_text"])
            if ref_list:
                P, R, F1 = bert_score_fn(cand_list, ref_list, lang="en", verbose=False)
                results[cond]["bertscore_f1_mean"] = round(F1.mean().item(), 4)
                results[cond]["bertscore_precision_mean"] = round(P.mean().item(), 4)
                results[cond]["bertscore_recall_mean"] = round(R.mean().item(), 4)
    except ImportError:
        print("\nbert_score not installed — skipping BERTScore. (pip install bert-score to enable)")
    except Exception as e:
        print(f"\nBERTScore failed: {e}")

    with open(out_dir / "exp2_results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Human-readable summary
    lines = ["="*78, "EXPERIMENT 2 — Stage 4b Response Quality", "="*78, ""]
    lines.append(f"References from RRGen original_response field: {n_with_ref}/100")
    lines.append("")
    metrics = ["bleu_1_mean", "bleu_2_mean", "bleu_3_mean", "bleu_4_mean",
               "rouge_l_mean", "bertscore_f1_mean",
               "response_length_mean_words", "distinct_1", "distinct_2"]
    cond_order = ["rrgen_baseline", "prompt_baseline", "reviewagent_no_spec", "reviewagent_full"]
    lines.append(f"{'metric':28s}" + "".join(f"{c[:14]:>14}" for c in cond_order))
    lines.append("-"*84)
    for metric in metrics:
        row = f"{metric:28s}"
        for c in cond_order:
            v = results[c].get(metric, "—")
            if isinstance(v, float):
                row += f"{v:>14.4f}"
            else:
                row += f"{str(v):>14}"
        lines.append(row)

    summary = "\n".join(lines)
    print("\n" + summary)
    with open(out_dir / "exp2_summary.txt", "w") as f:
        f.write(summary)
    return results


if __name__ == "__main__":
    print("\nRunning Experiment 1...")
    e1 = run_experiment_1()
    print("\n\nRunning Experiment 2...")
    e2 = run_experiment_2()
    print("\n\nAll done.")
