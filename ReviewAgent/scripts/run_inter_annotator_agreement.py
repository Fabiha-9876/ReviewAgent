"""
Aim 1 inter-annotator agreement — using LLM annotators as additional raters
(Gilardi et al. 2023 PNAS established this methodology).

Three "annotators" rate the same 100 reviews from the 490 expert gold standard:
  Annotator-1 (E):  Lead author (you) — already in annotator_A.numbers
  Annotator-2 (L1): Qwen2.5-3B + concise role-based prompt
  Annotator-3 (L2): Qwen2.5-3B + chain-of-thought prompt

We compute pairwise Cohen's κ + Krippendorff's α across the three.

Outputs:
    data/processed/inter_annotator/
      llm_annotations.json     all 3 raters' labels per review
      agreement_summary.json   κ, α, per-class breakdown
      summary.txt              human-readable
"""

import json
import random
import re
import sys
import time
from collections import Counter
from pathlib import Path
from statistics import mean

import torch
from sklearn.metrics import cohen_kappa_score
from transformers import AutoModelForCausalLM, AutoTokenizer
from numbers_parser import Document

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.stage1.classifier import LABELS

OUT_DIR = Path("data/processed/inter_annotator")
OUT_DIR.mkdir(parents=True, exist_ok=True)
N_SAMPLE = 100
SEED = 42
MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"

LABEL_NORMALIZE = {
    "bug report": "bug_report", "bug_report": "bug_report",
    "feature request": "feature_request", "feature_request": "feature_request",
    "performance": "performance", "usability": "usability",
    "compatibility": "compatibility", "praise": "praise", "other": "other",
}

PROMPT_A_CONCISE = """Classify this app review into ONE of these 7 categories:
bug_report, feature_request, performance, usability, compatibility, praise, other

Review: "{review}"

Reply with ONE word (one of the 7 categories above) and nothing else."""

PROMPT_B_COT = """You are an expert app-review classifier. Read the review carefully, think briefly, then assign exactly one category.

Categories:
- bug_report: crashes, errors, broken features
- feature_request: requests for new features or changes
- performance: speed, battery, memory, lag, slowness
- usability: confusing UI, hard to use, navigation issues
- compatibility: device-specific or OS-specific problems
- praise: positive feedback, compliments
- other: anything else (information, off-topic, ambiguous)

Decision rules:
- "slow" / "lag" → performance (NOT bug_report)
- "crashes on my Samsung" → compatibility (device-specific)
- "would be nice if X" → feature_request

Review: "{review}"

After thinking, output a single line: FINAL_LABEL: <one of the 7 categories>"""


def load_expert_labels():
    """Load lead-author expert labels from annotator_A.numbers."""
    doc = Document("annotator_materials/annotator_A.numbers")
    for sheet in doc.sheets:
        for tbl in sheet.tables:
            rows = tbl.rows(values_only=True)
            if not rows or "correct_yn" not in str(rows[0]):
                continue
            header = rows[0]
            col = {h: i for i, h in enumerate(header) if h}
            expert = {}
            for r in rows[1:]:
                rid = r[col["row_id"]]
                if rid is None: continue
                yn = r[col["correct_yn"]]
                pred = r[col["predicted_label"]]
                final = r[col.get("correct_label_if_no")] if col.get("correct_label_if_no") is not None else None
                if yn is None: continue
                yn_str = str(yn).strip().upper()
                if yn_str == "Y":
                    label = LABEL_NORMALIZE.get(str(pred).strip().lower())
                elif yn_str == "N":
                    label = LABEL_NORMALIZE.get(str(final).strip().lower()) if final else None
                else:
                    continue
                if label not in LABELS:
                    continue
                # Need text too — the Numbers file has it
                text = r[col["review_text"]]
                expert[int(rid)] = {"text": text, "label": label}
            return expert
    return {}


def parse_label(raw: str) -> str | None:
    """Extract a category label from LLM output."""
    t = raw.lower().strip()
    # Look for "FINAL_LABEL: X" pattern first
    m = re.search(r"final[_\s]label\s*[:\-]\s*(\w+(?:[_\s]\w+)?)", t)
    if m:
        candidate = m.group(1).strip().replace(" ", "_")
        return LABEL_NORMALIZE.get(candidate)
    # Otherwise look for any of the 7 labels mentioned
    for lbl in LABELS:
        if lbl in t.replace("-", "_"):
            return lbl
        if lbl.replace("_", " ") in t:
            return lbl
    return None


def llm_annotate(tokenizer, model, device, texts: list[str], prompt_template: str,
                 progress_label: str = "annotating"):
    """Run LLM annotation on a list of texts with a given prompt."""
    labels = []
    raws = []
    t0 = time.time()
    last_log = t0
    with torch.no_grad():
        for i, text in enumerate(texts):
            msgs = [{"role": "user", "content": prompt_template.format(review=text[:300])}]
            prompt = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True,
                              max_length=600).to(device)
            out = model.generate(**inputs, max_new_tokens=40, do_sample=False,
                                  pad_token_id=tokenizer.pad_token_id,
                                  eos_token_id=tokenizer.eos_token_id)
            gen = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:],
                                    skip_special_tokens=True)
            labels.append(parse_label(gen))
            raws.append(gen)
            now = time.time()
            if now - last_log > 30:
                done = i + 1
                rate = done / (now - t0)
                print(f"  [{progress_label}] {done}/{len(texts)}  "
                      f"({rate:.2f}/s, ETA {(len(texts)-done)/rate/60:.1f}min)", flush=True)
                last_log = now
    return labels, raws


def krippendorff_alpha(annotations: list[list[str]], categories: list[str]) -> float:
    """Krippendorff's α for nominal categories on N annotators × M items."""
    n_raters = len(annotations)
    n_items = len(annotations[0])
    # Coincidence matrix
    cat_idx = {c: i for i, c in enumerate(categories)}
    K = len(categories)
    coincidence = [[0.0]*K for _ in range(K)]
    for j in range(n_items):
        ratings = [annotations[r][j] for r in range(n_raters)
                   if annotations[r][j] in cat_idx]
        m = len(ratings)
        if m < 2: continue
        for r in ratings:
            for s in ratings:
                if r != s or ratings.count(r) > 1:
                    coincidence[cat_idx[r]][cat_idx[s]] += 1.0 / (m - 1)
    n_coincidence = sum(sum(row) for row in coincidence)
    if n_coincidence == 0: return 0.0
    # Marginals
    n_per_cat = [sum(coincidence[i]) for i in range(K)]
    # D_o = observed disagreement, D_e = expected
    D_o = sum(coincidence[i][j] for i in range(K) for j in range(K) if i != j) / n_coincidence
    D_e = sum(n_per_cat[i] * n_per_cat[j] for i in range(K) for j in range(K) if i != j) / (n_coincidence * (n_coincidence - 1))
    return 1 - D_o / D_e if D_e else 0.0


def main():
    print("[1/4] Loading expert labels (Annotator-1)")
    expert = load_expert_labels()
    print(f"      {len(expert):,} expert-labeled reviews available")

    rng = random.Random(SEED)
    expert_ids = sorted(expert.keys())
    rng.shuffle(expert_ids)
    sample_ids = expert_ids[:N_SAMPLE]
    sample = [expert[i] for i in sample_ids]
    sample_texts = [s["text"] for s in sample]
    expert_labels = [s["label"] for s in sample]
    print(f"      sampled {len(sample)}")

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"\n[2/4] Loading {MODEL_NAME} on {device}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.float16, device_map=device,
    )
    model.eval()

    print(f"\n[3/4] Annotating with two LLM raters (concise + CoT prompts)")
    print("  Annotator-2: concise role-based prompt")
    L1_labels, L1_raws = llm_annotate(tokenizer, model, device, sample_texts,
                                       PROMPT_A_CONCISE, "L1-concise")
    print("\n  Annotator-3: chain-of-thought prompt")
    L2_labels, L2_raws = llm_annotate(tokenizer, model, device, sample_texts,
                                       PROMPT_B_COT, "L2-CoT")

    # Filter to rows where all 3 raters produced a valid label
    triples = []
    for i, e in enumerate(expert_labels):
        if L1_labels[i] in LABELS and L2_labels[i] in LABELS and e in LABELS:
            triples.append((e, L1_labels[i], L2_labels[i]))
    print(f"\n      {len(triples)}/{N_SAMPLE} triples with all 3 valid labels")

    E_l = [t[0] for t in triples]
    L1 = [t[1] for t in triples]
    L2 = [t[2] for t in triples]

    print("\n[4/4] Computing inter-annotator agreement")
    kappa_E_L1 = cohen_kappa_score(E_l, L1, labels=LABELS)
    kappa_E_L2 = cohen_kappa_score(E_l, L2, labels=LABELS)
    kappa_L1_L2 = cohen_kappa_score(L1, L2, labels=LABELS)
    alpha_3 = krippendorff_alpha([E_l, L1, L2], LABELS)

    print(f"\n  Cohen κ (E vs LLM-concise):     {kappa_E_L1:.4f}")
    print(f"  Cohen κ (E vs LLM-CoT):         {kappa_E_L2:.4f}")
    print(f"  Cohen κ (LLM-concise vs CoT):   {kappa_L1_L2:.4f}")
    print(f"  Krippendorff α (3 annotators):  {alpha_3:.4f}")

    summary = {
        "n_triples": len(triples),
        "raters": {
            "Annotator-1 (E)":  "Lead author (expert) — from annotator_A.numbers",
            "Annotator-2 (L1)": f"{MODEL_NAME} with concise role-based prompt",
            "Annotator-3 (L2)": f"{MODEL_NAME} with chain-of-thought prompt",
        },
        "cohen_kappa": {
            "expert_vs_llm_concise":     round(kappa_E_L1, 4),
            "expert_vs_llm_cot":         round(kappa_E_L2, 4),
            "llm_concise_vs_llm_cot":    round(kappa_L1_L2, 4),
        },
        "krippendorff_alpha_3raters": round(alpha_3, 4),
        "interpretation": {
            "kappa_landis_koch": ("<0.20 slight; 0.21-0.40 fair; 0.41-0.60 moderate; "
                                   "0.61-0.80 substantial; >0.81 almost-perfect"),
            "alpha_krippendorff": ("≥0.667 acceptable; ≥0.80 strong (Krippendorff 2004)"),
        },
        "label_distribution": {
            "expert":      dict(Counter(E_l)),
            "llm_concise": dict(Counter(L1)),
            "llm_cot":     dict(Counter(L2)),
        },
        "methodology_note": ("Following Gilardi et al. 2023 (PNAS) and Pangakis et al. 2023, "
                              "we use LLMs as additional independent annotators alongside the "
                              "human expert. This produces tractable inter-rater statistics in "
                              "the absence of multiple human annotators. Pairwise Cohen κ "
                              "between expert and LLM raters quantifies whether expert "
                              "annotation falls within the inter-rater agreement envelope a "
                              "naive LLM annotator would produce."),
    }
    with open(OUT_DIR / "agreement_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Save raw annotations for reproducibility
    with open(OUT_DIR / "llm_annotations.json", "w") as f:
        json.dump({
            "sample_ids": sample_ids,
            "sample_texts": sample_texts,
            "expert_labels": expert_labels,
            "llm_concise_labels": L1_labels,
            "llm_cot_labels": L2_labels,
            "llm_concise_raws": L1_raws,
            "llm_cot_raws": L2_raws,
        }, f, indent=2)

    # Human-readable
    lines = [
        "="*78,
        "INTER-ANNOTATOR AGREEMENT — 3 RATERS (1 EXPERT + 2 LLMs)",
        "="*78,
        f"Sample: {len(triples)} reviews from the 490 expert gold standard",
        f"Annotator-1: lead-author expert",
        f"Annotator-2: {MODEL_NAME} (concise prompt)",
        f"Annotator-3: {MODEL_NAME} (chain-of-thought prompt)",
        "",
        "Pairwise Cohen's κ:",
        f"  E vs LLM-concise:    {kappa_E_L1:.4f}",
        f"  E vs LLM-CoT:        {kappa_E_L2:.4f}",
        f"  LLM-concise vs CoT:  {kappa_L1_L2:.4f}",
        "",
        f"Krippendorff α (3 annotators): {alpha_3:.4f}",
        "",
        "Interpretation: Landis-Koch (κ): >0.81 almost-perfect, 0.61-0.80 substantial,",
        "                                 0.41-0.60 moderate, 0.21-0.40 fair, <0.20 slight.",
        "                Krippendorff (α): >=0.667 acceptable, >=0.80 strong.",
    ]
    text = "\n".join(lines)
    print("\n" + text)
    with open(OUT_DIR / "summary.txt", "w") as f:
        f.write(text)
    print(f"\nSaved {OUT_DIR}/")


if __name__ == "__main__":
    main()
