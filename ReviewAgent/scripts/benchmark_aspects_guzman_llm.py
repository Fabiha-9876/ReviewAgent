"""
Benchmark local-LLM (Qwen2.5-3B-Instruct) aspect extraction on a stratified
200-sentence sample from the Guzman & Maalej 2014 gold standard, then compare
side-by-side with the heuristic benchmark.

Outputs:
  data/processed/guzman_benchmark/
    llm_sample_results.json    per-sentence extraction + match status
    llm_summary.json           precision/recall/F1 (LLM)
    comparison.txt             side-by-side vs heuristic
"""

import json
import random
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean

import torch
import spacy
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent.parent))
# Reuse normalization + matching from the heuristic benchmark
from scripts.benchmark_aspects_guzman import normalize, lemmatize_word, match_aspects, metrics


GUZMAN_PATH = Path("data/raw/guzman/guzman_reviews.json")
OUT_DIR     = Path("data/processed/guzman_benchmark")
N_SAMPLE    = 200
MAX_PER_APP = 30
SEED        = 42
MODEL_NAME  = "Qwen/Qwen2.5-3B-Instruct"

PROMPT_TEMPLATE = """Extract the specific aspects (features, components, things) explicitly mentioned in this app review sentence.

Rules:
- Output a JSON array of short aspect phrases (1-3 words each).
- Use lowercase. No duplicates.
- Only include aspects clearly mentioned (a feature, component, or named thing), not inferred.
- If no aspect is mentioned, return [].
- Return ONLY the JSON array.

Examples:
Review: "Too many ads and the interface is erratic."
Aspects: ["ads", "interface"]

Review: "Login does not work after the latest update."
Aspects: ["login", "update"]

Review: "Cant get to install"
Aspects: ["install"]

Review: "May be i can check"
Aspects: []

Review: "{review}"
Aspects:"""


def parse_aspects(raw: str) -> list[str]:
    m = re.search(r'\[(.*?)\]', raw, re.DOTALL)
    if not m:
        return []
    inside = m.group(1)
    try:
        return [str(x).strip().lower() for x in json.loads("[" + inside + "]")
                if x and 1 <= len(str(x)) <= 50]
    except json.JSONDecodeError:
        items = re.findall(r'["\']([^"\']{1,50})["\']', inside)
        return [s.strip().lower() for s in items if s.strip()]


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    random.seed(SEED)
    torch.manual_seed(SEED)

    print(f"Loading GUZMAN")
    data = json.load(open(GUZMAN_PATH))

    # Stratified sample 200 from the 971 gold-bearing sentences (with per-app cap)
    by_app = defaultdict(list)
    for r in data:
        if r.get("aspects"):
            by_app[r["app_id"]].append(r)
    print(f"  {sum(len(v) for v in by_app.values())} sentences with gold across {len(by_app)} apps")

    sample = []
    for app, rows in by_app.items():
        random.shuffle(rows)
        sample.extend(rows[:MAX_PER_APP])
    random.shuffle(sample)
    sample = sample[:N_SAMPLE]
    print(f"  sampled {len(sample)} sentences")
    apps_sampled = Counter(r["app_id"] for r in sample)
    print(f"  apps: {dict(apps_sampled)}")

    # Load model
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"\nLoading {MODEL_NAME} on {device}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.float16, device_map=device,
    )
    model.eval()

    nlp = spacy.load("en_core_web_sm", disable=["ner"])

    print(f"\nGenerating aspects for {len(sample)} sentences (~5-10s per sentence)")
    per_sentence = []
    t0 = time.time()
    last_log = t0

    with torch.no_grad():
        for i, r in enumerate(sample):
            text = r["text"][:300]
            gold_aspects = [a["aspect"] for a in r.get("aspects", [])]

            msgs = [{"role": "user", "content": PROMPT_TEMPLATE.format(review=text)}]
            prompt = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True,
                              max_length=600).to(device)
            out = model.generate(**inputs, max_new_tokens=60, do_sample=False,
                                  pad_token_id=tokenizer.pad_token_id,
                                  eos_token_id=tokenizer.eos_token_id)
            gen = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:],
                                    skip_special_tokens=True)
            predicted = parse_aspects(gen)
            result = match_aspects(predicted, gold_aspects, nlp)

            per_sentence.append({
                "sentence_id": f"{r['review_id']}_{r.get('sentence_id', 0)}",
                "app_id": r.get("app_id"),
                "text": text,
                "raw_llm_output": gen,
                "n_gold": len(gold_aspects),
                "n_predicted": len(predicted),
                **result,
            })

            now = time.time()
            if now - last_log > 30:
                done = i + 1
                rate = done / (now - t0)
                eta = (len(sample) - done) / rate
                print(f"  {done}/{len(sample)}  ({rate:.2f} sent/s, ETA {eta/60:.1f}min)", flush=True)
                last_log = now

    print(f"\nDone in {(time.time()-t0)/60:.1f} min")

    # Compute metrics
    summary = {}
    for level in ["exact", "lemma", "substring"]:
        tp = sum(len(s[level]["tp"]) for s in per_sentence)
        fp = sum(len(s[level]["fp"]) for s in per_sentence)
        fn = sum(len(s[level]["fn"]) for s in per_sentence)
        micro = metrics(tp, fp, fn)

        per_sent_p, per_sent_r, per_sent_f1 = [], [], []
        for s in per_sentence:
            tp_s = len(s[level]["tp"])
            fp_s = len(s[level]["fp"])
            fn_s = len(s[level]["fn"])
            if tp_s + fp_s + fn_s == 0:
                continue
            m = metrics(tp_s, fp_s, fn_s)
            per_sent_p.append(m["precision"])
            per_sent_r.append(m["recall"])
            per_sent_f1.append(m["f1"])
        summary[level] = {
            "micro": {**micro, "tp": tp, "fp": fp, "fn": fn},
            "macro": {
                "precision": round(mean(per_sent_p), 4) if per_sent_p else 0,
                "recall":    round(mean(per_sent_r), 4) if per_sent_r else 0,
                "f1":        round(mean(per_sent_f1), 4) if per_sent_f1 else 0,
                "n": len(per_sent_p),
            },
        }

    out = {
        "model": MODEL_NAME,
        "n_sampled": len(sample),
        "summary_by_match_level": summary,
    }
    with open(OUT_DIR / "llm_summary.json", "w") as f:
        json.dump(out, f, indent=2)
    with open(OUT_DIR / "llm_sample_results.json", "w") as f:
        json.dump(per_sentence, f)

    # Side-by-side comparison
    h = json.load(open(OUT_DIR / "summary.json"))
    lines = [
        "="*78,
        "GUZMAN ASPECT-EXTRACTION: HEURISTIC vs LOCAL-LLM (Qwen2.5-3B)",
        "="*78,
        "",
        f"Sample: {len(sample)} sentences (stratified by app, max {MAX_PER_APP}/app, seed {SEED})",
        f"All sampled sentences have at least one gold aspect annotation.",
        f"LLM: {MODEL_NAME}, run on {device}",
        "",
        "="*78,
        "SIDE-BY-SIDE (substring match level)",
        "="*78,
        f"{'method':28s} {'micro_P':>10} {'micro_R':>10} {'micro_F1':>10}  "
        f"{'macro_P':>9} {'macro_R':>9} {'macro_F1':>10}",
        "-"*78,
    ]
    h_micro = h["summary_by_match_level"]["substring"]["micro"]
    h_macro = h["summary_by_match_level"]["substring"]["macro_gold_only_sentences"]
    lines.append(f"{'heuristic (full GUZMAN)':28s} "
                 f"{h_micro['precision']:>10.4f} {h_micro['recall']:>10.4f} {h_micro['f1']:>10.4f}  "
                 f"{h_macro['precision']:>9.4f} {h_macro['recall']:>9.4f} {h_macro['f1']:>10.4f}")
    l_micro = summary["substring"]["micro"]
    l_macro = summary["substring"]["macro"]
    lines.append(f"{'local-LLM Qwen2.5-3B (n=200)':28s} "
                 f"{l_micro['precision']:>10.4f} {l_micro['recall']:>10.4f} {l_micro['f1']:>10.4f}  "
                 f"{l_macro['precision']:>9.4f} {l_macro['recall']:>9.4f} {l_macro['f1']:>10.4f}")
    lines.append("")
    lines.append("All match levels — Local LLM:")
    lines.append(f"{'level':12s} {'micro_P':>10} {'micro_R':>10} {'micro_F1':>10}")
    for level in ["exact", "lemma", "substring"]:
        m = summary[level]["micro"]
        lines.append(f"{level:12s} {m['precision']:>10.4f} {m['recall']:>10.4f} {m['f1']:>10.4f}")

    text = "\n".join(lines)
    print("\n" + text)
    with open(OUT_DIR / "comparison.txt", "w") as f:
        f.write(text)
    print(f"\nSaved {OUT_DIR}/llm_summary.json")
    print(f"Saved {OUT_DIR}/comparison.txt")


if __name__ == "__main__":
    main()
