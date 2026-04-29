"""
Phase 3 of free-tier Stage 2: local LLM aspect extraction (no API needed).

Uses Qwen2.5-3B-Instruct (Apache-2.0, no HF token required) to extract aspects
from a sample of reviews. Quality comparable to GPT-4o-mini for aspect tasks
but slow — ~5-10 sec/review on MPS, so we run on a sample (~1000 by default)
and use this as gold-standard validation for Phase 2 heuristics.

Usage:
    python3 scripts/extract_aspects_local_llm.py \
        --input data/processed/rrgen_v5_relabeled/rrgen_v5_relabeled.json \
        --label-field v5_label \
        --sample 1000 \
        --out-dir data/processed/aspects_local_llm

Output:
    aspects_per_review.json     {idx: ["login button", "battery", ...]}
    sample_indices.json         which review indices were sampled
    raw_outputs.json            raw LLM responses (debugging)
"""

import argparse
import json
import random
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent.parent))

PROMPT_TEMPLATE = """Extract the specific aspects (features, components, behaviors) explicitly mentioned in this app review.

Rules:
- Output a JSON array of short aspect phrases (1-4 words each).
- Use lowercase. No duplicates.
- Only include aspects that are clearly mentioned, not inferred.
- Return ONLY the JSON array, nothing else.

Examples:
Review: "The login button does not work after the latest update."
Aspects: ["login button", "update"]

Review: "App is super slow on my Pixel 8 and battery drains fast."
Aspects: ["loading speed", "battery", "pixel 8"]

Review: "Please add dark mode and a search bar in settings."
Aspects: ["dark mode", "search bar", "settings"]

Review: "{review}"
Aspects:"""


def parse_aspects(raw: str) -> list[str]:
    """Extract a JSON list of aspects from raw LLM output."""
    # Find first JSON array
    m = re.search(r'\[(.*?)\]', raw, re.DOTALL)
    if not m:
        return []
    inside = m.group(1)
    try:
        parsed = json.loads("[" + inside + "]")
        if isinstance(parsed, list):
            return [str(x).strip().lower() for x in parsed if x and 1 <= len(str(x)) <= 50]
    except json.JSONDecodeError:
        # Fall back to regex split on quoted strings
        items = re.findall(r'["\']([^"\']{1,50})["\']', inside)
        return [s.strip().lower() for s in items if s.strip()]
    return []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path,
                    default=Path("data/processed/rrgen_v5_relabeled/rrgen_v5_relabeled.json"))
    ap.add_argument("--label-field", default="v5_label")
    ap.add_argument("--out-dir", type=Path,
                    default=Path("data/processed/aspects_local_llm"))
    ap.add_argument("--model-name", default="Qwen/Qwen2.5-3B-Instruct",
                    help="HF model to use. Apache-2.0/MIT models recommended.")
    ap.add_argument("--sample", type=int, default=1000,
                    help="Number of reviews to sample (random within actionable types).")
    ap.add_argument("--max-new-tokens", type=int, default=80)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--actionable-only", action="store_true", default=True)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    if not args.input.exists():
        fb = Path("data/processed/rrgen_corrected_v2/rrgen_corrected_v2.json")
        print(f"Input not found, falling back to {fb}")
        args.input = fb
        if args.label_field == "v5_label":
            args.label_field = "final_label"

    device = "mps" if torch.backends.mps.is_available() else (
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print(f"Device: {device}")

    print(f"\nLoading dataset: {args.input}")
    with open(args.input) as f:
        rows = json.load(f)
    print(f"  {len(rows):,} rows")

    # Sample
    actionable = {"bug_report", "feature_request", "performance", "usability", "compatibility"}
    eligible = [i for i, r in enumerate(rows)
                if (not args.actionable_only or r.get(args.label_field) in actionable)
                and len(r["text"]) >= 20]  # skip very short reviews
    print(f"  eligible reviews: {len(eligible):,}")
    sample_idxs = random.sample(eligible, min(args.sample, len(eligible)))
    sample_idxs.sort()
    print(f"  sampled {len(sample_idxs):,}")

    print(f"\nLoading {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.float16,
        device_map=device,
    )
    model.eval()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"\nExtracting aspects on {len(sample_idxs)} reviews (batch_size={args.batch_size})")
    aspects_per_review = {}
    raw_outputs = {}
    t0 = time.time()
    last_log = t0

    with torch.no_grad():
        for i in range(0, len(sample_idxs), args.batch_size):
            batch_idxs = sample_idxs[i : i + args.batch_size]
            batch_texts = [rows[idx]["text"][:500] for idx in batch_idxs]  # cap input length

            # Format using chat template
            prompts = []
            for text in batch_texts:
                msgs = [{"role": "user", "content": PROMPT_TEMPLATE.format(review=text)}]
                prompt = tokenizer.apply_chat_template(
                    msgs, tokenize=False, add_generation_prompt=True
                )
                prompts.append(prompt)

            inputs = tokenizer(prompts, return_tensors="pt", padding=True,
                               truncation=True, max_length=1024).to(device)
            out = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            for j, idx in enumerate(batch_idxs):
                # Decode only the new tokens
                input_len = inputs["input_ids"][j].shape[0]
                gen_tokens = out[j][input_len:]
                gen_text = tokenizer.decode(gen_tokens, skip_special_tokens=True)
                raw_outputs[idx] = gen_text
                aspects = parse_aspects(gen_text)
                if aspects:
                    aspects_per_review[idx] = aspects

            now = time.time()
            if now - last_log > 30:
                done = i + len(batch_idxs)
                pct = 100 * done / len(sample_idxs)
                rate = done / (now - t0)
                eta = (len(sample_idxs) - done) / rate
                print(f"  {done:>5,} / {len(sample_idxs):,}  ({pct:5.1f}%)  "
                      f"{rate:.2f} reviews/s  ETA {eta/60:.1f} min", flush=True)
                last_log = now

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed/60:.1f} min ({len(sample_idxs)/elapsed:.2f} reviews/s)")
    print(f"  Reviews with parsed aspects: {len(aspects_per_review):,} / {len(sample_idxs):,}")

    # Save outputs (keys are stringified ints for JSON)
    with open(args.out_dir / "aspects_per_review.json", "w") as f:
        json.dump({str(k): v for k, v in aspects_per_review.items()}, f, indent=2)
    with open(args.out_dir / "raw_outputs.json", "w") as f:
        json.dump({str(k): v for k, v in raw_outputs.items()}, f)
    with open(args.out_dir / "sample_indices.json", "w") as f:
        json.dump(sample_idxs, f)

    aspect_freq = Counter()
    for aspects in aspects_per_review.values():
        for a in aspects:
            aspect_freq[a] += 1

    print(f"\nTop 30 LLM-extracted aspects:")
    for asp, n in aspect_freq.most_common(30):
        print(f"  {n:>4,}  {asp}")

    print(f"\nOutputs: {args.out_dir}/")


if __name__ == "__main__":
    main()
