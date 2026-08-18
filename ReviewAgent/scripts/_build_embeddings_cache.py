"""Generate the embeddings cache needed by ablation_a1b_fine_flat_vs_kg.py.

Embeds all reviews in rrgen_v5_relabeled.json whose v5_label is actionable.
Saves a single (N, 384) float32 array + an index mapping to data/processed/.
Uses MPS on Apple Silicon if available, else CUDA, else CPU.
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path

import numpy as np

ACTIONABLE = {"bug_report", "feature_request", "performance", "usability", "compatibility"}
BASE = Path(__file__).resolve().parent.parent
INP = BASE / "data/processed/rrgen_v5_relabeled/rrgen_v5_relabeled.json"
OUT_EMB = BASE / "data/processed/embeddings_cache.npy"
OUT_IDX = BASE / "data/processed/embeddings_cache_index.json"

print("Loading", INP, file=sys.stderr)
with open(INP) as f:
    rows = json.load(f)
print(f"  {len(rows):,} rows", file=sys.stderr)

# Filter to actionable + collect texts
indices, texts, labels = [], [], []
for i, r in enumerate(rows):
    if r.get("v5_label") in ACTIONABLE:
        indices.append(i)
        texts.append(r.get("text", ""))
        labels.append(r["v5_label"])
print(f"  actionable subset: {len(texts):,}", file=sys.stderr)

# Detect device
import torch
if torch.backends.mps.is_available():
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
print(f"  device: {device}", file=sys.stderr)

from sentence_transformers import SentenceTransformer
model = SentenceTransformer("all-MiniLM-L6-v2", device=device)

t0 = time.time()
emb = model.encode(texts, batch_size=128, show_progress_bar=True,
                   convert_to_numpy=True, normalize_embeddings=False)
emb = emb.astype(np.float32)
print(f"  encoded in {time.time()-t0:.1f}s, shape={emb.shape}", file=sys.stderr)

np.save(OUT_EMB, emb)
json.dump({"actionable_indices": indices, "labels": labels,
           "n": len(indices), "shape": list(emb.shape)},
          open(OUT_IDX, "w"))
print(f"Saved -> {OUT_EMB} and {OUT_IDX}", file=sys.stderr)
