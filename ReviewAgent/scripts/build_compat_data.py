"""
Fix the compatibility class data problem.

Two sources:
  1. Synthetic templates — generate ~150 device/OS-specific compat reviews
     (covers patterns the model has never seen: Samsung-specific crashes,
     Android 13 update breakage, foldable issues, etc.)
  2. RRGen mining — extract reviews from rrgen_full_labeled.json that contain
     compat keywords (specific device names, OS versions, "after update")
     but were NOT labeled compatibility by the V2 LLM. These are the
     classifier's confident-but-wrong cases — perfect training material.

Output:
    data/processed/compat_augmentation.json   ~250 compat-labeled reviews
    data/processed/compat_augmentation.csv    flat view for spot-checking

After running this, build the V5 training corpus by combining:
    rrgen_corrected_v2.json + compat_augmentation.json
"""

import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.stage1.classifier import LABELS

random.seed(42)

# --------------------------------------------------------------------------
# 1. Synthetic compat review templates
# --------------------------------------------------------------------------

DEVICES = [
    "Samsung Galaxy S22", "Samsung Galaxy S23", "Samsung Galaxy A52",
    "Samsung Galaxy Note 20", "Google Pixel 6", "Google Pixel 7",
    "Google Pixel 8", "Pixel 4a", "OnePlus 9", "OnePlus 10 Pro",
    "OnePlus Nord", "Xiaomi Mi 11", "Xiaomi Redmi Note 12",
    "Huawei P30", "Huawei P40", "Huawei Mate 40", "Motorola Moto G",
    "Motorola Edge", "Nokia G50", "Realme 9", "Vivo V25", "Oppo Reno 8",
    "LG V60", "Asus ROG Phone", "Sony Xperia 1",
]

OS_VERSIONS = [
    "Android 10", "Android 11", "Android 12", "Android 13", "Android 14",
    "Android 9", "Android 8.0 Oreo", "Android 7 Nougat",
    "MIUI 13", "ColorOS 12", "OneUI 5", "OxygenOS 13", "EMUI 11",
]

SCREEN_TYPES = [
    "tablet", "foldable phone", "Galaxy Fold", "small screen device",
    "landscape mode", "split-screen mode", "Z Flip", "tablet mode",
]

APPS = [
    "com.spotify.music", "com.facebook.katana", "com.whatsapp",
    "com.instagram.android", "com.netflix.mediaclient", "com.snapchat.android",
    "com.twitter.android", "com.zhiliaoapp.musically", "com.amazon.mShop.android.shopping",
    "com.king.candycrushsaga", "com.ubercab", "com.airbnb.android",
]

DEVICE_TEMPLATES = [
    "App keeps crashing on my {device}. Worked fine on my old phone.",
    "Doesn't work properly on {device}. Force closes every time I open it.",
    "Crashes only on my {device} — my friend's iPhone runs it fine.",
    "Cannot install this app on my {device}. Says my device is not supported.",
    "App freezes on {device}. Other apps work fine, just this one.",
    "Login screen never loads on {device}. Tried reinstalling, same issue.",
    "{device} user here — app force closes immediately on launch.",
    "Tried on {device} and the app is completely broken. Black screen on open.",
    "My {device} can't run this app at all. Crashes within 2 seconds.",
    "App is not optimized for {device}. UI is cut off and buttons don't work.",
    "Installed on {device}, app shows 'device not compatible' even though specs are fine.",
    "Specifically broken on {device} — works on every other phone I've tried.",
]

OS_TEMPLATES = [
    "App stopped working after the {os} update.",
    "Broken on {os}. Was fine before the update.",
    "Force closes since I updated to {os}.",
    "App crashes constantly after upgrading to {os}.",
    "Not compatible with {os}. Keeps erroring out.",
    "Since {os} update, app won't even open.",
    "Worked fine on the previous OS but {os} broke everything.",
    "Login broken since {os}. Stuck on loading screen forever.",
    "{os} user — app is unusable since the update.",
    "After updating to {os}, half the features stopped working.",
    "Please fix the {os} compatibility issue, app crashes on startup.",
    "Used to love this app but {os} update killed it for me.",
]

SCREEN_TEMPLATES = [
    "App layout is broken on {screen}. Buttons cut off the screen.",
    "Doesn't work properly in {screen}. Half the UI is missing.",
    "On my {screen}, the interface is misaligned and unusable.",
    "Not optimized for {screen}. Looks terrible and elements overlap.",
    "Please add proper {screen} support. Currently unusable on it.",
    "Switching to {screen} crashes the app every time.",
    "{screen} layout issues — text gets cut off, can't tap buttons.",
]


def gen_synthetic(n: int = 200) -> list[dict]:
    """Generate synthetic compatibility reviews using templates."""
    out = []
    n_device = int(n * 0.5)
    n_os = int(n * 0.35)
    n_screen = n - n_device - n_os

    for _ in range(n_device):
        device = random.choice(DEVICES)
        text = random.choice(DEVICE_TEMPLATES).format(device=device)
        out.append({
            "text": text,
            "rating": random.choice([1, 1, 1, 2, 2, 3]),
            "app_id": random.choice(APPS),
            "timestamp": "2026-04-26",
            "labels": ["compatibility"],
            "source": "synthetic_compat_v2",
            "subtype": "device",
        })

    for _ in range(n_os):
        os = random.choice(OS_VERSIONS)
        text = random.choice(OS_TEMPLATES).format(os=os)
        out.append({
            "text": text,
            "rating": random.choice([1, 1, 2, 2, 3]),
            "app_id": random.choice(APPS),
            "timestamp": "2026-04-26",
            "labels": ["compatibility"],
            "source": "synthetic_compat_v2",
            "subtype": "os",
        })

    for _ in range(n_screen):
        screen = random.choice(SCREEN_TYPES)
        text = random.choice(SCREEN_TEMPLATES).format(screen=screen)
        out.append({
            "text": text,
            "rating": random.choice([1, 2, 2, 3]),
            "app_id": random.choice(APPS),
            "timestamp": "2026-04-26",
            "labels": ["compatibility"],
            "source": "synthetic_compat_v2",
            "subtype": "screen",
        })

    random.shuffle(out)
    return out


# --------------------------------------------------------------------------
# 2. Mine compat candidates from existing RRGen 215K labeled data
# --------------------------------------------------------------------------

# Tight keyword patterns that strongly signal compatibility
DEVICE_KEYWORDS = [
    r"\bsamsung\b", r"\bgalaxy\b", r"\bpixel\b", r"\boneplus\b",
    r"\bxiaomi\b", r"\bredmi\b", r"\bhuawei\b", r"\bmotorola\b",
    r"\bnokia\b", r"\brealme\b", r"\boppo\b", r"\bvivo\b",
    r"\bnote ?\d+\b", r"\bs ?\d+\s+(ultra|plus|pro)?",
]
OS_KEYWORDS = [
    r"android\s*\d+", r"\boreo\b", r"\bnougat\b", r"\bmarshmallow\b",
    r"\blollipop\b", r"\bpie\b", r"\bmiui\b", r"\boneui\b",
    r"\bcoloros\b", r"\boxygenos\b",
]
COMPAT_PHRASES = [
    r"after.*update", r"since.*update", r"after.*upgrade",
    r"not\s+compatible", r"incompatible", r"won.?t\s+(install|open|run)",
    r"doesn.?t\s+(work|run)\s+on", r"crash(es)?\s+on\s+my",
    r"only\s+(crash|break)s\s+on",
]

ALL_PATTERNS = DEVICE_KEYWORDS + OS_KEYWORDS + COMPAT_PHRASES


def looks_like_compat(text: str) -> int:
    """Count how many compat patterns the text matches."""
    t = text.lower()
    return sum(1 for p in ALL_PATTERNS if re.search(p, t))


def mine_rrgen(noisy_path: Path, target: int = 100) -> list[dict]:
    """Extract reviews not currently labeled compat but with strong compat signal."""
    print(f"  loading noisy 215K from {noisy_path}")
    with open(noisy_path) as f:
        rows = json.load(f)

    candidates = []
    for r in rows:
        if r["predicted_label"] == "compatibility":
            continue  # already compat
        score = looks_like_compat(r["text"])
        if score >= 2:  # need at least 2 compat patterns to reduce false positives
            candidates.append((score, r))

    # Sort by signal strength (most-compat-like first)
    candidates.sort(key=lambda x: -x[0])
    print(f"  found {len(candidates):,} candidates with compat-signal >=2")

    selected = candidates[:target]
    out = []
    for score, r in selected:
        out.append({
            "text": r["text"],
            "rating": r.get("rating"),
            "app_id": r.get("app_id"),
            "timestamp": r.get("timestamp"),
            "labels": ["compatibility"],
            "source": "rrgen_mined_compat",
            "original_label": r["predicted_label"],
            "compat_signal_score": score,
        })
    return out


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main():
    out_dir = Path("data/processed")
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[1/3] Generating synthetic compatibility reviews")
    synthetic = gen_synthetic(n=200)
    print(f"      {len(synthetic):,} synthetic samples")
    sub_counts = Counter(r["subtype"] for r in synthetic)
    print(f"      subtypes: {dict(sub_counts)}")

    print("\n[2/3] Mining compat candidates from RRGen 215K")
    mined = mine_rrgen(
        Path("data/processed/rrgen_full_labeled/rrgen_full_labeled.json"),
        target=100,
    )
    print(f"      {len(mined):,} mined samples")
    if mined:
        original_counts = Counter(r["original_label"] for r in mined)
        print(f"      original LLM labels: {dict(original_counts)}")
        print(f"\n      Top 5 mined examples:")
        for r in mined[:5]:
            print(f"        [{r['compat_signal_score']}] {r['original_label']:15s} | {r['text'][:90]}")

    print("\n[3/3] Writing combined augmentation set")
    all_compat = synthetic + mined
    random.shuffle(all_compat)

    out_json = out_dir / "compat_augmentation.json"
    with open(out_json, "w") as f:
        json.dump(all_compat, f, indent=2)

    import csv
    out_csv = out_dir / "compat_augmentation.csv"
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["text", "rating", "app_id", "labels", "source", "original_label", "subtype"])
        for r in all_compat:
            w.writerow([
                r["text"], r.get("rating"), r.get("app_id"),
                ",".join(r["labels"]), r["source"],
                r.get("original_label", ""), r.get("subtype", ""),
            ])

    print(f"\nWrote {out_json}  ({len(all_compat):,} compat-labeled reviews)")
    print(f"Wrote {out_csv}")
    print(f"\nBreakdown:")
    src_counts = Counter(r["source"] for r in all_compat)
    for s, n in src_counts.items():
        print(f"  {s:30s} {n}")


if __name__ == "__main__":
    main()
