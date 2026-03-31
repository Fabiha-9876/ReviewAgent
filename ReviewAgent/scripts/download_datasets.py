"""
Dataset Download Script for ReviewAgent
========================================

Downloads and prepares 3 datasets:
1. MAALEJ (~3,691 labeled reviews) - Direct download from Mendeley Data
2. RRGen (~309K review-response pairs) - Requires academic access request
3. GUZMAN (~1,820 aspect-annotated reviews) - Requires author contact

Usage:
    python3 scripts/download_datasets.py              # Download all available
    python3 scripts/download_datasets.py maalej       # Download MAALEJ only
    python3 scripts/download_datasets.py rrgen        # Setup RRGen (instructions)
    python3 scripts/download_datasets.py guzman       # Setup GUZMAN (instructions)
    python3 scripts/download_datasets.py sample       # Generate synthetic sample data
"""

import json
import csv
import os
import sys
import zipfile
import io
from pathlib import Path
from datetime import datetime, timedelta
import random
import urllib.request
import ssl


BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"


def ensure_dirs():
    """Create all data directories."""
    for subdir in ["raw/maalej", "raw/rrgen", "raw/guzman", "processed", "gold_standard", "feedback"]:
        (DATA_DIR / subdir).mkdir(parents=True, exist_ok=True)
    print("Data directories created.")


# ============================================================
# MAALEJ Dataset - Direct Download from Mendeley Data
# ============================================================

MAALEJ_URL = "https://data.mendeley.com/public-files/datasets/5fk732vkwr/files/a68e43a3-0fa0-4ee4-8ef5-bff8f245a714/file_downloaded"
MAALEJ_BACKUP_URLS = [
    "https://prod-dcd-datasets-cache-zipfiles.s3.eu-west-1.amazonaws.com/5fk732vkwr-2.zip",
]


def download_maalej():
    """Download and process the MAALEJ dataset (3,691 labeled app reviews)."""
    print("\n" + "=" * 60)
    print("MAALEJ Dataset (~3,691 labeled reviews)")
    print("Source: Maalej et al. (2016) - Mendeley Data")
    print("DOI: 10.17632/5fk732vkwr.2")
    print("License: CC BY 4.0")
    print("=" * 60)

    output_dir = RAW_DIR / "maalej"
    processed_file = output_dir / "maalej_reviews.json"

    if processed_file.exists():
        data = json.loads(processed_file.read_text())
        print(f"Already downloaded: {len(data)} reviews in {processed_file}")
        return True

    # Try downloading
    print("\nAttempting download from Mendeley Data...")
    downloaded = False
    all_urls = [MAALEJ_URL] + MAALEJ_BACKUP_URLS

    # Create SSL context that doesn't verify (some academic servers have cert issues)
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE

    for url in all_urls:
        try:
            print(f"  Trying: {url[:80]}...")
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            response = urllib.request.urlopen(req, timeout=30, context=ctx)
            content = response.read()

            # Check if it's a ZIP file
            if content[:2] == b'PK':
                print("  Downloaded ZIP file, extracting...")
                with zipfile.ZipFile(io.BytesIO(content)) as zf:
                    zf.extractall(str(output_dir))
                    print(f"  Extracted to: {output_dir}")
                    print(f"  Files: {[f.filename for f in zf.filelist]}")
                downloaded = True
                break
            else:
                # Might be CSV or other format
                save_path = output_dir / "maalej_raw.csv"
                save_path.write_bytes(content)
                print(f"  Saved to: {save_path}")
                downloaded = True
                break

        except Exception as e:
            print(f"  Failed: {e}")
            continue

    if not downloaded:
        print("\n  Automatic download failed. Manual steps:")
        print("  1. Go to: https://data.mendeley.com/datasets/5fk732vkwr/2")
        print("  2. Click 'Download' (may need free Mendeley account)")
        print("  3. Extract the ZIP into: data/raw/maalej/")
        print("  4. Re-run this script")

        # Create manual instruction file
        (output_dir / "DOWNLOAD_INSTRUCTIONS.txt").write_text(
            "MAALEJ Dataset Download Instructions\n"
            "====================================\n\n"
            "1. Visit: https://data.mendeley.com/datasets/5fk732vkwr/2\n"
            "2. Click the 'Download' button\n"
            "3. Extract the ZIP file into this directory (data/raw/maalej/)\n"
            "4. Re-run: python3 scripts/download_datasets.py maalej\n\n"
            "Citation:\n"
            "Maalej, W., Kurtanovic, Z., Nabil, H., & Stanik, C. (2016).\n"
            "On the automatic classification of app reviews.\n"
            "Requirements Engineering, 21(3), 311-331.\n"
        )

    # Try to process whatever files we have
    _process_maalej(output_dir, processed_file)
    return downloaded


def _process_maalej(input_dir: Path, output_file: Path):
    """Process MAALEJ raw files into standardized JSON."""
    reviews = []

    # Look for CSV files
    csv_files = list(input_dir.glob("**/*.csv")) + list(input_dir.glob("**/*.CSV"))
    txt_files = list(input_dir.glob("**/*.txt"))
    xlsx_files = list(input_dir.glob("**/*.xlsx"))

    for csv_file in csv_files:
        if "INSTRUCTION" in csv_file.name.upper():
            continue
        try:
            print(f"  Processing: {csv_file.name}")
            with open(csv_file, "r", encoding="utf-8", errors="replace") as f:
                # Try different delimiters
                sample = f.read(2048)
                f.seek(0)
                delimiter = "," if sample.count(",") > sample.count("\t") else "\t"
                reader = csv.DictReader(f, delimiter=delimiter)

                for row in reader:
                    # Try common column names
                    text = (
                        row.get("review", "") or row.get("text", "") or
                        row.get("Review", "") or row.get("comment", "") or
                        row.get("reviewText", "") or ""
                    )
                    label = (
                        row.get("class", "") or row.get("label", "") or
                        row.get("Class", "") or row.get("category", "") or
                        row.get("Classification", "") or ""
                    )
                    rating = row.get("rating", row.get("Rating", row.get("star", "3")))

                    if text.strip():
                        # Map MAALEJ labels to our schema
                        label_map = {
                            "Bug": "bug_report", "bug": "bug_report", "PD": "bug_report",
                            "Problem Discovery": "bug_report",
                            "Feature": "feature_request", "feature": "feature_request",
                            "FR": "feature_request", "Feature Request": "feature_request",
                            "UserExperience": "usability", "UE": "usability",
                            "User Experience": "usability",
                            "Rating": "praise", "RT": "praise",
                        }
                        mapped_label = label_map.get(label.strip(), "other")

                        reviews.append({
                            "text": text.strip(),
                            "rating": int(float(rating)) if rating and rating.strip() else 3,
                            "app_id": row.get("app", row.get("App", "maalej_app")),
                            "labels": [mapped_label],
                            "source": "maalej",
                            "original_label": label.strip(),
                        })
        except Exception as e:
            print(f"  Error processing {csv_file.name}: {e}")

    if reviews:
        output_file.write_text(json.dumps(reviews, indent=2))
        print(f"\n  Processed {len(reviews)} MAALEJ reviews -> {output_file}")

        # Print label distribution
        from collections import Counter
        dist = Counter(r["labels"][0] for r in reviews)
        print("  Label distribution:")
        for label, count in dist.most_common():
            print(f"    {label}: {count}")
    else:
        print("  No reviews processed yet. Download the dataset first.")


# ============================================================
# RRGen Dataset - Requires Academic Access
# ============================================================

def setup_rrgen():
    """Provide instructions for obtaining the RRGen dataset."""
    print("\n" + "=" * 60)
    print("RRGen Dataset (~309K review-response pairs)")
    print("Source: Gao et al. (ASE 2019)")
    print("GitHub: https://github.com/armor-ai/RRGen")
    print("=" * 60)

    output_dir = RAW_DIR / "rrgen"
    processed_file = output_dir / "rrgen_reviews.json"

    if processed_file.exists():
        data = json.loads(processed_file.read_text())
        print(f"Already processed: {len(data)} review-response pairs in {processed_file}")
        return True

    # Check if user has already downloaded raw files
    existing_files = list(output_dir.glob("**/*.csv")) + list(output_dir.glob("**/*.json")) + list(output_dir.glob("**/*.txt"))
    if existing_files:
        print(f"\n  Found files in {output_dir}:")
        for f in existing_files[:10]:
            print(f"    {f.name}")
        print("  Attempting to process...")
        _process_rrgen(output_dir, processed_file)
        return True

    print("\n  This dataset requires academic access request.")
    print("\n  Steps to obtain:")
    print("  1. Visit: https://github.com/armor-ai/RRGen")
    print("  2. Read the README for the Google Form link")
    print("  3. Fill out the form with your academic details")
    print("  4. Wait for approval (usually 1-3 days)")
    print("  5. Download and extract into: data/raw/rrgen/")
    print("  6. Re-run: python3 scripts/download_datasets.py rrgen")

    instructions = (
        "RRGen Dataset Access Instructions\n"
        "==================================\n\n"
        "1. Visit: https://github.com/armor-ai/RRGen\n"
        "2. Find the Google Form link in the README\n"
        "3. Fill in your name, institution, and research purpose\n"
        "4. Wait for email approval (1-3 business days)\n"
        "5. Download the dataset and extract here\n"
        "6. Re-run: python3 scripts/download_datasets.py rrgen\n\n"
        "Expected contents:\n"
        "- 58 apps with Google Play reviews and developer responses\n"
        "- ~309,246 review-response pairs\n"
        "- Used for Stage 1 (classification training) and Stage 4b (RAG)\n\n"
        "Citation:\n"
        "Gao, C., Zeng, J., Xia, X., Lo, D., Lyu, M. R., & King, I. (2019).\n"
        "Automating App Review Response Generation. ASE 2019.\n"
    )
    (output_dir / "DOWNLOAD_INSTRUCTIONS.txt").write_text(instructions)
    print(f"\n  Instructions saved to: {output_dir}/DOWNLOAD_INSTRUCTIONS.txt")
    return False


def _process_rrgen(input_dir: Path, output_file: Path):
    """Process RRGen raw files into standardized JSON."""
    reviews = []
    csv_files = list(input_dir.glob("**/*.csv"))
    json_files = list(input_dir.glob("**/*.json"))

    for f in csv_files:
        try:
            with open(f, "r", encoding="utf-8", errors="replace") as fh:
                reader = csv.DictReader(fh)
                for row in reader:
                    review_text = row.get("review", row.get("Review", row.get("text", "")))
                    response_text = row.get("response", row.get("Response", row.get("reply", "")))
                    if review_text.strip():
                        reviews.append({
                            "text": review_text.strip(),
                            "response": response_text.strip() if response_text else "",
                            "rating": int(float(row.get("rating", row.get("Rating", "3")))) if row.get("rating", row.get("Rating")) else 3,
                            "app_id": row.get("app", row.get("App", f.stem)),
                            "source": "rrgen",
                        })
        except Exception as e:
            print(f"  Error processing {f.name}: {e}")

    for f in json_files:
        if f.name == "rrgen_reviews.json":
            continue
        try:
            data = json.loads(f.read_text())
            if isinstance(data, list):
                for item in data:
                    if isinstance(item, dict) and ("review" in item or "text" in item):
                        reviews.append({
                            "text": (item.get("review") or item.get("text", "")).strip(),
                            "response": (item.get("response") or item.get("reply", "")).strip(),
                            "rating": item.get("rating", 3),
                            "app_id": item.get("app", item.get("app_id", "rrgen_app")),
                            "source": "rrgen",
                        })
        except Exception as e:
            print(f"  Error processing {f.name}: {e}")

    if reviews:
        output_file.write_text(json.dumps(reviews, indent=2))
        print(f"  Processed {len(reviews)} RRGen review-response pairs -> {output_file}")


# ============================================================
# GUZMAN Dataset - Contact Authors
# ============================================================

def setup_guzman():
    """Provide instructions for obtaining the GUZMAN dataset."""
    print("\n" + "=" * 60)
    print("GUZMAN Dataset (~1,820 aspect-annotated reviews)")
    print("Source: Guzman & Maalej (RE 2014)")
    print("=" * 60)

    output_dir = RAW_DIR / "guzman"
    processed_file = output_dir / "guzman_reviews.json"

    if processed_file.exists():
        data = json.loads(processed_file.read_text())
        print(f"Already processed: {len(data)} reviews in {processed_file}")
        return True

    existing_files = list(output_dir.glob("**/*.csv")) + list(output_dir.glob("**/*.json"))
    if existing_files:
        print(f"\n  Found files in {output_dir}, attempting to process...")
        _process_guzman(output_dir, processed_file)
        return True

    print("\n  This dataset is not publicly hosted. You need to contact the authors.")
    print("\n  Steps to obtain:")
    print("  1. Email Prof. Walid Maalej at: maalej@informatik.uni-hamburg.de")
    print("  2. Or check: https://mast.informatik.uni-hamburg.de/app-review-analysis/")
    print("  3. Mention your research and cite Guzman & Maalej (RE 2014)")
    print("  4. Download and extract into: data/raw/guzman/")
    print("  5. Re-run: python3 scripts/download_datasets.py guzman")

    instructions = (
        "GUZMAN Dataset Access Instructions\n"
        "====================================\n\n"
        "This dataset is not publicly hosted.\n\n"
        "Option 1: Contact the authors\n"
        "  Email: maalej@informatik.uni-hamburg.de (Prof. Walid Maalej)\n"
        "  Subject: Request for app review dataset (Guzman & Maalej, RE 2014)\n\n"
        "Option 2: Check the MAST lab page\n"
        "  URL: https://mast.informatik.uni-hamburg.de/app-review-analysis/\n\n"
        "Expected contents:\n"
        "- 7 apps (3 iOS, 4 Android)\n"
        "- ~260 reviews per app (~1,820 total)\n"
        "- Aspect-level sentiment annotations\n"
        "- Used for Stage 1 (aspect-based sentiment training)\n\n"
        "Citation:\n"
        "Guzman, E., & Maalej, W. (2014).\n"
        "How Do Users Like This Feature? A Fine Grained Sentiment\n"
        "Analysis of App Reviews. RE 2014.\n"
        "DOI: 10.1109/RE.2014.6912257\n"
    )
    (output_dir / "DOWNLOAD_INSTRUCTIONS.txt").write_text(instructions)
    print(f"\n  Instructions saved to: {output_dir}/DOWNLOAD_INSTRUCTIONS.txt")
    return False


def _process_guzman(input_dir: Path, output_file: Path):
    """Process GUZMAN raw files into standardized JSON."""
    reviews = []
    for f in list(input_dir.glob("**/*.csv")):
        try:
            with open(f, "r", encoding="utf-8", errors="replace") as fh:
                reader = csv.DictReader(fh)
                for row in reader:
                    text = row.get("review", row.get("text", row.get("Review", "")))
                    aspect = row.get("aspect", row.get("feature", row.get("Aspect", "")))
                    sentiment = row.get("sentiment", row.get("Sentiment", ""))
                    if text.strip():
                        reviews.append({
                            "text": text.strip(),
                            "aspect": aspect.strip(),
                            "sentiment": sentiment.strip(),
                            "app_id": row.get("app", row.get("App", f.stem)),
                            "source": "guzman",
                        })
        except Exception as e:
            print(f"  Error processing {f.name}: {e}")

    if reviews:
        output_file.write_text(json.dumps(reviews, indent=2))
        print(f"  Processed {len(reviews)} GUZMAN reviews -> {output_file}")


# ============================================================
# Synthetic Sample Data Generator
# ============================================================

def generate_sample_data():
    """Generate a larger synthetic dataset for development and testing."""
    print("\n" + "=" * 60)
    print("Generating Synthetic Sample Data")
    print("(For development/testing when real datasets aren't available)")
    print("=" * 60)

    random.seed(42)

    # Templates for each category
    templates = {
        "bug_report": [
            "App crashes when I {action}. This started after {version}.",
            "Can't {action} anymore. It just {symptom} every time on my {device}.",
            "{feature} is completely broken since the last update. {symptom}!",
            "Getting a {symptom} whenever I try to {action} on {device} running {os}.",
            "The app {symptom} every time I open {feature}. Please fix ASAP!",
            "Since updating to {version}, the {feature} {symptom}. Terrible!",
            "{feature} won't work on my {device}. It just {symptom}.",
            "Keep getting {symptom} when using {feature}. Very frustrating.",
        ],
        "feature_request": [
            "I wish this app had {feature}. Would make it so much better!",
            "Please add {feature}! Every other app has it.",
            "It would be great if you could add {feature} to the app.",
            "Can you please implement {feature}? I really need it.",
            "The app needs {feature}. Without it, I'm switching to {competitor}.",
            "Suggestion: add {feature}. It would improve the experience a lot.",
        ],
        "performance": [
            "App is super slow on my {device}. Takes {time} to load.",
            "The {feature} is extremely laggy. My {device} heats up like crazy.",
            "Battery drain is insane with this app on {device} running {os}.",
            "App uses too much memory. My {device} keeps killing it in background.",
            "Loading time is {time} for {feature}. Way too slow on {device}.",
            "The app makes my {device} overheat and drains battery in {time}.",
        ],
        "usability": [
            "The {feature} is so confusing. I can't figure out how to {action}.",
            "UI is terrible. Can't find where to {action}. Too many menus!",
            "The new design is awful. I can't {action} anymore. Make it simpler!",
            "Text is too small on {feature}. Can't read anything on my {device}.",
            "Buttons are too close together on {feature}. Keep tapping the wrong one.",
            "Navigation is very confusing. Took me {time} to find {feature}.",
        ],
        "compatibility": [
            "App doesn't work on my {device} with {os}. Just shows a blank screen.",
            "Crashes immediately on {device} running {os}. Works fine on other devices.",
            "Not compatible with {os} on {device}. Please update for newer devices!",
            "The {feature} looks broken on my {device} with {os}. Layout is messed up.",
        ],
        "praise": [
            "Love this app! The {feature} is amazing. 5 stars!",
            "Best app ever! The {feature} works perfectly on my {device}.",
            "Great update! The new {feature} is exactly what I needed.",
            "Keep up the great work! {feature} is so smooth and fast.",
            "Finally an app that does {feature} right. Highly recommended!",
        ],
    }

    actions = ["login", "sign up", "checkout", "upload a photo", "send a message",
               "search", "update my profile", "sync my data", "make a payment",
               "open settings", "share a post", "download content"]
    symptoms = ["crashes", "freezes", "shows an error", "force closes",
                "goes to a blank screen", "hangs for minutes", "restarts",
                "loses my data", "shows a white screen"]
    features = ["login", "camera", "dark mode", "notifications", "payment",
                "search", "profile page", "settings", "home screen", "chat",
                "photo editor", "file manager", "calendar", "maps", "voice input"]
    requested_features = ["dark mode", "widget support", "offline mode", "biometric login",
                         "custom themes", "export to PDF", "multi-language support",
                         "cloud backup", "split screen", "desktop sync"]
    devices = ["iPhone 15 Pro", "iPhone 14", "Samsung Galaxy S24", "Google Pixel 8",
               "OnePlus 12", "Samsung Galaxy A54", "iPad Pro", "Xiaomi 14"]
    os_versions = ["iOS 18.2", "iOS 17.5", "Android 15", "Android 14", "Android 13"]
    app_versions = ["v3.2", "v3.2.1", "v3.1", "v3.0", "v2.9"]
    times = ["10 seconds", "30 seconds", "a minute", "2 minutes", "forever"]
    competitors = ["the competitor", "another app", "AppX", "the other app"]

    reviews = []
    base_time = datetime(2026, 1, 1)

    for i in range(500):
        category = random.choices(
            list(templates.keys()),
            weights=[0.30, 0.20, 0.15, 0.15, 0.10, 0.10],
        )[0]

        template = random.choice(templates[category])
        text = template.format(
            action=random.choice(actions),
            symptom=random.choice(symptoms),
            feature=random.choice(features) if category != "feature_request" else random.choice(requested_features),
            device=random.choice(devices),
            os=random.choice(os_versions),
            version=random.choice(app_versions),
            time=random.choice(times),
            competitor=random.choice(competitors),
        )

        rating_map = {
            "bug_report": random.choice([1, 1, 1, 2]),
            "feature_request": random.choice([2, 3, 3]),
            "performance": random.choice([1, 2, 2]),
            "usability": random.choice([1, 2, 2, 3]),
            "compatibility": random.choice([1, 1, 2]),
            "praise": random.choice([4, 5, 5, 5]),
        }

        reviews.append({
            "text": text,
            "rating": rating_map[category],
            "app_id": "com.example.reviewagent",
            "timestamp": (base_time + timedelta(hours=random.randint(0, 2000))).isoformat(),
            "labels": [category],
            "source": "synthetic",
        })

    # Save
    output_file = RAW_DIR / "sample_reviews.json"
    output_file.write_text(json.dumps(reviews, indent=2))
    print(f"\n  Generated {len(reviews)} synthetic reviews -> {output_file}")

    # Also save as training-ready format
    train_file = PROCESSED_DIR / "training_data.json"
    train_data = {
        "texts": [r["text"] for r in reviews],
        "labels": [r["labels"] for r in reviews],
        "ratings": [r["rating"] for r in reviews],
    }
    train_file.write_text(json.dumps(train_data, indent=2))
    print(f"  Training-ready format -> {train_file}")

    # Print distribution
    from collections import Counter
    dist = Counter(r["labels"][0] for r in reviews)
    print("\n  Category distribution:")
    for cat, count in dist.most_common():
        print(f"    {cat}: {count} ({count/len(reviews)*100:.1f}%)")

    return True


# ============================================================
# Dataset Summary
# ============================================================

def print_summary():
    """Print summary of all available datasets."""
    print("\n" + "=" * 60)
    print("DATASET SUMMARY")
    print("=" * 60)

    datasets = {
        "MAALEJ": RAW_DIR / "maalej" / "maalej_reviews.json",
        "RRGen": RAW_DIR / "rrgen" / "rrgen_reviews.json",
        "GUZMAN": RAW_DIR / "guzman" / "guzman_reviews.json",
        "Sample (synthetic)": RAW_DIR / "sample_reviews.json",
    }

    print(f"\n{'Dataset':<25} {'Status':<15} {'Reviews':<10} {'Path'}")
    print("-" * 90)
    for name, path in datasets.items():
        if path.exists():
            data = json.loads(path.read_text())
            count = len(data)
            print(f"{name:<25} {'AVAILABLE':<15} {count:<10} {path.relative_to(BASE_DIR)}")
        else:
            instruction = path.parent / "DOWNLOAD_INSTRUCTIONS.txt"
            status = "INSTRUCTIONS" if instruction.exists() else "MISSING"
            print(f"{name:<25} {status:<15} {'--':<10} {path.relative_to(BASE_DIR)}")

    print(f"\nData directory: {DATA_DIR}")


# ============================================================
# Main
# ============================================================

def main():
    ensure_dirs()

    if len(sys.argv) > 1:
        target = sys.argv[1].lower()
        if target == "maalej":
            download_maalej()
        elif target == "rrgen":
            setup_rrgen()
        elif target == "guzman":
            setup_guzman()
        elif target == "sample":
            generate_sample_data()
        elif target == "summary":
            print_summary()
        else:
            print(f"Unknown target: {target}")
            print("Usage: python3 scripts/download_datasets.py [maalej|rrgen|guzman|sample|summary]")
            return
    else:
        # Download/setup everything
        print("Setting up all datasets...\n")
        download_maalej()
        setup_rrgen()
        setup_guzman()
        generate_sample_data()

    print_summary()


if __name__ == "__main__":
    main()
