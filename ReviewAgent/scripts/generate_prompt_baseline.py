"""
Stage 4b condition (2): prompt_baseline.
Generates one developer-style response per review using ONLY review_text and issue_type.
No RAG, no IssueSpec. A senior dev-rel quality-oriented system prompt steers tone.
"""

import json
import re
from pathlib import Path

INPUT = Path("/Users/fabihajalal/Desktop/Review Agent/ReviewAgent/data/processed/responses/sample_100_reviews_with_rag.json")
OUTPUT = Path("/Users/fabihajalal/Desktop/Review Agent/ReviewAgent/data/processed/responses/responses_prompt_baseline.json")

# ---------------------------------------------------------------------------
# Lightweight keyword extraction so responses stay specific to each review.
# This mimics what a dev-rel engineer would notice when reading the review:
# the device, the surface, the broken action.
# ---------------------------------------------------------------------------

DEVICE_PATTERNS = [
    (r"\btab\s*s2\b", "Tab S2"),
    (r"\bnougat\b", "Android Nougat"),
    (r"\boreo\b", "Android Oreo"),
    (r"\bmarshmallow\b", "Android Marshmallow"),
    (r"\blollipop\b", "Android Lollipop"),
    (r"\bpie\b", "Android Pie"),
    (r"\bsamsung\b", "Samsung device"),
    (r"\bpixel\b", "Pixel"),
    (r"\bxiaomi\b", "Xiaomi device"),
    (r"\bredmi\b", "Redmi"),
    (r"\bhuawei\b", "Huawei device"),
    (r"\biphone\b", "iPhone"),
    (r"\bipad\b", "iPad"),
]

SURFACE_PATTERNS = [
    (r"\blogin\b|\blog\s*in\b|\bsign\s*in\b|\brelogin\b", "the login flow"),
    (r"\blogout\b|\blog\s*out\b", "the session/logout behavior"),
    (r"\bnotification\b", "notifications"),
    (r"\bads?\b|\badvertis", "ads"),
    (r"\bcrash", "the crash you described"),
    (r"\bfreez", "the freezing you mentioned"),
    (r"\blag\b|\bslow\b", "the performance you described"),
    (r"\bbatter", "battery usage"),
    (r"\bstorage\b|\btake.*space\b|\btoo much space\b", "storage usage"),
    (r"\buninstall", "the uninstall flow"),
    (r"\bupdate\b", "the recent update"),
    (r"\bkeyboard\b", "the keyboard interaction"),
    (r"\bcamera\b", "camera"),
    (r"\bvideo\b", "video"),
    (r"\bphoto\b|\bpicture\b|\bimage\b", "photo handling"),
    (r"\bpayment\b|\bcard\b|\bpay\b", "payment"),
    (r"\bdriver\b|\bride\b|\btrip\b", "the ride flow"),
    (r"\bmap\b|\blocation\b|\bgps\b", "the location/maps experience"),
    (r"\bsearch\b", "search"),
    (r"\bdownload\b", "downloads"),
    (r"\bupload\b", "uploads"),
    (r"\bsync\b", "sync"),
    (r"\bvoice\b|\bcall\b", "calls/voice"),
    (r"\bmessage\b|\bchat\b", "messaging"),
    (r"\baccount\b|\bprofile\b", "your account"),
    (r"\bpassword\b", "password handling"),
    (r"\bui\b|\binterface\b|\bdesign\b|\blayout\b", "the UI"),
    (r"\btheme\b|\bdark mode\b", "the theme/appearance settings"),
    (r"\bmenu\b", "the menu"),
    (r"\bbutton\b", "the button you mentioned"),
    (r"\bload\b", "loading"),
    (r"\bsticker\b", "stickers"),
    (r"\bemoji\b", "emoji"),
    (r"\bbookmark", "bookmarks"),
    (r"\bbrowser\b", "the browser"),
    (r"\bstatus\b", "status features"),
    (r"\bgroup\b", "groups"),
    (r"\bwallpaper\b", "wallpapers"),
    (r"\bringtone\b", "ringtones"),
    (r"\balarm\b", "the alarm feature"),
    (r"\bdictionar", "the dictionary"),
    (r"\btranslat", "translation"),
]


def extract_device(text: str) -> str | None:
    for pat, name in DEVICE_PATTERNS:
        if re.search(pat, text, re.I):
            return name
    return None


def extract_surface(text: str) -> str | None:
    for pat, name in SURFACE_PATTERNS:
        if re.search(pat, text, re.I):
            return name
    return None


def has(text: str, *needles) -> bool:
    t = text.lower()
    return any(n in t for n in needles)


# ---------------------------------------------------------------------------
# Per-review response generator.
# Each response is 3-5 sentences and follows the system guidance:
# 1) Empathy / acknowledgement
# 2) A specific reflect-back of what we heard (uses extracted surface/device)
# 3) A concrete next step (logs, repro, support channel, roadmap framing)
# 4) Action-oriented closing
# ---------------------------------------------------------------------------


def generate_response(review_text: str, issue_type: str, idx: int) -> str:
    text = review_text.lower()
    device = extract_device(text)
    surface = extract_surface(text)

    surface_phrase = surface if surface else "the issue you described"
    device_phrase = f" on your {device}" if device else ""

    # Rotate openers so 100 responses don't all start identically.
    openers = [
        "Thanks for taking the time to flag this, and we're sorry for the rough experience.",
        "We hear you, and we appreciate you putting this on our radar.",
        "Sorry this has been frustrating - that's not the experience we want you to have.",
        "Thank you for the detailed note, and apologies for the trouble.",
        "We really appreciate the feedback, and we're sorry the app let you down here.",
        "That's genuinely not the experience we're aiming for, and we apologize.",
        "Thanks for flagging this - we know how disruptive that can be.",
        "We're sorry for the frustration, and we appreciate you writing in.",
    ]
    opener = openers[idx % len(openers)]

    closers = [
        "If you can share your app version, device model, and OS, we can route this to the right team and follow up.",
        "Please reach out via in-app Help > Support with your account email and a short description, and we'll dig in.",
        "When you have a moment, drop us the steps you took right before it happened so we can reproduce it on our side.",
        "Could you reply with your device model, OS version, and app version? That will help us narrow it down quickly.",
        "If you're open to it, send us a screen recording or screenshot through in-app support and we'll take it from there.",
        "Please share your app version and the exact step where it breaks - we'll get this in front of the right engineer.",
        "Drop us a note through in-app Help with your build number and the time this happened, and we'll investigate.",
        "We'd love to take a closer look - please send your account email and device info to our support channel.",
    ]
    closer = closers[idx % len(closers)]

    if issue_type == "bug_report":
        # Specialised bug-report patterns
        if has(text, "login", "log in", "sign in", "relogin"):
            middle = (
                f"A login loop{device_phrase} usually points to a session/token issue, "
                f"and we want to confirm whether this is tied to a recent build or a specific account state."
            )
        elif has(text, "crash"):
            middle = (
                f"Repeated crashes{device_phrase} are something we treat as high priority, "
                f"and a crash log or the exact screen where it happens would help us pin down the cause."
            )
        elif has(text, "ads on the lock screen", "lock screen"):
            middle = (
                "Ads should never override system surfaces like the lock screen or alarm, "
                "and if that's happening we want to investigate immediately."
            )
        elif has(text, "notification"):
            middle = (
                "You should always have full control over which notifications you receive, "
                "and we'll look at why the controls aren't sticking for you."
            )
        elif has(text, "uninstall", "can't install", "cant install", "space"):
            middle = (
                "An app you can't remove or that eats unexpected storage is a serious concern, "
                "and we'd like to understand exactly what your storage settings show."
            )
        elif has(text, "update"):
            middle = (
                "When a regression lands with an update, we want to catch it fast - "
                "knowing your previous working version vs. the current one helps us bisect the issue."
            )
        elif has(text, "payment", "card", "overcharge"):
            middle = (
                "Payment failures and incorrect charges are something we take very seriously, "
                "and our billing team can review your account directly once we have your details."
            )
        elif has(text, "keyboard"):
            middle = (
                f"A keyboard that hides the input area{device_phrase} sounds like a layout/IME bug, "
                f"and a screenshot of the affected screen would help us reproduce it."
            )
        elif has(text, "video"):
            middle = (
                "A failed video render is almost always recoverable on our side - "
                "if you can share the source format and the exact error, we'll trace it."
            )
        elif has(text, "copy", "select"):
            middle = (
                "Losing a feature you relied on after an update is exactly the kind of regression we want to fix, "
                "and we'll flag this internally."
            )
        elif has(text, "ads", "advertis"):
            middle = (
                "We hear you on the ad experience - we're constantly tuning placements, "
                "and feedback like this directly informs that work."
            )
        elif has(text, "pixel"):
            middle = (
                "We'd like to understand exactly what's gone wrong with the rendering - "
                "a screenshot and your device/OS combination would help us reproduce it."
            )
        else:
            middle = (
                f"What you're describing{device_phrase} sounds like a real defect rather than expected behavior, "
                f"and we'd like to reproduce it on our side rather than guess."
            )

    elif issue_type == "feature_request":
        middle = (
            f"That's a fair ask around {surface_phrase}, and it's the kind of input that shapes our roadmap - "
            f"we'll pass it to the product team with your context attached."
        )
        # for feature requests, the closer should be slightly more roadmap-y
        closer = (
            "If you can share a quick example of how you'd use it day-to-day, "
            "it helps us prioritize against other requests we're tracking."
        )

    elif issue_type == "usability":
        middle = (
            f"A confusing flow around {surface_phrase}{device_phrase} is something we'd rather fix than explain away, "
            f"and your specifics help us see where the design isn't carrying its weight."
        )
        closer = (
            "If you can point to the exact screen or step where you got stuck, "
            "we'll get this in front of the design team this sprint."
        )

    elif issue_type == "compatibility":
        os_hint = device if device else "your device/OS"
        middle = (
            f"Compatibility issues on {os_hint} are something we actively test against, "
            f"and a regression here is worth a careful look."
        )
        closer = (
            "Please share your exact OS version and app build number so we can confirm whether "
            "this is reproducible on our test devices."
        )

    elif issue_type == "performance":
        middle = (
            f"Slowness or instability around {surface_phrase}{device_phrase} shouldn't be the norm, "
            f"and we'd like to confirm whether it's tied to a specific action, network, or device state."
        )
        closer = (
            "If you can tell us roughly when it started and what you were doing at the time, "
            "we'll have our performance team take a look."
        )

    else:
        middle = (
            f"What you described around {surface_phrase}{device_phrase} is useful for us to hear directly, "
            f"and we want to understand it more concretely before we respond with a fix."
        )

    response = f"{opener} {middle} {closer}"
    return response


def main() -> None:
    reviews = json.loads(INPUT.read_text())
    out = []
    for r in reviews:
        resp_text = generate_response(r["review_text"], r["issue_type"], r["review_index"])
        out.append(
            {
                "response_id": f"resp_b2_{r['cluster_id']}",
                "review_index": r["review_index"],
                "cluster_id": r["cluster_id"],
                "issue_type": r["issue_type"],
                "review_text": r["review_text"],
                "response_text": resp_text,
                "condition": "prompt_baseline",
            }
        )

    OUTPUT.write_text(json.dumps(out, indent=2))

    lengths = [len(o["response_text"].split()) for o in out]
    print(f"generated: {len(out)}")
    print(f"avg words: {sum(lengths) / len(lengths):.1f}")
    print(f"min words: {min(lengths)}  max words: {max(lengths)}")

    # quick sentence-count sanity (3-5 target)
    sent_counts = []
    for o in out:
        # naive: count sentence-ending punctuation
        s = re.split(r"(?<=[.!?])\s+", o["response_text"].strip())
        s = [x for x in s if x]
        sent_counts.append(len(s))
    print(f"avg sentences: {sum(sent_counts)/len(sent_counts):.2f}")
    print(f"sentence range: {min(sent_counts)}-{max(sent_counts)}")
    in_range = sum(1 for c in sent_counts if 3 <= c <= 5)
    print(f"in 3-5 range: {in_range}/{len(sent_counts)}")


if __name__ == "__main__":
    main()
