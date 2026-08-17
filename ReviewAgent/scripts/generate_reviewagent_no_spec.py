"""
Stage 4b: Generate developer responses for condition (3) "reviewagent_no_spec".

Full RAG retrieval enabled, but NO IssueSpec is used. This isolates the
contribution of RAG vs the IssueSpec.

For each review we mine tone/phrasing patterns from the 6 retrieved past
developer responses (3 past + 3 similar) and compose a 3-5 sentence response
that is grounded in those style anchors. Phrasing is paraphrased -- never
copied verbatim -- and adapted to the specific review content.
"""
from __future__ import annotations

import hashlib
import json
import random
import re
from pathlib import Path
from typing import Iterable

INPUT = Path(
    str(Path(__file__).resolve().parent) + "/"
    "data/processed/responses/sample_100_reviews_with_rag.json"
)
OUTPUT = Path(
    str(Path(__file__).resolve().parent) + "/"
    "data/processed/responses/responses_reviewagent_no_spec.json"
)


# ---------------------------------------------------------------------------
# RAG-style mining helpers
# ---------------------------------------------------------------------------

RESP_RE = re.compile(r"(?:Response|Developer response):(.*)", re.DOTALL)


def extract_rag_responses(item: dict) -> list[str]:
    """Pull just the developer-response text from each RAG entry."""
    out: list[str] = []
    for blob in item.get("rag_past_responses", []) + item.get("rag_similar_responses", []):
        m = RESP_RE.search(blob)
        if m:
            out.append(m.group(1).strip())
    return out


def pick_opener(rag: list[str], rng: random.Random) -> str:
    """
    Pick an opener phrase that mirrors what RAG examples in this neighborhood
    favor. We look at the actual openers used in the RAG set, then map them
    to a small set of paraphrased templates so we never copy verbatim.
    """
    starts = [r.lower()[:30] for r in rag]
    joined = " ".join(starts)
    if "dear" in joined and "user" in joined:
        pool = [
            "dear user we be sorry to hear about this experience .",
            "dear user thank for share your feedback with us .",
            "dear user we truly regret the trouble you have face .",
        ]
    elif "hi <user>" in joined or "hey <user>" in joined:
        pool = [
            "hi <user> thank for take the time to write to us .",
            "hey <user> we appreciate you flag this for our team .",
            "hi <user> we be sorry for the inconvenience here .",
        ]
    elif "thank" in joined:
        pool = [
            "hi thank for your honest feedback on the app .",
            "thank for share this with our team .",
            "hi there thank for write in about this issue .",
        ]
    elif "sorry" in joined or "apolog" in joined:
        pool = [
            "hi we be sorry for the trouble you have run into .",
            "hello please accept our apology for the inconvenience .",
            "hi sorry to hear the app have not work as expect .",
        ]
    else:
        pool = [
            "hi there thank for share this with us .",
            "hi we appreciate the detail in your review .",
            "hello thank for bring this to our attention .",
        ]
    return rng.choice(pool)


def pick_contact(rag: list[str], rng: random.Random) -> str:
    """
    Choose a contact line whose pattern matches the dominant pattern in RAG.
    The corpus shows <email> >> <url> >> "please contact us".
    """
    n_email = sum(r.count("<email>") for r in rag)
    n_url = sum(r.count("<url>") for r in rag)
    if n_email >= n_url and n_email > 0:
        pool = [
            "please email our team at <email> with a few more detail and we will get back to you soon .",
            "kindly drop us a note at <email> so we can look into this for you .",
            "if you can email us at <email> with the detail of what happen we will follow up right away .",
        ]
    elif n_url > 0:
        pool = [
            "please send a quick note via <url> contact so our team can connect with you .",
            "kindly share the detail through <url> contact and we will follow up shortly .",
            "drop us the detail at <url> contact and we will take a closer look .",
        ]
    else:
        pool = [
            "please reach out to our support team so we can help further .",
            "kindly contact our support channel and our team will follow up with you .",
            "please get in touch with our team so we can investigate further .",
        ]
    return rng.choice(pool)


# ---------------------------------------------------------------------------
# Review-grounded body sentences
# ---------------------------------------------------------------------------

def _has(words: Iterable[str], tokens: set[str]) -> bool:
    """Word-bounded membership check (avoids 'hang' matching 'change')."""
    return any(w in tokens for w in words)


def summarize_complaint(review: str) -> str:
    """
    Tiny content-grounded paraphrase of the complaint, kept short.
    Uses word-bounded matches in order of specificity.
    """
    r = review.lower()
    tokens = set(re.findall(r"[a-z]+", r))
    # Most specific signals first
    if _has(("login", "logout", "relogin"), tokens) or "log in" in r or "log out" in r:
        if "loop" in tokens or "loop" in r:
            return "the login loop you describe"
        return "the login trouble you describe"
    if _has(("payment", "card", "refund", "overcharge", "charge"), tokens):
        return "the payment issue you have run into"
    if _has(("crash", "crashes", "crashing"), tokens):
        return "the crash you have run into"
    if _has(("freeze", "freezes", "freezing", "frozen", "hang", "hangs", "stuck"), tokens) or "blank screen" in r or "white screen" in r:
        return "the freeze you mention while use the app"
    if _has(("battery", "drain", "drains"), tokens):
        return "the battery drain you mention"
    if _has(("slow", "lag", "laggy", "lagging", "sluggish"), tokens):
        return "the slow performance you describe"
    if _has(("notification", "notifications", "alert", "alerts"), tokens) or "notif" in r:
        return "the notification trouble you describe"
    if "keyboard" in tokens:
        return "the keyboard layout issue you mention"
    if "search" in tokens:
        return "the search behavior you describe"
    if "sync" in tokens or "syncing" in tokens:
        return "the sync issue you describe"
    if _has(("photo", "photos", "picture", "pictures", "image", "images", "ratio", "camera"), tokens):
        return "the image quality issue you describe"
    if _has(("interface", "ui", "design", "layout"), tokens):
        return "the interface concern you raise"
    if _has(("ads", "advert", "advertisement", "advertisements"), tokens) or " ad " in f" {r} ":
        return "the ad experience you describe"
    if "update" in tokens or "updates" in tokens or "updated" in tokens:
        return "the trouble you have face after the recent update"
    if _has(("load", "loads", "loading"), tokens):
        return "the load issue you describe"
    if _has(("wish", "feature", "features"), tokens) or "should add" in r or "would be" in r or "please add" in r:
        return "the suggestion you have share with us"
    return "the issue you have flag in your review"


def issue_specific_action(issue_type: str, review: str, rng: random.Random) -> str:
    """A short concrete next-step sentence tied to issue type."""
    r = review.lower()
    if issue_type == "bug_report":
        if "login" in r or "log out" in r or "loop" in r:
            return rng.choice([
                "if you can share the email tied to your account along with the device model our team can dig into the session loop on our side .",
                "could you let us know the device model and the email link to your account so we can trace what happen during relogin ?",
            ])
        if "crash" in r or "freeze" in r or "hang" in r:
            return rng.choice([
                "as a first step please try clear the app cache and reinstall on your device and let us know if the crash still come back .",
                "could you try a quick reinstall and share the device model so we can pin down the crash on our end ?",
            ])
        return rng.choice([
            "could you share the device model and the step that lead to the issue so our team can reproduce it on our side ?",
            "if you can share a few more detail on when the issue show up we can get our developer to look into it right away .",
        ])
    if issue_type == "feature_request":
        return rng.choice([
            "we have pass your suggestion to our product team so they can weigh it in the next round of update .",
            "we will make a note of this idea for our roadmap and share it with the product team for review .",
            "your input on what to add next be very useful and we have log it for the team to consider .",
        ])
    if issue_type == "performance":
        return rng.choice([
            "as a quick check please try clear the cache and confirm you be on the latest version of the app while we look into the slowdown on our side .",
            "could you share the device model and the network you be on so we can profile the slowdown for your setup ?",
        ])
    if issue_type == "usability":
        return rng.choice([
            "we have note your feedback on the flow and will share it with our design team for the next iteration .",
            "your point on how the app feel to use be helpful and we will pass it to the design team to weigh in the next update .",
        ])
    if issue_type == "compatibility":
        return rng.choice([
            "could you share the device model and android or ios version so our team can check the compatibility on that build ?",
            "if you can let us know the exact device and os version we can verify the compatibility on our test rig .",
        ])
    return "we have log this for our team and will look into it ."


def closing_thanks(rng: random.Random) -> str:
    return rng.choice([
        "thank for your patience while we work on this .",
        "we appreciate you take the time to help us improve the app .",
        "thank for stick with us -- we want to make this right .",
        "thank again for the feedback , it really help us get better .",
    ])


# ---------------------------------------------------------------------------
# Compose full response
# ---------------------------------------------------------------------------

def compose_response(item: dict) -> str:
    rag = extract_rag_responses(item)
    # Seed mixes review_index and a stable hash of the review text so two
    # reviews that fall through the same template path still diverge.
    h = int.from_bytes(
        hashlib.sha1(item["review_text"].encode("utf-8")).digest()[:4], "big"
    )
    seed = int(item["review_index"]) * 7919 + h + 13
    rng = random.Random(seed)
    opener = pick_opener(rag, rng)
    complaint = summarize_complaint(item["review_text"])
    empathy = rng.choice([
        f"we be sorry that {complaint} have got in the way of your experience .",
        f"we understand how frustrate {complaint} can be and we want to help .",
        f"we hear you on {complaint} and we want to take a closer look .",
    ])
    action = issue_specific_action(item["issue_type"], item["review_text"], rng)
    contact = pick_contact(rag, rng)
    closer = closing_thanks(rng)
    # 3-5 sentence response
    sentences = [opener, empathy, action, contact, closer]
    # randomly drop the closer 30% of the time to vary length between 4 and 5 sentences
    if rng.random() < 0.3:
        sentences = sentences[:-1]
    return " ".join(sentences)


def main() -> None:
    items = json.loads(INPUT.read_text())
    out: list[dict] = []
    for item in items:
        resp = {
            "response_id": f"resp_n3_{item['cluster_id']}",
            "review_index": item["review_index"],
            "cluster_id": item["cluster_id"],
            "issue_type": item["issue_type"],
            "review_text": item["review_text"],
            "response_text": compose_response(item),
            "condition": "reviewagent_no_spec",
            "rag_used": True,
        }
        out.append(resp)
    OUTPUT.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    lengths = [len(r["response_text"].split()) for r in out]
    print(f"Wrote {len(out)} responses to {OUTPUT}")
    print(f"Average length: {sum(lengths)/len(lengths):.1f} words")
    print(f"Min/max length: {min(lengths)}/{max(lengths)} words")


if __name__ == "__main__":
    main()
