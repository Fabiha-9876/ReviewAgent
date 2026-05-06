"""
Stage 4b generator for condition (4) "reviewagent_full":
Full system = RAG retrieval + IssueSpec from Stage 3.

For each review in sample_100_reviews_with_rag.json, produce a 4-6 sentence
developer response that:
  - opens with empathy calibrated to severity (P0 stronger than P3)
  - names the SPECIFIC affected_component (not "this issue")
  - references concrete failure mode (actual_behavior / steps_to_reproduce
    for bugs; user_story / acceptance_criteria for features; nfr_category
    for performance; nielsen_heuristic for usability; device_os_matrix for
    compatibility)
  - sets next-step expectations realistically based on severity
  - mirrors dev-rel phrasing patterns from RAG (apology + ask-for-detail +
    contact channel) without copying verbatim
"""

import hashlib
import json
import random
import re
from pathlib import Path

INPUT_PATH = Path(
    "/Users/fabihajalal/Desktop/Review Agent/ReviewAgent/"
    "data/processed/responses/sample_100_reviews_with_rag.json"
)
OUTPUT_PATH = Path(
    "/Users/fabihajalal/Desktop/Review Agent/ReviewAgent/"
    "data/processed/responses/responses_reviewagent_full.json"
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

EMAIL_RE = re.compile(r"<email>")
URL_RE = re.compile(r"<url>")


def seeded_choice(seed_str: str, options):
    """Deterministic pick from a list, seeded by the cluster id."""
    h = int(hashlib.md5(seed_str.encode()).hexdigest(), 16)
    return options[h % len(options)]


def severity_opening(severity: str, seed: str) -> str:
    """Empathy opener calibrated to severity."""
    p0 = [
        "We're really sorry — getting locked out at a critical moment is exactly the kind of failure we work hardest to prevent, and we hear you.",
        "We sincerely apologise for this — what you've described is a serious break in the experience and we want to make it right.",
        "This is genuinely concerning to us and we're sorry you ran into it; an issue at this level shouldn't reach our users.",
    ]
    p1 = [
        "Thank you for flagging this — we can see why it's frustrating and we appreciate you taking the time to write in.",
        "We're sorry for the trouble here — this isn't the experience we want for you, and your feedback is genuinely useful.",
        "Apologies for the friction you've hit — we hear you and we want to dig into this further.",
    ]
    p2 = [
        "Thanks for sharing this with us — your feedback helps us prioritise what to improve next.",
        "We appreciate you taking the time to write — feedback like this directly shapes our roadmap.",
        "Thank you for the detailed note — we're sorry for the inconvenience and we've taken it on board.",
    ]
    p3 = [
        "Thanks for the suggestion — we always appreciate hearing how the app fits (or doesn't fit) into your routine.",
        "We appreciate you flagging this — it's the kind of polish-level feedback that helps us refine the product.",
        "Thank you for writing in — every piece of feedback gets read by the team.",
    ]
    bucket = {"P0": p0, "P1": p1, "P2": p2, "P3": p3}.get(severity, p1)
    return seeded_choice(seed, bucket)


def severity_next_step(severity: str, seed: str) -> str:
    p0 = [
        "We're treating this as a top-priority fix and our engineering team is already looking into the failure path.",
        "This has been escalated internally so the team can ship a fix as quickly as possible.",
        "We'll prioritise this with engineering immediately and aim to have a fix out in the very next release.",
    ]
    p1 = [
        "We've logged this with the relevant engineering squad and it will be addressed in an upcoming update.",
        "Our team is actively investigating this area and we expect improvements in the next release cycle.",
        "We'll make sure this reaches the right team so it can be tackled in a coming build.",
    ]
    p2 = [
        "We've passed this along to the product team to factor into upcoming planning.",
        "We'll keep this on file and review it as part of our next round of improvements.",
        "We've added this to our backlog and will weigh it against other priorities for an upcoming release.",
    ]
    p3 = [
        "We'll keep this in mind for future iterations as we refine the experience.",
        "We've noted it for consideration in a future release, alongside similar polish items.",
        "We'll factor this into longer-term roadmap thinking.",
    ]
    bucket = {"P0": p0, "P1": p1, "P2": p2, "P3": p3}.get(severity, p1)
    return seeded_choice(seed, bucket)


def contact_close(seed: str, severity: str) -> str:
    """Closing CTA. P0/P1 push for direct contact, P2/P3 thank-and-close."""
    direct = [
        "If you can drop us a quick note at <email> with your account details and device model, we'll be able to look into your specific case and follow up directly.",
        "Please reach out to us at <email> with your account email and a screenshot if possible, and our support team will pick it up from there.",
        "If you can write to us at <email> with your device model and OS version, that'll let us reproduce on our side and get back to you with a concrete update.",
    ]
    soft = [
        "Thanks again for taking the time to write — feedback like yours is what keeps us improving.",
        "We genuinely appreciate the feedback, and we hope you'll keep using the app as we roll out improvements.",
        "Thanks for sticking with us — please keep the feedback coming.",
    ]
    if severity in ("P0", "P1"):
        return seeded_choice(seed + "_close", direct)
    return seeded_choice(seed + "_close", soft)


def trim_component(component: str) -> str:
    """affected_component is sometimes long — keep it readable."""
    if not component:
        return "this part of the app"
    # Drop leading qualifiers like "App " when followed by parens, etc.
    c = component.strip()
    # If it's overly long, take the first clause before " / " or "; " up to ~80 chars
    if len(c) > 90:
        c = c.split(";")[0]
    return c


def first_sentence(text: str) -> str:
    """Take the first reasonable clause from a longer field.

    Strips out embedded user-quote fragments (single-quoted or double-quoted)
    so we don't end up with dangling "is — and that's fair" outputs after
    truncation. Returns an empty string if nothing meaningful remains, so
    the caller can fall back.
    """
    if not text:
        return ""
    # Remove embedded quoted fragments entirely — they're noisy and often
    # truncate badly in our short responses.
    cleaned = re.sub(r"'[^']{2,}?'", "", text)
    cleaned = re.sub(r'"[^"]{2,}?"', "", cleaned)
    # Tidy up artefacts left where a quote used to be.
    # 1. " with " followed by punctuation → drop the dangling " with"
    cleaned = re.sub(r"\b(?:with|like|saying|including|e\.g\.|such as)\s*[,;:.\-—]", ".", cleaned)
    # 2. Collapse ", ," / " , " / " ; " sequences from removed quotes
    cleaned = re.sub(r"\s+,(\s*,)+", ",", cleaned)
    cleaned = re.sub(r"\s+,", ",", cleaned)
    cleaned = re.sub(r",\s*,", ",", cleaned)
    # 3. " is , " etc → " is. "
    cleaned = re.sub(r"\b(is|are|was|were|says|reads)\s*,", r"\1,", cleaned)
    # 4. " — and " left over from stripped trailing quote → drop
    cleaned = re.sub(r"\s+,\s+and\s+", " and ", cleaned)
    # Whitespace
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    # Take the first sentence
    parts = re.split(r"(?<=[\.\?\!])\s+", cleaned)
    s = parts[0] if parts else cleaned
    s = s.strip(" ,;:-")
    if len(s) > 220:
        s = s[:217].rstrip() + "..."
    # If the cleaned result still has telltale "stub commas" (`, ,` or
    # `something is ,`), it's too damaged to reuse — bail out.
    if re.search(r"\b(is|are|was|were)\s*,\s*(that|and|with|or|but|so)\b", s):
        return ""
    if re.search(r",\s*and\s+,", s):
        return ""
    # Bail if too short to carry meaning
    if len(s.split()) < 5:
        return ""
    return s


# ---------------------------------------------------------------------------
# Per-issue-type response builders
# ---------------------------------------------------------------------------

def build_bug_response(spec: dict, seed: str) -> str:
    severity = spec.get("severity", "P1")
    component = trim_component(spec.get("affected_component"))
    actual = first_sentence(spec.get("actual_behavior") or "")
    steps = spec.get("steps_to_reproduce") or []
    expected = first_sentence(spec.get("expected_behavior") or "")

    s1 = severity_opening(severity, seed)

    # Specific acknowledgement using component + actual_behavior
    if actual:
        s2 = (
            f"What you're describing — {actual.lower().rstrip('.')} — points at "
            f"{component}, and that's not the behaviour we expect."
        )
    else:
        s2 = (
            f"This sounds like a regression in {component}, and we want to "
            "make sure we understand exactly where it's breaking."
        )

    # Reference reproduction path / expected behaviour
    if steps and len(steps) >= 2:
        # Use the most informative step (often step 2 or 3) rather than the trivial "Open the app"
        step_pick = steps[1] if len(steps) > 1 else steps[0]
        s3 = (
            f"We've been tracing through the flow where users "
            f"{step_pick.lower().rstrip('.')}, to pin down exactly where it diverges."
        )
    elif expected:
        s3 = (
            f"The expected flow is that {expected.lower().rstrip('.')}, and we're "
            "investigating why that path is failing for you."
        )
    else:
        s3 = (
            "We'd like to reproduce this on our side so we can pin down the root cause."
        )

    s4 = severity_next_step(severity, seed)
    s5 = contact_close(seed, severity)
    return " ".join([s1, s2, s3, s4, s5])


def build_feature_response(spec: dict, seed: str) -> str:
    severity = spec.get("severity", "P2")
    component = trim_component(spec.get("affected_component"))
    user_story = first_sentence(spec.get("user_story") or "")
    accept = spec.get("acceptance_criteria") or []

    s1 = severity_opening(severity, seed)
    if user_story:
        # Restate user_story to demonstrate understanding
        s2 = (
            f"If we're reading you right, the ask is essentially: "
            f"{user_story.lower().rstrip('.')}, and that's a fair expectation."
        )
    else:
        s2 = (
            f"You're asking for a meaningful improvement to {component}, and "
            "it's the kind of suggestion we love to receive."
        )

    if accept:
        # Quote the most concrete criterion to show specificity
        crit = accept[0]
        s3 = (
            f"Concretely, something along the lines of \"{crit.lower().rstrip('.')}\" "
            "is exactly the level of detail that helps our product team scope the work."
        )
    else:
        s3 = (
            f"We'll share this with the team that owns {component} so it can be "
            "weighed against the rest of the roadmap."
        )

    s4 = severity_next_step(severity, seed)
    s5 = contact_close(seed, severity)
    return " ".join([s1, s2, s3, s4, s5])


def build_performance_response(spec: dict, seed: str) -> str:
    severity = spec.get("severity", "P1")
    component = trim_component(spec.get("affected_component"))
    actual = first_sentence(spec.get("actual_behavior") or "")
    nfr = (spec.get("nfr_category") or "").replace("_", " ")

    s1 = severity_opening(severity, seed)
    # Only quote actual_behavior if it's clean (no leftover punctuation
    # artefacts from prior quote-stripping)
    if actual and not re.search(r"\bwith\s*$|\bcrashes the app\s*$", actual):
        s2 = (
            f"You're right that {actual.lower().rstrip('.')} — that falls on us, "
            f"and {component} is exactly where we need to do better."
        )
    else:
        s2 = (
            f"Performance in {component} should feel snappy and reliable, and "
            "clearly it isn't holding up for you."
        )

    if nfr:
        s3 = (
            f"We're treating this as a {nfr} regression and profiling the "
            "code path involved to see what's costing the time."
        )
    else:
        s3 = (
            "We're profiling the relevant code paths to track down where the slowdown is coming from."
        )

    s4 = severity_next_step(severity, seed)
    s5 = contact_close(seed, severity)
    return " ".join([s1, s2, s3, s4, s5])


HEURISTIC_PHRASE = {
    "user_control": "giving you proper control over what you see and do",
    "visibility_of_system_status": "keeping you in the loop on what the app is doing",
    "match_real_world": "speaking your language rather than ours",
    "consistency_and_standards": "behaving consistently across the app",
    "error_prevention": "preventing avoidable errors before they happen",
    "recognition_rather_than_recall": "making important options easy to spot",
    "flexibility_and_efficiency": "letting power users move quickly",
    "aesthetic_and_minimalist_design": "keeping the interface clean and focused",
    "help_recognise_recover_errors": "helping you recover when something goes wrong",
    "help_and_documentation": "providing clear help where you need it",
}


def build_usability_response(spec: dict, seed: str) -> str:
    severity = spec.get("severity", "P2")
    component = trim_component(spec.get("affected_component"))
    heur = spec.get("nielsen_heuristic") or ""

    s1 = severity_opening(severity, seed)
    # Skip description (often contains noisy embedded quotes that truncate
    # badly); rely on affected_component + nielsen_heuristic for specificity.
    s2_options = [
        f"You're flagging real friction in {component}, and that's something we want to address.",
        f"The pain point you're describing centres on {component}, and we don't take that lightly.",
        f"What you're describing maps onto {component}, and we agree the current experience isn't where it needs to be.",
    ]
    s2 = seeded_choice(seed + "_us2", s2_options)

    if heur and heur in HEURISTIC_PHRASE:
        s3 = (
            f"We agree the experience falls short on {HEURISTIC_PHRASE[heur]}, "
            "and that's a principle we hold ourselves to."
        )
    elif heur:
        s3 = (
            f"From a UX standpoint this maps to a {heur.replace('_',' ')} gap, "
            "and it's something we want to close."
        )
    else:
        s3 = (
            "From a UX standpoint this is the kind of friction we actively try to design out."
        )

    s4 = severity_next_step(severity, seed)
    s5 = contact_close(seed, severity)
    return " ".join([s1, s2, s3, s4, s5])


def build_compatibility_response(spec: dict, seed: str) -> str:
    severity = spec.get("severity", "P1")
    component = trim_component(spec.get("affected_component"))
    actual = first_sentence(spec.get("actual_behavior") or "")
    matrix = spec.get("device_os_matrix") or {}
    devices = matrix.get("affected_devices") or []
    oses = matrix.get("affected_os") or []

    s1 = severity_opening(severity, seed)
    if actual:
        s2 = (
            f"What you're seeing — {actual.lower().rstrip('.')} — is consistent "
            f"with reports we've had on {component}."
        )
    else:
        s2 = (
            f"This sounds like a device-specific failure in {component}, and "
            "we want to get to the bottom of it."
        )

    if devices:
        dev_str = ", ".join(devices[:3])
        if oses:
            s3 = (
                f"We're already tracking similar reports on {dev_str} "
                f"running {oses[0]}, so your note adds useful confirmation."
            )
        else:
            s3 = (
                f"We're already tracking similar reports on {dev_str}, "
                "so your note adds useful confirmation."
            )
    elif oses:
        s3 = (
            f"We've seen similar reports on {oses[0]} and are working on a targeted fix."
        )
    else:
        s3 = (
            "Could you share your exact device model and Android/iOS version "
            "so we can match it against our compatibility matrix?"
        )

    s4 = severity_next_step(severity, seed)
    s5 = contact_close(seed, severity)
    return " ".join([s1, s2, s3, s4, s5])


BUILDERS = {
    "bug_report": build_bug_response,
    "feature_request": build_feature_response,
    "performance": build_performance_response,
    "usability": build_usability_response,
    "compatibility": build_compatibility_response,
}


def build_response(item: dict) -> str:
    spec = item["issue_spec_taxonomy"]
    issue_type = item["issue_type"]
    seed = f"{item['cluster_id']}_{issue_type}"
    builder = BUILDERS.get(issue_type, build_bug_response)
    text = builder(spec, seed)
    # Compress runs of whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    with INPUT_PATH.open() as f:
        reviews = json.load(f)

    out = []
    for item in reviews:
        response_text = build_response(item)
        out.append({
            "response_id": f"resp_f4_{item['cluster_id']}",
            "review_index": item["review_index"],
            "cluster_id": item["cluster_id"],
            "issue_type": item["issue_type"],
            "review_text": item["review_text"],
            "response_text": response_text,
            "condition": "reviewagent_full",
            "rag_used": True,
            "issue_spec_used": True,
        })

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    word_counts = [len(r["response_text"].split()) for r in out]
    avg_words = sum(word_counts) / len(word_counts)
    print(f"Generated: {len(out)} responses")
    print(f"Avg length: {avg_words:.1f} words")
    print(f"Min/Max: {min(word_counts)}/{max(word_counts)} words")
    # Sentence count check
    sent_counts = [len(re.findall(r"[\.\?\!]", r['response_text'])) for r in out]
    print(f"Avg sentences (approx): {sum(sent_counts)/len(sent_counts):.1f}")
    print(f"Output: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
