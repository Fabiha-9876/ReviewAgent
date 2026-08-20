# Volunteer Verification Task — README

Hi! Thanks for helping with this annotation task.

## What's in this folder

- `calibration_set.xlsx` — 20 reviews. Do these FIRST. We'll discuss disagreements
  before moving on, so we're all aligned on the categories.
- `annotator_X.xlsx` — your assigned main task. The 'X' will be A, B, or C — Fabiha
  will tell you which one is yours.
- This README.

## Task in one paragraph

You'll see ~500 mobile-app reviews. Each one already has a **predicted label**
from our AI classifier. Your job is to read each review and decide whether the
AI's label is correct (Y) or wrong (N). If wrong, write what the correct label
should be.

## Steps

1. Open `calibration_set.xlsx`. Read the "Instructions" sheet first.
2. Fill in the 20 calibration reviews (Y/N for each).
3. Send the calibration sheet back to Fabiha. After we discuss any
   disagreements, you'll be cleared to start the main task.
4. Open your `annotator_X.xlsx`. Same format, ~500 reviews.
5. Estimated time: ~3 hours total. You can do it in chunks — just save the file.

## Important rules

- Do **not** look at what other annotators are filling in. We're measuring
  inter-annotator agreement, so independent judgments are essential.
- Don't worry about being "right" — we adjudicate disagreements at the end.
- For ambiguous reviews, mark Y if the label is reasonable; only mark N if
  you're confident it's wrong.

## Categories (cheat sheet)

| label | example |
|---|---|
| bug_report | "App keeps crashing when I open it" |
| feature_request | "Please add a dark mode" |
| performance | "Super slow on my phone, takes forever to load" |
| usability | "Hard to find the settings menu" |
| compatibility | "Doesn't work on Samsung Galaxy S22" |
| praise | "Best app ever, love it!" |
| other | "Hi, just downloaded this" |

## Decision rules (read these — they prevent the most common mistakes)

1. **slow / lag** → `performance`, NOT bug_report. (App is degraded, not broken.)
2. **crash on my Samsung** → `compatibility` (device-specific). **crash** alone → `bug_report`.
3. **"would be nice if X"** → `feature_request`, even if it sounds like a complaint.
4. **"hard to find the X button"** → `usability`, not bug_report.
5. **Multi-aspect reviews** → pick the *primary* category. If genuinely two-headed,
   mark Y if the AI picked one of them.
6. **Spam / non-English / nonsense** → `other`.

## Compensation

Per the project agreement, you'll receive co-authorship or acknowledgment on
the resulting paper, depending on contribution scale. Fabiha will discuss
specifics.

## Contact

Questions? Email Fabiha at fabihajalal@iut-dhaka.edu.

## Once you're done

Save the file (no need to rename) and email it back to Fabiha.

Thank you!
