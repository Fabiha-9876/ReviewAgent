# SpecCov Validation Task - README

One file per rater: `speccov_validation_A.xlsx`, `_D.xlsx`, `_E.xlsx`. Fabiha will tell you
which is yours. All three contain the same 60 rows in the same order.

## What you are judging

Each row gives you the source reviews a specification was written from, and the
specification. You judge whether the spec is *grounded* in those reviews. You are not
judging whether the spec is well written, well formatted, or useful.

Fill two columns:

- `faithfulness_1_to_5` - 5 means everything in the spec traces back to the reviews,
  1 means the spec describes something the reviews do not say.
- `unsupported_details` - list anything the spec asserts that the reviews never mention.
  Write `none` if there is nothing. This column is the point of the whole task.

The Instructions sheet inside the file has the full scale and the rules of thumb. Read it
before starting.

## Why we are asking

We built an automatic score that tries to detect ungrounded specifications. We do not yet
know whether it detects what a person would call a hallucination. Your `unsupported_details`
column is the ground truth we compare it against, so guessing hurts more than leaving a row
blank. If you cannot decide, leave the row blank and say why in `notes`.

## Ground rules

- Do not discuss individual rows with the other raters while the round is open.
- Do not re-sort the rows.
- Save the file under its original name and send it back.

About 45 minutes.
