# Stage-4 rating round, LLM versus LLM

You have one file. `stage4_llm_rerun_D.xlsx` or `stage4_llm_rerun_E.xlsx`. Fabiha will
tell you which is yours. Both files contain the same 200 rows in the same order.

## What is different about this round

You may have rated app-review replies for us before. This round is not the same task and
the replies are not the same text. Previously the candidate replies were produced by two
different programs, so one of them read as broken English and the other as polished prose,
and it was easy to tell them apart. That made the comparison unfair.

This time both replies for a review were written by the same language model, with the same
retrieved context. They differ in exactly one thing: one of them was given a structured
issue specification for that review and the other was not. They are similar in length and
in register, roughly 100 words each.

So you cannot judge by polish. Judge by content.

## What to fill in

Three columns per row.

- `quality_1_to_5` - 1 is an unusable reply, 5 is the reply you would want to receive as
  the user who wrote that review.
- `specificity_1_to_5` - 1 is generic boilerplate that could answer any review, 5 names
  the actual problem in this review.
- `helpful_y_n` - Y if this reply would actually move the user's problem forward.

What tends to separate a good reply from a weak one here: does it name the specific
failure, does it identify which part of the app is involved, does it say what happens
next, and does it avoid promising something no team could deliver.

## Ground rules

- Do not discuss individual rows with the other rater while the round is open. We compute
  agreement between you, and that number is meaningless if the sheets are related.
- If you genuinely cannot decide on a row, leave it blank and say why in `notes`. A blank
  row is excluded from the analysis. A guessed row is not, and it does damage.
- Do not re-sort the rows.

About one hour. Save as you go, and send the file back under its original name.
