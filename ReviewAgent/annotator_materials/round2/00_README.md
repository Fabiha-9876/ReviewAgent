# Round-2 Verification Task - README

Thanks for helping with this annotation round.

## What's in this folder

- `calibration_set_round2.xlsx` - 20 reviews. Do these FIRST. We compare and
  discuss disagreements before the main task starts, so everyone is aligned.
- `annotator_D.xlsx` or `annotator_E.xlsx` - your assigned main task, 490 reviews.
  Fabiha will tell you which letter is yours.
- This README.

## Task in one paragraph

You will see 490 mobile-app reviews. Each one already carries a predicted label
from our classifier. Read the review and decide whether that label is correct (Y)
or wrong (N). If wrong, write the correct label.

## Steps

1. Open the calibration file, read the Instructions sheet, fill in the 20 rows.
2. Send the calibration file back. After we discuss disagreements you are cleared
   to start the main task.
3. Open your `annotator_X.xlsx` and work through the 490 rows.
4. Estimated time: about 3 hours total. Work in chunks, just save the file.

## Ground rules

- Do NOT look at any other annotator's file, and do not discuss individual rows
  while the round is open. We are measuring inter-annotator agreement, so the
  judgements have to be independent.
- Do not re-sort the rows. The row order matches the earlier annotation sheets so
  the supervisor can cross-check the columns side by side.
- Use the exact lowercase label strings listed in the Instructions sheet.
- If you are unsure, mark Y when the label is reasonable. Mark N only when you are
  confident it is wrong.
- Leave a row blank rather than guessing if you truly cannot decide, and note it in
  `comments`. Blank rows are excluded from the analysis.

## Labels

| label | example |
|---|---|
| bug_report | "App keeps crashing when I open it" |
| feature_request | "Please add a dark mode" |
| performance | "Super slow on my phone, takes forever to load" |
| usability | "Hard to find the settings menu" |
| compatibility | "Doesn't work on Samsung Galaxy S22" |
| praise | "Best app ever, love it!" |
| other | "Hi, just downloaded this" |

## When you are done

Save the file under its original name and send it back. Do not rename it, and do
not save it over anyone else's copy.
