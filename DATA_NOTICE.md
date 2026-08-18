# Data notice

This repository contains original code and derived research artifacts. It does **not**
redistribute the third-party review corpora the work is built on, because we do not hold the
right to redistribute them. An earlier version of this repository did, and we removed them.

## What was removed, and why

| File | Reason |
|---|---|
| `RRGen_Full_Dataset.csv`, `RRGen_Full_Annotator_Review.xlsx`, `RRGen_LLM_Verification_59K.xlsx` | The RRGen release (github.com/armor-ai/RRGen) carries no licence, is access-gated behind a request form, and states academic use only. It grants no redistribution right. |
| `data/raw/maalej/` | Redistributed without a permission basis, and the files contain unredacted personal data: email addresses, phone numbers, and personal names of identifiable review authors. |
| `data/raw/guzman/` | The accompanying MIT notice was not travelling with the data, and each record carries a persistent Google Play review identifier that re-links to a named public reviewer. |

The underlying content is user-generated Google Play reviews. Copyright in each review rests with
its author, and the Google Play Terms of Service prohibit redistribution of Play content and
harvesting of user data.

## How to obtain the data

- **RRGen**: request it from the authors at https://github.com/armor-ai/RRGen (request form).
- **Maalej and Nabil**: request from the original authors. If you obtain it, note that the raw
  file contains personal data and should not be re-published.
- **Guzman / Dabrowski**: available from Dabrowski's replication package under MIT. Keep the
  `LICENSE.txt` with the data if you redistribute it.

Place each under `ReviewAgent/data/raw/<name>/` and the pipeline scripts will run unchanged.

## What this repository does contain

Original code, model checkpoints, annotation instruments, and derived result files. Derived
artifacts that embed verbatim third-party review text carry the same restriction as the source
and are not offered under any open licence. Code is licensed separately; see `LICENSE`.

## Residual identifiers we did not remove

Where we release derived files containing review text, the text arrives from RRGen already
lowercased and lemmatised with digits masked. That masking is imperfect. We are aware of a small
number of surviving transaction identifiers (for example ride-booking CRNs) in the source corpus.
Anyone re-releasing derived text should re-scan for these.
