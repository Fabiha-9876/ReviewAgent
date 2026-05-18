# Help with a Research Paper, ~1 to 1.5 Hours of Your Time

Hi, thanks for taking the time to look at this. I'm submitting a paper to CIKM 2026 (Conference on Information and Knowledge Management) and the methodology requires a rater who is independent of the system I built. You're it.

There are **two short tasks**, you can do either or both. Each is fully self-contained.

---

## Task 1, Pairwise rating (~20–30 min, 30 pairs)

Open a terminal, `cd` into this folder, and run:

```bash
python3 rate.py
```

You'll be shown 30 user app-store reviews. For each review, two candidate developer responses are shown as A and B (in random order, you don't know which generator produced which). For each pair you'll be asked five quick questions:

1. Response A quality (1–5)
2. Response A helpful? (Y/N)
3. Response B quality (1–5)
4. Response B helpful? (Y/N)
5. Which is better overall? (A / B / EQUAL)

**Quality scale:** 1 = generic / off-topic, 2 = vague, 3 = basic, 4 = specific & empathetic, 5 = excellent.

**Tip:** read the review first, then both responses, then rate each on its own merits relative to the review. Don't anchor A's score against B's. The pairwise preference at the end is where the comparison happens.

Progress saves after every rating. You can quit with `Ctrl+C` and resume by rerunning the script. When you're done, send me the `ratings.json` file that the script creates in this folder.

---

## Task 2, Cluster curation (~40–60 min, 20 clusters)

Open `curation_rubric.md` first, it's a one-page rubric explaining what to do.

Then open `curation_sheet.csv` in any spreadsheet (Excel, Google Sheets, Numbers, whatever you have). Each of the 20 rows is one cluster of 5 user reviews with an auto-generated label.

For each cluster, fill in the four empty columns:

- `operation_KEEP_RENAME_MERGE_SPLIT`, pick one (rubric explains)
- `new_name_if_RENAME`, only if you picked RENAME
- `merge_target_cluster_id_if_MERGE`, only if you picked MERGE
- `split_note_if_SPLIT`, only if you picked SPLIT

When done, send me the filled-in CSV.

---

## What this is being used for

I'm comparing your judgments to mine to compute inter-rater agreement (Cohen's kappa). The numbers go straight into the paper as a methodology defense, "we don't only have the lead author's ratings, here's an independent rater agreeing/disagreeing on a sample." The honest answer matters more than agreement; if you disagree with my labels, that's data, not a problem.

Please **don't** discuss with me until you've completed whichever task(s) you're doing. After that, happy to chat about anything you noticed.

Thanks!
