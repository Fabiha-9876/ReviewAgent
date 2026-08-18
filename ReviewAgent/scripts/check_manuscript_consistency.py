"""
Numeric consistency sweep over the manuscript.

Why this exists
---------------
This paper was revised many times, including one withdrawal that touched a dozen sections.
Each revision pass fixed the section it was aimed at and left the same quantity written a
different way somewhere else: the pooled Stage-4 gain appeared as +0.03 and +0.04, the
per-rater gains appeared in both adjusted and unadjusted form in the same table, and the
ablation count was given as six, seven and eight. Reviewers found all of it before we did.

This script pins one canonical value per quantity and fails if any other rendering appears.
Run it before every build.

Usage
    python3 scripts/check_manuscript_consistency.py paper/IssueSpec_IST/main_ist.tex
"""

import re
import sys
from pathlib import Path

# quantity -> (pattern that finds every rendering, the set of renderings that are allowed,
#              contexts in which a non-canonical value is legitimate)
CANON = {
    "pooled Stage-4 gain": (r"[+]0\.0[34]", {"+0.04"},
                            ["slot-stratified", "[-0.33,", "[-0.10,"]),
    "rater A gain": (r"[+]0\.5[36]", {"+0.56"}, []),
    "rater D gain": (r"[-−]0\.2[69]", {"-0.29"}, []),
    "rater E gain": (r"[-−]0\.1[58](?![0-9])", {"-0.15"}, ["[-0.33"]),
    "kappa progression": (r"0\.16[0-9]|0\.33[0-9]|0\.59[0-9]",
                          {"0.163", "0.333", "0.592", "0.165", "0.334", "0.590"}, []),
    "held-out kappa": (r"0\.616", {"0.616"}, []),
    "template fill, routed": (r"0\.96(?![0-9])", {"0.96"}, []),
    "rubric, full set": (r"3\.9[34]", {"3.94", "3.93"}, []),
    "SpecCov unfloored": (r"4\.19|3\.38|4\.47|3\.45", {"4.19", "3.38", "4.47", "3.45"}, []),
    "lambda max": (r"0\.65(?![0-9])", {"0.65"}, []),
    # 0.1369 / 0.0896 - 1 = 52.79%, so +53%. The old "+52%" came from dividing the
    # rounded table values (0.137 / 0.090) and must not come back.
    "constrained-proxy BLEU-1 gain": (r"\+5[23]\\%", {"+53\\%"}, []),
    # The 0.451 was produced by a Krippendorff implementation that credited each rating
    # with a coincidence against itself. Correct value 0.285, matching the reference
    # package. It may appear only in the sentence that documents the correction.
    "LLM-panel Krippendorff alpha": (r"\\alpha = 0\.(?:285|451)", {"\\alpha = 0.285"},
                                    ["mis-implemented", "We reported this row"]),
    # Table 12 mean cluster sizes, recomputed from per-cluster review counts.
    "mean cluster sizes": (r"471\.0|15\.9", {"471.0", "15.9"}, []),
}

# strings that belonged to a withdrawn claim and must never reappear
STALE = [
    "+2.35 Likert points for",
    "justifies the IR",
    "significantly below a no-context",
    "the headline gain reproduces individually",
    "raw concatenation scores 5.00",
    "is what carries downstream response quality",
    "The intermediate carries the work",
]


def main():
    path = Path(sys.argv[1] if len(sys.argv) > 1
                else "paper/IssueSpec_IST/main_ist.tex")
    tex = path.read_text()
    lines = tex.splitlines()
    problems = 0

    print(f"checking {path}\n")
    for name, (pattern, allowed, exempt_contexts) in CANON.items():
        offenders = []
        for i, line in enumerate(lines, 1):
            for m in re.finditer(pattern, line):
                val = m.group(0).replace("−", "-")
                if val in allowed:
                    continue
                if any(ctx in line for ctx in exempt_contexts):
                    continue
                offenders.append((i, val, line.strip()[:70]))
        if offenders:
            problems += len(offenders)
            print(f"  [FIX] {name}")
            for i, val, snippet in offenders[:4]:
                print(f"        line {i}: {val}   {snippet}")
        else:
            print(f"  [OK ] {name}")

    print()
    for s in STALE:
        hits = [i for i, line in enumerate(lines, 1) if s in line]
        if hits:
            problems += len(hits)
            print(f"  [FIX] withdrawn claim reappeared at line(s) {hits}: {s!r}")
    if not any(s in tex for s in STALE):
        print("  [OK ] no withdrawn claim reappears")

    print()
    print("CONSISTENT" if not problems else f"{problems} issue(s) to fix")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
