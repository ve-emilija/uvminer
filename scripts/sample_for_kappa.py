"""
sample_for_kappa.py — Sample 50 commits for human annotation (Cohen's kappa)
Paper: "When RTL Changes, What Breaks?"
Author: Emilija Velinova

How the 50 commits are selected:
  - Stratified random sampling from llm_labeled_final.csv
  - Proportional to category distribution (so rare categories are represented)
  - Fixed random seed (42) for full reproducibility
  - Ground truth commits always included (they are your anchor points)
  - Output is a readable text file showing commit message + RTL diff + DV diff
    so you can label without seeing the LLM's answer

Usage:
    python3 sample_for_kappa.py \
        --labeled ../data/llm_labeled_final.csv \
        --commits ../data/commits_final.csv \
        --output  ../data/kappa_sample.txt \
        --human   ../data/human_labels.csv

Output files:
    kappa_sample.txt  — readable file for manual labeling (you read this)
    human_labels.csv  — template CSV you fill in with your labels
"""

import argparse
import json
import os
import random
import pandas as pd
from collections import Counter

# Fixed seed — guarantees same 50 commits every time script is run
RANDOM_SEED = 42
SAMPLE_SIZE = 50

# Ground truth commits — always included in the 50
GROUND_TRUTH_HASHES = [
    "b058bfe507",  # INTERFACE_DRIFT
    "b13dc6ff8e",  # ASSERTION_ORPHANING
    "da7fc3571b",  # COVERAGE_REGRESSION
    "57f9776639",  # CHECKER_DESYNC
    "6ee0cf0193",  # SEQUENCE_INVALIDATION
]

CATEGORIES = [
    "INTERFACE_DRIFT",
    "SEQUENCE_INVALIDATION",
    "CHECKER_DESYNC",
    "ASSERTION_ORPHANING",
    "COVERAGE_REGRESSION",
    "RAL_DESYNC",
    "OTHER",
]


def load_data(labeled_path: str, commits_path: str):
    labeled = pd.read_csv(labeled_path)
    commits = pd.read_csv(commits_path)

    # Filter out ERROR rows
    labeled = labeled[
        labeled["categories"].apply(lambda x: json.loads(x)[0] != "ERROR")
    ].reset_index(drop=True)

    print(f"Loaded {len(labeled)} successfully labeled commits")
    return labeled, commits


def stratified_sample(labeled: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    """
    Stratified random sample proportional to dominant category distribution.
    Ground truth commits are always included.
    """
    random.seed(seed)

    # Get dominant category per commit (first in the list)
    labeled["dominant_cat"] = labeled["categories"].apply(
        lambda x: json.loads(x)[0]
    )

    # Always include ground truth commits
    gt_mask = labeled["commit_hash"].apply(
        lambda h: any(h.startswith(g) for g in GROUND_TRUTH_HASHES)
    )
    gt_commits = labeled[gt_mask].copy()
    gt_hashes  = set(gt_commits["commit_hash"].tolist())

    print(f"Ground truth commits found: {len(gt_commits)}/5")

    # Remaining pool (exclude ground truth)
    pool = labeled[~gt_mask].copy()

    # Calculate how many more we need
    remaining = n - len(gt_commits)

    # Stratified sampling from pool
    cat_counts = Counter(pool["dominant_cat"])
    total_pool = len(pool)

    sampled_indices = []
    for cat, count in cat_counts.items():
        # Proportional allocation
        n_cat = max(1, round(remaining * count / total_pool))
        cat_pool = pool[pool["dominant_cat"] == cat]
        n_cat = min(n_cat, len(cat_pool))
        sampled = cat_pool.sample(n=n_cat, random_state=seed)
        sampled_indices.extend(sampled.index.tolist())

    # If we have too many or too few, adjust
    random.seed(seed)
    if len(sampled_indices) > remaining:
        sampled_indices = random.sample(sampled_indices, remaining)
    elif len(sampled_indices) < remaining:
        leftover_pool = pool[~pool.index.isin(sampled_indices)]
        extra = leftover_pool.sample(
            n=remaining - len(sampled_indices),
            random_state=seed
        )
        sampled_indices.extend(extra.index.tolist())

    sampled = pool.loc[sampled_indices].copy()
    final   = pd.concat([gt_commits, sampled], ignore_index=True)

    # Shuffle so ground truth commits are not always first
    final = final.sample(frac=1, random_state=seed).reset_index(drop=True)

    print(f"Final sample: {len(final)} commits")
    print(f"Category breakdown in sample:")
    for cat, count in Counter(final["dominant_cat"]).most_common():
        print(f"  {cat:<30} {count}")

    return final


def get_diff(commits_df: pd.DataFrame, commit_hash: str) -> tuple[str, str]:
    """Get RTL and DV diffs from the commits CSV."""
    row = commits_df[
        commits_df["commit_hash"].astype(str).str.startswith(commit_hash[:10])
    ]
    if row.empty:
        return "DIFF NOT FOUND", "DIFF NOT FOUND"
    rtl = str(row.iloc[0].get("rtl_diff", ""))
    dv  = str(row.iloc[0].get("dv_diff", ""))
    return rtl[:3000], dv[:3000]  # truncate for readability


def write_review_file(sample: pd.DataFrame, commits_df: pd.DataFrame, output_path: str):
    """Write the human-readable labeling file."""
    lines = []
    lines.append("=" * 72)
    lines.append("HUMAN ANNOTATION FILE — 50 Commits for Cohen's Kappa")
    lines.append("Paper: When RTL Changes, What Breaks?")
    lines.append("Author: Emilija Velinova")
    lines.append("")
    lines.append("INSTRUCTIONS:")
    lines.append("  For each commit below:")
    lines.append("  1. Read the commit message")
    lines.append("  2. Read the RTL diff (what changed in the design)")
    lines.append("  3. Read the DV diff (what changed in the testbench)")
    lines.append("  4. Fill in your verdict in human_labels.csv")
    lines.append("")
    lines.append("CATEGORIES:")
    for cat in CATEGORIES:
        lines.append(f"  {cat}")
    lines.append("")
    lines.append("RULES:")
    lines.append("  - Assign ALL categories that apply (multi-label)")
    lines.append("  - Base decision on the DV diff, not the RTL diff")
    lines.append("  - RTL diff explains WHY; DV diff shows WHAT broke")
    lines.append("  - Do NOT look at human_labels.csv of others first")
    lines.append("  - If genuinely unclear, assign OTHER")
    lines.append("=" * 72)
    lines.append("")

    for i, row in sample.iterrows():
        h     = str(row["commit_hash"])
        block = str(row["ip_block"])
        msg   = str(row["commit_message"])
        is_gt = any(h.startswith(g) for g in GROUND_TRUTH_HASHES)

        rtl_diff, dv_diff = get_diff(commits_df, h)

        lines.append(f"{'=' * 72}")
        lines.append(f"COMMIT {i+1:02d}/50")
        if is_gt:
            lines.append(f"[GROUND TRUTH COMMIT — used for validation]")
        lines.append(f"Hash     : {h}")
        lines.append(f"IP Block : {block}")
        lines.append(f"Message  : {msg}")
        lines.append(f"{'=' * 72}")
        lines.append("")
        lines.append("--- RTL DIFF (what changed in the design) ---")
        lines.append(rtl_diff[:2000])
        if len(rtl_diff) > 2000:
            lines.append("[... truncated ...]")
        lines.append("")
        lines.append("--- DV DIFF (what changed in the testbench) ---")
        lines.append(dv_diff[:2000])
        if len(dv_diff) > 2000:
            lines.append("[... truncated ...]")
        lines.append("")
        lines.append("YOUR VERDICT:")
        lines.append("  Categories : ________________________________")
        lines.append("  Notes      : ________________________________")
        lines.append("")
        lines.append("")

    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"Review file written: {output_path}")


def write_human_labels_template(sample: pd.DataFrame, output_path: str):
    """Write the CSV template that Emili fills in with her labels."""
    rows = []
    for i, row in sample.iterrows():
        rows.append({
            "commit_number": i + 1,
            "commit_hash":   str(row["commit_hash"]),
            "ip_block":      str(row["ip_block"]),
            "commit_message": str(row["commit_message"])[:80],
            "categories":    ""  # ← Emili fills this in
        })

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    print(f"Human labels template written: {output_path}")
    print(f"Fill in the 'categories' column as JSON arrays, e.g.:")
    print(f'  ["INTERFACE_DRIFT"]')
    print(f'  ["CHECKER_DESYNC", "SEQUENCE_INVALIDATION"]')


def main():
    parser = argparse.ArgumentParser(
        description="Sample 50 commits for human annotation (Cohen's kappa)"
    )
    parser.add_argument("--labeled",  required=True,
                        help="Path to llm_labeled_final.csv")
    parser.add_argument("--commits",  required=True,
                        help="Path to commits_final.csv (for diffs)")
    parser.add_argument("--output",   default="../data/kappa_sample.txt",
                        help="Output readable text file")
    parser.add_argument("--human",    default="../data/human_labels.csv",
                        help="Output CSV template for your labels")
    parser.add_argument("--seed",     type=int, default=RANDOM_SEED,
                        help=f"Random seed (default: {RANDOM_SEED})")
    args = parser.parse_args()

    print(f"\nKappa Sample Generator")
    print(f"Labeled CSV : {args.labeled}")
    print(f"Commits CSV : {args.commits}")
    print(f"Random seed : {args.seed} (fixed for reproducibility)")
    print(f"Sample size : {SAMPLE_SIZE}")
    print()

    labeled, commits = load_data(args.labeled, args.commits)
    sample = stratified_sample(labeled, SAMPLE_SIZE, args.seed)

    write_review_file(sample, commits, args.output)
    write_human_labels_template(sample, args.human)

    print()
    print("=" * 50)
    print("NEXT STEPS:")
    print(f"  1. Open {args.output}")
    print(f"  2. Read each commit and fill in your labels")
    print(f"  3. Save your labels in {args.human}")
    print(f"  4. Run kappa computation:")
    print(f"     python3 llm_label.py \\")
    print(f"       --input ../data/commits_final.csv \\")
    print(f"       --output ../data/llm_labeled_final.csv \\")
    print(f"       --kappa-human {args.human}")
    print("=" * 50)


if __name__ == "__main__":
    main()