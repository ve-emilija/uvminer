"""
export_llm_labels_for_kappa.py — Export LLM labels for the 50 kappa commits
Paper: "When RTL Changes, What Breaks?"
Author: Emilija Velinova

This script extracts the LLM-generated labels for exactly the same 50 commits
that were sampled for human annotation (using the same fixed seed=42).
Output is a readable comparison file showing LLM label vs your human label.

Usage:
    python3 export_llm_labels_for_kappa.py \
        --labeled ../data/llm_labeled_final.csv \
        --output  ../data/llm_labels_kappa50.csv

This file is your local proof of what the LLM said for each commit,
BEFORE any comparison with your human labels.
"""

import argparse
import json
import os
import random
import pandas as pd

RANDOM_SEED = 42
SAMPLE_SIZE = 50

GROUND_TRUTH_HASHES = [
    "b058bfe507",
    "b13dc6ff8e",
    "da7fc3571b",
    "57f9776639",
    "6ee0cf0193",
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


def get_sample_hashes(labeled: pd.DataFrame) -> list[str]:
    """
    Reproduce the exact same 50 commits as sample_for_kappa.py.
    Uses identical logic and seed so results are guaranteed to match.
    """
    random.seed(RANDOM_SEED)

    # Filter errors
    labeled = labeled[
        labeled["categories"].apply(lambda x: json.loads(x)[0] != "ERROR")
    ].reset_index(drop=True)

    # Get dominant category
    labeled["dominant_cat"] = labeled["categories"].apply(
        lambda x: json.loads(x)[0]
    )

    # Ground truth always included
    gt_mask = labeled["commit_hash"].apply(
        lambda h: any(h.startswith(g) for g in GROUND_TRUTH_HASHES)
    )
    gt_commits = labeled[gt_mask].copy()
    pool = labeled[~gt_mask].copy()

    remaining = SAMPLE_SIZE - len(gt_commits)

    from collections import Counter
    cat_counts = Counter(pool["dominant_cat"])
    total_pool = len(pool)

    sampled_indices = []
    for cat, count in cat_counts.items():
        n_cat = max(1, round(remaining * count / total_pool))
        cat_pool = pool[pool["dominant_cat"] == cat]
        n_cat = min(n_cat, len(cat_pool))
        sampled = cat_pool.sample(n=n_cat, random_state=RANDOM_SEED)
        sampled_indices.extend(sampled.index.tolist())

    random.seed(RANDOM_SEED)
    if len(sampled_indices) > remaining:
        sampled_indices = random.sample(sampled_indices, remaining)
    elif len(sampled_indices) < remaining:
        leftover = pool[~pool.index.isin(sampled_indices)]
        extra = leftover.sample(
            n=remaining - len(sampled_indices),
            random_state=RANDOM_SEED
        )
        sampled_indices.extend(extra.index.tolist())

    sampled = pool.loc[sampled_indices].copy()
    final = pd.concat([gt_commits, sampled], ignore_index=True)
    final = final.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)

    return final


def main():
    parser = argparse.ArgumentParser(
        description="Export LLM labels for the 50 kappa commits"
    )
    parser.add_argument("--labeled", required=True,
                        help="Path to llm_labeled_final.csv")
    parser.add_argument("--output",  default="../data/llm_labels_kappa50.csv",
                        help="Output CSV file")
    args = parser.parse_args()

    print(f"\nExporting LLM labels for kappa sample")
    print(f"Input  : {args.labeled}")
    print(f"Output : {args.output}")
    print(f"Seed   : {RANDOM_SEED} (matches sample_for_kappa.py)")
    print()

    labeled = pd.read_csv(args.labeled)
    sample  = get_sample_hashes(labeled)

    print(f"Sample size: {len(sample)} commits")
    print()

    # Build output rows
    rows = []
    for i, row in sample.iterrows():
        h     = str(row["commit_hash"])
        is_gt = any(h.startswith(g) for g in GROUND_TRUTH_HASHES)
        cats  = json.loads(row["categories"])
        conf  = row["confidence"]
        justif = str(row.get("justification", ""))[:120]

        rows.append({
            "commit_number": len(rows) + 1,
            "commit_hash":   h,
            "ip_block":      row["ip_block"],
            "commit_message": str(row["commit_message"])[:80],
            "is_ground_truth": "YES" if is_gt else "NO",
            "llm_categories": json.dumps(cats),
            "llm_confidence": conf,
            "llm_justification": justif,
        })

    df_out = pd.DataFrame(rows)

    os.makedirs(
        os.path.dirname(args.output) if os.path.dirname(args.output) else ".",
        exist_ok=True
    )
    df_out.to_csv(args.output, index=False)

    # Print summary
    print(f"{'#':<4} {'Hash':<12} {'GT':<4} {'LLM Categories':<55} {'Conf'}")
    print("-" * 85)
    for _, r in df_out.iterrows():
        cats = json.loads(r["llm_categories"])
        print(f"{int(r['commit_number']):<4} {r['commit_hash'][:10]:<12} "
              f"{r['is_ground_truth']:<4} {str(cats):<55} {r['llm_confidence']}")

    print()
    print(f"Saved: {args.output}")
    print()
    print("This file contains what the LLM classified for each of your 50 commits.")
    print("Compare against your human_labels.csv to see agreements and disagreements.")
    print()
    print("To compute kappa run:")
    print("  python3 llm_label.py \\")
    print("    --input  ../data/commits_final.csv \\")
    print("    --output ../data/llm_labeled_final.csv \\")
    print("    --kappa-human ../data/human_labels.csv")


if __name__ == "__main__":
    main()