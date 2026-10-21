"""
sample_for_kappa.py — Sample 50 commits for human annotation (Cohen's kappa)
Paper: "When RTL Changes, What Breaks?"
Author: Emilija Velinova

How commits are selected:
    - Sample from the raw commit pool, never from either approach's predictions
    - Balance the sample across repositories, then sample randomly within each
    - Fixed random seed (42) for reproducibility
    - Output paired design and verification diffs for blind human annotation

Usage:
    python data_labeling/sample_for_kappa.py \
        --commits data/commits_final.csv \
        --source opentitan \
        --output data/kappa_sample.txt \
        --human data/human_labels.csv

Output files:
    kappa_sample.txt  — readable file for manual labeling (you read this)
    human_labels.csv  — protocol-versioned template CSV for adjudicated labels
    kappa_commits.csv — exact raw commit sample shared by all labeling methods
"""

import argparse
import os
import random
import pandas as pd
from collections import Counter
from uvminer.data_extraction.label_schema import CATEGORY_CODES, PROTOCOL_VERSION

# Fixed seed — guarantees same 50 commits every time script is run
RANDOM_SEED = 42
SAMPLE_SIZE = 50

CATEGORIES = list(CATEGORY_CODES)


def load_data(commits_path: str, source_repo: str | None = None) -> pd.DataFrame:
    path = os.fspath(commits_path)
    if os.path.isdir(path):
        input_paths = sorted(
            os.path.join(path, name)
            for name in os.listdir(path)
            if name.lower().endswith(".csv")
        )
        if not input_paths:
            raise ValueError(f"No commit CSV files found in directory: {path}")
        commits = pd.concat(
            [pd.read_csv(input_path) for input_path in input_paths],
            ignore_index=True,
        )
    else:
        commits = pd.read_csv(path)
    required = {"commit_hash", "commit_message"}
    missing = required - set(commits.columns)
    if missing:
        raise ValueError(f"Commit CSV missing columns: {missing}")
    if "design_diff" not in commits and "rtl_diff" in commits:
        commits["design_diff"] = commits["rtl_diff"]
    if "verification_diff" not in commits and "dv_diff" in commits:
        commits["verification_diff"] = commits["dv_diff"]
    if "design_diff" not in commits or "verification_diff" not in commits:
        raise ValueError(
            "Commit CSV must include design_diff and verification_diff "
            "(or legacy rtl_diff and dv_diff)."
        )
    if "source_repo" not in commits:
        commits["source_repo"] = ""
    commits["source_repo"] = commits["source_repo"].fillna("").astype(str).str.strip()
    if source_repo:
        commits.loc[commits["source_repo"] == "", "source_repo"] = source_repo
    if (commits["source_repo"] == "").any():
        raise ValueError(
            "Set --source or provide a non-empty source_repo column in the commit CSV."
        )
    return commits


def stratified_sample(commits: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    """
    Select a deterministic, approximately equal sample per source repository.
    """
    if n <= 0:
        raise ValueError("Sample size must be positive.")
    rng = random.Random(seed)
    groups = {
        source: group.index.tolist()
        for source, group in commits.groupby("source_repo", sort=True)
    }
    for indices in groups.values():
        rng.shuffle(indices)

    selected = []
    target = min(n, len(commits))
    while len(selected) < target:
        added = False
        for source in groups:
            if groups[source]:
                selected.append(groups[source].pop())
                added = True
                if len(selected) == target:
                    break
        if not added:
            break

    sample = commits.loc[selected].sample(frac=1, random_state=seed).reset_index(drop=True)
    print(f"Selected {len(sample)} of {len(commits)} raw commits")
    for source, count in Counter(sample["source_repo"]).most_common():
        print(f"  {source:<24} {count}")
    return sample


def get_diff(
    commits_df: pd.DataFrame,
    commit_hash: str,
    source_repo: str,
) -> tuple[str, str]:
    """Get paired design and verification diffs for a source commit."""
    row = commits_df[
        (commits_df["commit_hash"].astype(str) == commit_hash)
        & (commits_df["source_repo"] == source_repo)
    ]
    if row.empty:
        return "DIFF NOT FOUND", "DIFF NOT FOUND"
    design = str(row.iloc[0].get("design_diff", ""))
    verification = str(row.iloc[0].get("verification_diff", ""))
    return design[:3000], verification[:3000]  # truncate for readability


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
    lines.append("  2. Compare the design and verification diffs")
    lines.append("  3. Apply LABELING_PROTOCOL.md; cite paths and diff evidence")
    lines.append("  4. Fill label_status, categories, evidence, and justification")
    lines.append("")
    lines.append("CATEGORIES:")
    for cat in CATEGORIES:
        lines.append(f"  {cat}")
    lines.append("")
    lines.append("LABEL STATUSES: CLASSIFIED, OTHER, NOT_APPLICABLE, INSUFFICIENT_EVIDENCE")
    lines.append("RULES:")
    lines.append("  - Assign ALL supported categories (multi-label)")
    lines.append("  - Consider both diffs; co-change alone is not evidence")
    lines.append("  - OTHER means a concrete impact outside the six categories")
    lines.append("  - Do not use model labels as ground truth")
    lines.append("  - Do NOT look at human_labels.csv of others first")
    lines.append("=" * 72)
    lines.append("")

    for i, row in sample.iterrows():
        h     = str(row["commit_hash"])
        block = str(row.get("ip_block", ""))
        source = str(row.get("source_repo", "opentitan"))
        msg   = str(row["commit_message"])
        rtl_diff, dv_diff = get_diff(commits_df, h, source)

        lines.append(f"{'=' * 72}")
        lines.append(f"COMMIT {i+1:02d}/50")
        lines.append(f"Hash     : {h}")
        lines.append(f"Source   : {source}")
        lines.append(f"IP Block : {block}")
        lines.append(f"Message  : {msg}")
        lines.append(f"{'=' * 72}")
        lines.append("")
        lines.append("--- DESIGN DIFF (what changed in the design) ---")
        lines.append(rtl_diff[:2000])
        if len(rtl_diff) > 2000:
            lines.append("[... truncated ...]")
        lines.append("")
        lines.append("--- VERIFICATION DIFF (what changed in verification) ---")
        lines.append(dv_diff[:2000])
        if len(dv_diff) > 2000:
            lines.append("[... truncated ...]")
        lines.append("")
        lines.append("YOUR VERDICT:")
        lines.append("  Label status : ______________________________")
        lines.append("  Categories   : ______________________________")
        lines.append("  Evidence     : ______________________________")
        lines.append("  Justification: ______________________________")
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
            "protocol_version": PROTOCOL_VERSION,
            "source_repo":    str(row.get("source_repo", "opentitan")),
            "commit_number": i + 1,
            "commit_hash":   str(row["commit_hash"]),
            "ip_block":      str(row.get("ip_block", "")),
            "commit_message": str(row["commit_message"])[:80],
            "label_status":  "",
            "categories":    "",
            "evidence":      "",
            "justification": "",
            "annotator":     "",
        })

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    print(f"Human labels template written: {output_path}")
    print(f"Fill in the 'categories' column as JSON arrays, e.g.:")
    print(f'  ["INTERFACE_DRIFT"]')
    print(f'  ["CHECKER_DESYNC", "SEQUENCE_INVALIDATION"]')
    print("Fill label_status, evidence, and justification per LABELING_PROTOCOL.md.")


def main():
    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(project_dir, "data")
    parser = argparse.ArgumentParser(
        description="Sample 50 commits for human annotation (Cohen's kappa)"
    )
    parser.add_argument("--commits",  required=True,
                        help="Raw commit CSV with paired diffs; must not contain model-selected sampling")
    parser.add_argument("--source", default=None,
                        help="Repository identifier if the CSV has no source_repo column")
    parser.add_argument("--sample-size", type=int, default=SAMPLE_SIZE)
    parser.add_argument("--sampled-csv", default=os.path.join(data_dir, "kappa_commits.csv"),
                        help="Machine-readable raw sample shared by each approach")
    parser.add_argument("--output",   default=os.path.join(data_dir, "kappa_sample.txt"),
                        help="Output readable text file")
    parser.add_argument("--human",    default=os.path.join(data_dir, "human_labels.csv"),
                        help="Output CSV template for your labels")
    parser.add_argument("--seed",     type=int, default=RANDOM_SEED,
                        help=f"Random seed (default: {RANDOM_SEED})")
    args = parser.parse_args()

    print(f"\nKappa Sample Generator")
    print(f"Commits CSV : {args.commits}")
    print(f"Random seed : {args.seed} (fixed for reproducibility)")
    print(f"Sample size : {args.sample_size}")
    print()

    commits = load_data(args.commits, args.source)
    sample = stratified_sample(commits, args.sample_size, args.seed)

    os.makedirs(os.path.dirname(args.sampled_csv) or ".", exist_ok=True)
    sample.to_csv(args.sampled_csv, index=False)
    write_review_file(sample, commits, args.output)
    write_human_labels_template(sample, args.human)
    print(f"Raw sample CSV written: {args.sampled_csv}")

    print()
    print("=" * 50)
    print("Annotate the human template independently before comparing either model.")
    print("=" * 50)


if __name__ == "__main__":
    main()