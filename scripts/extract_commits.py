"""
extract_commits.py — Extract RTL+DV co-modification commits from OpenTitan
Outputs a CSV ready for llm_label.py

Paper: "When RTL Changes, What Breaks?"
Section 3.2 inclusion/exclusion criteria applied.

Usage:
    python3 extract_commits.py --repo ../opentitan --output data/commits.csv
    python3 extract_commits.py --repo ../opentitan --output data/commits.csv --stats
"""

import subprocess
import csv
import os
import argparse
from datetime import datetime

# ---------------------------------------------------------------------------
# Configuration — matches Section 3.2 exactly
# ---------------------------------------------------------------------------

DATE_FROM = "2021-01-01"
DATE_TO = "2026-04-30"

# Exclusion patterns — Section 3.2
EXCLUDE_DV_PATTERNS = [
    ".md",          # documentation only
]
EXCLUDE_PATH_PATTERNS = [
    "autogen",      # auto-generated files
    ".hjson",       # register description (auto-generates RAL)
]

DV_EXTENSIONS = {
    ".sv",   # SystemVerilog UVM components
    ".svh",  # SystemVerilog headers
    ".cc",   # C++ reference models and DPI checkers
    ".h",    # C/C++ headers for DV components
    ".py",   # Python ISS and test scripts (OTBN reference model)
    ".s",    # Assembly test sequences (OTBN)
}
MAX_DIFF_LINES = 200   # per side (RTL / DV) — keeps CSV manageable


# ---------------------------------------------------------------------------
# Git helpers
# ---------------------------------------------------------------------------

def git(cmd: list, repo: str) -> str:
    result = subprocess.run(
        ["git"] + cmd,
        cwd=repo,
        capture_output=True,
        text=True,
        timeout=120,
    )
    return result.stdout


def get_commit_list(repo: str) -> list[dict]:
    """
    Get all non-merge commits in the date range.
    Returns list of {hash, message, author_date}
    """
    log = git([
        "log",
        f"--after={DATE_FROM}",
        f"--before={DATE_TO}",
        "--no-merges",                  # exclude merge commits — Section 3.2
        "--pretty=format:%H\t%s\t%ai",  # hash TAB subject TAB date
        "master",
    ], repo)

    commits = []
    for line in log.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split("\t", 2)
        if len(parts) < 2:
            continue
        commits.append({
            "hash":    parts[0],
            "message": parts[1] if len(parts) > 1 else "",
            "date":    parts[2] if len(parts) > 2 else "",
        })
    return commits


def get_changed_files(repo: str, commit_hash: str) -> list[str]:
    """Return list of files changed in this commit."""
    out = git(["diff-tree", "--no-commit-id", "-r",
               "--name-only", commit_hash], repo)
    return [f.strip() for f in out.splitlines() if f.strip()]


def get_diff(repo: str, commit_hash: str, path_prefix: str) -> str:
    """Get diff for a specific path prefix, truncated to MAX_DIFF_LINES."""
    out = git(["show", commit_hash, "--",
               path_prefix], repo)
    lines = out.splitlines()
    if len(lines) <= MAX_DIFF_LINES:
        return "\n".join(lines)
    kept = lines[:MAX_DIFF_LINES]
    kept.append(f"... [{len(lines) - MAX_DIFF_LINES} lines truncated]")
    return "\n".join(kept)


# ---------------------------------------------------------------------------
# Filtering helpers — Section 3.2
# ---------------------------------------------------------------------------

def extract_ip_block(filepath: str) -> str | None:
    """
    Extract IP block name from path like hw/ip/<block>/rtl/...
    Returns None if path doesn't match expected structure.
    """
    parts = filepath.replace("\\", "/").split("/")
    # Expected: hw / ip / <block> / rtl|dv / ...
    if len(parts) >= 4 and parts[0] == "hw" and parts[1] == "ip":
        return parts[2]
    return None


def is_excluded(filepath: str) -> bool:
    """Return True if this file should be excluded per Section 3.2."""
    for pat in EXCLUDE_PATH_PATTERNS:
        if pat in filepath:
            return True
    return False


def is_rtl_file(filepath: str) -> bool:
    return "/rtl/" in filepath and filepath.endswith(".sv") and not is_excluded(filepath)


def is_dv_file(filepath: str) -> bool:
    if "/dv/" not in filepath:
        return False
    ext = "." + filepath.rsplit(".", 1)[-1] if "." in filepath else ""
    if ext not in DV_EXTENSIONS:
        return False
    if is_excluded(filepath):
        return False
    return True


# ---------------------------------------------------------------------------
# Main extraction
# ---------------------------------------------------------------------------

def extract(repo: str, output: str, stats: bool = False):

    print(f"\nOpenTitan commit extractor")
    print(f"Repo     : {repo}")
    print(f"Range    : {DATE_FROM} → {DATE_TO}")
    print(f"Output   : {output}\n")

    os.makedirs(os.path.dirname(output) if os.path.dirname(output) else ".", exist_ok=True)

    print("Fetching commit list from git log...", flush=True)
    all_commits = get_commit_list(repo)
    print(f"Total commits in range (non-merge): {len(all_commits)}")

    rows         = []
    n_checked    = 0
    n_no_rtl_dv  = 0
    n_no_overlap = 0
    n_excluded   = 0
    n_dv_only_md = 0

    for i, commit in enumerate(all_commits):
        h   = commit["hash"]
        msg = commit["message"]
        dt  = commit["date"]

        # Progress
        if (i + 1) % 500 == 0 or i == 0:
            print(f"  [{i+1:>5}/{len(all_commits)}] checked so far, "
                  f"{len(rows)} qualifying commits found...")

        files = get_changed_files(repo, h)
        n_checked += 1

        # Separate RTL and DV files
        rtl_files = [f for f in files if is_rtl_file(f)]
        dv_files  = [f for f in files if is_dv_file(f)]

        if not rtl_files or not dv_files:
            n_no_rtl_dv += 1
            continue

        # Check same-block overlap — Section 3.2
        rtl_blocks = set(b for f in rtl_files for b in [extract_ip_block(f)] if b)
        dv_blocks  = set(b for f in dv_files  for b in [extract_ip_block(f)] if b)
        overlap    = rtl_blocks & dv_blocks

        if not overlap:
            n_no_overlap += 1
            continue

        # Use the first overlapping block (most commits touch one block)
        ip_block = sorted(overlap)[0]

        # Pull diffs for this block only
        rtl_path = f"hw/ip/{ip_block}/rtl/"
        dv_path  = f"hw/ip/{ip_block}/dv/"

        rtl_diff = get_diff(repo, h, rtl_path)
        dv_diff  = get_diff(repo, h, dv_path)

        # Skip if DV diff is empty after filtering (e.g. only .md changes)
        if not dv_diff.strip():
            n_dv_only_md += 1
            continue

        rows.append({
            "commit_hash":    h,
            "ip_block":       ip_block,
            "commit_date":    dt,
            "commit_message": msg,
            "rtl_files":      "|".join(rtl_files),
            "dv_files":       "|".join(dv_files),
            "rtl_diff":       rtl_diff,
            "dv_diff":        dv_diff,
        })

    # Write CSV
    fieldnames = ["commit_hash", "ip_block", "commit_date", "commit_message",
                  "rtl_files", "dv_files", "rtl_diff", "dv_diff"]

    with open(output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    # Summary
    print(f"\n{'='*55}")
    print(f"  EXTRACTION COMPLETE")
    print(f"{'='*55}")
    print(f"  Date range                : {DATE_FROM} → {DATE_TO}")
    print(f"  Total commits checked     : {n_checked}")
    print(f"  No RTL+DV pair            : {n_no_rtl_dv}")
    print(f"  No same-block overlap     : {n_no_overlap}")
    print(f"  DV diff empty after filter: {n_dv_only_md}")
    print(f"  ─────────────────────────────────")
    print(f"  Qualifying commits        : {len(rows)}")
    print(f"  Output saved              : {output}")
    print(f"{'='*55}")

    if stats:
        import collections
        block_counts = collections.Counter(r["ip_block"] for r in rows)
        print(f"\n  Top 15 IP blocks:")
        for block, count in block_counts.most_common(15):
            bar = "█" * (count // 3)
            print(f"    {block:<25} {count:>4}  {bar}")

    print(f"\nNext step:")
    print(f"  python3 llm_label.py --input {output} "
          f"--output data/llm_labeled.csv --validate\n")

    return len(rows)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Extract RTL+DV co-modification commits from OpenTitan"
    )
    parser.add_argument("--repo",   required=True,
                        help="Path to local OpenTitan clone")
    parser.add_argument("--output", default="data/commits.csv",
                        help="Output CSV path (default: data/commits.csv)")
    parser.add_argument("--stats",  action="store_true",
                        help="Print per-IP-block distribution after extraction")
    args = parser.parse_args()

    if not os.path.isdir(args.repo):
        raise FileNotFoundError(f"Repo not found: {args.repo}")

    extract(repo=args.repo, output=args.output, stats=args.stats)


if __name__ == "__main__":
    main()