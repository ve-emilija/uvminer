#!/usr/bin/env python3
"""
VeriDelta — Stage 2a: file-extension inventory
stage2a_extension_inventory.py

Reads Stage 1's files.csv (+ commits.csv for commit sizes) and lists EVERY file
extension that occurs in the history of each repository, so you can decide which
ones matter.  Nothing is filtered or classified here.

Output (in --out-dir, default ./stage2_out)
    ext_inventory.csv   repo, ext, file_changes, commits, commits_nonbulk,
                        distinct_paths, first_seen, last_seen, added, deleted,
                        top_dirs, examples
    (+ a table printed per repo, sorted by number of non-bulk commits)

Extension rules
    * lower-cased last suffix:  foo.SV -> .sv
    * template wrappers keep the inner suffix:  foo.sv.tpl -> .sv.tpl
    * no suffix: the base name is used in angle brackets:  Makefile -> <Makefile>
    * dotfiles:  .gitignore -> <.gitignore>

"Bulk" commits touch more than --bulk-threshold files (default 500): repo-wide
imports, license-header rewrites, vendor drops.  They are counted in file_changes
but excluded from commits_nonbulk, so they cannot make a vendored extension
look important.

Usage
    python3 stage2a_extension_inventory.py
    python3 stage2a_extension_inventory.py --stage1-dir stage1_out --top 40
"""

from __future__ import annotations

import argparse
import csv
import os
import posixpath
import sys
from collections import Counter, defaultdict

TEMPLATE_SUFFIXES = {".tpl", ".in", ".j2", ".mako", ".template", ".orig", ".bak", ".tmpl"}


def ext_of(path: str) -> str:
    base = posixpath.basename(path)
    if base.startswith(".") and base.count(".") == 1:        # .gitignore, .clang-format
        return f"<{base}>"
    stem, dot, last = base.rpartition(".")
    if not dot or not stem:
        return f"<{base}>"                                   # Makefile, BUILD, LICENSE
    last = "." + last.lower()
    if last in TEMPLATE_SUFFIXES:
        inner = posixpath.splitext(stem)[1].lower()
        if inner:
            return inner + last                              # .sv.tpl
    return last


def dir_prefix(path: str, depth: int = 3) -> str:
    parts = path.split("/")[:-1]
    return "/".join(parts[:depth]) if parts else "."


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="VeriDelta Stage 2a: extension inventory")
    ap.add_argument("--stage1-dir", default="stage1_out")
    ap.add_argument("--out-dir", default="stage2_out")
    ap.add_argument("--bulk-threshold", type=int, default=500,
                    help="commits touching more files than this are 'bulk' (default 500)")
    ap.add_argument("--top", type=int, default=0,
                    help="print only the N most common extensions per repo (default: all)")
    args = ap.parse_args(argv)

    csv.field_size_limit(1 << 30)
    commits_csv = os.path.join(args.stage1_dir, "commits.csv")
    files_csv = os.path.join(args.stage1_dir, "files.csv")
    for p in (commits_csv, files_csv):
        if not os.path.exists(p):
            print(f"missing {p} — run stage1_extract_commits.py first")
            return 1

    # pass 1: which commits are bulk, and their dates
    bulk: set[tuple[str, str]] = set()
    date: dict[tuple[str, str], str] = {}
    with open(commits_csv, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            k = (r["repo"], r["hash"])
            date[k] = r["author_date"][:10]
            if int(r["n_files"] or 0) > args.bulk_threshold:
                bulk.add(k)
    print(f"bulk commits (> {args.bulk_threshold} files): {len(bulk)}")

    # pass 2: aggregate per (repo, ext)
    agg: dict[tuple[str, str], dict] = {}
    with open(files_csv, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            repo, h, path = r["repo"], r["hash"], r["path"]
            e = agg.setdefault((repo, ext_of(path)), {
                "changes": 0, "commits": set(), "nonbulk": set(), "paths": set(),
                "first": "9999", "last": "", "added": 0, "deleted": 0,
                "dirs": Counter(), "examples": [],
            })
            e["changes"] += 1
            e["commits"].add(h)
            if (repo, h) not in bulk:
                e["nonbulk"].add(h)
            e["paths"].add(path)
            d = date.get((repo, h), "")
            if d:
                e["first"], e["last"] = min(e["first"], d), max(e["last"], d)
            e["added"] += int(r["added"] or 0)
            e["deleted"] += int(r["deleted"] or 0)
            e["dirs"][dir_prefix(path)] += 1
            if len(e["examples"]) < 2 and (repo, h) not in bulk:
                e["examples"].append(path)

    os.makedirs(args.out_dir, exist_ok=True)
    rows = []
    for (repo, ext), e in agg.items():
        rows.append({
            "repo": repo, "ext": ext, "file_changes": e["changes"],
            "commits": len(e["commits"]), "commits_nonbulk": len(e["nonbulk"]),
            "distinct_paths": len(e["paths"]), "first_seen": e["first"], "last_seen": e["last"],
            "added": e["added"], "deleted": e["deleted"],
            "top_dirs": " | ".join(f"{d} ({n})" for d, n in e["dirs"].most_common(3)),
            "examples": " | ".join(e["examples"]),
        })
    rows.sort(key=lambda r: (r["repo"], -r["commits_nonbulk"], -r["file_changes"], r["ext"]))
    out_path = os.path.join(args.out_dir, "ext_inventory.csv")
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    by_repo: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_repo[r["repo"]].append(r)
    for repo, rs in by_repo.items():
        shown = rs[: args.top] if args.top else rs
        print(f"\n{repo}: {len(rs)} distinct extensions")
        print(f"  {'ext':<22} {'commits(nb)':>11} {'commits':>8} {'changes':>8} {'paths':>7}  top dirs")
        for r in shown:
            print(f"  {r['ext']:<22} {r['commits_nonbulk']:>11} {r['commits']:>8} "
                  f"{r['file_changes']:>8} {r['distinct_paths']:>7}  {r['top_dirs'].split(' | ')[0]}")

    names = list(by_repo)
    if len(names) >= 2:
        sets = {n: {r["ext"] for r in by_repo[n]} for n in names}
        for n in names:
            others = set().union(*(sets[m] for m in names if m != n))
            only = sorted(sets[n] - others, key=lambda x: -next(r["commits_nonbulk"] for r in by_repo[n] if r["ext"] == x))
            print(f"\nextensions only in {n}: {len(only)}   (top 25 by commits) {only[:25]}")

    print(f"\nWrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())