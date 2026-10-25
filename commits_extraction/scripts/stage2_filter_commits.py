#!/usr/bin/env python3
"""
VeriDelta — Stage 2: filter commits and label them design / verification / both
stage2_filter_commits.py

Reads Stage 1's commits.csv + files.csv and keeps only commits that change
HDL source files.

1. SELECTOR FILES    a changed file counts if it has one of
                         .sv .svh .v .vh .sva .sv.tpl .svh.tpl
                     (other extensions are ignored — they never select a commit).
2. SIDE OF EACH FILE decided by generic name rules (no repo-specific paths),
                     first match wins:
                       a. a directory named like a verification area (dv, tb, uvm,
                          uvmf*, fpv, formal, sva, coverage, asserts, test_suites ...)
                          → VERIFICATION
                       b. `.sva` extension → VERIFICATION
                       c. a verification word in the FILE NAME (uvm, cov, covergroups,
                          sva, assert*, bind, tb, bfm, scoreboard, vseq, env, *_reg_sample ...,
                          e.g. kv_reg_uvm.sv, prim_assert.sv, foo_tb.sv, xbar.dv.sv.tpl)
                          → VERIFICATION
                       d. a directory rtl / design / hdl → DESIGN
                       e. a file inside a vendor/third_party directory → DESIGN
                       f. any remaining template (.tpl) → DESIGN (a template takes the
                          side of what it generates; DV templates were caught by c.)
                       g. otherwise UNKNOWN
                     Files whose change is whitespace-only (git diff -w) are ignored.
3. COMMIT LABEL      design        only DESIGN files (plus unknown ones)
                     verification  only VERIFICATION files (plus unknown)
                     both          at least one of each
                     unknown       selector files exist but none could be placed
                     none          no selector file → dropped
   `neither`      every selector file change is whitespace-only → not selected
   `needs_review` = the commit has any UNKNOWN selector file; these (and a
   random sample of the confident ones) are what Stage 2c sends to the LLM and
   to the human annotator.
4. EXCLUSIONS        commits touching more than --bulk-threshold files (default
                     500: repo imports, license-header rewrites) are labelled but
                     not selected.  `vendored` marks commits that touch a
                     vendor/third_party directory; they stay selected.

Outputs (in --out-dir, default ./stage2_out)
    commits_labeled.csv   every commit: counts per side, label, selected, reason
    commits_selected.csv  selected == 1 only (the input to the next stage)
    selector_files.csv    every selector file with its side and the rule that fired
    summary.txt           counts per repo/label + the top directories left UNKNOWN,
                          so the rules can be tightened
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import subprocess
import sys
from collections import Counter, defaultdict

SELECTOR_SUFFIXES = (".sv", ".svh", ".v", ".vh", ".sva", ".sv.tpl", ".svh.tpl")

# A path segment matches if it IS one of these words or starts/ends with one
# separated by '_' or '-'  (dv, generic_dv, dv-classes, uvmf_soc_ifc, pre_dv ...)
VERIF_WORDS = ("dv", "tb", "uvm", "uvmf", "fpv", "formal", "sva", "verif", "verification",
               "testbench", "coverage", "asserts", "assertions", "properties", "pre_dv",
               "uvmdvgen", "fpvgen")
DESIGN_WORDS = ("rtl", "design", "hdl")
VENDOR_WORDS = ("vendor", "third_party", "thirdparty", "submodules")


def _seg_re(words):
    alt = "|".join(sorted(map(re.escape, words), key=len, reverse=True))
    return re.compile(rf"^(?:.*[_-])?(?:{alt})(?:[_-].*)?$")


VERIF_SEG = _seg_re(VERIF_WORDS)
DESIGN_SEG = _seg_re(DESIGN_WORDS)


# verification words that, in a FILE NAME (tokens split on _ - . ), mark a verification file
FILE_VERIF_TOKENS = {"uvm", "uvmf", "cov", "covergroups", "covergroup", "sva", "assert", "asserts",
                     "assertion", "assertions", "bind", "tb", "bfm", "scoreboard", "vseq", "vseqs",
                     "env", "fpv", "dv", "regsample"}
VERIF_WORDS = VERIF_WORDS + ("test_suites", "cov")
VERIF_SEG = _seg_re(VERIF_WORDS)


def stem_tokens(path: str) -> set:
    base = path.split("/")[-1].lower()
    for suf in sorted(SELECTOR_SUFFIXES, key=len, reverse=True):
        if base.endswith(suf):
            base = base[: -len(suf)]
            break
    toks = {t for t in re.split(r"[^a-z0-9]+", base) if t}
    if "sample" in toks and toks & {"reg", "csr"}:       # kv_reg_sample.svh: coverage sampling of a register model
        toks.add("regsample")
    toks.discard("dv")                                   # bare "dv" is Caliptra's DataVault
    if ".dv." in "." + base + ".":                       # but xbar.dv.sv.tpl is a DV template
        toks.add("dv")
    return toks


def is_template(path: str) -> bool:
    return path.lower().endswith(".tpl")


def is_selector(path: str) -> bool:
    return path.lower().endswith(SELECTOR_SUFFIXES)


def side_of(path: str) -> tuple[str, str]:
    """(side, rule) for a selector file.  side ∈ VERIFICATION | DESIGN | UNKNOWN."""
    segs = path.split("/")[:-1]
    for s in segs:
        if VERIF_SEG.match(s.lower()):
            return "VERIFICATION", f"dir:{s}"
    if path.lower().endswith(".sva"):
        return "VERIFICATION", "ext:.sva"
    hit = stem_tokens(path) & FILE_VERIF_TOKENS
    if hit:
        return "VERIFICATION", f"file:{sorted(hit)[0]}"
    for s in segs:
        if DESIGN_SEG.match(s.lower()):
            return "DESIGN", f"dir:{s}"
    if is_vendored(path):
        return "DESIGN", "vendor"
    if is_template(path):
        return "DESIGN", "template"
    return "UNKNOWN", ""


def is_vendored(path: str) -> bool:
    return any(s.lower() in VENDOR_WORDS for s in path.split("/")[:-1])


def label_of(n_d: int, n_v: int, n_u: int, n_w: int = 0) -> str:
    if n_d and n_v:
        return "both"
    if n_v:
        return "verification"
    if n_d:
        return "design"
    if n_u:
        return "unknown"
    return "neither" if n_w else "none"


def whitespace_only_changes(sources_dir: str, repo: str, cand: set) -> set:
    """(hash, path) pairs among `cand` whose change is whitespace-only: the numstat is 0/0 when
    whitespace and blank lines are ignored (git -w --ignore-blank-lines)."""
    path = os.path.join(sources_dir, repo)
    if not os.path.isdir(path) or not cand:
        return set()
    cmd = ["git", "-C", path, "-c", "core.quotepath=off", "log", "--no-merges", "--full-history",
           "--no-renames", "-w", "--ignore-blank-lines", "--numstat", "--format=%x01%H", "--",
           "*.sv", "*.svh", "*.v", "*.vh", "*.sva", "*.tpl"]
    headers, shown, cur = set(), set(), None
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, text=True, encoding="utf-8", errors="replace")
    for line in proc.stdout:
        if line.startswith("\x01"):
            cur = line[1:].strip()
            headers.add(cur)
        elif "\t" in line and cur:
            a, d, p = line.rstrip("\n").split("\t", 2)
            if (cur, p) in cand and not (a == "0" and d == "0"):
                shown.add((cur, p))              # still has a real change with whitespace ignored
    proc.wait()
    # git prints nothing for a file whose diff is empty once whitespace is ignored
    return {(h, p) for (h, p) in cand if h in headers and (h, p) not in shown}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="VeriDelta Stage 2: filter + label commits")
    ap.add_argument("--stage1-dir", default="stage1_out")
    ap.add_argument("--out-dir", default="stage2_out")
    ap.add_argument("--sources-dir", default="sources",
                    help="git clones, used only for the whitespace-only check")
    ap.add_argument("--no-whitespace-check", action="store_true")
    ap.add_argument("--bulk-threshold", type=int, default=500)
    ap.add_argument("--compare-labels", metavar="CSV",
                    help="CSV with repo,hash,human_label: print how the rule labels agree with it")
    args = ap.parse_args(argv)

    csv.field_size_limit(1 << 30)
    commits_path = os.path.join(args.stage1_dir, "commits.csv")
    files_path = os.path.join(args.stage1_dir, "files.csv")
    for p in (commits_path, files_path):
        if not os.path.exists(p):
            print(f"missing {p} — run stage1_extract_commits.py first")
            return 1
    os.makedirs(args.out_dir, exist_ok=True)

    # selector file changes
    sel = []
    cand: dict[str, set] = defaultdict(set)          # repo -> {(hash, path)} modified files with real churn
    with open(files_path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if not is_selector(r["path"]):
                continue
            sel.append(r)
            if r["status"].startswith("M") and not int(r["binary"] or 0) \
                    and int(r["added"] or 0) + int(r["deleted"] or 0) > 0:
                cand[r["repo"]].add((r["hash"], r["path"]))
    ws: dict[str, set] = {}
    if args.no_whitespace_check:
        print("whitespace check skipped")
    else:
        for repo in sorted({r["repo"] for r in sel}):
            if not os.path.isdir(os.path.join(args.sources_dir, repo)):
                print(f"note: {args.sources_dir}/{repo} not found — whitespace check skipped for {repo}")
                continue
            ws[repo] = whitespace_only_changes(args.sources_dir, repo, cand[repo])
            print(f"{repo}: {len(ws[repo])} whitespace-only file changes (of {len(cand[repo])} checked)")

    acc: dict[tuple[str, str], dict] = defaultdict(
        lambda: {"DESIGN": 0, "VERIFICATION": 0, "UNKNOWN": 0, "WS": 0, "vendored": 0})
    unknown_dirs: dict[str, Counter] = defaultdict(Counter)
    sel_rows = []
    for r in sel:
        p = r["path"]
        a = acc[(r["repo"], r["hash"])]
        if (r["hash"], p) in ws.get(r["repo"], ()):
            side, rule = "WS", "whitespace-only"
            a["WS"] += 1
        else:
            side, rule = side_of(p)
            a[side] += 1
            a["vendored"] |= int(is_vendored(p))
            if side == "UNKNOWN":
                unknown_dirs[r["repo"]]["/".join(p.split("/")[:-1][:4])] += 1
        sel_rows.append({"repo": r["repo"], "hash": r["hash"], "status": r["status"],
                         "path": p, "side": side, "rule": rule})

    cols = ["repo", "hash", "author_date", "subject", "n_files", "n_selector_files",
            "n_design", "n_verification", "n_unknown", "n_whitespace", "label", "needs_review",
            "vendored", "selected", "reason"]
    counts: dict[str, Counter] = defaultdict(Counter)
    n_selected: Counter = Counter()
    label_by_commit: dict[tuple[str, str], str] = {}
    with open(commits_path, newline="", encoding="utf-8") as fin, \
         open(os.path.join(args.out_dir, "commits_labeled.csv"), "w", newline="", encoding="utf-8") as fall, \
         open(os.path.join(args.out_dir, "commits_selected.csv"), "w", newline="", encoding="utf-8") as fsel:
        wall = csv.DictWriter(fall, fieldnames=cols)
        wsel = csv.DictWriter(fsel, fieldnames=cols)
        wall.writeheader()
        wsel.writeheader()
        for r in csv.DictReader(fin):
            a = acc.get((r["repo"], r["hash"]))
            n_d, n_v, n_u, n_w = (a["DESIGN"], a["VERIFICATION"], a["UNKNOWN"], a["WS"]) if a else (0, 0, 0, 0)
            label = label_of(n_d, n_v, n_u, n_w)
            label_by_commit[(r["repo"], r["hash"])] = label
            n_files = int(r["n_files"] or 0)
            if label == "none":
                selected, reason = 0, "no HDL selector file"
            elif label == "neither":
                selected, reason = 0, "whitespace-only change"
            elif n_files > args.bulk_threshold:
                selected, reason = 0, f"bulk commit (> {args.bulk_threshold} files)"
            else:
                selected, reason = 1, ""
            row = {
                "repo": r["repo"], "hash": r["hash"], "author_date": r["author_date"],
                "subject": r["subject"], "n_files": n_files,
                "n_selector_files": n_d + n_v + n_u + n_w, "n_design": n_d, "n_verification": n_v,
                "n_unknown": n_u, "n_whitespace": n_w, "label": label,
                "needs_review": int(selected and n_u > 0),
                "vendored": a["vendored"] if a else 0, "selected": selected, "reason": reason,
            }
            wall.writerow(row)
            counts[r["repo"]][(label, "selected" if selected else "excluded")] += 1
            if selected:
                wsel.writerow(row)
                n_selected[r["repo"]] += 1

    with open(os.path.join(args.out_dir, "selector_files.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["repo", "hash", "status", "path", "side", "rule"])
        w.writeheader()
        w.writerows(sel_rows)

    lines = [f"selector suffixes: {', '.join(SELECTOR_SUFFIXES)}",
             f"bulk threshold: {args.bulk_threshold} files", ""]
    for repo in sorted(counts):
        tot = sum(counts[repo].values())
        n_rv = 0
        lines.append(f"{repo}: {tot} commits, {n_selected[repo]} selected")
        for label in ("design", "verification", "both", "unknown", "neither", "none"):
            s_, e = counts[repo][(label, "selected")], counts[repo][(label, "excluded")]
            if s_ or e:
                lines.append(f"    {label:<13} selected {s_:>6}   excluded {e:>6}")
        lines.append("    top directories with UNKNOWN selector files (tighten rules here):")
        for d, n in unknown_dirs[repo].most_common(12):
            lines.append(f"        {n:>5}  {d}")
        lines.append("")
    summary = "\n".join(lines)
    with open(os.path.join(args.out_dir, "summary.txt"), "w") as f:
        f.write(summary)
    print(summary)
    print(f"Wrote commits_labeled.csv, commits_selected.csv, selector_files.csv, summary.txt → {args.out_dir}")

    if args.compare_labels:
        human = list(csv.DictReader(open(args.compare_labels, newline="", encoding="utf-8")))
        ok, bad = 0, []
        for h in human:
            rl = label_by_commit.get((h["repo"], h["hash"]), "?")
            if rl == h["human_label"]:
                ok += 1
            else:
                bad.append((h["hash"][:12], h["human_label"], rl))
        print(f"\nrule label vs {args.compare_labels}: {ok}/{len(human)} agree")
        for b in bad:
            print(f"   {b[0]}  human={b[1]:<12} rule={b[2]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())