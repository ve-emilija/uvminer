#!/usr/bin/env python33
"""
VeriDelta — Stage 1: extract commits
stage1_extract_commits.py

Clones (or reuses) the source repositories and dumps their full commit history
to flat files.  NO FILTERING happens here: every commit reachable from the
chosen revision is written, and Stage 2 decides what is design, verification
or neither.

Default repositories
    opentitan     https://github.com/lowRISC/opentitan.git
    caliptra-rtl  https://github.com/chipsalliance/caliptra-rtl.git

Outputs  (in --out-dir, default ./stage1_out)
    commits.csv     one row per commit:
                    repo, hash, parents, n_parents, author_date, commit_date,
                    subject, body, n_files, insertions, deletions
                    (+ author_name, author_email with --include-authors)
    files.csv       one row per (commit, changed file):
python3                    repo, hash, status, path, old_path, added, deleted, binary
                    status ∈ A M D T R C ; old_path is set for R/C only
    manifest.json   repo URL, resolved ref, pinned HEAD sha, counts, git version,
                    sha256 of this script, timestamp — the provenance record
                    (the exact commit hash the README asks for)

Design notes
    * Read-only on the working tree: only `git clone/fetch` and `git log`.
    * Reproducible: use --pin NAME=SHA (or --pin NAME=<tag>) to freeze a
      revision; without it the remote default branch is used and its SHA is
      recorded in manifest.json.
    * Commits are written oldest-first.  Merge commits are skipped by default
      (their content appears in the commits they merge); --keep-merges keeps
      them as rows with no files.  Rename detection (-M) is on, so renames
      come out as one R row — important because a renamed RTL module is
      exactly the trigger for Assertion Orphaning.
    * Author identity is OFF by default (double-blind artifact).
    * After extraction the script cross-checks the commit count against
      `git rev-list --count` and re-derives the file list of a random sample of
      commits with `git diff-tree`; any mismatch aborts with a non-zero exit.

Usage
    python3 stage1_extract_commits.py                        # both repos
    python3 stage1_extract_commits.py --repos caliptra-rtl
    python3 stage1_extract_commits.py --pin opentitan=<sha> --pin caliptra-rtl=<sha>
    python3 stage1_extract_commits.py --sources-dir sources --out-dir stage1_out
    python3 stage1_extract_commits.py --add myrepo=https://github.com/org/repo.git
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import re
import subprocess
import sys
from datetime import datetime, timezone

DEFAULT_REPOS = {
    "opentitan":    "https://github.com/lowRISC/opentitan.git",
    "caliptra-rtl": "https://github.com/chipsalliance/caliptra-rtl.git",
}

SOH, STX, ETX = b"\x01", b"\x02", b"\x03"
LOG_FORMAT = "%x01%H%x02%P%x02%aI%x02%cI%x02%an%x02%ae%x02%s%x02%b%x03"
_NUMSTAT = re.compile(rb"^(\d+|-)\t(\d+|-)\t(.*)$", re.S)

COMMIT_FIELDS = ["repo", "hash", "parents", "n_parents", "author_date", "commit_date",
                 "subject", "body", "n_files", "insertions", "deletions"]
FILE_FIELDS = ["repo", "hash", "status", "path", "old_path", "added", "deleted", "binary"]


# ── git helpers ───────────────────────────────────────────────────────────────

def _git_cmd(repo_dir: str, *args: str) -> list[str]:
    return ["git", "-C", repo_dir, "-c", "core.quotepath=false",
            "-c", "diff.renameLimit=5000", *args]


def git(repo_dir: str, *args: str, check: bool = True) -> str:
    p = subprocess.run(_git_cmd(repo_dir, *args), capture_output=True, text=True,
                       encoding="utf-8", errors="replace")
    if check and p.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed in {repo_dir}:\n{p.stderr.strip()}")
    return p.stdout.strip()


def ensure_repo(name: str, url: str, sources_dir: str, fetch: bool) -> str:
    path = os.path.join(sources_dir, name)
    if not os.path.isdir(os.path.join(path, ".git")):
        os.makedirs(sources_dir, exist_ok=True)
        print(f"[{name}] cloning {url} → {path}")
        subprocess.run(["git", "clone", "--no-tags", url, path], check=True)
        subprocess.run(["git", "-C", path, "fetch", "--tags", "--quiet", "origin"], check=False)
    elif fetch:
        print(f"[{name}] fetching origin")
        subprocess.run(["git", "-C", path, "fetch", "--tags", "--quiet", "origin"], check=True)
    if git(path, "rev-parse", "--is-shallow-repository") == "true":
        print(f"[{name}] shallow clone detected → unshallowing")
        subprocess.run(["git", "-C", path, "fetch", "--unshallow", "--quiet"], check=True)
    return path


def resolve_rev(path: str, pin: str | None) -> tuple[str, str]:
    """(ref description, full sha)."""
    if pin:
        return pin, git(path, "rev-parse", "--verify", f"{pin}^{{commit}}")
    for cand in ("origin/HEAD", "origin/main", "origin/master", "HEAD"):
        p = subprocess.run(_git_cmd(path, "rev-parse", "--verify", "--quiet", f"{cand}^{{commit}}"),
                           capture_output=True, text=True)
        if p.returncode == 0:
            return cand, p.stdout.strip()
    raise RuntimeError(f"no usable ref in {path}")


# ── parsing ───────────────────────────────────────────────────────────────────

def parse_log_stream(stream, repo: str, keep_merges: bool):
    """
    Yield (commit_dict, [file_dict, ...]) from `git log --raw --numstat -z`
    output (format: SOH hash STX … STX body ETX NUL <raw entries> <numstat entries>).
    """
    buf = b""
    while True:
        chunk = stream.read(1 << 20)
        if chunk:
            buf += chunk
        parts = buf.split(SOH)
        # last part may be incomplete unless EOF
        complete, buf = (parts, b"") if not chunk else (parts[:-1], parts[-1])
        for rec in complete:
            if rec:
                out = _parse_record(rec, repo)
                if out and (keep_merges or out[0]["n_parents"] <= 1):
                    yield out
        if not chunk:
            return


def _dec(b: bytes) -> str:
    return b.decode("utf-8", errors="replace")


def _parse_record(rec: bytes, repo: str):
    head, sep, tail = rec.partition(ETX)
    if not sep:
        return None
    f = head.split(STX)
    if len(f) < 8:
        return None
    h, parents, adate, cdate, an, ae, subject = (_dec(x) for x in f[:7])
    body = _dec(STX.join(f[7:]))
    commit = {
        "repo": repo, "hash": h, "parents": parents, "n_parents": len(parents.split()),
        "author_date": adate, "commit_date": cdate, "author_name": an, "author_email": ae,
        "subject": subject, "body": body.strip("\n"),
    }

    toks = [t for t in tail.split(b"\x00") if t != b""]
    if toks and toks[0].startswith(b"\n"):          # git puts a newline before the first raw entry
        toks[0] = toks[0][1:]

    files: dict[tuple[str, str], dict] = {}
    order: list[tuple[str, str]] = []
    i = 0
    while i < len(toks):
        t = toks[i]
        if t.startswith(b":"):                         # raw entry
            status = _dec(t.split(b" ")[-1])
            code = status[0]
            if code in "RC":
                old, new = _dec(toks[i + 1]), _dec(toks[i + 2])
                i += 3
            else:
                old, new = "", _dec(toks[i + 1])
                i += 2
            key = (old, new)
            files[key] = {"repo": repo, "hash": h, "status": code, "path": new,
                          "old_path": old, "added": "", "deleted": "", "binary": 0}
            order.append(key)
            continue
        m = _NUMSTAT.match(t)
        if m:                                          # numstat entry
            a, d, p = m.group(1), m.group(2), m.group(3)
            if p == b"":                               # rename/copy: two path tokens follow
                old, new = _dec(toks[i + 1]), _dec(toks[i + 2])
                i += 3
            else:
                old, new = "", _dec(p)
                i += 1
            fr = files.get((old, new)) or files.get(("", new))
            if fr is not None:
                if a == b"-":
                    fr["binary"] = 1
                else:
                    fr["added"], fr["deleted"] = int(a), int(d)
            continue
        i += 1                                         # unknown token: skip defensively

    flist = [files[k] for k in order]
    commit["n_files"] = len(flist)
    commit["insertions"] = sum(x["added"] for x in flist if x["added"] != "")
    commit["deletions"] = sum(x["deleted"] for x in flist if x["deleted"] != "")
    return commit, flist


# ── extraction + verification ─────────────────────────────────────────────────

def extract_repo(name, path, sha, out_commits, out_files, include_authors, keep_merges):
    cmd = _git_cmd(path, "log", sha, "--reverse", "--no-color", f"--format={LOG_FORMAT}",
                   "--raw", "--numstat", "-z", "-M", "--no-abbrev", "--root")
    if not keep_merges:
        cmd.insert(cmd.index("--reverse"), "--no-merges")
    fields = COMMIT_FIELDS + (["author_name", "author_email"] if include_authors else [])
    n = 0
    sample_pool: list[tuple[str, list[dict]]] = []
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    cw = csv.DictWriter(out_commits, fieldnames=fields, extrasaction="ignore")
    fw = csv.DictWriter(out_files, fieldnames=FILE_FIELDS, extrasaction="ignore")
    rng = random.Random(0)
    seen_first, seen_last = None, None
    for commit, flist in parse_log_stream(proc.stdout, name, keep_merges):
        cw.writerow(commit)
        fw.writerows(flist)
        n += 1
        seen_first = seen_first or commit["author_date"]
        seen_last = commit["author_date"]
        if flist:                                      # reservoir sample for verification
            if len(sample_pool) < 25:
                sample_pool.append((commit["hash"], flist))
            else:
                j = rng.randrange(n)
                if j < 25:
                    sample_pool[j] = (commit["hash"], flist)
        if n % 2000 == 0:
            print(f"[{name}]   {n} commits…", flush=True)
    err = proc.stderr.read().decode("utf-8", "replace")
    if proc.wait() != 0:
        raise RuntimeError(f"git log failed for {name}:\n{err}")
    return n, sample_pool, seen_first, seen_last


def verify(name, path, sha, n_written, sample_pool, keep_merges) -> dict:
    expected = int(git(path, "rev-list", "--count", *([] if keep_merges else ["--no-merges"]), sha))
    ok_count = (expected == n_written)
    bad = []
    for h, flist in sample_pool:
        raw = subprocess.run(_git_cmd(path, "diff-tree", "--root", "-r", "-M", "-z", "--name-status",
                                      "--no-commit-id", h), capture_output=True).stdout
        toks = [t for t in raw.split(b"\x00") if t]
        ref, i = set(), 0
        while i < len(toks):
            code = _dec(toks[i])[0]
            if code in "RC":
                ref.add((code, _dec(toks[i + 1]), _dec(toks[i + 2])))
                i += 3
            else:
                ref.add((code, "", _dec(toks[i + 1])))
                i += 2
        mine = {(f["status"], f["old_path"], f["path"]) for f in flist}
        if ref != mine:
            bad.append(h)
    result = {"commits_expected": expected, "commits_written": n_written,
              "count_ok": ok_count, "sample_checked": len(sample_pool), "sample_mismatches": bad}
    if not ok_count or bad:
        raise RuntimeError(f"[{name}] verification FAILED: {result}")
    return result


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="VeriDelta Stage 1: extract commits")
    ap.add_argument("--repos", nargs="+", default=list(DEFAULT_REPOS),
                    help="names to extract (default: all configured)")
    ap.add_argument("--add", action="append", default=[], metavar="NAME=URL",
                    help="register an additional repository")
    ap.add_argument("--pin", action="append", default=[], metavar="NAME=REV",
                    help="freeze a repo at a sha/tag (recorded in manifest.json)")
    ap.add_argument("--sources-dir", default="sources")
    ap.add_argument("--out-dir", default="stage1_out")
    ap.add_argument("--no-fetch", action="store_true", help="do not fetch existing clones")
    ap.add_argument("--keep-merges", action="store_true")
    ap.add_argument("--include-authors", action="store_true")
    args = ap.parse_args(argv)

    repos = dict(DEFAULT_REPOS)
    for a in args.add:
        k, _, v = a.partition("=")
        repos[k] = v
    pins = dict(p.partition("=")[::2] for p in args.pin)
    unknown = [r for r in args.repos if r not in repos]
    if unknown:
        print(f"unknown repo(s): {unknown}; known: {list(repos)}")
        return 2

    os.makedirs(args.out_dir, exist_ok=True)
    fields = COMMIT_FIELDS + (["author_name", "author_email"] if args.include_authors else [])
    script_sha = hashlib.sha256(open(__file__, "rb").read()).hexdigest()
    manifest = {
        "stage": 1, "script_sha256": script_sha,
        "extracted_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_version": subprocess.run(["git", "--version"], capture_output=True, text=True).stdout.strip(),
        "options": {"keep_merges": args.keep_merges, "include_authors": args.include_authors,
                    "rename_detection": True},
        "repos": {},
    }

    tmp_c = os.path.join(args.out_dir, "commits.csv.tmp")
    tmp_f = os.path.join(args.out_dir, "files.csv.tmp")
    with open(tmp_c, "w", newline="", encoding="utf-8") as fc, \
         open(tmp_f, "w", newline="", encoding="utf-8") as ff:
        csv.DictWriter(fc, fieldnames=fields).writeheader()
        csv.DictWriter(ff, fieldnames=FILE_FIELDS).writeheader()
        for name in args.repos:
            path = ensure_repo(name, repos[name], args.sources_dir, not args.no_fetch)
            ref, sha = resolve_rev(path, pins.get(name))
            print(f"[{name}] extracting history up to {ref} = {sha}")
            n, sample, first, last = extract_repo(name, path, sha, fc, ff,
                                                  args.include_authors, args.keep_merges)
            v = verify(name, path, sha, n, sample, args.keep_merges)
            manifest["repos"][name] = {
                "url": repos[name], "ref": ref, "head_sha": sha, "pinned": name in pins,
                "commits": n, "first_author_date": first, "last_author_date": last,
                "verification": v,
            }
            print(f"[{name}] {n} commits  ({first} → {last})  verified: "
                  f"count ok, {v['sample_checked']} sampled file lists identical")

    os.replace(tmp_c, os.path.join(args.out_dir, "commits.csv"))
    os.replace(tmp_f, os.path.join(args.out_dir, "files.csv"))
    with open(os.path.join(args.out_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nWrote commits.csv, files.csv, manifest.json → {args.out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())