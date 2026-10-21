"""Extract commit metadata and design/verification diffs from local Git repos.

Repository paths and classification globs come from ``sources.toml``. Each
repository is written to its own CSV (default ``default=project_root / "extracted_raw_commits",``) as soon as
it finishes, and ``manifest.json`` records what was extracted and from which
commit. This step does not assign failure labels or discard commits that do not
match the design/verification profiles. Merge commits are skipped unless asked
for, because a merge repeats the changes of the commits it merges.
"""

import argparse
import csv
import fnmatch
import hashlib
import json
import re
import subprocess
import time
import tomllib
from datetime import datetime, timezone
from pathlib import Path
from label_schema import CATEGORY_APPLICABILITY_STATUSES, FAILURE_CATEGORY_CODES


FIELDS = [
    "protocol_version",
    "source_repo",
    "source_ref",
    "source_head",
    "profile_hash",
    "category_profile_status",
    "category_profile_note",
    "category_applicability",
    "commit_hash",
    "parent_hashes",
    "is_merge",
    "diff_base_hash",
    "commit_date",
    "commit_message",
    "changed_files",
    "design_files",
    "verification_files",
    "change_scope",
    "design_diff",
    "verification_diff",
    "design_diff_lines",
    "verification_diff_lines",
    "design_diff_truncated",
    "verification_diff_truncated",
]

# List-valued fields are JSON-encoded so each commit remains one CSV row.
# The category profile is context only; it never filters commits or labels.
PROTOCOL_VERSION = "1.0.0"


def git(repo: Path, *args: str) -> str:
    """Run Git in one source checkout and return stdout or a useful error."""
    result = subprocess.run(
        ["git", "-c", "core.quotepath=false", *args],
        cwd=repo,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if result.returncode:
        raise RuntimeError(result.stderr.strip() or f"git {' '.join(args)} failed")
    return result.stdout


def load_sources(config_path: Path) -> dict:
    """Resolve source checkout paths and validate that each is a Git repo."""
    with config_path.open("rb") as config_file:
        config = tomllib.load(config_file)
    root = (config_path.parent / config.get("source_root", "sources")).resolve()
    repositories = {}
    for name, profile in config.get("repositories", {}).items():
        applicability = profile.get("category_applicability", {})
        missing_categories = set(FAILURE_CATEGORY_CODES) - set(applicability)
        extra_categories = set(applicability) - set(FAILURE_CATEGORY_CODES)
        if missing_categories or extra_categories:
            raise ValueError(
                f"Source '{name}' must define all six failure categories; "
                f"missing={sorted(missing_categories)}, extra={sorted(extra_categories)}"
            )
        invalid_statuses = set(applicability.values()) - set(CATEGORY_APPLICABILITY_STATUSES)
        if invalid_statuses:
            raise ValueError(
                f"Source '{name}' has invalid category applicability status(s): "
                f"{sorted(invalid_statuses)}"
            )
        repo = (root / profile["checkout_dir"]).resolve()
        if not repo.is_dir():
            raise FileNotFoundError(f"Source '{name}' is missing: {repo}")
        git(repo, "rev-parse", "--git-dir")
        repositories[name] = (repo, profile)
    return repositories


def list_commits(repo: Path, ref: str, since: str | None, until: str | None,
                 include_merges: bool, limit: int | None) -> list[dict]:
    """Read commit metadata reachable from ``ref``, preserving full messages."""
    # Git walks the selected ref's history, not every branch or tag in the repo.
    args = ["log", "--date=iso-strict"]
    if not include_merges:
        args.append("--no-merges")
    if since:
        args.append(f"--since={since}")
    if until:
        args.append(f"--until={until}")
    if limit:
        args.append(f"--max-count={limit}")
    args.append(ref)

    commits = []
    raw_log = git(repo, *args, "--format=%H%x1f%P%x1f%aI%x1f%B%x00")
    for record in raw_log.split("\x00"):
        line = record.strip("\r\n")
        if not line:
            continue
        fields = line.split("\x1f", 3)
        if len(fields) != 4:
            continue
        commit_hash, parents, date, message = fields
        commits.append({
            "commit_hash": commit_hash,
            "parent_hashes": parents.split() if parents else [],
            "commit_date": date,
            "commit_message": message,
        })
    return commits


def names_by_commit(repo: Path, ref: str, since: str | None, until: str | None,
                    limit: int | None) -> dict[str, list[str]]:
    """Map every non-merge commit to its changed paths with a single Git call."""
    args = ["log", "--no-merges", "--name-only", "--no-renames", "--format=@@%H"]
    if since:
        args.append(f"--since={since}")
    if until:
        args.append(f"--until={until}")
    if limit:
        args.append(f"--max-count={limit}")
    args.append(ref)

    names: dict[str, list[str]] = {}
    current = None
    for line in git(repo, *args).splitlines():
        if line.startswith("@@"):
            current = line[2:]
            names[current] = []
        elif line and current:
            names[current].append(line)
    return names


def changed_files(repo: Path, commit: dict) -> list[str]:
    """List changed paths relative to the first parent (or empty tree for root)."""
    parents = commit["parent_hashes"]
    if parents:
        # Merge commits are compared with their first parent for a consistent diff.
        names = git(repo, "diff", "--name-only", "--no-renames", parents[0],
                    commit["commit_hash"])
    else:
        names = git(repo, "diff-tree", "--root", "--no-commit-id", "-r",
                    "--name-only", "--no-renames", commit["commit_hash"])
    return [name for name in names.splitlines() if name]


def expand_braces(pattern: str) -> list[str]:
    """Expand `{a,b}` groups, e.g. `*.{sv,svh}` -> `*.sv`, `*.svh`."""
    start = pattern.find("{")
    if start == -1:
        return [pattern]
    end = pattern.index("}", start)
    return [
        expanded
        for option in pattern[start + 1:end].split(",")
        for expanded in expand_braces(pattern[:start] + option + pattern[end + 1:])
    ]


def compile_patterns(patterns: list[str]):
    """Return a path matcher for profile globs (`*` also matches `/`)."""
    expanded = [item for pattern in patterns for item in expand_braces(pattern)]
    if not expanded:
        return lambda path: False
    regex = re.compile("|".join(fnmatch.translate(item) for item in expanded))
    return lambda path: regex.match(path) is not None


def diff_for_paths(repo: Path, commit: dict, paths: list[str],
                   max_lines: int) -> tuple[str, bool, int]:
    """Return (diff text, truncated, total diff lines) for the selected paths.

    ``max_lines=0`` keeps the complete diff. The total is always the real
    length, so a truncated row still shows how much was cut.
    """
    if not paths:
        return "", False, 0

    parents = commit["parent_hashes"]
    kept: list[str] = []
    total = 0
    # Keep each Git command small enough for repositories with many changed files.
    for start in range(0, len(paths), 32):
        pathspecs = [f":(literal){path}" for path in paths[start:start + 32]]
        if parents:
            diff = git(repo, "diff", "--no-ext-diff", "--no-renames", "--unified=3",
                       parents[0], commit["commit_hash"], "--", *pathspecs)
        else:
            diff = git(repo, "show", "--root", "--format=", "--no-ext-diff",
                       "--no-renames", "--unified=3", commit["commit_hash"],
                       "--", *pathspecs)

        lines = diff.splitlines()
        total += len(lines)
        room = max_lines - len(kept) if max_lines > 0 else len(lines)
        kept.extend(lines[:max(room, 0)])

    truncated = max_lines > 0 and total > max_lines
    if truncated:
        kept.append("... [diff truncated at configured line limit]")
    return "\n".join(kept), truncated, total


def resolve_head(repo: Path, ref: str) -> str:
    """Pin a ref (e.g. HEAD) to a full commit hash so the run is reproducible."""
    return git(repo, "rev-parse", "--verify", f"{ref}^{{commit}}").strip()


def profile_digest(profile: dict) -> str:
    """Short hash of a repository profile, to tell which rules produced a CSV."""
    text = json.dumps(profile, sort_keys=True)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


def format_duration(seconds: float) -> str:
    seconds = int(seconds)
    return f"{seconds // 3600}h{seconds % 3600 // 60:02d}m" if seconds >= 3600 \
        else f"{seconds // 60}m{seconds % 60:02d}s"


def extract_repository(name: str, repo: Path, profile: dict, ref: str, head: str,
                       since: str | None, until: str | None,
                       include_merges: bool, limit: int | None,
                       max_diff_lines: int) -> list[dict]:
    """Build one raw row per commit using this repository's path profile.

    A commit's scope is ``paired``, ``design_only``, ``verification_only``, or
    ``other``; the scope describes available paths and does not filter rows.
    """
    # All Git calls use the pinned hash, so a mid-run fetch cannot change results.
    commits = list_commits(repo, head, since, until, include_merges, limit)
    names = names_by_commit(repo, head, since, until, limit)
    is_design = compile_patterns(profile.get("design_paths", []))
    is_verification = compile_patterns(profile.get("verification_paths", []))
    is_excluded = compile_patterns(profile.get("exclude_paths", []))
    category_applicability = profile["category_applicability"]
    profile_hash = profile_digest(profile)
    records = []
    started = time.monotonic()

    for index, commit in enumerate(commits, start=1):
        # Merge commits (opt-in) are not in the batched listing, so ask Git directly.
        files = names[commit["commit_hash"]] if commit["commit_hash"] in names \
            else changed_files(repo, commit)
        # Precedence: excluded, then verification, then design.
        usable = [path for path in files if not is_excluded(path)]
        verification_files = [path for path in usable if is_verification(path)]
        verification_set = set(verification_files)
        design_files = [
            path for path in usable
            if path not in verification_set and is_design(path)
        ]
        design_diff, design_truncated, design_lines = diff_for_paths(
            repo, commit, design_files, max_diff_lines
        )
        verification_diff, verification_truncated, verification_lines = diff_for_paths(
            repo, commit, verification_files, max_diff_lines
        )

        # Pairing here means both path types changed in this exact commit/repo.
        # It does not correlate commits across separate repositories.
        if design_files and verification_files:
            scope = "paired"
        elif design_files:
            scope = "design_only"
        elif verification_files:
            scope = "verification_only"
        else:
            scope = "other"

        records.append({
            "protocol_version": PROTOCOL_VERSION,
            "source_repo": name,
            "source_ref": ref,
            "source_head": head,
            "profile_hash": profile_hash,
            "category_profile_status": profile.get("category_profile_status", "pending_review"),
            "category_profile_note": profile.get("category_profile_note", ""),
            # Keep profile states with the evidence; never use them to suppress labels.
            "category_applicability": json.dumps(category_applicability, sort_keys=True),
            **commit,
            "parent_hashes": json.dumps(commit["parent_hashes"]),
            "is_merge": str(len(commit["parent_hashes"]) > 1).lower(),
            "diff_base_hash": commit["parent_hashes"][0] if commit["parent_hashes"] else "",
            "changed_files": json.dumps(files),
            "design_files": json.dumps(design_files),
            "verification_files": json.dumps(verification_files),
            "change_scope": scope,
            "design_diff": design_diff,
            "verification_diff": verification_diff,
            "design_diff_lines": design_lines,
            "verification_diff_lines": verification_lines,
            "design_diff_truncated": str(design_truncated).lower(),
            "verification_diff_truncated": str(verification_truncated).lower(),
        })
        if index % 200 == 0 or index == len(commits):
            elapsed = time.monotonic() - started
            remaining = elapsed / index * (len(commits) - index)
            print(f"  {name}: {index}/{len(commits)} commits "
                  f"({index * 100 // len(commits)}%), elapsed {format_duration(elapsed)}, "
                  f"about {format_duration(remaining)} left", flush=True)

    return records


def write_csv(path: Path, records: list[dict]) -> None:
    """Write via a temporary file so an interrupted run never leaves a partial CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(records)
    temporary.replace(path)


def update_manifest(path: Path, name: str, entry: dict) -> None:
    """Record one repository's run in manifest.json, keeping other repos' entries."""
    manifest = {"repositories": {}}
    if path.exists():
        try:
            manifest = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            pass
    manifest.update({
        "tool": "extract_raw_commits.py",
        "protocol_version": PROTOCOL_VERSION,
        "updated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    })
    manifest.setdefault("repositories", {})[name] = entry
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    """Parse user options, mine selected repos, and write one CSV per repo."""
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description=(
            "Extract raw commits from local repositories. Docs-only and one-sided "
            "commits are kept; merge commits are skipped unless --include-merges."
        ),
        epilog=(
            "Output: one CSV per repository plus manifest.json (pinned commit, "
            "profile hash, counts). List columns are JSON text; *_diff_lines is "
            "the real diff length and *_diff_truncated shows when the cap was hit."
        ),
    )
    parser.add_argument("--config", type=Path, default=project_root / "sources.toml",
                        help="TOML file defining checkout paths and design/verification globs")
    parser.add_argument("--repo", action="append", dest="repositories",
                        help="Repository key from sources.toml; repeat to select multiple")
    parser.add_argument("--ref", default=None, help="Override configured ref for selected repos")
    parser.add_argument("--since", default=None, help="Optional Git --since date")
    parser.add_argument("--until", default=None, help="Optional Git --until date")
    parser.add_argument("--include-merges", action="store_true",
                        help="Also extract merge commits (diffed against their first parent); "
                             "off by default because a merge repeats the commits it merges")
    parser.add_argument("--limit-per-repo", type=int, default=None,
                        help="Smoke-test limit; omitted means all commits reachable from each ref")
    parser.add_argument("--max-diff-lines", type=int, default=2000,
                        help="Maximum extracted diff lines per side; use 0 for untruncated diffs")
    parser.add_argument("--output-dir", type=Path,
                        default=project_root / "extracted_raw_commits",
                        help="Directory for one CSV per repository")
    parser.add_argument("--force", action="store_true",
                        help="Replace existing per-repository CSVs")
    args = parser.parse_args()

    if args.limit_per_repo is not None and args.limit_per_repo <= 0:
        parser.error("--limit-per-repo must be positive")
    if args.max_diff_lines < 0:
        parser.error("--max-diff-lines cannot be negative")
    configured = load_sources(args.config.resolve())
    selected = args.repositories or list(configured)
    unknown = set(selected) - set(configured)
    if unknown:
        parser.error(f"Unknown repository key(s): {', '.join(sorted(unknown))}")

    output_paths = {name: args.output_dir / f"{name}.csv" for name in selected}
    existing_outputs = [path for path in output_paths.values() if path.exists()]
    if existing_outputs and not args.force:
        parser.error(
            "Output file(s) already exist; use another --output-dir or --force: "
            + ", ".join(str(path) for path in existing_outputs)
        )

    records_by_repo = {}
    output_dir = args.output_dir.resolve()
    manifest_path = output_dir / "manifest.json"
    print(f"Output directory: {output_dir}", flush=True)

    for name in selected:
        repo, profile = configured[name]
        ref = args.ref or profile.get("ref", "HEAD")
        head = resolve_head(repo, ref)
        print(f"Extracting {name} from {repo} at {ref} ({head[:12]})", flush=True)
        started = time.monotonic()
        records = extract_repository(
            name, repo, profile, ref, head, args.since, args.until,
            args.include_merges, args.limit_per_repo, args.max_diff_lines,
        )
        scope_counts = {}
        for record in records:
            scope_counts[record["change_scope"]] = (
                scope_counts.get(record["change_scope"], 0) + 1
            )
        scope_summary = ", ".join(
            f"{scope}={count}" for scope, count in sorted(scope_counts.items())
        ) or "no commits in selected range"
        print(f"  {name}: {len(records)} commits ({scope_summary})", flush=True)

        output_path = output_paths[name].resolve()
        write_csv(output_path, records)
        update_manifest(manifest_path, name, {
            "csv": output_path.name,
            "ref": ref,
            "head": head,
            "profile_hash": profile_digest(profile),
            "commits": len(records),
            "scope_counts": scope_counts,
            "options": {
                "since": args.since,
                "until": args.until,
                "include_merges": args.include_merges,
                "limit_per_repo": args.limit_per_repo,
                "max_diff_lines": args.max_diff_lines,
            },
            "seconds": round(time.monotonic() - started, 1),
            "finished_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        })
        records_by_repo[name] = output_path
        print(f"  wrote {output_path}", flush=True)

    print("\nDone. Files:")
    for name, path in records_by_repo.items():
        print(f"  {name}: {path}")
    print(f"  manifest: {manifest_path}")


if __name__ == "__main__":
    main()