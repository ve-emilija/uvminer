"""
llm_label.py — LLM-assisted commit classifier for UVM DV failure taxonomy
Paper: "When RTL Changes, What Breaks?"
Author: Emilija Velinova

Usage:
    python data_labeling/llm_label.py --source opentitan --input data/commits.csv --output data/labeled_commits.csv
    python data_labeling/llm_label.py --source opentitan --input data/commits.csv --output data/labeled_commits.csv --validate
    python data_labeling/llm_label.py --source opentitan --input data/commits.csv --output data/labeled_commits.csv --limit 10

Input CSV columns required:
    commit_hash, commit_message, design_diff, verification_diff
Optional columns: source_repo, ip_block
Legacy rtl_diff and dv_diff columns are also accepted.

Output CSV adds:
    protocol_version, method, source_repo, label_status, categories, evidence,
    at_risk_dv_files, justification, confidence, model, cost_usd

Requirements: requests, pandas, python-dotenv
API key:       set ANTHROPIC_API_KEY in environment or .env file
"""

import os
import json
import time
import argparse
import csv
import re
from datetime import datetime
from pathlib import Path
import requests
import pandas as pd
from dotenv import load_dotenv
from uvminer.data_extraction.label_schema import (
    CATEGORY_CODES,
    LABEL_STATUSES,
    PROTOCOL_VERSION,
    RUN_STATUSES,
    normalize_label_result,
)

load_dotenv()

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

API_URL    = "https://api.anthropic.com/v1/messages"
MODEL      = "claude-haiku-4-5-20251001"
MAX_TOKENS = 600
RETRY_MAX  = 3
RETRY_WAIT = 5

PRICE_IN   = 1.00   # Haiku input  $/M tokens
PRICE_OUT  = 5.00   # Haiku output $/M tokens

# FIX 2: increased from 150 to 300 — reduces truncation-driven mislabeling
MAX_DIFF_LINES = 300

# ---------------------------------------------------------------------------
# Versioned annotation contract
# ---------------------------------------------------------------------------

CATEGORIES = CATEGORY_CODES

LABELING_PROTOCOL = Path(__file__).with_name("LABELING_PROTOCOL.md").read_text(
    encoding="utf-8"
)

# Ground-truth commits from Section 4
GROUND_TRUTH = [
    {
        "commit_hash":        "b058bfe507",
        "expected_categories": ["INTERFACE_DRIFT"],
        "note":               "SPI host command length 9->20 bit, env_pkg updated",
    },
    {
        "commit_hash":        "b13dc6ff8e",
        "expected_categories": ["ASSERTION_ORPHANING"],
        "note":               "I2C module rename broke $assertoff path",
    },
    {
        "commit_hash":        "da7fc3571b",
        "expected_categories": ["COVERAGE_REGRESSION"],
        "note":               "lc_flash_rma -> lc_nvm_rma rename broke cov_bind",
    },
    {
        "commit_hash":        "57f9776639",
        "expected_categories": ["CHECKER_DESYNC"],
        "note":               "entropy_src Ack SM change, scoreboard realigned",
    },
    {
        "commit_hash":        "6ee0cf0193",
        "expected_categories": ["SEQUENCE_INVALIDATION"],
        "note":               "SPI host configopts multi-register removed",
    },
]

# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = f"""You are an expert hardware design-verification annotator.
Apply the following versioned protocol exactly. Classify only mechanisms
supported by the supplied commit evidence; do not infer a failure from a
co-change or filename alone.

{LABELING_PROTOCOL}

Return strict JSON with this shape and no surrounding prose:
{{
    "label_status": "{' | '.join(LABEL_STATUSES)}",
    "categories": ["CATEGORY_CODE"],
    "evidence": ["path: concrete change and its verification consequence"],
    "at_risk_dv_files": ["path/to/affected/file"],
    "justification": "Brief explanation grounded in the diffs.",
    "confidence": "HIGH | MEDIUM | LOW"
}}
Use category codes from the protocol. For CLASSIFIED, list one or more of the
six failure categories. For OTHER use categories ["OTHER"]. For
NOT_APPLICABLE or INSUFFICIENT_EVIDENCE use an empty categories list. Evidence
must reference paths and facts visible in the supplied diffs.
"""


def build_user_prompt(
    source_repo: str,
    commit_hash: str,
    ip_block: str,
    commit_message: str,
    design_diff: str,
    verification_diff: str,
) -> str:
    return f"""Classify this commit using protocol {PROTOCOL_VERSION}.

Source repository : {source_repo}
Commit hash       : {commit_hash}
Component / block : {ip_block or 'unspecified'}
Message           : {commit_message}

Design diff (first {MAX_DIFF_LINES} lines):
```
{_truncate_diff(design_diff)}
```

Verification diff (first {MAX_DIFF_LINES} lines):
```
{_truncate_diff(verification_diff)}
```

Respond with JSON only."""


def _truncate_diff(diff: str) -> str:
    lines = diff.splitlines()
    if len(lines) <= MAX_DIFF_LINES:
        return diff
    kept = lines[:MAX_DIFF_LINES]
    kept.append(f"... [{len(lines) - MAX_DIFF_LINES} lines truncated]")
    return "\n".join(kept)


def _diff_paths(diff: str) -> set[str]:
    return {
        path for path in re.findall(r"^(?:\+\+\+ b/|--- a/)(.+)$", diff, re.MULTILINE)
        if path != "/dev/null"
    }


# ---------------------------------------------------------------------------
# API call
# ---------------------------------------------------------------------------

def call_api(system: str, user: str, api_key: str) -> dict:
    headers = {
        "x-api-key":         api_key,
        "anthropic-version": "2023-06-01",
        "content-type":      "application/json",
    }
    payload = {
        "model":      MODEL,
        "max_tokens": MAX_TOKENS,
        "system":     system,
        "messages":   [{"role": "user", "content": user}],
    }

    for attempt in range(1, RETRY_MAX + 1):
        try:
            resp = requests.post(API_URL, headers=headers,
                                 json=payload, timeout=60)
            resp.raise_for_status()
            data = resp.json()

            raw_text = data["content"][0]["text"].strip()
            if raw_text.startswith("```"):
                raw_text = raw_text.split("```")[1]
                if raw_text.startswith("json"):
                    raw_text = raw_text[4:]
            parsed = json.loads(raw_text)

            usage    = data.get("usage", {})
            in_tok   = usage.get("input_tokens",  0)
            out_tok  = usage.get("output_tokens", 0)
            cost_usd = (in_tok * PRICE_IN + out_tok * PRICE_OUT) / 1_000_000

            return {
                "result":     parsed,
                "in_tokens":  in_tok,
                "out_tokens": out_tok,
                "cost_usd":   cost_usd,
            }

        except requests.exceptions.HTTPError as e:
            status = e.response.status_code if e.response else "?"
            if status in (429, 529) and attempt < RETRY_MAX:
                wait = RETRY_WAIT * attempt
                print(f"  [rate limit {status}] waiting {wait}s (attempt {attempt})")
                time.sleep(wait)
                continue
            raise
        except (json.JSONDecodeError, KeyError) as e:
            if attempt < RETRY_MAX:
                print(f"  [parse error] retrying (attempt {attempt}): {e}")
                time.sleep(RETRY_WAIT)
                continue
            raise

    raise RuntimeError("Max retries exceeded")


# ---------------------------------------------------------------------------
# FIX 3: Resume — load already-processed hashes from existing output
# ---------------------------------------------------------------------------

def load_existing(
    output_path: str,
    default_source_repo: str | None = None,
) -> tuple[list[dict], set[tuple[str, str]]]:
    if not os.path.exists(output_path):
        return [], set()
    try:
        df = pd.read_csv(output_path)
        all_results = df.to_dict("records")
        results = []
        done = set()
        for row in all_results:
            try:
                categories = json.loads(row.get("categories", "[]"))
            except (TypeError, json.JSONDecodeError):
                categories = []
            status = row.get("label_status", "")
            if pd.isna(status) or not status:
                status = "ERROR" if "ERROR" in categories else ""
            if status in RUN_STATUSES or {"ERROR", "DRY_RUN"} & set(categories):
                continue

            row_source = row.get("source_repo", "")
            if pd.isna(row_source) or not str(row_source).strip():
                row_source = default_source_repo or "unknown"
            row_source = str(row_source).strip()
            row["source_repo"] = row_source
            if pd.isna(row.get("protocol_version", "")) or not row.get("protocol_version"):
                row["protocol_version"] = "legacy"
            results.append(row)
            done.add((row_source, str(row["commit_hash"])))
        print(
            f"[resume] Found {len(all_results)} rows, {len(done)} completed rows, "
            f"{len(all_results) - len(done)} incomplete rows will be retried"
        )
        return results, done
    except Exception:
        return [], set()

# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_ground_truth(labeled_df: pd.DataFrame) -> dict:
    results = []
    for gt in GROUND_TRUTH:
        h   = gt["commit_hash"]
        row = labeled_df[labeled_df["commit_hash"].str.startswith(h)]
        if row.empty:
            results.append({
                "commit_hash": h, "status": "NOT_FOUND",
                "expected": gt["expected_categories"], "got": [],
                "note": gt["note"],
            })
            continue
        got_raw  = row.iloc[0]["categories"]
        got      = json.loads(got_raw) if isinstance(got_raw, str) else got_raw
        expected = set(gt["expected_categories"])
        match    = expected.issubset(set(got))
        results.append({
            "commit_hash": h,
            "status":      "PASS" if match else "FAIL",
            "expected":    gt["expected_categories"],
            "got":         got,
            "note":        gt["note"],
        })
    passed   = sum(1 for r in results if r["status"] == "PASS")
    total    = sum(1 for r in results if r["status"] != "NOT_FOUND")
    accuracy = passed / total if total else 0.0
    return {"accuracy": accuracy, "n_pass": passed,
            "n_total": total, "details": results}


# ---------------------------------------------------------------------------
# Cohen's kappa
# ---------------------------------------------------------------------------

def cohen_kappa(llm_labels: list, human_labels: list, all_cats: list) -> float:
    kappas = []
    for cat in all_cats:
        llm_bin   = [1 if cat in labels else 0 for labels in llm_labels]
        human_bin = [1 if cat in labels else 0 for labels in human_labels]
        n = len(llm_bin)
        if n == 0:
            continue
        po = sum(l == h for l, h in zip(llm_bin, human_bin)) / n
        pe = ((sum(llm_bin) / n) * (sum(human_bin) / n) +
              ((n - sum(llm_bin)) / n) * ((n - sum(human_bin)) / n))
        kappa = (po - pe) / (1 - pe) if pe < 1.0 else 1.0
        kappas.append(kappa)
    return sum(kappas) / len(kappas) if kappas else 0.0


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run(
    input_path: str,
    output_path: str,
    api_key: str,
    limit: int | None = None,
    validate: bool = False,
    dry_run: bool = False,
    source_repo: str | None = None,
):

    print(f"\n{'='*60}")
    print(f"  LLM Commit Classifier — {MODEL}")
    print(f"  Input  : {input_path}")
    print(f"  Output : {output_path}")
    print(f"  Diff window : {MAX_DIFF_LINES} lines per side")
    print(f"{'='*60}\n")

    df = pd.read_csv(input_path)
    required = {"commit_hash", "commit_message"}
    missing  = required - set(df.columns)
    if missing:
        raise ValueError(f"Input CSV missing columns: {missing}")
    design_column = "design_diff" if "design_diff" in df else "rtl_diff"
    verification_column = (
        "verification_diff" if "verification_diff" in df else "dv_diff"
    )
    if design_column not in df or verification_column not in df:
        raise ValueError(
            "Input CSV must include design_diff and verification_diff "
            "(or legacy rtl_diff and dv_diff)."
        )
    df["design_diff"] = df[design_column].fillna("").astype(str)
    df["verification_diff"] = df[verification_column].fillna("").astype(str)

    if "ip_block" not in df:
        df["ip_block"] = ""
    if "source_repo" not in df:
        df["source_repo"] = ""
    df["source_repo"] = df["source_repo"].fillna("").astype(str).str.strip()
    if source_repo:
        df.loc[df["source_repo"] == "", "source_repo"] = source_repo
    if (df["source_repo"] == "").any():
        raise ValueError(
            "Set --source or provide a non-empty source_repo column in the input CSV."
        )

    if limit:
        df = df.head(limit)
        print(f"[limit] Processing first {limit} commits\n")

    # FIX 3: load existing results for resume
    results, already_done = load_existing(output_path, source_repo)

    total_cost    = sum(r.get("cost_usd", 0) for r in results)
    total_in_tok  = sum(r.get("in_tokens", 0) for r in results)
    total_out_tok = sum(r.get("out_tokens", 0) for r in results)
    n_total       = len(df)
    n_skipped     = 0

    for i, row in df.iterrows():
        idx   = len(results) + 1 - n_skipped
        h     = str(row["commit_hash"])
        source = str(row["source_repo"])
        block = str(row.get("ip_block", ""))
        msg   = str(row["commit_message"])
        design_diff = str(row["design_diff"])
        verification_diff = str(row["verification_diff"])

        # Skip already processed
        if (source, h) in already_done:
            n_skipped += 1
            continue

        processed = len(results) + 1
        print(f"[{processed:>3}/{n_total}] {h[:12]} | {block:<20}",
              end=" ", flush=True)

        if dry_run:
            print("DRY RUN — skipping API call")
            results.append({
                "protocol_version": PROTOCOL_VERSION,
                "method":            "llm",
                "source_repo":      source,
                "commit_hash":      h, "ip_block": block,
                "commit_message":   msg,
                "label_status":     "DRY_RUN",
                "categories":       json.dumps([]),
                "evidence":         json.dumps([]),
                "at_risk_dv_files": json.dumps([]),
                "justification":    "dry run", "confidence": "LOW",
                "model":            MODEL, "in_tokens": 0,
                "out_tokens":       0, "cost_usd": 0.0,
            })
            continue

        user_prompt = build_user_prompt(
            source, h, block, msg, design_diff, verification_diff
        )

        try:
            resp   = call_api(SYSTEM_PROMPT, user_prompt, api_key)
            parsed = resp["result"]

            annotation = normalize_label_result(
                parsed.get("label_status", ""),
                parsed.get("categories", []),
            )
            evidence_raw = parsed.get("evidence", [])
            evidence = (
                [str(item).strip() for item in evidence_raw if str(item).strip()]
                if isinstance(evidence_raw, list) else []
            )
            visible_paths = _diff_paths(_truncate_diff(design_diff)) | _diff_paths(
                _truncate_diff(verification_diff)
            )
            evidence = [
                item for item in evidence
                if any(path in item for path in visible_paths)
            ]
            at_risk_raw = parsed.get("at_risk_dv_files", [])
            at_risk = (
                [str(item).strip() for item in at_risk_raw if str(item).strip()]
                if isinstance(at_risk_raw, list) else []
            )
            if annotation["label_status"] in ("CLASSIFIED", "OTHER") and not evidence:
                annotation = normalize_label_result("INSUFFICIENT_EVIDENCE", [])
            justif = str(parsed.get("justification", ""))
            confidence = str(parsed.get("confidence", "LOW")).upper()
            if confidence not in {"HIGH", "MEDIUM", "LOW"}:
                confidence = "LOW"

            total_cost    += resp["cost_usd"]
            total_in_tok  += resp["in_tokens"]
            total_out_tok += resp["out_tokens"]

            label_str = " | ".join(annotation["categories"]) or annotation["label_status"]
            print(f"→ {label_str:<45} [{confidence}] ${resp['cost_usd']:.4f}")

            results.append({
                "protocol_version": PROTOCOL_VERSION,
                "method":            "llm",
                "source_repo":      source,
                "commit_hash":      h, "ip_block": block,
                "commit_message":   msg,
                "label_status":     annotation["label_status"],
                "categories":       json.dumps(annotation["categories"]),
                "evidence":         json.dumps(evidence),
                "at_risk_dv_files": json.dumps(at_risk),
                "justification":    justif, "confidence": confidence,
                "model":            MODEL,
                "in_tokens":        resp["in_tokens"],
                "out_tokens":       resp["out_tokens"],
                "cost_usd":         resp["cost_usd"],
            })

        except Exception as e:
            print(f"ERROR: {e}")
            results.append({
                "protocol_version": PROTOCOL_VERSION,
                "method":            "llm",
                "source_repo":      source,
                "commit_hash":      h, "ip_block": block,
                "commit_message":   msg,
                "label_status":     "ERROR",
                "categories":       json.dumps([]),
                "evidence":         json.dumps([]),
                "at_risk_dv_files": json.dumps([]),
                "justification":    str(e), "confidence": "LOW",
                "model":            MODEL, "in_tokens": 0,
                "out_tokens":       0, "cost_usd": 0.0,
            })

        # Save after every commit — protects against crashes
        out_df = pd.DataFrame(results)
        out_df.to_csv(output_path, index=False)

        time.sleep(5.0)

    # Final save
    out_df = pd.DataFrame(results)
    out_df.to_csv(output_path, index=False)

    # Summary
    print(f"\n{'='*60}")
    print(f"  SUMMARY")
    print(f"{'='*60}")
    print(f"  Commits processed : {len(results)}")
    print(f"  Tokens in         : {total_in_tok:,}")
    print(f"  Tokens out        : {total_out_tok:,}")
    print(f"  Total cost        : ${total_cost:.4f}")
    print(f"  Output saved      : {output_path}")

    from collections import Counter
    all_cats_flat = []
    for r in results:
        try:
            cats = json.loads(r["categories"])
        except Exception:
            cats = [r["categories"]]
        all_cats_flat.extend(cats)

    print(f"\n  Category distribution:")
    dist = Counter(all_cats_flat)
    for cat, count in dist.most_common():
        pct = count / len(results) * 100
        print(f"    {cat:<30} {count:>4}  ({pct:.1f}%)")

    if validate:
        sources = set(out_df.get("source_repo", pd.Series(dtype=str)).dropna())
        if sources == {"opentitan"}:
            print(f"\n  Ground truth validation:")
            val = validate_ground_truth(out_df)
            for d in val["details"]:
                icon = "✓" if d["status"] == "PASS" else ("?" if d["status"] == "NOT_FOUND" else "✗")
                print(f"    {icon} {d['commit_hash']} | expected {d['expected']} | got {d['got']}")
            print(f"\n  Accuracy: {val['n_pass']}/{val['n_total']} "
                  f"({val['accuracy']*100:.0f}%)")
        else:
            print("\n  OpenTitan-only reference validation skipped for this source set.")

    print(f"\n{'='*60}\n")
    return out_df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="LLM-assisted UVM DV failure classifier"
    )
    parser.add_argument("--input",       required=True)
    parser.add_argument("--output",      required=True)
    parser.add_argument("--source",      default=None,
                        help="Repository identifier if input rows have no source_repo column")
    parser.add_argument("--limit",       type=int, default=None)
    parser.add_argument("--validate",    action="store_true")
    parser.add_argument("--dry-run",     action="store_true")
    parser.add_argument("--kappa-human", default=None)
    args = parser.parse_args()

    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key and not args.dry_run:
        raise EnvironmentError(
            "ANTHROPIC_API_KEY not set. "
            "Add it to your .env file or export it in your shell."
        )

    out_df = run(
        input_path  = args.input,
        output_path = args.output,
        api_key     = api_key or "",
        limit       = args.limit,
        validate    = args.validate,
        dry_run     = args.dry_run,
        source_repo = args.source,
    )

    if args.kappa_human and not args.dry_run:
        print("Computing Cohen's kappa against human annotations...")
        human_df = pd.read_csv(args.kappa_human)
        merge_keys = ["commit_hash"]
        if "source_repo" in out_df and "source_repo" in human_df:
            merge_keys.insert(0, "source_repo")
        merged = out_df.merge(
            human_df,
            on=merge_keys,
            suffixes=("_llm", "_human"),
        )

        if "label_status_llm" in merged:
            eligible_statuses = {"CLASSIFIED", "OTHER"}
            merged = merged[merged["label_status_llm"].isin(eligible_statuses)]
        if "label_status_human" in merged:
            eligible_statuses = {"CLASSIFIED", "OTHER"}
            merged = merged[merged["label_status_human"].isin(eligible_statuses)]
        if merged.empty:
            raise ValueError("No comparable classified rows found for kappa calculation.")

        llm_labels   = [json.loads(r) for r in merged["categories_llm"]]
        human_labels = [json.loads(r) for r in merged["categories_human"]]
        all_cats     = [k for k in CATEGORIES if k != "OTHER"]

        kappa = cohen_kappa(llm_labels, human_labels, all_cats)
        print(f"\n  Cohen's kappa (LLM vs human): κ = {kappa:.3f}")
        if   kappa >= 0.80: print("  → Almost perfect agreement")
        elif kappa >= 0.60: print("  → Substantial agreement")
        elif kappa >= 0.40: print("  → Moderate agreement")
        else:               print("  → Fair/slight — review prompt and taxonomy")

        kappa_path = args.output.replace(".csv", "_kappa.json")
        with open(kappa_path, "w") as f:
            json.dump({
                "kappa":     round(kappa, 4),
                "n_commits": len(merged),
                "model":     MODEL,
                "timestamp": datetime.utcnow().isoformat(),
            }, f, indent=2)
        print(f"  Kappa result saved: {kappa_path}")


if __name__ == "__main__":
    main()