"""
VeriDelta — Ground Truth Mapper (v2)
ground_truth_mapper.py

Maps commit-level DV failure category labels onto graph node labels.

Run type_annotator.py FIRST (with updated UVM_KEYWORDS) to regenerate
node_types.json before running this script.
"""

import ast
import csv
import json
import os
from collections import defaultdict, Counter

# ── Category normalisation ────────────────────────────────────────────────────
RAW_TO_DISPLAY: dict[str, str | None] = {
    "INTERFACE_DRIFT":       "Interface Drift",
    "SEQUENCE_INVALIDATION": "Sequence Invalidation",
    "CHECKER_DESYNC":        "Checker Desync",
    "ASSERTION_ORPHANING":   "Assertion Orphaning",
    "RAL_DESYNC":            "RAL Desync",
    "COVERAGE_REGRESSION":   "Coverage Regression",
    "OTHER":                 None,
}

# Paper Table 2 proportions (% of 652 commits, multi-label)
PAPER_PROPORTIONS = {
    "Interface Drift":       48.5,
    "Sequence Invalidation": 43.1,
    "Checker Desync":        35.0,
    "Assertion Orphaning":   13.5,
    "RAL Desync":             7.8,
    "Coverage Regression":    6.9,
}

ALL_CATEGORIES = list(PAPER_PROPORTIONS.keys())


def normalise_categories(raw_list: list[str]) -> list[str]:
    result = []
    for r in raw_list:
        display = RAW_TO_DISPLAY.get(r.upper())
        if display:
            result.append(display)
    return result


# ── Class name extraction with variants ───────────────────────────────────────

def class_name_from_path(filepath: str) -> str:
    """Strip directory and extension to get candidate class name."""
    base = os.path.basename(filepath.strip())
    for ext in (".sv", ".svh"):
        if base.endswith(ext):
            return base[: -len(ext)]
    return base


def class_name_variants(stem: str, known_nodes: set[str]) -> list[str]:
    """
    Return all known-node matches for a filename stem.

    For most files this is just [stem] if it exists in the graph.

    Special cases:
      _pkg files: aes_env_pkg → also try aes_env, aes_env_cfg
                  aes_ral_pkg → also try aes_reg_block
                  aes_test_pkg → skip (no useful graph match)
      _model files: skip (ISS/model files, not UVM components)

    Always returns a list — empty if nothing matched.
    """
    candidates = []

    # direct match first
    if stem in known_nodes:
        candidates.append(stem)

    # pkg file heuristics
    if stem.endswith("_pkg"):
        base = stem[:-4]  # strip _pkg

        # aes_env_pkg → aes_env
        if base in known_nodes:
            candidates.append(base)

        # aes_env_pkg → aes_env_cfg
        cfg = base + "_cfg"
        if cfg in known_nodes:
            candidates.append(cfg)

        # aes_ral_pkg → aes_reg_block
        if base.endswith("_ral"):
            reg = base[:-4] + "_reg_block"
            if reg in known_nodes:
                candidates.append(reg)

    return list(dict.fromkeys(candidates))  # deduplicate, preserve order


# ── Type-constrained ground truth ────────────────────────────────────────────
# A node can only carry categories consistent with its component type.
# This filters multi-label bleed-through from co-occurring commit labels.
#
# CFG gets two allowed categories because env_cfg holds both virtual interface
# handles (Interface Drift) and the RAL model handle (RAL Desync) — either
# field can change independently. All other types are single-category.

TYPE_ALLOWED_CATEGORIES: dict[str, list[str]] = {
    "CFG":   ["Interface Drift", "RAL Desync"],
    "ENV":   ["Interface Drift"],
    "AGENT": ["Interface Drift"],
    "SEQ":   ["Sequence Invalidation"],
    "SCBD":  ["Checker Desync"],
    "RAL":   ["RAL Desync"],
    "COV":   ["Coverage Regression"],
    "SVA":   ["Assertion Orphaning"],
    "OTHER": [
        "Interface Drift", "Sequence Invalidation", "Checker Desync",
        "Assertion Orphaning", "RAL Desync", "Coverage Regression",
    ],
}


def apply_type_constraints(
    ground_truth: dict,
    node_types:   dict,
) -> tuple[dict, int]:
    """
    Remove categories from each node that are inconsistent with its type.

    Returns the filtered ground_truth and the count of removed labels.
    """
    removed = 0
    filtered: dict = {}

    for node, data in ground_truth.items():
        node_type   = node_types.get(node, {}).get("type", "OTHER")
        allowed     = TYPE_ALLOWED_CATEGORIES.get(node_type, [])
        kept        = [c for c in data["categories"] if c in allowed]
        removed    += len(data["categories"]) - len(kept)

        if kept:
            filtered[node] = {**data, "categories": kept}
        # nodes whose only labels were noise are dropped entirely

    return filtered, removed

def classify_dv_file(stem: str) -> str:
    """Classify why a file didn't match — used only in the audit section."""
    if stem.endswith("_pkg"):
        return "pkg_file"
    if "vseq" in stem or "virtual_seq" in stem:
        return "vseq_file"
    if stem.endswith("_test") or "_test_" in stem:
        return "test_file"
    if any(k in stem for k in [
        "scoreboard", "_scb", "env_cfg", "_env", "_agent",
        "_driver", "_monitor", "cov_if", "cov_bind", "_checker",
    ]):
        return "uvm_sv_miss"
    return "other_sv"


# ── Load helpers ──────────────────────────────────────────────────────────────

def load_labeled_commits(path: str) -> dict[str, dict]:
    labeled: dict[str, dict] = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                raw_cats = ast.literal_eval(row["categories"])
            except (ValueError, SyntaxError):
                raw_cats = []
            try:
                at_risk = ast.literal_eval(row["at_risk_dv_files"])
            except (ValueError, SyntaxError):
                at_risk = []
            display_cats = normalise_categories(raw_cats)
            if not display_cats:
                continue
            labeled[row["commit_hash"]] = {
                "raw_categories":   raw_cats,
                "categories":       display_cats,
                "ip_block":         row.get("ip_block", ""),
                "at_risk_patterns": at_risk,
                "confidence":       row.get("confidence", "HIGH"),
            }
    return labeled


def load_dv_files(path: str, target_hashes: set[str]) -> dict[str, list[str]]:
    result: dict[str, list[str]] = {}
    remaining = set(target_hashes)
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            h = row["commit_hash"]
            if h not in remaining:
                continue
            raw = row.get("dv_files", "")
            result[h] = [p for p in raw.split("|") if p.strip()]
            remaining.discard(h)
            if not remaining:
                break
    return result


# ── Core mapping ──────────────────────────────────────────────────────────────

def build_ground_truth(
    labeled:          dict[str, dict],
    dv_files_by_hash: dict[str, list[str]],
    known_nodes:      set[str],
) -> tuple[dict, dict, dict]:
    """
    Returns (ground_truth, commit_index, audit).

    audit contains per-file-type counts and miss examples for inspection.
    """
    node_data: dict[str, dict] = defaultdict(
        lambda: {"categories": set(), "commit_hashes": []}
    )
    commit_index: dict[str, dict] = {}

    # audit accumulators
    audit_counts  = Counter()
    audit_miss_ex: dict[str, list[str]] = defaultdict(list)
    audit_matched_ex: list[str] = []

    matched_commits = 0
    total_pairs     = 0

    for commit_hash, meta in labeled.items():
        dv_paths = dv_files_by_hash.get(commit_hash, [])
        if not dv_paths:
            continue

        changed_nodes: list[str] = []

        for path in dv_paths:
            stem = class_name_from_path(path)
            if not stem:
                continue

            # is it an SV/SVH file?
            base = os.path.basename(path.strip())
            is_sv = base.endswith(".sv") or base.endswith(".svh")

            if not is_sv:
                audit_counts["non_sv"] += 1
                continue

            matched = class_name_variants(stem, known_nodes)

            if matched:
                changed_nodes.extend(matched)
                audit_counts["matched"] += 1
                if len(audit_matched_ex) < 10:
                    audit_matched_ex.append(f"{stem} → {matched}")
            else:
                bucket = classify_dv_file(stem)
                audit_counts[f"miss_{bucket}"] += 1
                if len(audit_miss_ex[bucket]) < 8:
                    audit_miss_ex[bucket].append(stem)

        # deduplicate
        changed_nodes = list(dict.fromkeys(changed_nodes))

        for node in changed_nodes:
            node_data[node]["categories"].update(meta["categories"])
            node_data[node]["commit_hashes"].append(commit_hash)

        if changed_nodes:
            matched_commits += 1
            total_pairs += len(changed_nodes)

        commit_index[commit_hash] = {
            "categories":       meta["categories"],
            "raw_categories":   meta["raw_categories"],
            "changed_nodes":    changed_nodes,
            "ip_block":         meta["ip_block"],
            "at_risk_patterns": meta["at_risk_patterns"],
            "confidence":       meta["confidence"],
        }

    ground_truth: dict[str, dict] = {
        node: {
            "categories":    sorted(data["categories"]),
            "commit_count":  len(data["commit_hashes"]),
            "commit_hashes": data["commit_hashes"],
        }
        for node, data in node_data.items()
    }

    audit = {
        "counts":      dict(audit_counts),
        "miss_examples": dict(audit_miss_ex),
        "matched_examples": audit_matched_ex,
        "matched_commits": matched_commits,
        "total_commits":   len(labeled),
        "total_node_label_pairs": total_pairs,
    }

    return ground_truth, commit_index, audit


# ── Audit printing ────────────────────────────────────────────────────────────

def print_audit(audit: dict, ground_truth: dict, node_types: dict) -> None:

    counts = audit["counts"]
    total_sv = sum(v for k, v in counts.items() if k != "non_sv")

    print("\nAudit: SV file match breakdown (602 labeled commits)")
    print("─" * 60)
    print(f"  Matched to graph node:          {counts.get('matched', 0):>5}")
    print(f"  Miss — vseq files:              {counts.get('miss_vseq_file', 0):>5}  (add 'vseq' to type_annotator keywords)")
    print(f"  Miss — pkg files:               {counts.get('miss_pkg_file', 0):>5}  (heuristic applied)")
    print(f"  Miss — UVM component (no node): {counts.get('miss_uvm_sv_miss', 0):>5}  (see examples below)")
    print(f"  Miss — test files:              {counts.get('miss_test_file', 0):>5}")
    print(f"  Miss — other SV:                {counts.get('miss_other_sv', 0):>5}  (model/tracer/tb files)")
    print(f"  Skipped — non-SV files:         {counts.get('non_sv', 0):>5}  (expected)")

    print("\nMiss examples — UVM components not in graph:")
    for ex in audit["miss_examples"].get("uvm_sv_miss", []):
        print(f"  {ex}")

    print("\nMiss examples — vseq files:")
    for ex in audit["miss_examples"].get("vseq_file", []):
        print(f"  {ex}")

    print("\nSuccessful match examples:")
    for ex in audit["matched_examples"]:
        print(f"  {ex}")

    # coverage by type
    type_total:   Counter = Counter()
    type_labeled: Counter = Counter()
    for node, info in node_types.items():
        t = info.get("type", "OTHER")
        type_total[t] += 1
        if node in ground_truth:
            type_labeled[t] += 1

    print("\nNode type coverage (labeled / total in graph):")
    print("─" * 50)
    for t in ["CFG", "ENV", "AGENT", "SEQ", "SCBD", "RAL", "COV", "SVA", "OTHER"]:
        total   = type_total[t]
        labeled = type_labeled[t]
        pct     = 100 * labeled / total if total else 0.0
        flag    = " ← LOW" if pct < 15 and total > 5 else ""
        print(f"  {t:<8}  {labeled:>4} / {total:<4}  ({pct:>5.1f}%){flag}")


def print_distribution_vs_paper(ground_truth: dict) -> None:
    cat_counter: Counter = Counter()
    for data in ground_truth.values():
        for cat in data["categories"]:
            cat_counter[cat] += 1

    total_labeled = len(ground_truth)

    print("\nDistribution comparison — paper Table 2 vs ground_truth.json")
    print("─" * 66)
    print(f"  {'Category':<28} {'Paper %':>8}  {'Our nodes':>9}  {'Our %':>7}")
    print("─" * 66)
    for cat in ALL_CATEGORIES:
        paper_pct = PAPER_PROPORTIONS[cat]
        our_count = cat_counter.get(cat, 0)
        our_pct   = 100 * our_count / total_labeled if total_labeled else 0
        delta     = our_pct - paper_pct
        flag      = " ✓" if abs(delta) < 20 else " ← large gap"
        print(f"  {cat:<28} {paper_pct:>7.1f}%  {our_count:>9}  {our_pct:>6.1f}%{flag}")
    print(f"\n  Total labeled nodes: {total_labeled}")

    multi = sum(1 for d in ground_truth.values() if len(d["categories"]) > 1)
    print(f"  Multi-label nodes (≥2 categories): {multi} ({100*multi/total_labeled:.1f}%)")
    print(f"  Paper multi-label commits: 50.0%")


def print_sample_nodes(ground_truth: dict, node_types: dict, n: int = 3) -> None:
    by_cat: dict[str, list[str]] = defaultdict(list)
    for node, data in ground_truth.items():
        for cat in data["categories"]:
            by_cat[cat].append(node)

    print("\nSample labeled nodes per category (spot-check these in OpenTitan):")
    print("─" * 60)
    for cat in ALL_CATEGORIES:
        nodes = by_cat.get(cat, [])[:n]
        types = [node_types.get(nd, {}).get("type", "?") for nd in nodes]
        pairs = [f"{nd} ({t})" for nd, t in zip(nodes, types)]
        print(f"  {cat:<28}  {', '.join(pairs) if pairs else '— none —'}")


# ── Entrypoint ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    SCRIPTS_DIR       = os.path.dirname(os.path.abspath(__file__))
    BASE_DIR          = os.path.dirname(SCRIPTS_DIR)
    DATA_DIR          = os.path.join(BASE_DIR, "data")
    LLM_LABELED_PATH  = os.path.join(DATA_DIR, "llm_labeled_final.csv")
    COMMITS_PATH      = os.path.join(DATA_DIR, "commits_final.csv")
    NODE_TYPES_PATH   = os.path.join(SCRIPTS_DIR, "node_types.json")
    GT_OUTPUT_PATH    = os.path.join(DATA_DIR, "ground_truth.json")
    INDEX_OUTPUT_PATH = os.path.join(DATA_DIR, "commit_index.json")
    AUDIT_OUTPUT_PATH = os.path.join(DATA_DIR, "audit_report.json")

    print("=" * 60)
    print("VeriDelta  —  ground_truth_mapper.py  v2")
    print("=" * 60)

    for path, label in [
        (LLM_LABELED_PATH, "llm_labeled_final.csv"),
        (COMMITS_PATH,     "commits_final.csv"),
        (NODE_TYPES_PATH,  "node_types.json"),
    ]:
        if not os.path.exists(path):
            print(f"\nMissing: {label}\n  Expected: {path}")
            sys.exit(1)
    print("\nAll input files found.")

    print("\nStep 1: Loading labeled commits")
    print("─" * 40)
    labeled = load_labeled_commits(LLM_LABELED_PATH)
    high_conf = sum(1 for m in labeled.values() if m["confidence"] == "HIGH")
    print(f"  Qualifying commits (non-OTHER): {len(labeled)}")
    print(f"  HIGH confidence:                {high_conf}")
    print(f"  MEDIUM/LOW confidence:          {len(labeled) - high_conf}")

    print("\nStep 2: Loading graph nodes")
    print("─" * 40)
    with open(NODE_TYPES_PATH) as f:
        node_types = json.load(f)
    known_nodes: set[str] = set(node_types.keys())
    type_counts = Counter(v["type"] for v in node_types.values())
    print(f"  Known nodes: {len(known_nodes)}")
    for t in ["CFG", "ENV", "AGENT", "SEQ", "SCBD", "RAL", "COV", "SVA", "OTHER"]:
        print(f"    {t:<8} {type_counts.get(t, 0)}")

    print("\nStep 3: Loading DV file paths from commits_final.csv")
    print("─" * 40)
    dv_files_by_hash = load_dv_files(COMMITS_PATH, set(labeled.keys()))
    print(f"  Commits with DV file data: {len(dv_files_by_hash)}")

    print("\nStep 4: Building ground truth (with pkg heuristics)")
    print("─" * 40)
    ground_truth, commit_index, audit = build_ground_truth(
        labeled, dv_files_by_hash, known_nodes
    )
    print(f"  Commits processed:            {audit['total_commits']}")
    print(f"  Commits with graph matches:   {audit['matched_commits']}")
    print(f"  Total node-label pairs:       {audit['total_node_label_pairs']}")
    print(f"  Unique nodes (raw):           {len(ground_truth)}")

    print("\nStep 4b: Applying type constraints")
    print("─" * 40)
    ground_truth, removed = apply_type_constraints(ground_truth, node_types)
    print(f"  Labels removed (type mismatch): {removed}")
    print(f"  Unique nodes after filtering:   {len(ground_truth)}")

    print("\nStep 5: Full audit")
    print_audit(audit, ground_truth, node_types)

    print("\nStep 6: Distribution vs paper")
    print_distribution_vs_paper(ground_truth)

    print("\nStep 7: Sample nodes per category")
    print_sample_nodes(ground_truth, node_types)

    print("\nStep 8: Saving outputs")
    print("─" * 40)
    with open(GT_OUTPUT_PATH, "w") as f:
        json.dump(ground_truth, f, indent=2)
    print(f"  ground_truth.json  → {GT_OUTPUT_PATH}")

    with open(INDEX_OUTPUT_PATH, "w") as f:
        json.dump(commit_index, f, indent=2)
    print(f"  commit_index.json  → {INDEX_OUTPUT_PATH}")

    with open(AUDIT_OUTPUT_PATH, "w") as f:
        json.dump(audit, f, indent=2)
    print(f"  audit_report.json  → {AUDIT_OUTPUT_PATH}")

    print("\nDone.")
    print("\nNext steps if match rate is still low:")
    print("  1. Rerun type_annotator.py (vseq + cov keywords added)")
    print("  2. Rerun this script with fresh node_types.json")
    print("  3. Check 'uvm_sv_miss' examples — are they in your local repo?")