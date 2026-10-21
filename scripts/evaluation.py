"""
VeriDelta — Evaluation
evaluation.py

Evaluates VeriDelta against three baselines on 80 held-out test commits.

SYSTEMS EVALUATED
─────────────────────────────────────────────────────────────────────
1. VeriDelta (GraphRAG)     Stage 1 + Stage 2 + Stage 3 (full pipeline)
2. LLM without history      Stage 1 + Stage 3, no few-shot retrieval
3. Structural prior only    Stage 1 only, τ(node)→category, no LLM
4. LLM without graph        LLM sees only changed_node name, no subgraph

The last two are the ablation baselines. The comparison demonstrates:
  - Graph structure adds value (VeriDelta > LLM without graph)
  - Commit history adds value (VeriDelta > LLM without history)
  - LLM reasoning adds value (VeriDelta > structural prior)

EVALUATION PROTOCOL
─────────────────────────────────────────────────────────────────────
Split: 80/20 stratified by category.
  Train (retrieval pool): 323 commits
  Test  (evaluation):      80 commits

For each test commit:
  For each changed_node in the commit:
    Run Stage 1 → blast radius
    For each blast radius node that has a ground truth label:
      Compare predicted categories vs ground truth categories
      Record TP / FP / FN per category

Metrics: per-category Precision, Recall, F1 + Macro-F1.
Unit of evaluation: (blast_radius_node, category) pairs.

OUTPUT
─────────────────────────────────────────────────────────────────────
  data/evaluation_results.json  — full per-node results
  data/table1.txt               — paper-ready Table 1 (copy-paste into LaTeX)
"""

import json
import os
import random
import re
import sys
import time
from collections import Counter, defaultdict

import anthropic

# Add scripts dir to path so we can import from graphrag_pipeline
SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPTS_DIR)

from graphrag_pipeline import (
    MODEL,
    VALID_CATEGORIES,
    TYPE_PRIMARY_CATEGORY,
    load_data,
    get_blast_radius,
    build_stage1_context,
    retrieve_similar_commits,
    build_prompt,
    call_llm,
)

# ── Config ────────────────────────────────────────────────────────────────────

TEST_RATIO  = 0.20
RANDOM_SEED = 42
MAX_COMMITS = None    


# ── Data paths ────────────────────────────────────────────────────────────────

BASE_DIR   = os.path.dirname(SCRIPTS_DIR)
DATA_DIR   = os.path.join(BASE_DIR, "data")
GT_PATH    = os.path.join(DATA_DIR, "ground_truth.json")
RESULTS_PATH = os.path.join(DATA_DIR, "evaluation_results.json")
TABLE_PATH   = os.path.join(DATA_DIR, "table1.txt")


# ── Train / test split ────────────────────────────────────────────────────────

def split_commits(
    commit_index: dict,
    test_ratio:   float = TEST_RATIO,
    seed:         int   = RANDOM_SEED,
) -> tuple[set[str], set[str]]:
    """
    80/20 stratified split by primary category.

    Only commits with at least one changed_node are included.
    Stratification ensures each category has representation in the test set.

    Returns (train_hashes, test_hashes).
    """
    eligible = {h: d for h, d in commit_index.items() if d.get("changed_nodes")}

    by_category: dict[str, list[str]] = defaultdict(list)
    for h, d in eligible.items():
        cats = d.get("categories", [])
        primary = cats[0] if cats else "uncategorized"
        by_category[primary].append(h)

    rng = random.Random(seed)
    test_hashes: set[str] = set()
    for cat, hashes in by_category.items():
        rng.shuffle(hashes)
        n_test = max(1, int(len(hashes) * test_ratio))
        test_hashes.update(hashes[:n_test])

    train_hashes = set(eligible.keys()) - test_hashes
    return train_hashes, test_hashes


# ── Baselines ─────────────────────────────────────────────────────────────────

def predict_structural_prior(
    blast_radius: list[tuple[str, int]],
    node_types:   dict,
) -> dict:
    """
    Baseline 1 — Structural prior only.

    Assigns τ(node) → primary DV failure category for each node in the
    blast radius. No LLM call. Represents the lower bound: what we can
    predict from graph structure alone without any language reasoning.
    """
    predictions: dict = {}
    for node, depth in blast_radius:
        ntype   = node_types.get(node, {}).get("type", "OTHER")
        primary = TYPE_PRIMARY_CATEGORY.get(ntype)
        if primary:
            predictions[node] = {
                "categories": [primary],
                "confidence": 0.70,
                "node_type":  ntype,
                "source":     "structural_prior",
            }
    return predictions


def predict_llm_no_graph(
    changed_node:  str,
    ip_block:      str,
    blast_radius:  list[tuple[str, int]],
    node_types:    dict,
    client:        anthropic.Anthropic,
) -> dict:
    """
    Baseline 2 — LLM without graph context.

    The LLM receives only the changed node name and ip_block — no subgraph,
    no type annotations, no few-shot examples. Its predicted categories are
    applied uniformly to all blast radius nodes.

    Tests whether subgraph context in the prompt adds value vs. a naive
    LLM-only approach. The blast radius itself is still computed from the
    graph (Stage 1), since without it there are no nodes to evaluate.
    """
    ntype   = node_types.get(changed_node, {}).get("type", "OTHER")
    primary = TYPE_PRIMARY_CATEGORY.get(ntype, "Interface Drift")

    prompt = (
        f"You are a UVM verification expert.\n"
        f"A DV engineer changed component: {changed_node}\n"
        f"  Component type: {ntype}\n"
        f"  IP block: {ip_block}\n\n"
        f"Which DV failure categories will be at risk in downstream components?\n"
        f"Valid categories (use exact strings only):\n"
        f"  {', '.join(VALID_CATEGORIES)}\n\n"
        f"Output ONLY valid JSON — no markdown, no explanation:\n"
        f'{{"categories": ["Category1"], "confidence": 0.85}}'
    )

    try:
        message = client.messages.create(
            model     = MODEL,
            max_tokens= 256,
            messages  = [{"role": "user", "content": prompt}],
        )
        raw  = message.content[0].text.strip()
        raw  = re.sub(r"^```(?:json)?\s*", "", raw)
        raw  = re.sub(r"\s*```$",          "", raw)
        parsed = json.loads(raw)
        cats = [c for c in parsed.get("categories", []) if c in VALID_CATEGORIES]
        conf = float(parsed.get("confidence", 0.5))
    except Exception:
        cats = [primary]
        conf = 0.50

    if not cats:
        cats = [primary]

    # Apply uniform prediction to all blast radius nodes
    predictions: dict = {}
    for node, depth in blast_radius:
        ntype_node = node_types.get(node, {}).get("type", "OTHER")
        predictions[node] = {
            "categories": cats,
            "confidence": max(0.30, conf - depth * 0.04),
            "node_type":  ntype_node,
            "source":     "llm_no_graph",
        }
    return predictions


def predict_llm_no_history(
    changed_node: str,
    ip_block:     str,
    blast_radius: list[tuple[str, int]],
    node_types:   dict,
    bind_at_risk: list[str],
    client:       anthropic.Anthropic,
) -> dict:
    """
    Baseline 3 — LLM without commit history (no Stage 2 retrieval).

    Full graph context is given in the prompt but no historical few-shot
    examples from commit_index. Tests whether commit history retrieval
    adds value beyond graph structure alone.
    """
    stage1 = build_stage1_context(
        changed_node, blast_radius, node_types, bind_at_risk
    )
    prompt     = build_prompt(stage1, [], ip_block, node_types)   # empty few-shot list
    llm_output = call_llm(prompt, client)

    predictions: dict = {}
    for item in stage1["blast_radius"]:
        node  = item["node"]
        ntype = item["type"]
        if node in llm_output:
            predictions[node] = {
                **llm_output[node],
                "node_type": ntype,
                "source":    "llm_no_history",
            }
        elif item["primary"]:
            predictions[node] = {
                "categories": [item["primary"]],
                "confidence": 0.40,
                "node_type":  ntype,
                "source":     "structural_prior_fallback",
            }
    return predictions


# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_metrics(eval_records: list[dict]) -> dict:
    """
    Compute per-category and macro Precision / Recall / F1.

    eval_records: list of {predicted: [cats], actual: [cats]}

    Multi-label: for each (node, category), independently assess TP/FP/FN.
    """
    tp: Counter = Counter()
    fp: Counter = Counter()
    fn: Counter = Counter()

    for rec in eval_records:
        pred   = set(rec["predicted"])
        actual = set(rec["actual"])
        for cat in VALID_CATEGORIES:
            in_pred   = cat in pred
            in_actual = cat in actual
            if in_pred and in_actual:
                tp[cat] += 1
            elif in_pred and not in_actual:
                fp[cat] += 1
            elif not in_pred and in_actual:
                fn[cat] += 1

    metrics: dict = {}
    for cat in VALID_CATEGORIES:
        denom_p  = tp[cat] + fp[cat]
        denom_r  = tp[cat] + fn[cat]
        p  = tp[cat] / denom_p if denom_p > 0 else 0.0
        r  = tp[cat] / denom_r if denom_r > 0 else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        metrics[cat] = {
            "precision": round(p,  4),
            "recall":    round(r,  4),
            "f1":        round(f1, 4),
            "support":   tp[cat] + fn[cat],
            "tp": tp[cat], "fp": fp[cat], "fn": fn[cat],
        }

    # Macro average over categories with at least one support sample
    active = [m for m in metrics.values() if m["support"] > 0]
    if active:
        macro_f1 = sum(m["f1"]        for m in active) / len(active)
        macro_p  = sum(m["precision"] for m in active) / len(active)
        macro_r  = sum(m["recall"]    for m in active) / len(active)
    else:
        macro_f1 = macro_p = macro_r = 0.0

    metrics["macro"] = {
        "precision": round(macro_p,  4),
        "recall":    round(macro_r,  4),
        "f1":        round(macro_f1, 4),
    }
    return metrics


# ── Evaluation loop ───────────────────────────────────────────────────────────

def run_evaluation(
    test_hashes:   set[str],
    train_hashes:  set[str],
    data:          dict,
    ground_truth:  dict,
    client:        anthropic.Anthropic,
    max_commits:   int | None = None,
) -> dict:
    """
    Run all four systems on every test commit and collect evaluation records.

    Returns dict: {system_name: [eval_records]}
    """
    adjacency    = data["adjacency"]
    node_types   = data["node_types"]
    bind_index   = data["bind_index"]
    commit_index = data["commit_index"]
    gt_nodes     = set(ground_truth.keys())

    # Training commit pool — exclude test commits from retrieval
    train_commit_index = {
        h: d for h, d in commit_index.items()
        if h in train_hashes
    }

    systems = ["graphrag", "llm_no_history", "structural_prior", "llm_no_graph"]
    records: dict[str, list] = {s: [] for s in systems}
    skipped = 0

    test_list = sorted(test_hashes)
    if max_commits:
        test_list = test_list[:max_commits]

    print(f"\n  Evaluating {len(test_list)} test commits across 4 systems...")
    print(f"  {'Commit':<12} {'Nodes':>6} {'GT hits':>8} {'Status'}")
    print("  " + "─" * 50)

    for idx, chash in enumerate(test_list, 1):
        commit = commit_index[chash]
        changed_nodes = commit.get("changed_nodes", [])
        ip_block      = commit.get("ip_block", "unknown")

        commit_gt_hits = 0

        for changed_node in changed_nodes:
            # Stage 1: blast radius (shared across all systems)
            blast_radius = get_blast_radius(changed_node, adjacency)
            if not blast_radius:
                skipped += 1
                continue

            bind_at_risk = []  # RTL module not known from commit_index

            # Find GT nodes in blast radius
            gt_in_blast = [(n, d) for n, d in blast_radius if n in gt_nodes]
            if not gt_in_blast:
                continue

            commit_gt_hits += len(gt_in_blast)
            actual_by_node = {n: ground_truth[n]["categories"] for n, _ in gt_in_blast}

            # ── System 1: VeriDelta (full GraphRAG) ───────────────────────────
            similar = retrieve_similar_commits(
                changed_node, node_types.get(changed_node, {}).get("type", "OTHER"),
                ip_block, train_commit_index, node_types
            )
            stage1 = build_stage1_context(
                changed_node, blast_radius, node_types, bind_at_risk
            )
            prompt = build_prompt(stage1, similar, ip_block, node_types)
            llm_out = call_llm(prompt, client)
            # Merge with structural fallback
            graphrag_preds: dict = {}
            for item in stage1["blast_radius"]:
                n, ntype = item["node"], item["type"]
                if n in llm_out:
                    graphrag_preds[n] = llm_out[n]["categories"]
                elif item["primary"]:
                    graphrag_preds[n] = [item["primary"]]

            # ── System 2: LLM without history ─────────────────────────────────
            nohist_preds_raw = predict_llm_no_history(
                changed_node, ip_block, blast_radius, node_types, bind_at_risk, client
            )
            nohist_preds = {n: v["categories"] for n, v in nohist_preds_raw.items()}

            # ── System 3: Structural prior ─────────────────────────────────────
            prior_raw = predict_structural_prior(blast_radius, node_types)
            prior_preds = {n: v["categories"] for n, v in prior_raw.items()}

            # ── System 4: LLM without graph ────────────────────────────────────
            nograph_raw = predict_llm_no_graph(
                changed_node, ip_block, blast_radius, node_types, client
            )
            nograph_preds = {n: v["categories"] for n, v in nograph_raw.items()}

            # ── Record results for GT nodes ────────────────────────────────────
            preds_by_system = {
                "graphrag":        graphrag_preds,
                "llm_no_history":  nohist_preds,
                "structural_prior": prior_preds,
                "llm_no_graph":    nograph_preds,
            }

            for node, actual in actual_by_node.items():
                for sys_name, preds in preds_by_system.items():
                    predicted = preds.get(node, [])
                    records[sys_name].append({
                        "commit":       chash[:10],
                        "changed_node": changed_node,
                        "node":         node,
                        "node_type":    node_types.get(node, {}).get("type", "OTHER"),
                        "predicted":    predicted,
                        "actual":       actual,
                        "ip_block":     ip_block,
                    })

        status = "ok" if commit_gt_hits > 0 else "no GT"
        print(f"  {chash[:10]}  {len(changed_nodes):>6}  {commit_gt_hits:>8}  {status}")

    print(f"\n  Skipped (no blast radius): {skipped}")
    return records


# ── Output formatting ─────────────────────────────────────────────────────────

def print_results(all_metrics: dict[str, dict]) -> None:
    """Print Table 1 to stdout."""
    CAT_SHORT = {
        "Interface Drift":       "IF",
        "Sequence Invalidation": "SI",
        "Checker Desync":        "CD",
        "RAL Desync":            "RD",
        "Coverage Regression":   "CR",
        "Assertion Orphaning":   "AO",
    }
    SYSTEM_LABELS = {
        "graphrag":         "VeriDelta (GraphRAG)",
        "llm_no_history":   "LLM — no commit history",
        "structural_prior": "Structural prior only",
        "llm_no_graph":     "LLM — no graph context",
    }

    cats = [c for c in VALID_CATEGORIES if c != "Assertion Orphaning"]
    header = f"{'System':<26} " + " ".join(f"{CAT_SHORT[c]:>6}" for c in cats) + f"  {'Macro':>6}"
    sep = "─" * len(header)

    lines = [
        "Table 1: VeriDelta category-level F1 vs baselines",
        "(IF=Interface Drift, SI=Sequence Invalidation, CD=Checker Desync,",
        " RD=RAL Desync, CR=Coverage Regression | Macro = avg over active categories)",
        "",
        sep, header, sep,
    ]

    for sys_name in ["graphrag", "llm_no_history", "structural_prior", "llm_no_graph"]:
        if sys_name not in all_metrics:
            continue
        m = all_metrics[sys_name]
        row = f"{SYSTEM_LABELS[sys_name]:<26} "
        for cat in cats:
            f1 = m.get(cat, {}).get("f1", 0.0)
            row += f"{f1:>6.2f} "
        macro = m.get("macro", {}).get("f1", 0.0)
        row += f"  {macro:>6.2f}"
        lines.append(row)

    lines += [sep, ""]

    # Per-system detail
    for sys_name in ["graphrag", "llm_no_history", "structural_prior", "llm_no_graph"]:
        if sys_name not in all_metrics:
            continue
        m = all_metrics[sys_name]
        lines.append(f"\n{SYSTEM_LABELS[sys_name]} — Precision / Recall / F1 / Support")
        lines.append("─" * 58)
        for cat in cats:
            cm = m.get(cat, {})
            p, r, f1, sup = cm.get("precision",0), cm.get("recall",0), cm.get("f1",0), cm.get("support",0)
            lines.append(f"  {cat:<28} P={p:.2f}  R={r:.2f}  F1={f1:.2f}  n={sup}")
        macro = m.get("macro", {})
        lines.append(f"  {'MACRO':<28} P={macro.get('precision',0):.2f}  R={macro.get('recall',0):.2f}  F1={macro.get('f1',0):.2f}")

    output = "\n".join(lines)
    print(output)
    return output


# ── Entrypoint ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("VeriDelta  —  evaluation.py")
    print("=" * 60)

    # ── Load everything ───────────────────────────────────────────────────────
    print("\nLoading data files...")
    data = load_data(SCRIPTS_DIR, DATA_DIR)

    if not os.path.exists(GT_PATH):
        print(f"Missing: {GT_PATH}")
        sys.exit(1)
    with open(GT_PATH) as f:
        ground_truth = json.load(f)

    print(f"  Graph nodes:       {len(data['node_types'])}")
    print(f"  Commit index:      {len(data['commit_index'])} commits")
    print(f"  Ground truth:      {len(ground_truth)} labeled nodes")

    # ── Split ─────────────────────────────────────────────────────────────────
    print("\nSplitting commits (80/20 stratified)...")
    train_hashes, test_hashes = split_commits(data["commit_index"])
    print(f"  Train: {len(train_hashes)} commits (retrieval pool)")
    print(f"  Test:  {len(test_hashes)} commits (evaluation set)")

    if MAX_COMMITS:
        print(f"\n  Note: capped at {MAX_COMMITS} commits for this run")

    # Cost estimate
    n_eval = min(len(test_hashes), MAX_COMMITS or len(test_hashes))
    print(f"\n  Estimated LLM calls: ~{n_eval * 3} (3 systems × {n_eval} commits)")
    print(f"  Estimated cost: ~${n_eval * 3 * 0.001:.2f} (Claude Haiku)")

    # ── API client ────────────────────────────────────────────────────────────
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("\nError: ANTHROPIC_API_KEY not set.")
        sys.exit(1)
    client = anthropic.Anthropic(api_key=api_key)

    # ── Run evaluation ────────────────────────────────────────────────────────
    print("\nRunning evaluation...")
    t_start = time.time()

    records = run_evaluation(
        test_hashes, train_hashes, data, ground_truth, client, MAX_COMMITS
    )

    elapsed = time.time() - t_start
    print(f"\n  Completed in {elapsed:.0f}s")

    # ── Compute metrics ───────────────────────────────────────────────────────
    print("\nComputing metrics...")
    all_metrics: dict[str, dict] = {}
    for sys_name, recs in records.items():
        if recs:
            all_metrics[sys_name] = compute_metrics(recs)
            n = len(recs)
            macro_f1 = all_metrics[sys_name]["macro"]["f1"]
            print(f"  {sys_name:<20} {n:>5} eval points  macro-F1={macro_f1:.3f}")

    # ── Print Table 1 ─────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    table_str = print_results(all_metrics)

    # ── Save ──────────────────────────────────────────────────────────────────
    results_out = {
        "config": {
            "test_ratio": TEST_RATIO,
            "seed":       RANDOM_SEED,
            "n_test":     len(test_hashes),
            "n_train":    len(train_hashes),
            "max_commits": MAX_COMMITS,
            "model":      MODEL,
        },
        "metrics":  all_metrics,
        "records":  records,
    }

    with open(RESULTS_PATH, "w") as f:
        json.dump(results_out, f, indent=2)
    print(f"\nFull results → {RESULTS_PATH}")

    with open(TABLE_PATH, "w", encoding="utf-8") as f:
        f.write(table_str)
    print(f"Table 1      → {TABLE_PATH}")

    print("\nDone. table1.txt is ready for the paper.")