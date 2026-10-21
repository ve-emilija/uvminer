"""
VeriDelta — GraphRAG Pipeline
graphrag_pipeline.py

Three-stage pipeline for category-aware blast radius prediction.

STAGE 1 — Structural subgraph retrieval
  Given a changed UVM component name, traverse the typed dependency graph
  (graph_edges.json) to identify all downstream nodes (blast radius).
  Each node is annotated with its type τ and primary DV failure category
  from node_types.json. Assertion Orphaning is handled separately via
  bind_index.json when the changed file is an RTL module.

STAGE 2 — Historical commit retrieval
  From commit_index.json (602 labeled commits), retrieve the top-k most
  similar historical commits using a scoring function over:
    - ip_block match         (+3)
    - changed node exact match (+3)
    - same node type match   (+2)
    - category overlap       (+1 each)
  These form the few-shot examples in the LLM prompt.

STAGE 3 — LLM category inference
  Claude Haiku receives the typed blast radius + few-shot examples and
  outputs per-node category predictions with confidence scores as JSON.
  Model: claude-haiku-4-5-20251001
  (same model validated at κ=0.673 in the companion empirical study)

INPUTS
  scripts/graph_edges.json   — adjacency list (type_annotator.py)
  scripts/node_types.json    — typed nodes   (type_annotator.py)
  scripts/bind_index.json    — RTL→bind map  (type_annotator.py)
  data/commit_index.json     — 602 commits   (ground_truth_mapper.py)

OUTPUT (per prediction call)
  {
    "changed_node":  "uart_scoreboard",
    "ip_block":      "uart",
    "blast_radius":  ["uart_env", "uart_base_vseq", ...],
    "bind_at_risk":  [],
    "predictions": {
      "uart_env": {
        "categories":  ["Interface Drift"],
        "confidence":  0.91,
        "node_type":   "ENV"
      },
      ...
    },
    "model": "claude-haiku-4-5-20251001",
    "few_shot_commits_used": 3
  }
"""

import json
import os
import re
import sys
import argparse
import subprocess
from collections import defaultdict
import time

import anthropic

# ── Constants ─────────────────────────────────────────────────────────────────

MODEL           = "claude-haiku-4-5-20251001"
MAX_FEW_SHOT    = 5      # top-k similar commits for few-shot context
MAX_BLAST_NODES = 60     # cap blast radius in prompt; prioritise by graph depth

VALID_CATEGORIES = [
    "Interface Drift",
    "Sequence Invalidation",
    "Checker Desync",
    "Assertion Orphaning",
    "RAL Desync",
    "Coverage Regression",
]

# Primary category per node type — structural prior before LLM inference
TYPE_PRIMARY_CATEGORY = {
    "CFG":   "Interface Drift",
    "ENV":   "Interface Drift",
    "AGENT": "Interface Drift",
    "SEQ":   "Sequence Invalidation",
    "SCBD":  "Checker Desync",
    "RAL":   "RAL Desync",
    "COV":   "Coverage Regression",
    "SVA":   "Assertion Orphaning",
    "OTHER": None,
}


# ── Data loading ──────────────────────────────────────────────────────────────

def load_data(
    scripts_dir: str,
    data_dir: str,
    require_commit_index: bool = True,
) -> dict:
    """
    Load all four JSON files needed by the pipeline.
    Returns a single dict with keys: adjacency, node_types, bind_index, commit_index.
    """
    paths = {
        "adjacency":    os.path.join(data_dir, "graph_edges.json"),
        "node_types":   os.path.join(data_dir, "node_types.json"),
        "bind_index":   os.path.join(data_dir, "bind_index.json"),
        "commit_index": os.path.join(data_dir,    "commit_index.json"),
    }
    data = {}
    for key, path in paths.items():
        if not os.path.exists(path):
            if key == "commit_index" and not require_commit_index:
                data[key] = {}
                continue
            print(f"Missing: {path}")
            sys.exit(1)
        with open(path, encoding="utf-8") as f:
            data[key] = json.load(f)
    return data


# ── Stage 1: Structural subgraph retrieval ────────────────────────────────────

def get_blast_radius(
    changed_node: str,
    adjacency:    dict[str, list[str]],
    max_nodes:    int = MAX_BLAST_NODES,
) -> list[tuple[str, int]]:
    """
    BFS traversal of the dependency graph from changed_node.

    Returns a list of (node_name, depth) tuples, sorted by depth then name.
    Depth = number of inheritance hops from the changed node.
    Capped at max_nodes to keep the LLM prompt manageable.
    """
    if changed_node not in adjacency:
        return []

    visited: dict[str, int] = {}   # node → depth
    queue   = [(changed_node, 0)]

    while queue and len(visited) < max_nodes:
        node, depth = queue.pop(0)
        if node in visited:
            continue
        visited[node] = depth
        for child in adjacency.get(node, []):
            if child not in visited:
                queue.append((child, depth + 1))

    # Return children only (exclude the changed node itself)
    result = [(n, d) for n, d in visited.items() if n != changed_node]
    return sorted(result, key=lambda x: (x[1], x[0]))


def get_bind_at_risk(
    rtl_module_stem: str,
    bind_index:      dict[str, list[str]],
) -> list[str]:
    """
    Given an RTL module name (stem of changed RTL file), return any
    bind files that reference it — Assertion Orphaning candidates.
    """
    return bind_index.get(rtl_module_stem, [])


def build_stage1_context(
    changed_node:    str,
    blast_radius:    list[tuple[str, int]],
    node_types:      dict,
    bind_at_risk:    list[str],
) -> dict:
    """
    Annotate each blast radius node with its type and primary category.
    Returns structured context dict for Stage 3 prompt construction.
    """
    annotated = []
    for node, depth in blast_radius:
        info     = node_types.get(node, {})
        ntype    = info.get("type", "OTHER")
        primary  = TYPE_PRIMARY_CATEGORY.get(ntype)
        annotated.append({
            "node":     node,
            "depth":    depth,
            "type":     ntype,
            "primary":  primary,
        })

    changed_info = node_types.get(changed_node, {})
    return {
        "changed_node":  changed_node,
        "changed_type":  changed_info.get("type", "OTHER"),
        "blast_radius":  annotated,
        "bind_at_risk":  bind_at_risk,
    }


# ── Stage 2: Historical commit retrieval ─────────────────────────────────────

def score_commit(
    commit:       dict,
    changed_node: str,
    node_type:    str,
    ip_block:     str,
    node_types:   dict,
) -> int:
    """
    Score a historical commit for similarity to the current prediction task.

    Scoring:
      +3 same ip_block
      +3 changed_node appears in this commit's changed_nodes
      +2 any changed node in this commit has same type as changed_node
      +1 per shared category (max 3)
    """
    score = 0

    if commit.get("ip_block") == ip_block:
        score += 3

    changed_nodes = commit.get("changed_nodes", [])
    if changed_node in changed_nodes:
        score += 3

    for n in changed_nodes:
        if node_types.get(n, {}).get("type") == node_type:
            score += 2
            break

    primary = TYPE_PRIMARY_CATEGORY.get(node_type)
    if primary and primary in commit.get("categories", []):
        score += 1

    return score


def retrieve_similar_commits(
    changed_node:  str,
    node_type:     str,
    ip_block:      str,
    commit_index:  dict,
    node_types:    dict,
    k:             int = MAX_FEW_SHOT,
) -> list[dict]:
    """
    Return the top-k most similar historical commits from commit_index.

    Only commits with at least one changed_node (i.e. commits that touched
    graph nodes) are candidates, since they provide concrete node-level context.
    """
    scored = []
    for chash, commit in commit_index.items():
        if not commit.get("changed_nodes"):
            continue
        s = score_commit(commit, changed_node, node_type, ip_block, node_types)
        if s > 0:
            scored.append((s, chash, commit))

    scored.sort(key=lambda x: -x[0])

    # top-3 by score, then fill with commits bringing unseen categories
    selected  = scored[:3]
    seen_cats = set()
    for _, _, c in selected:
        seen_cats.update(c.get("categories", []))

    for s, h, c in scored[3:]:
        if len(selected) >= k:
            break
        if set(c.get("categories", [])) - seen_cats:
            selected.append((s, h, c))
            seen_cats.update(c.get("categories", []))

    if len(selected) < k:                      # pad if short
        chosen = {h for _, h, _ in selected}
        for s, h, c in scored:
            if len(selected) >= k:
                break
            if h not in chosen:
                selected.append((s, h, c))

    return [{"hash": h[:10], **c} for _, h, c in selected]


# ── Stage 3: LLM category inference ──────────────────────────────────────────

def build_prompt(
    stage1:          dict,
    similar_commits: list[dict],
    ip_block:        str,
    node_types:      dict,       
) -> str:
    """
    Construct the structured prompt for Claude Haiku.

    Structure:
      1. System context — what VeriDelta is
      2. Current task — changed node + blast radius with types
      3. Few-shot examples — similar historical commits
      4. Output specification — strict JSON format
    """
    changed_node  = stage1["changed_node"]
    changed_type  = stage1["changed_type"]
    changed_prim  = TYPE_PRIMARY_CATEGORY.get(changed_type, "unknown")
    blast_radius  = stage1["blast_radius"]
    bind_at_risk  = stage1["bind_at_risk"]

    lines = [
        "You are VeriDelta, an AI system for UVM verification change impact analysis.",
        "Your task: given a changed UVM component, predict which DV failure categories",
        "will manifest in each downstream component in the blast radius.",
        "",
        f"CHANGED COMPONENT: {changed_node}",
        f"  Type: {changed_type} | Primary risk: {changed_prim}",
        f"  IP block: {ip_block}",
        "",
    ]

    # Blast radius
    lines.append(f"BLAST RADIUS ({len(blast_radius)} downstream components):")
    for item in blast_radius:
        prim = item["primary"] or "—"
        lines.append(
            f"  depth={item['depth']}  {item['node']:<45} "
            f"type={item['type']:<8} structural_prior={prim}"
        )

    # Bind-level Assertion Orphaning
    if bind_at_risk:
        lines.append("")
        lines.append("BIND FILES AT RISK (Assertion Orphaning):")
        for bf in bind_at_risk:
            lines.append(f"  {bf}")

    # Few-shot examples
    if similar_commits:
        lines.append("")
        lines.append(f"HISTORICAL EXAMPLES (top-{len(similar_commits)} similar commits):")
        for i, c in enumerate(similar_commits, 1):
            nodes_str    = ", ".join(f"{n} ({node_types.get(n, {}).get('type', '?')})" for n in c.get("changed_nodes", [])[:5]) or "—"
            patterns_str = ", ".join(c.get("at_risk_patterns", [])[:4]) or "—"
            cats_str     = ", ".join(c.get("categories", []))
            lines.append(
                f"  Example {i} [{c['hash']}] ip={c.get('ip_block','?')} "
                f"cats=[{cats_str}]"
            )
            lines.append(f"    changed: {nodes_str}")
            lines.append(f"    at_risk: {patterns_str}")

    # Output spec
    lines += [
        "",
        "VALID CATEGORIES (use exact strings):",
        "  Interface Drift, Sequence Invalidation, Checker Desync,",
        "  Assertion Orphaning, RAL Desync, Coverage Regression",
        "",
        "OUTPUT: Respond with ONLY a valid JSON object. No explanation, no markdown.",
        "Format:",
        '{',
        '  "node_name": {"categories": ["Category1"], "confidence": 0.85},',
        '  ...',
        '}',
        "",
        "Rules:",
        "- Include every node from the blast radius list above.",
        "- Categories must be from the valid list only.",
        "- Confidence: 0.0–1.0 (use lower values when type mismatch or uncertain).",
        "- A node may have multiple categories if the commit context supports it.",
        "- Use the structural_prior as your baseline; adjust based on examples.",
        "- Categories in the historical examples are commit-level: they are split",
        "  across the changed nodes by component type, not applied to every node.",
        "  Assign each blast radius node only categories consistent with its type.",
    ]

    return "\n".join(lines)


def call_llm(prompt: str, client: anthropic.Anthropic) -> dict:
    """
    Call Claude Haiku and parse the JSON response.

    Returns parsed dict of {node_name: {categories, confidence}}.
    Falls back to empty dict on parse failure.
    """
    message = client.messages.create(
        model=MODEL,
        max_tokens=4096,
        messages=[{"role": "user", "content": prompt}],
    )

    raw = message.content[0].text.strip()

    # Strip markdown code fences if present
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        # Try to extract JSON object from response
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            try:
                parsed = json.loads(match.group())
            except json.JSONDecodeError:
                print(f"  Warning: LLM response not valid JSON. Raw:\n{raw[:300]}")
                return {}
        else:
            print(f"  Warning: No JSON found in LLM response. Raw:\n{raw[:300]}")
            return {}

    # Validate and normalise each entry
    clean: dict = {}
    for node, data in parsed.items():
        if not isinstance(data, dict):
            continue
        cats = [c for c in data.get("categories", []) if c in VALID_CATEGORIES]
        conf = float(data.get("confidence", 0.5))
        conf = max(0.0, min(1.0, conf))
        if cats:
            clean[node] = {"categories": cats, "confidence": conf}

    return clean


# ── Main prediction function ──────────────────────────────────────────────────

def predict(
    changed_node: str,
    ip_block:     str,
    data:         dict,
    client:       anthropic.Anthropic | None,
    rtl_module:   str | None = None,
    verbose:      bool = False,
) -> dict:
    """
    Run the full three-stage VeriDelta pipeline for a single changed node.

    Args:
        changed_node: Name of the UVM class that was modified
        ip_block:     OpenTitan IP block name (e.g. 'uart', 'aes')
        data:         Loaded data dict from load_data()
        client:       Anthropic client instance
        rtl_module:   Optional RTL module name (stem of changed RTL file)
                      If provided, bind_index is checked for Assertion Orphaning
        verbose:      Print stage progress

    Returns:
        Prediction result dict (see module docstring for structure)
    """
    adjacency    = data["adjacency"]
    node_types   = data["node_types"]
    bind_index   = data["bind_index"]
    commit_index = data["commit_index"]

    # ── Stage 1 ───────────────────────────────────────────────────────────────
    if verbose:
        print(f"\n[Stage 1] Computing blast radius from '{changed_node}'...")

    blast_radius = get_blast_radius(changed_node, adjacency)
    bind_at_risk = get_bind_at_risk(rtl_module, bind_index) if rtl_module else []
    stage1       = build_stage1_context(
        changed_node, blast_radius, node_types, bind_at_risk
    )

    if verbose:
        print(f"  Blast radius: {len(blast_radius)} nodes")
        if bind_at_risk:
            print(f"  Bind files at risk: {bind_at_risk}")

    if not blast_radius:
        if verbose:
            print(f"  '{changed_node}' not in graph or has no descendants.")
        bind_predictions = {
            bind_file: {
                "categories": ["Assertion Orphaning"],
                "confidence": 0.95,
                "node_type": "SVA",
                "source": "bind_index",
            }
            for bind_file in bind_at_risk
        }
        return {
            "changed_node": changed_node,
            "ip_block":     ip_block,
            "blast_radius": [],
            "bind_at_risk": bind_at_risk,
            "predictions":  bind_predictions,
            "model":        MODEL if client else None,
            "few_shot_commits_used": 0,
        }

    # ── Stage 2 ───────────────────────────────────────────────────────────────
    changed_type = node_types.get(changed_node, {}).get("type", "OTHER")

    if verbose:
        print(f"\n[Stage 2] Retrieving similar commits (changed_type={changed_type})...")

    similar_commits = []
    if client:
        similar_commits = retrieve_similar_commits(
            changed_node, changed_type, ip_block, commit_index, node_types
        )

    if verbose:
        print(f"  Retrieved {len(similar_commits)} similar commits")
        for c in similar_commits:
            print(f"    {c['hash']} [{c.get('ip_block','?')}] "
                  f"→ {c.get('categories', [])}")

    # ── Stage 3 ───────────────────────────────────────────────────────────────
    llm_output = {}
    if client:
        if verbose:
            print(f"\n[Stage 3] Calling {MODEL}...")
        prompt = build_prompt(stage1, similar_commits, ip_block, node_types)
        llm_output = call_llm(prompt, client)

    if verbose:
        print(f"  LLM returned predictions for {len(llm_output)} nodes")

    # Merge LLM predictions with structural priors
    # If LLM didn't predict a node, fall back to the structural prior
    predictions: dict = {}
    for item in stage1["blast_radius"]:
        node  = item["node"]
        ntype = item["type"]

        if node in llm_output:
            predictions[node] = {
                **llm_output[node],
                "node_type": ntype,
                "source":    "llm",
            }
        elif item["primary"]:
            # Structural fallback — lower confidence
            predictions[node] = {
                "categories": [item["primary"]],
                "confidence": 0.40,
                "node_type":  ntype,
                "source":     "structural_prior",
            }

    # Add Assertion Orphaning for bind files (from Stage 1 bind index)
    for bf in bind_at_risk:
        predictions[bf] = {
            "categories": ["Assertion Orphaning"],
            "confidence": 0.95,
            "node_type":  "SVA",
            "source":     "bind_index",
        }

    return {
        "changed_node":          changed_node,
        "ip_block":              ip_block,
        "blast_radius":          [n for n, _ in blast_radius],
        "bind_at_risk":          bind_at_risk,
        "predictions":           predictions,
        "model":                 MODEL if client else None,
        "few_shot_commits_used": len(similar_commits),
    }


def git_output(repo: str, *args: str) -> str:
    result = subprocess.run(
        ["git", *args], cwd=repo, capture_output=True, text=True, check=True
    )
    return result.stdout.strip()


def changed_classes(repo: str, commit: str, known_nodes: set[str]) -> tuple[list[str], list[str]]:
    resolved_commit = git_output(repo, "rev-parse", "--verify", f"{commit}^{{commit}}")
    changed_paths = git_output(
        repo, "diff-tree", "--root", "--no-commit-id", "-r", "--name-only", resolved_commit
    ).splitlines()
    parent = None
    try:
        parent = git_output(repo, "rev-parse", f"{resolved_commit}^")
    except subprocess.CalledProcessError:
        pass

    class_pattern = re.compile(r"\bclass\s+(\w+)\s+extends\b", re.IGNORECASE)
    found_nodes = set()
    for path in changed_paths:
        if not path.lower().endswith((".sv", ".svh")):
            continue
        revisions = [resolved_commit]
        if parent:
            revisions.append(parent)
        for revision in revisions:
            try:
                source = git_output(repo, "show", f"{revision}:{path}")
            except subprocess.CalledProcessError:
                continue
            found_nodes.update(
                name for name in class_pattern.findall(source) if name in known_nodes
            )

    return sorted(found_nodes), changed_paths


def commit_ip_block(paths: list[str]) -> str:
    for path in paths:
        parts = path.replace("\\", "/").split("/")
        if len(parts) > 2 and parts[0:2] == ["hw", "ip"]:
            return parts[2]
    return ""


def main() -> None:
    scripts_dir = os.path.dirname(os.path.abspath(__file__))
    default_data_dir = os.path.join(os.path.dirname(scripts_dir), "data")
    parser = argparse.ArgumentParser(
        description="Analyze a Git commit's UVM change impact with VeriDelta."
    )
    parser.add_argument("--repo", required=True, help="Path to a local Git repository")
    parser.add_argument("--commit", required=True, help="Commit hash or Git ref to analyze")
    parser.add_argument(
        "--data-dir", default=default_data_dir,
        help="Directory containing graph_edges.json, node_types.json, bind_index.json, and commit_index.json",
    )
    parser.add_argument("--llm", action="store_true", help="Add category inference using Claude")
    parser.add_argument("--output", default="-", help="JSON output path (default: stdout)")
    args = parser.parse_args()

    repo = os.path.abspath(args.repo)
    if not os.path.isdir(repo):
        parser.error(f"Repository path does not exist: {repo}")

    data = load_data(
        scripts_dir,
        os.path.abspath(args.data_dir),
        require_commit_index=args.llm,
    )
    nodes, paths = changed_classes(repo, args.commit, set(data["node_types"]))
    rtl_modules = sorted({
        os.path.splitext(os.path.basename(path))[0]
        for path in paths
        if path.lower().endswith((".sv", ".svh", ".v"))
        and os.path.splitext(os.path.basename(path))[0] in data["bind_index"]
    })

    client = None
    if args.llm:
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            parser.error("--llm requires the ANTHROPIC_API_KEY environment variable.")
        client = anthropic.Anthropic(api_key=api_key)

    block = commit_ip_block(paths)
    analyses = [
        predict(node, block, data, client=client, verbose=False)
        for node in nodes
    ]
    analyses.extend(
        predict(module, block, data, client=client, rtl_module=module, verbose=False)
        for module in rtl_modules
    )
    report = {
        "repository": repo,
        "commit": git_output(repo, "rev-parse", "--verify", f"{args.commit}^{{commit}}"),
        "changed_files": paths,
        "changed_nodes": nodes,
        "changed_rtl_modules": rtl_modules,
        "analyses": analyses,
    }
    rendered = json.dumps(report, indent=2)
    if args.output == "-":
        print(rendered)
    else:
        with open(args.output, "w", encoding="utf-8") as output_file:
            output_file.write(rendered + "\n")


if __name__ == "__main__":
    main()