"""
VeriDelta — Type Annotator
type_annotator.py

Assigns component type τ(v) to every node in the UVM dependency graph.
Seven types map directly to the six DV failure categories established by
the companion empirical study.

τ → DV failure category
───────────────────────────────────────────────────────────────
CFG    _env_cfg, _cfg, _env_pkg            Interface Drift
ENV    _env, _base_env                     Interface Drift
AGENT  _agent, _driver, _monitor, _if      Interface Drift
SEQ    _vseq, _seq, _seq_item, _sequencer  Sequence Invalidation
SCBD   _scoreboard, _scb, _checker         Checker Desync
RAL    _reg_block, _ral_pkg, _reg          RAL Desync
COV    _cov_if, _cov_bind, _cov            Coverage Regression
SVA    _bind, _sva_if, _sva                Assertion Orphaning
OTHER  (no match)                          —
───────────────────────────────────────────────────────────────

Rules are evaluated top-to-bottom; first match wins.
More specific patterns must come before general ones (e.g. CFG before ENV,
RAL before CFG, COV before SVA) to avoid false matches.

Usage as module:
    from type_annotator import classify_type, annotate_graph, CATEGORY

Usage standalone:
    python type_annotator.py
    → runs unit tests, annotates the live graph, prints distribution,
      saves node_types.json
"""

import os
import re
import json
from collections import Counter
import networkx as nx


# ── Taxonomy: τ type → DV failure category ───────────────────────────────────
# Directly mirrors Table 3 of the companion empirical study.
# OTHER nodes carry no primary category prediction; they still appear
# in the blast radius but are not labelled in category-level evaluation.

CATEGORY: dict[str, str | None] = {
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

# Ordered list (used for consistent display/iteration)
ALL_TYPES: list[str] = list(CATEGORY.keys())

# Reverse mapping: category string → list of τ types that produce it
CATEGORY_TO_TYPES: dict[str, list[str]] = {}
for _t, _c in CATEGORY.items():
    if _c is not None:
        CATEGORY_TO_TYPES.setdefault(_c, []).append(_t)


# ── Classification rules ──────────────────────────────────────────────────────
# Each rule is a tuple: (type_str, suffixes, contains, prefixes)
#   suffix   — name.endswith(s)     (checked first, most reliable signal)
#   contains — s in name            (substring match, use sparingly)
#   prefix   — name.startswith(s)   (for bind_ files and ral_ packages)
#
# ORDER MATTERS — first match wins:
#   RAL   before CFG/ENV  (catches _ral_pkg, _reg_block before _env_cfg)
#   CFG   before ENV      (catches _env_cfg before plain _env)
#   COV   before SVA      (catches _cov_bind before plain _bind)
#   SVA   before AGENT    (catches _sva_if before plain _if)
#   SEQ   before ENV      (seq rules are specific, but listed before ENV)
#   ENV   before AGENT    (catches _env before _if or loose patterns)

_RULES: list[tuple[str, list[str], list[str], list[str]]] = [
    (
        "RAL",
        ["_reg_block", "_ral_pkg", "_reg_model", "_reg"],
        ["_ral_"],
        [],
    ),
    (
        "CFG",
        ["_env_cfg", "_env_pkg", "_cfg"],
        [],
        [],
    ),
    (
        "SCBD",
        ["_scoreboard", "_scb", "_checker"],
        [],
        [],
    ),
    (
        "COV",
        ["_cov_if", "_cov_bind", "_coverage", "_cov"],
        ["cov_if", "cov_bind"],
        [],
    ),
    (
        "SVA",
        ["_sva_if", "_assert_if", "_sva", "_bind"],
        ["sva_"],
        ["bind_"],
    ),
    (
        "SEQ",
        ["_vseq", "_base_vseq", "_seq_item", "_seq", "_sequencer",
         "_sequence", "_item"],
        [],
        [],
    ),
    (
        "ENV",
        ["_env", "_base_env"],
        [],
        [],
    ),
    (
        "AGENT",
        ["_agent", "_agt", "_driver", "_monitor", "_if"],
        [],
        [],
    ),
]


# ── Core classification function ──────────────────────────────────────────────

def classify_type(name: str) -> str:
    """
    Return the UVM component type τ for a class name.

    Matching is case-insensitive and checks (in order):
      1. suffix patterns  (name ends with pattern)
      2. contains patterns (pattern appears anywhere in name)
      3. prefix patterns  (name starts with pattern)

    Args:
        name: UVM class name string, e.g. 'cip_base_env_cfg'

    Returns:
        One of: CFG | ENV | AGENT | SEQ | SCBD | RAL | COV | SVA | OTHER
    """
    n = name.lower()
    for type_str, suffixes, contains, prefixes in _RULES:
        if any(n.endswith(s) for s in suffixes):
            return type_str
        if any(c in n for c in contains):
            return type_str
        if any(n.startswith(p) for p in prefixes):
            return type_str
    return "OTHER"


# ── Graph annotation ──────────────────────────────────────────────────────────

def annotate_graph(G: nx.DiGraph) -> nx.DiGraph:
    """
    Add 'type' and 'category' node attributes to every node in G.

    Modifies the graph in-place and also returns it, so callers can chain:
        G = annotate_graph(build_graph(files))

    After annotation each node satisfies:
        G.nodes[n]['type']     ∈ ALL_TYPES
        G.nodes[n]['category'] ∈ set(CATEGORY.values()) | {None}
    """
    for node in G.nodes():
        t = classify_type(node)
        G.nodes[node]["type"]     = t
        G.nodes[node]["category"] = CATEGORY[t]
    return G


# ── Query helpers ─────────────────────────────────────────────────────────────

def type_distribution(G: nx.DiGraph) -> dict[str, int]:
    """Count nodes per type. Graph must already be annotated."""
    return dict(Counter(G.nodes[n].get("type", "OTHER") for n in G.nodes()))


def nodes_by_type(G: nx.DiGraph, type_str: str) -> list[str]:
    """Return all node names with a given type."""
    return [n for n in G.nodes() if G.nodes[n].get("type") == type_str]


def predict_categories_for_blast_radius(
    blast_radius: set[str], G: nx.DiGraph
) -> dict[str, list[str]]:
    """
    For each node in the blast radius, return its predicted DV failure
    categories (may be empty list for OTHER nodes).

    This is the structured input Stage 3 (LLM inference) uses to
    build the category-prediction prompt.

    Args:
        blast_radius: set of node names reachable from the changed node
        G: annotated dependency graph

    Returns:
        dict { node_name: [category_string, ...] }
        (list because multi-label is possible — empirical study shows
        50% co-occurrence rate across categories)
    """
    result: dict[str, list[str]] = {}
    for node in blast_radius:
        cat = G.nodes[node].get("category") if node in G else None
        result[node] = [cat] if cat else []
    return result


# ── Output utilities ──────────────────────────────────────────────────────────

def print_distribution(G: nx.DiGraph) -> None:
    """Pretty-print type distribution with category mapping."""
    dist  = type_distribution(G)
    total = sum(dist.values())

    print(f"\nVeriDelta τ type distribution  ({total} nodes total)")
    print("─" * 58)
    print(f"  {'Type':<8} {'Count':>5}  {'%':>5}  Category")
    print("─" * 58)
    for t in ALL_TYPES:
        count = dist.get(t, 0)
        pct   = 100 * count / total if total else 0.0
        cat   = CATEGORY[t] or "—"
        print(f"  {t:<8} {count:>5}  {pct:>4.1f}%  {cat}")
    print("─" * 58)
    interface_drift_n = sum(dist.get(t, 0) for t in ["CFG", "ENV", "AGENT"])
    print(f"\n  Interface Drift pool (CFG + ENV + AGENT): {interface_drift_n} nodes")
    print(f"  Unlabelled (OTHER):                       {dist.get('OTHER', 0)} nodes")


def save_annotations(G: nx.DiGraph, output_path: str = "node_types.json") -> None:
    """
    Persist node type annotations to JSON for use by
    ground_truth_mapper.py and graphrag_pipeline.py.

    JSON structure:
        {
            "cip_base_env_cfg": {
                "type": "CFG",
                "category": "Interface Drift"
            },
            ...
        }
    """
    annotations = {
        node: {
            "type":     G.nodes[node].get("type", "OTHER"),
            "category": G.nodes[node].get("category"),
        }
        for node in G.nodes()
    }
    with open(output_path, "w") as fh:
        json.dump(annotations, fh, indent=2)
    print(f"\nAnnotations saved → {output_path}  ({len(annotations)} nodes)")


# ── Bind index ────────────────────────────────────────────────────────────────

def build_bind_index(files: list[str]) -> dict[str, list[str]]:
    """
    Parse _bind.sv files and sva/ directory files for SystemVerilog
    'bind <rtl_module_name>' statements.

    Returns dict: { rtl_module_name: [bind_file_stem, ...] }

    WHY THIS EXISTS
    ───────────────
    Assertion Orphaning occurs when RTL module names or signal hierarchies
    change and the bind statements referencing them become stale.  Unlike
    class inheritance — which the dependency graph captures — bind statements
    are MODULE-level constructs.  A bind file targeting a renamed RTL module
    will compile and elaborate silently while every assertion inside it
    evaluates against non-existent or wrong signals.

    From OpenTitan DV methodology:
      "Unlike design assertions, in DV assertions are typically created within
       SV interfaces bound to the DUT. This way assertions and any collateral
       code don't affect the design, and can reach any internal design signal
       if needed."

    The bind statement form is:
      bind <rtl_module_or_instance> <checker_module> <instance_name> (ports);

    We capture only <rtl_module_or_instance> — the first token after 'bind' —
    which is the RTL module type being targeted.  Hierarchical paths (e.g.
    module.sub_instance) are handled correctly because \\w+ stops at the dot.

    USAGE BY graphrag_pipeline.py (Stage 1 extension)
    ──────────────────────────────────────────────────
    When a commit changes an RTL file, extract the RTL module name from the
    file stem (e.g. hw/ip/sysrst_ctrl/rtl/sysrst_ctrl.sv → 'sysrst_ctrl')
    and look it up in bind_index.  Any matching bind files are flagged as
    Assertion Orphaning risk.
    """
    bind_stmt = re.compile(r'\bbind\s+(\w+)', re.IGNORECASE)

    # Scan any file that either:
    #   (a) has "_bind" at the end of its stem, OR
    #   (b) lives in a /sva/ directory (OpenTitan puts SVA files there)
    def is_bind_candidate(path: str) -> bool:
        stem = os.path.basename(path).replace(".sv", "").replace(".svh", "")
        in_sva_dir = os.sep + "sva" + os.sep in path or path.endswith(os.sep + "sva")
        return stem.endswith("_bind") or in_sva_dir

    bind_index: dict[str, list[str]] = {}

    for f in files:
        if not is_bind_candidate(f):
            continue
        stem = os.path.basename(f).replace(".sv", "").replace(".svh", "")
        try:
            with open(f, encoding="utf-8", errors="ignore") as fh:
                content = fh.read()
            for m in bind_stmt.finditer(content):
                rtl_target = m.group(1)
                # skip UVM/SV keywords that can follow 'bind' in other contexts
                if rtl_target.lower() in {
                    "module", "interface", "class", "function",
                    "task", "begin", "end", "this",
                    "assertion", "csr", "the", "bind",  # false positives from SV comments
                }:
                    continue
                bind_index.setdefault(rtl_target, [])
                if stem not in bind_index[rtl_target]:
                    bind_index[rtl_target].append(stem)
        except Exception:
            pass

    return bind_index


def save_graph_edges(G: nx.DiGraph, output_path: str = "graph_edges.json") -> None:
    """
    Save the dependency graph as a successor adjacency list.

    JSON structure:
        { "parent_class": ["child_class_1", "child_class_2", ...], ... }

    This is the minimal representation needed by graphrag_pipeline.py
    to compute blast radius via graph traversal — without requiring
    NetworkX or access to the OpenTitan source at inference time.
    """
    adjacency = {node: list(G.successors(node)) for node in G.nodes()}
    with open(output_path, "w") as fh:
        json.dump(adjacency, fh, indent=2)
    print(f"\nGraph edges saved → {output_path}  ({G.number_of_edges()} edges)")


def save_bind_index(
    bind_index: dict[str, list[str]],
    output_path: str = "bind_index.json",
) -> None:
    """
    Persist the bind index to JSON for use by graphrag_pipeline.py.

    JSON structure:
        {
            "sysrst_ctrl": ["sysrst_ctrl_bind"],
            "aes":         ["aes_bind", "aes_sec_cm_bind"],
            ...
        }
    """
    with open(output_path, "w") as fh:
        json.dump(bind_index, fh, indent=2)
    total_refs = sum(len(v) for v in bind_index.values())
    print(
        f"\nBind index saved → {output_path}  "
        f"({len(bind_index)} RTL modules, {total_refs} bind-file references)"
    )


# ── Unit tests ────────────────────────────────────────────────────────────────

def _run_unit_tests() -> bool:
    """
    Validate classify_type() against known OpenTitan class names.
    Returns True if all tests pass.
    """
    cases = [
        # CFG — more specific than ENV, must win
        ("dv_base_env_cfg",        "CFG"),
        ("cip_base_env_cfg",       "CFG"),
        ("aes_env_cfg",            "CFG"),
        ("tl_agent_env_cfg",       "CFG"),   # ends _env_cfg, not _env
        ("xbar_env_cfg",           "CFG"),
        ("chip_env_cfg",           "CFG"),
        # ENV
        ("cip_base_env",           "ENV"),
        ("dv_base_env",            "ENV"),
        ("aes_env",                "ENV"),
        ("kmac_env",               "ENV"),
        # AGENT / DRIVER / MONITOR
        ("tl_agent",               "AGENT"),
        ("dv_base_driver",         "AGENT"),
        ("dv_base_monitor",        "AGENT"),
        ("i2c_driver",             "AGENT"),
        ("uart_if",                "AGENT"),  # interface used by agent
        # SEQ
        ("rv_dm_smoke_vseq",       "SEQ"),
        ("dv_base_vseq",           "SEQ"),
        ("cip_base_vseq",          "SEQ"),
        ("dv_base_seq_item",       "SEQ"),
        ("aes_base_seq",           "SEQ"),
        # SCBD
        ("uart_scoreboard",        "SCBD"),
        ("aes_scb",                "SCBD"),
        ("dv_base_checker",        "SCBD"),
        # RAL — must beat CFG/ENV
        ("aes_reg_block",          "RAL"),
        ("cip_base_reg",           "RAL"),   # ends _reg
        ("aes_ral_pkg",            "RAL"),   # contains _ral_
        # COV — must beat SVA (cov_bind before _bind)
        ("aes_cov_if",             "COV"),
        ("aes_cov_bind",           "COV"),
        ("aes_cov",                "COV"),
        # SVA
        ("bind_aes_sec_cm",        "SVA"),
        ("aes_sva",                "SVA"),
        ("aes_sva_if",             "SVA"),
        # RAL — confirmed in OpenTitan
        ("aes_reg_block",          "RAL"),
        ("cip_base_reg",           "RAL"),
        ("aes_ral_pkg",            "RAL"),   # contains _ral_
        # SEQ — _item suffix (e.g. csrng_item, mbx_seq_item)
        ("csrng_item",             "SEQ"),
        ("aes_message_item",       "SEQ"),
        ("dma_seq_item",           "SEQ"),
        # OTHER
        ("dv_base_vif_proxy",      "OTHER"), # ends _proxy, not _if
        ("dv_base_object",         "OTHER"),
        ("dv_base_component",      "OTHER"),
        # vseq_list files are list files, not classes — should be OTHER
        # but they contain "vseq" so they get scanned; class pattern
        # won't find them since they have no 'class X extends Y'
    ]

    passed = failed = 0
    for name, expected in cases:
        got = classify_type(name)
        if got == expected:
            passed += 1
        else:
            print(f"  FAIL  {name:<38} expected={expected:<6}  got={got}")
            failed += 1

    print(f"  {passed} passed, {failed} failed out of {len(cases)} cases")
    return failed == 0


# ── Standalone entrypoint ─────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    print("=" * 60)
    print("VeriDelta  —  type_annotator.py")
    print("=" * 60)

    # Step 1: unit tests
    print("\nStep 1: Unit tests")
    print("─" * 40)
    ok = _run_unit_tests()
    if not ok:
        print("\nFix failing cases before running on the full graph.")
        sys.exit(1)
    print("All unit tests passed.\n")

    # Step 2: load graph from OpenTitan
    OPENTITAN_PATH = os.environ.get(
        "OPENTITAN_PATH",
        os.path.join(os.path.dirname(__file__), "..", "opentitan", "hw")
    )
    EXTENSIONS   = (".sv", ".svh")
    UVM_KEYWORDS = (
        # core UVM components — from lowRISC DVCodingStyle.md spec
        "agent", "env", "driver", "monitor", "scoreboard",
        "sequencer", "sequence", "interface", "checker",
        # additional — confirmed in OpenTitan commit history
        "vseq",   # virtual sequences (_vseq) — 'vseq' != 'sequence'
        "cov",    # coverage interfaces (_cov_if, _env_cov, _agent_cov)
        "sva",    # SVA interface files (_sva_if, _sva)
        "bind",   # assertion bind files (_bind, _ctrl_bind, _cov_bind)
        "reg",    # RAL register block files (_reg_block, _reg)
        "ral",    # RAL package files (_ral_pkg)
        "item",   # sequence items (_item, _seq_item)
    )

    print("Step 2: Building dependency graph")
    print("─" * 40)
    files = []
    for dirpath, _, filenames in os.walk(OPENTITAN_PATH):
        for f in filenames:
            if f.endswith(EXTENSIONS) and any(k in f.lower() for k in UVM_KEYWORDS):
                files.append(os.path.join(dirpath, f))
    print(f"  Found {len(files)} UVM-relevant source files")

    G = nx.DiGraph()
    # Handles both plain and parameterized class declarations:
    #   class X extends Y                          ← plain
    #   class X #(type T = base) extends Y        ← single-line params
    #   class X #(\n  type T = base\n) extends Y  ← multi-line params
    class_pattern = re.compile(
        r'\bclass\s+(\w+)\s*(?:#\s*\([^)]*\))?\s*extends\s+(\w+)',
        re.IGNORECASE | re.DOTALL
    )
    for f in files:
        try:
            with open(f, encoding="utf-8", errors="ignore") as fh:
                content = fh.read()
            for m in class_pattern.finditer(content):
                G.add_edge(m.group(2), m.group(1))
        except Exception:
            pass
    print(f"  Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    # Step 3: annotate
    print("\nStep 3: Annotating types")
    print("─" * 40)
    annotate_graph(G)
    print_distribution(G)

    # Step 4: sample nodes per type
    print("\nStep 4: Sample nodes per type")
    print("─" * 40)
    for t in ALL_TYPES:
        examples = sorted(nodes_by_type(G, t))[:5]
        if examples:
            print(f"  {t:<8}  {examples}")

    # Step 5: blast radius spot-check with category predictions
    print("\nStep 5: Blast radius spot-check — cip_base_env_cfg")
    print("─" * 40)
    ROOT = "cip_base_env_cfg"
    if ROOT in G:
        br = nx.descendants(G, ROOT)
        preds = predict_categories_for_blast_radius(br, G)
        by_cat: dict[str, list[str]] = {}
        for node, cats in preds.items():
            for c in cats:
                by_cat.setdefault(c, []).append(node)
        print(f"  Blast radius: {len(br)} nodes")
        for cat, nodes in sorted(by_cat.items()):
            print(f"  {cat:<25}  {len(nodes)} nodes")
    else:
        print(f"  Node '{ROOT}' not found in graph — check OpenTitan path.")

    # Step 6: bind index — Assertion Orphaning coverage
    print("\nStep 6: Building bind index (Assertion Orphaning)")
    print("─" * 40)
    bind_index = build_bind_index(files)
    total_refs = sum(len(v) for v in bind_index.values())
    print(f"  RTL modules referenced by bind statements: {len(bind_index)}")
    print(f"  Total bind-file references:                {total_refs}")
    if bind_index:
        print("\n  Sample entries (rtl_module → bind_files):")
        for rtl_mod, bfiles in list(bind_index.items())[:8]:
            print(f"    {rtl_mod:<35} {bfiles}")
    else:
        print("  No bind statements found — check that 'bind'/'sva' are in UVM_KEYWORDS")

    save_bind_index(bind_index, "bind_index.json")

    # Step 7: save class graph annotations
    print()
    save_graph_edges(G, "graph_edges.json")
    save_annotations(G, "node_types.json")
    print("\nDone.")
    print("  graph_edges.json → adjacency list for blast radius traversal")
    print("  node_types.json  → class dependency graph with type annotations")
    print("  bind_index.json  → RTL module → bind file mapping (Assertion Orphaning)")