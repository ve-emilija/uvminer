# VeriDelta — Anonymous Artifact Repository

Supporting artifact for the paper *"VeriDelta: GraphRAG-Grounded Change Impact
Analysis for UVM Verification Environments"* (submitted, double-blind review).

This repository contains the full pipeline used to produce every number
reported in the paper — the typed UVM dependency graph, the ground-truth
mapping, the GraphRAG pipeline, and the four-system evaluation ablation.

---

## Repository structure

```
README.md
scripts/
  type_annotator.py        # Stage 0: builds the typed dependency graph
  ground_truth_mapper.py   # Stage 0: maps [9]'s labeled commits onto graph nodes
  graphrag_pipeline.py     # Stages 1–3: BFS blast radius, retrieval, LLM inference
  evaluation.py            # Runs the 4-system ablation, produces Table III
data/
  llm_labeled_final.csv    # input from [9] — per-commit failure category labels
  commits_final.csv        # input from [9] — commit metadata
  node_types.json          # output of type_annotator.py — 1,295 typed nodes
  graph_edges.json         # output of type_annotator.py — 1,270 inheritance edges
  bind_index.json          # output of type_annotator.py — 63 bind targets
  ground_truth.json        # output of ground_truth_mapper.py — 345 labeled nodes
  commit_index.json        # output of ground_truth_mapper.py — 602 retrieval-pool commits
  evaluation_results.json  # output of evaluation.py — full per-node predictions + metrics
```

---

## Prerequisites

- Python 3.11 (or later)
- Dependencies: see `requirements.txt`
- A local clone of [OpenTitan](https://github.com/lowRISC/opentitan) at commit
  `<FILL IN — exact commit hash used for this study>`
- API access to Claude Haiku (`claude-haiku-4-5-20251001`) for Stage 3 inference.
  Set the API key via the `ANTHROPIC_API_KEY` environment variable:
  ```
  export ANTHROPIC_API_KEY=your-key-here
  ```
  Do **not** commit a key to this repository.

---

## Run order

The pipeline has a strict dependency chain. Run in this order:

```
1. type_annotator.py       (reads: OpenTitan source tree, via OPENTITAN_PATH env var)
                            (writes: node_types.json, graph_edges.json, bind_index.json)

2. ground_truth_mapper.py  (reads: node_types.json, llm_labeled_final.csv, commits_final.csv)
                            (writes: ground_truth.json, commit_index.json)

3. evaluation.py           (reads: all of the above; imports graphrag_pipeline.py)
                            (writes: evaluation_results.json)
```

`graphrag_pipeline.py` is not run standalone — it is imported by
`evaluation.py`, which calls it once per (test commit, changed node) pair
across all four ablation systems (VeriDelta, LLM-no-history, Structural
prior, LLM-no-graph).

Set `OPENTITAN_PATH` to point at your local OpenTitan clone before running
`type_annotator.py`:
```
export OPENTITAN_PATH=/path/to/opentitan/hw
```
If unset, the script falls back to `../opentitan/hw` relative to `scripts/`.

---

## Expected outputs

| File | Key figures it should reproduce |
|---|---|
| `node_types.json` | 1,295 nodes — Table I: RAL 13, CFG 69, SCBD 46, COV 64, SVA 0, SEQ 880, ENV 44, AGENT 102, OTHER 77 |
| `graph_edges.json` | 1,270 inheritance edges |
| `bind_index.json` | 63 bind targets (rtl_module → bind file mapping) |
| `ground_truth.json` / `commit_index.json` | 602 retrieval-pool commits; 403 graph-resolvable commits; 345 labeled nodes |
| `evaluation_results.json` | Table III: per-category precision/recall/F1 for all four systems; 419 evaluation points; 432 category-support sum |

Note: Stage 3 uses the API default temperature, so predictions are
stochastic. Small run-to-run variation in Table III is expected; the
qualitative conclusions (graph structure required for discrimination,
retrieval required for RAL Desync detection) are stable across runs.

---

## Data provenance

`llm_labeled_final.csv` and `commits_final.csv` are the labeled dataset
released by the companion empirical study [9], included here so results —
including the 92.3% co-modification-failure rate reported in the
Introduction — can be independently recomputed by reviewers, since [9] is
not yet publicly available at time of submission.

---
