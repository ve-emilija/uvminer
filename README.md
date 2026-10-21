# VeriDelta

Commit-impact analysis for SystemVerilog and UVM repositories.

VeriDelta maps changed UVM classes to a dependency graph and reports reachable
downstream components as a structural blast-radius estimate. RTL changes can
also surface indexed bind files. Optional Claude inference adds failure-category
labels to the affected components.

## Quick Start

Requirements: Python 3.11 or later, Git, and a local clone of the source
repository to analyze.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Source Repositories

`sources.toml` defines the shared checkout root and the upstream repositories
that VeriDelta will analyze. Clone each repository into its configured
directory under `sources/`:

```powershell
git clone https://github.com/lowRISC/opentitan.git sources/opentitan
git clone https://github.com/openhwfoundation/core-v-verif.git sources/core-v-verif
git clone https://github.com/chipsalliance/caliptra-rtl.git sources/caliptra-rtl
git clone https://github.com/chipsalliance/riscv-dv.git sources/riscv-dv
```

The checkouts are local inputs and are excluded from VeriDelta's Git history.
Keep analysis indexes separate for each repository; graph and commit data from
one project should not be reused as another project's index.

The checked-in `data/` indexes describe the OpenTitan study dataset. To analyze
another repository structurally, generate graph indexes for that repository.
The OpenTitan `commit_index.json` is only needed for `--llm`; meaningful
history-backed inference on another project requires labeled history for it.

```powershell
python scripts/graphrag_pipeline.py `
  --repo C:\src\opentitan `
  --commit 5989ce15 `
  --data-dir data
```

The report is JSON on stdout. Save it with `--output impact.json`. The `--llm`
option adds category inference and requires `ANTHROPIC_API_KEY`:

```powershell
$env:ANTHROPIC_API_KEY = "your-key"
python scripts/graphrag_pipeline.py --repo C:\src\opentitan --commit HEAD --llm
```

## Data Labeling

The annotation and dataset-preparation workflow lives in
[`data_labeling/`](data_labeling/README.md). It includes commit extraction,
LLM-assisted labeling, reviewer sampling/export, and ground-truth mapping.
The versioned [`labeling protocol`](data_labeling/LABELING_PROTOCOL.md) and
shared [`label schema`](data_labeling/label_schema.py) define the common output
contract. [`extract_raw_commits.py`](data_labeling/extract_raw_commits.py)
builds a repository-profiled raw commit corpus from configured local sources;
the older extractor remains available for reproducing the OpenTitan study's
narrower selection criteria.

## Data Layout

`--data-dir` must contain the three graph indexes below. `commit_index.json` is
needed only when `--llm` is enabled.

| File | Purpose |
| --- | --- |
| `graph_edges.json` | Directed class-inheritance graph |
| `node_types.json` | UVM component types and primary categories |
| `bind_index.json` | RTL module to assertion-bind mapping |
| `commit_index.json` | Historical examples for `--llm` inference (optional otherwise) |

The `type_annotator.py` script builds graph indexes from SystemVerilog files.
Set `OPENTITAN_PATH` to the source tree's hardware directory and run it with
`data/` as the working directory so generated indexes are written there:

```powershell
$env:OPENTITAN_PATH = "C:\src\opentitan\hw"
Push-Location data
python ..\scripts\type_annotator.py
Pop-Location
```

The historical commit index depends on labeled commit data and must be prepared
separately before using `--llm` with a new source repository.

## Output And Scope

Each report includes the resolved commit, changed files, matched classes and
RTL modules, and one analysis per matched change. A class's blast radius is
computed by traversing its downstream inheritance edges; a leaf class can
therefore have an empty radius. Commits with no graph-matched changes produce
an empty analysis list.

The structural radius is a graph-based candidate set, not a guarantee that a
component will fail. LLM category labels are advisory and require a compatible
graph and historical index. Source repositories are read-only inputs; reports
are written to stdout unless `--output` is provided.

## Research Artifacts

The prediction and evaluation scripts and study datasets remain in `scripts/`
and `data/`. See [README.txt](README.txt) for artifact provenance and
reproduction notes.