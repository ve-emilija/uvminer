# Data Labeling Workflow

This folder contains scripts for preparing commit data, producing LLM labels,
sampling commits for human review, and mapping reviewed labels onto graph nodes.
Run commands from the VeriDelta project root so input and output paths resolve
consistently.

[`LABELING_PROTOCOL.md`](LABELING_PROTOCOL.md) is the versioned annotation
specification. `label_schema.py` is the shared category/status contract for the
LLM annotator and the planned Python baseline. The LLM's results are proposals,
not ground truth; comparison requires a separate human-reviewed, adjudicated
label set.

## Extract Raw Commits

`extract_raw_commits.py` reads profiles from `sources.toml`, mines commits
reachable from each configured ref, and writes one CSV per repository under
`data/raw_commits/`. It does not filter commits by category or by whether both
sides changed. `change_scope` and the matched path lists describe available
evidence; docs-only and one-sided commits remain in the corpus.

Extract all configured repositories into one raw corpus:

```powershell
python data_labeling/extract_raw_commits.py --repo opentitan --repo core-v-verif --repo caliptra-rtl --repo riscv-dv --output-dir data/raw_commits
```

The default output directory is `data/raw_commits/`, containing
`<repository>.csv`; existing per-repository files are not overwritten unless
`--force` is supplied. The sampler accepts the directory and combines its CSVs
in memory for balanced sampling. Diff text is capped at 2,000 lines per side by
default and truncation is recorded. Use `--max-diff-lines 0` for full diffs or
`--limit-per-repo 1` for a smoke test. Every repository profile explicitly
records all six category states:

- `in_scope`: this checkout contains the relevant design and verification scope.
- `paired_design_required`: the verification mechanism is present, but a
	companion design repository/change is needed to assess the failure.
- `not_present_in_checkout`: the mechanism is absent from this source checkout;
	this does not claim it is absent from the wider project.

`category_profile_status` and `category_profile_note` describe how mature the
repository assessment is. These profile values are context for extraction and
review; they do not automatically assign or suppress a commit label.

`extract_commits.py` remains the legacy OpenTitan study extractor with its
narrower RTL/DV co-change criteria.

## Blind Comparison Sample

Sample raw commits independently of either labeling method:

```powershell
python data_labeling/sample_for_kappa.py --commits data/raw_commits
```

After adjudicating the human reference, run the LLM on the exact same sample.
The Python baseline should use this input and emit the shared schema:

```powershell
python data_labeling/llm_label.py --input data/kappa_commits.csv --output data/llm_kappa.csv
python data_labeling/llm_label.py --input data/kappa_commits.csv --output data/llm_kappa.csv --kappa-human data/human_labels.csv
```

For corpus labeling outside the blind evaluation sample, run the labeler once
per source CSV:

```powershell
python data_labeling/llm_label.py --input data/raw_commits/opentitan.csv --output data/llm_labeled/opentitan.csv
```

LLM labeling requires `ANTHROPIC_API_KEY`. The sample script supports explicit
output paths; its defaults write under the project `data/`
directory. `ground_truth_mapper.py` consumes the prepared label and commit CSVs
along with `data/node_types.json`, then writes ground-truth and retrieval-index
artifacts under `data/`.

The sampler selects from raw commits, not LLM/Python predictions, and writes
`data/kappa_commits.csv` as the shared comparison input. Annotate
`data/human_labels.csv` independently before running either method on that
sample. Keep the human adjudications separate from both methods' output files.