# Commit corpus pipeline (VeriDelta)

Builds a corpus of OpenTitan and Caliptra commits that touch HDL files, each labelled
design, verification or both. Labels come from path rules, a 3-run LLM vote for commits
the rules cannot place, and a 50-commit human-annotated pilot.

## Requirements
- Python 3.13 with the venv in `.venv` (`source .venv/bin/activate`)
- `pip install anthropic` (only for the LLM step)
- `export ANTHROPIC_API_KEY=...` in the terminal (never put the key in a file)
- Git clones in `sources/opentitan` and `sources/caliptra-rtl`
  (extracted at OpenTitan d76cd324f213, Caliptra 5e577ee820ff, on 2026-10-24)

Run everything from this folder, not from `scripts/`.

## Run order
1. Extract history (merges skipped):
   `python3 scripts/stage1_extract_commits.py`          -> stage1_out/
   (options: `--help`)
2. Optional extension inventory:
   `python3 scripts/stage2a_extension_inventory.py --out-dir stage2_out_v2`     -> stage2_out_v2/ext_inventory.csv
3. Rules (v2): select HDL commits, label them:
   `python3 scripts/stage2_filter_commits.py --stage1-dir stage1_out --out-dir stage2_out_v2 --sources-dir sources`
4. LLM labels for commits the rules cannot place (3 runs, majority; results cached in llm_cache.jsonl):
   `python3 scripts/stage2c_review.py --stage2-dir stage2_out_v2 llm --scope review`
5. Final corpus table (human > LLM for uncertain > rules):
   `python3 scripts/stage2c_review.py --stage2-dir stage2_out_v2 finalize --audit 0 --human human_labels/human_labels.csv`
   -> stage2_out_v2/commits_verified.csv

Steps 3 to 5 print counts; step 5 should report human 49, rules 13,540, LLM 41,
corpus 13,629 (OpenTitan 13,147, Caliptra 482).

## Key outputs
| File | Meaning |
|---|---|
| stage1_out/commits.csv, files.csv | every commit and the files it changed |
| stage2_out_v2/commits_selected.csv | commits touching HDL files (13,630) |
| stage2_out_v2/commits_labeled.csv | rule labels for those commits |
| stage2_out_v2/llm_cache.jsonl | cached LLM answers (model claude-haiku-4-5-20251001) |
| stage2_out_v2/commits_verified.csv | final corpus: rule, LLM, human and final label, source, in_corpus |

## Known limits
- The 50 pilot commits are a development set; rule and LLM labels outside them are not independently checked.
- Template-only (.tpl) commits, a 1,000-commit rules-vs-LLM comparison, a second annotator and an audit are future work.
