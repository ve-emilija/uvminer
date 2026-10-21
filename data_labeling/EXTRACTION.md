# Commit Extraction: Scope, Repository Structure, and Decision Rules

This explains what `extract_raw_commits.py` extracts, how each source
repository is laid out, and how the script decides what counts as design or
verification. The rules live in `sources.toml`; this file explains why.

## Purpose

The extractor is the first step. It reads Git history and saves one row per
commit. It does not label commits and does not decide which commits matter.
Its aim is to capture changes to **RTL, UVM/verification code, registers,
coverage, assertions and reference models**, and to leave out software,
file lists, documentation, build output and generated or vendored files.

## What the script does

1. Pin the configured ref (`HEAD`) to its commit hash, then list the commits
   reachable from it. Merge commits are skipped unless `--include-merges` is
   passed, because a merge repeats the changes of the commits it merges.
2. List each commit's changed files against its first parent. Renames appear
   as a delete plus an add. Non-merge commits are listed in one Git call per
   repository, which is much faster than one call per commit.
3. Drop files matching `exclude_paths`.
4. A file is **verification** if it matches `verification_paths`.
5. A file is **design** if it matches `design_paths` and is not already
   verification. Verification wins when both match.
6. Set `change_scope` from what is left:

   | Scope | Meaning |
   | --- | --- |
   | `paired` | the commit changed at least one design file and one verification file |
   | `design_only` | design files only |
   | `verification_only` | verification files only |
   | `other` | neither (software, docs, build, and so on) |

7. Save the diff of the design files and of the verification files
   separately, capped at 2,000 lines per side. `*_diff_truncated` flags a cut
   and `*_diff_lines` always holds the real length, so a truncated row still
   shows how much was left out.
8. Write each repository's CSV as soon as it finishes, then record it in
   `manifest.json`.

Every changed file is still listed in `changed_files`, including excluded and
unmatched ones, so nothing is hidden. Only classified files get a diff.

### Provenance

Each row carries `source_ref`, `source_head` (the full commit hash the ref was
pinned to) and `profile_hash` (a short hash of that repository's profile in
`sources.toml`). `manifest.json`, written next to the CSVs, lists per
repository the pinned commit, profile hash, commit count, scope counts, the
options used and the run time. Two CSVs with the same `source_head` and
`profile_hash` came from the same data and the same rules.

### Pattern syntax

- `*` matches any characters **including `/`**. There is no separate `**`.
- `{a,b}` expands to alternatives, for example `*.{sv,svh}`.
- Matching is case-sensitive.

### What "paired" means

Paired means one commit in one repository changed both sides. It does not link
a design commit to a later verification fix, and it does not link commits
across repositories. It is a candidate signal, not proof of a failure: a
refactor can touch both sides without breaking anything.

## Repository structures and the rules chosen

Findings come from the full history of each repository, not only the current
checkout. Checkouts audited: OpenTitan `8cfac43aa6`, core-v-verif `083728bd`,
Caliptra RTL `0f311d30`, riscv-dv `0d71c2b`.

### OpenTitan

Hardware and verification live together under `hw/`.

| Location | Contents | Side |
| --- | --- | --- |
| `hw/ip/<ip>/rtl`, `hw/top_*/rtl`, `hw/top_*/ip/*/rtl` | RTL | design |
| `hw/ip/prim*/` | primitive cells (rtl, abstract) | design |
| `hw/ip_templates/` | RTL and register templates (`.sv.tpl`, `.hjson`) | design |
| `hw/*/data/*.hjson`, `hw/*/doc/*.hjson` | register and IP specifications | design |
| `hw/*/templates/`, `util/{reggen,topgen,tlgen,design}` | RTL and register generators | design |
| `hw/ip/*/dv`, `hw/dv/` | UVM environments, agents, coverage, `.el`/`.cfg` coverage files | verification |
| `hw/*/{fpv,pre_dv,generic_dv,dv_bkp}` | formal, unit testbenches, shared DV | verification |
| `hw/ip/*/model`, `hw/ip/otbn/util` | reference models and OTBN tools | verification |
| `*testplan*.hjson`, reggen UVM/FPV templates, `util/{uvmdvgen,fpvgen}` | testplans, RAL and DV generators | verification |

Notes:

- RAL models are generated from `.hjson`. The `.hjson` is the design-side
  source; the generated RAL is not committed.
- Some testbench RTL sits inside a `dv` folder, for example
  `hw/ip/otbn/dv/tracer/rtl/`. It is classified as verification because
  verification wins.
- Excluded: `hw/vendor/` (imported projects such as Ibex), `*_autogen/` and
  `autogen/` (generated copies of template output), `pre_sca/`, `sw/`,
  `third_party/`, `*.prj.hjson`.

### Caliptra RTL

RTL and verification are both in `src/`, organised per block.

| Location | Contents | Side |
| --- | --- | --- |
| `src/<block>/rtl/` | RTL and register RTL | design |
| `src/**/*.rdl`, `src/*/data/*.hjson` | register specifications | design |
| `src/<block>/rtl/*_uvm.sv` | **RAL models**, generated from the `.rdl`, kept inside `rtl/` | verification |
| `src/<block>/{tb,formal,coverage,asserts}/` | testbenches, formal, coverage, SVA | verification |
| `src/<block>/uvmf*/` | UVMF-generated environments | verification |
| `tools/dv-classes/`, `tools/templates/` | UVM base classes, RAL templates | verification |
| `coverage/`, `src/*/coverage/*.{cfg,yml}` | coverage configuration | verification |
| `tools/scripts/{reg_gen,rdl_post_process,reg_json,sparse_fsm_encode}.py` | register and RTL generators | design |
| `tools/scripts/gen_*covergroups*.py` | coverage generator | verification |

Notes:

- The RAL model path rule exists because the old design pattern matched
  everything in `src/**/*.sv`, so RAL models and UVMF files counted as both
  sides. About two thirds of old paired commits came from that overlap.
- Excluded: `docs/`, `.github/`. Not matched: `.vf` file lists,
  `src/integration/test_suites` C and assembly tests, `.yml` regression lists.

### core-v-verif

Verification only. The CPU RTL lives in separate CORE-V-CORES repositories;
`core-v-cores/` in this checkout is empty.

| Location | Contents | Side |
| --- | --- | --- |
| `lib/uvm_agents`, `lib/uvm_libs`, `lib/cv_dv_utils`, `lib/uvm` | UVM agents and libraries | verification |
| `cv32*/{env,tb}`, `cva6/{env,tb}` | per-core environments and testbench tops | verification |
| `vendor_lib/` | vendor SystemVerilog models used by the environments | verification |

Rule: any `.sv`, `.svh`, `.v`, `.vh`, `.vhd` file, except under
`vendor/`, `docs/`, `*/results/`. Excluded: the vendored Spike simulator in
`vendor/riscv/`, C and assembly test programs, Makefiles, file lists, and
committed reports and build artifacts.

Consequence: this repository cannot produce `paired` commits until its commits
are linked to the RTL revision they target.

### riscv-dv

An instruction-sequence generator. It has no DUT RTL, bind files or RAL.

| Location | Contents | Side |
| --- | --- | --- |
| `src/`, `test/`, `target/`, `setting/` | SystemVerilog UVM generator, coverage, per-target settings | verification |
| `pygen/pygen_src` | Python port of the generator | verification |
| `euvm/`, `riscv/` | D-language port | verification |
| `scripts/*.py` | trace converters and comparison | verification |
| `yaml/`, `target/**/*.yaml` | CSR and register descriptions | verification |

Excluded: `docs/` (including committed build output), `deprecated/`, `exp/`,
`experimental/`. Like core-v-verif, it cannot produce `paired` commits.

## What is deliberately not captured

- Device software and firmware (C, C++, Rust, assembly tests).
- File lists and build files: `.core`, `.vf`, `.flist`, Makefiles, `.bazel`,
  `.mk`.
- Simulation, lint, CDC and formal flow configuration (`*_sim_cfg.hjson`,
  tool `.tcl`, `.vlt`, waivers) and FPGA constraints (`.xdc`).
- Documentation, images, fonts and binaries.
- Vendored trees and generated copies of template output.

If a category above turns out to matter, add its pattern to `sources.toml`
and rerun the extractor.

## Known limits

- Same-commit pairing misses the case where RTL changes first and the
  environment is repaired in a later commit. `design_only` commits are
  therefore not clean negatives.
- Only failures that someone noticed and fixed show up. Silent mismatches do
  not.
- Commits are not linked across repositories.
- The patterns are hand-written from directory surveys. A new repository
  needs its own profile and a fresh audit.

## Audit results at the time of writing

An ad hoc script compared each profile against every file ever changed in
each repository (merge commits excluded). The script was run inline and is not
saved in the repository.

| Repo | Commits | Paired | Design only | Verification only | Other |
| --- | ---: | ---: | ---: | ---: | ---: |
| OpenTitan | 32,216 | 1,516 | 6,265 | 8,402 | 16,033 |
| Caliptra RTL | 791 | 192 | 92 | 209 | 298 |
| core-v-verif | 3,120 | 0 | 0 | 1,365 | 1,755 |
| riscv-dv | 877 | 0 | 0 | 720 | 157 |

Hardware-type files still not captured in any repository: 5 OpenTitan
`util/example` SystemVerilog files, Caliptra's `tools/scripts/demo.rdl`, and
one build-artifact `.d` file in core-v-verif. All were left out on purpose.

OpenTitan's paired count is not comparable with the 652 commits in the earlier
study, which covered a limited date range and required RTL and DV changes in
the same IP block.

## Running it

From the `uvminer` directory:

```bash
python data_labeling/extract_raw_commits.py --output-dir data_extracted
```

It prints the absolute output directory first, then progress with an estimated
time left, and lists every file it wrote at the end. Each repository's CSV
appears as soon as that repository finishes; it is written to a temporary file
and renamed, so an interrupted run never leaves a half-written CSV. Existing
output files are never overwritten unless `--force` is passed. To redo one
repository, pass `--repo <name>`.

Only `data/raw_commits/` is Git-ignored; add any other output directory to
`.gitignore`, since the CSVs contain source diffs.
