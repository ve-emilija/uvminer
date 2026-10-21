# VeriDelta Commit Labeling Protocol

Protocol version: `1.0.0`

## Purpose And Unit

The unit of annotation is one commit in one named source repository. Annotate
the verification-impact mechanisms supported by the paired design and
verification diffs. A label means the change creates or requires the described
verification adaptation; it does not, by itself, claim that a simulation was
run or that a failure was observed.

Use the RTL/design diff to identify what changed and the verification diff to
identify whether and how the testbench adapted. Consider both sides together.
Do not infer a label from a filename, commit message, or co-change alone.

## Failure Categories

Assign every category independently when its mechanism is supported. Multiple
categories may apply to one commit.

| Code | Category | Label when the change... |
| --- | --- | --- |
| `INTERFACE_DRIFT` | Interface Drift (IF) | Changes signal names, widths, layouts, or protocol parameters in a way that invalidates a verification interface, transaction representation, or agent/driver/monitor assumptions. |
| `SEQUENCE_INVALIDATION` | Sequence Invalidation (SI) | Invalidates stimulus assumptions so a sequence can generate illegal, incomplete, or no-longer-meaningful transactions. |
| `CHECKER_DESYNC` | Checker Desync (CD) | Changes design behavior without a corresponding update to the scoreboard, checker, or reference model semantics. |
| `ASSERTION_ORPHANING` | Assertion Orphaning (AO) | Renames or restructures a design module or signal so a bind, assertion, or hierarchical assertion control no longer targets the intended design object. |
| `RAL_DESYNC` | RAL Desync (RD) | Changes registers, fields, reset values, access behavior, or addresses so the verification register model no longer represents the design. This includes generated models when the generated output is stale. |
| `COVERAGE_REGRESSION` | Coverage Regression (CR) | Makes a coverage point unreachable, vacuous, disconnected, or insufficient for the changed design behavior. |

`OTHER` is for a concrete verification impact outside these six mechanisms; it
is not a fallback for uncertainty or an unrelated documentation-only change.

## Decision Rules

- **IF vs. SI:** IF concerns the interface contract or transaction shape; SI
  concerns the legality or adequacy of generated stimulus.
- **CD vs. IF:** CD concerns expected behavior in a checker or reference model;
  IF concerns communication and representation at the boundary.
- **RD vs. SI:** RD concerns the register model's representation of the design;
  sequences that use a now-invalid register contract may additionally receive
  SI when the stimulus itself is affected.
- **AO:** Include stale bind targets and stale hierarchical assertion controls.
  Do not require that the simulator silently accept the stale reference.
- **CR:** Assign only when the coverage model or collection path is affected;
  a change that merely adds stimulus is not enough by itself.
- Do not assign a category simply because its usual component filename appears
  in the diff. Cite the changed behavior and its verification consequence.

## Label Status

`label_status` is separate from the failure-category list:

- `CLASSIFIED`: one or more of the six categories apply.
- `OTHER`: a concrete verification impact exists but is outside the six.
- `NOT_APPLICABLE`: the commit is outside the protocol's design/verification
  change scope, for example a docs-only commit.
- `INSUFFICIENT_EVIDENCE`: the commit is in scope, but the supplied diffs do not
  establish a defensible category.
- `ERROR` and `DRY_RUN` are execution statuses, not annotation labels.

`categories` contains only category codes. It is empty for
`NOT_APPLICABLE` and `INSUFFICIENT_EVIDENCE`; it is `OTHER` for the `OTHER`
status.

## Evidence And Confidence

Each category must have concise evidence tied to paths or changes visible in
the supplied diffs. Never invent files, test outcomes, or line references.
`at_risk_dv_files` identifies affected verification files; `evidence` explains
the mechanism. Confidence is `HIGH` for direct paired-diff evidence, `MEDIUM`
for a clear but partly indirect consequence, and `LOW` when the mechanism is
plausible but only weakly supported. Use `INSUFFICIENT_EVIDENCE` rather than a
low-confidence category when the evidence is too incomplete to classify.

## Comparison And Adjudication

The LLM and Python approaches must receive the same commit records and emit the
same schema: `protocol_version`, `method`, `source_repo`, `commit_hash`,
`label_status`, `categories`, `evidence`, `at_risk_dv_files`, `justification`,
and `confidence`. Neither approach's predictions are ground truth. For evaluation,
maintain a human-reviewed, adjudicated label set using this protocol, record the
protocol version, and keep model/rule predictions separate from those labels.
Resolve annotator disagreements against the cited diff evidence before
freezing a dataset release. Select human-review commits from the raw commit pool
without conditioning selection on either approach's predictions; preserve the
sampled commit CSV so both methods see the exact same inputs.