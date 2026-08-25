# First-20 rerun result index

Scope: raw JSON order, indices 0 through 19, exactly 20 examples per dataset.

| Dataset | Python baseline | Text baseline | Optimized Python | Delta vs Python (percentage points) |
|---|---:|---:|---:|---:|
| WTQ | 7/20 (35%) | 18/20 (90%) | 12/20 (60%) | +25 pp |
| TabFact | 13/20 (65%) | 19/20 (95%) | 17/20 (85%) | +20 pp |

Method definitions:

- Python baseline: one base-model Python generation, no semantic optimization.
- Text baseline: one direct text-reasoning generation, no Python and no semantic optimization.
- Optimized Python: optimized Python prompt plus semantic audit and at most two conservative audit-repair cycles.

Attribution note:

- WTQ optimized generation starts at 8/20 and conservative semantic repair raises it to 12/20, with no within-run degradation.
- TabFact optimized generation starts and ends at 17/20. Its 4-case gain over the plain Python baseline comes from the optimized Python generation setup on this slice; semantic repair adds 0 net cases.
- Residual audit exceptions remain on 11 WTQ and 5 TabFact examples. They are retained as diagnostics and are not allowed to overwrite an answer unless the candidate passes the conservative acceptance gate.

Pairwise checks against Python baseline:

- WTQ: 5 baseline errors corrected, 0 baseline-correct cases degraded; within optimized run, repair improved 4 and degraded 0.
- TabFact: 4 baseline errors corrected, 0 baseline-correct cases degraded; within optimized run, repair improved 0 and degraded 0.

Files:

- WTQ python_baseline: outputs/20_recheck_v4/wtq/python_baseline/result.jsonl
- WTQ text_baseline: outputs/20_recheck_v4/wtq/text_baseline/result.jsonl
- WTQ optimized_python: outputs/20_recheck_v4/wtq/optimized_python/result.jsonl
- TabFact python_baseline: outputs/20_recheck_v4/tabfact/python_baseline/result.jsonl
- TabFact text_baseline: outputs/20_recheck_v4/tabfact/text_baseline/result.jsonl
- TabFact optimized_python: outputs/20_recheck_v4/tabfact/optimized_python/result.jsonl
- Recomputed metrics: outputs/20_recheck_v4/metrics_recomputed.json

Verification:

- Every result file has 20 rows and the same source-order IDs within each dataset.
- Every row begins with idx and then answer.
- Python and text baselines use exactly one model call per example.
- nu-2928 is idx 6, answer is ["5h 29' 10\""], audit passed, and select evidence points to Time.
- The first-paper datasets were read only; outputs are stored under the second_paper workspace.

Caveat: this is a 20-example diagnostic slice, not a statistically stable full-dataset result.
