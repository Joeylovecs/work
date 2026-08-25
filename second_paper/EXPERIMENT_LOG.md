# Experiment log

## Stage 0 - 2026-08-19

- Remote workspace verified at `/root/shared-nvme/wangjie`.
- `work` environment: Python 3.10.20.
- Paper-1 data adapters read existing `Rethinking/data/wtq.json` and `TabFact/data/tabfact.json` in their existing `sampled_indices` order.
- Existing paper-1 utility files copied read-only into `second_paper/paper1_reuse/`; no model files or credentials copied.
- WTQ: 421 table objects, 837 sampled instances, 4,344 full question instances.
- TabFact: 298 table objects, 550 sampled instances, 2,024 full question instances.

## Stage 3 - synthetic semantic bug tests

- Added 12 tests covering sum/mean, strict comparator, target column, missing filter, wrong entity, ranking direction, highest/lowest operator errors, answer type, cardinality, arithmetic sign, and boolean polarity.
- Initial result: 11 passed, 1 failed due to a lexical boundary bug in `not` detection.
- General fix: AST analyzer now uses a word-boundary match for boolean negation.
- Final result: `12 passed in 0.41s`.

## API smoke status

- WTQ baseline and TabFact baseline runner guards were exercised on five-sample debug slices.
- Both stopped before API calls because `PARATERA_API_KEY` is not set in the server environment.
- No accuracy, silent-failure, repair, or latency result is claimed until the key is set and the paired runs complete.


## Stage 4 - 2026-08-19: first 100 samples and paper-inspired iteration

- WTQ and TabFact were run in first-paper order on flat index `[0,100)` for baseline, audit, and joint. Each method has exactly 100 unique records and the paired sample IDs are saved in `selected_samples.json`.
- `outputs/100_cycle1_clean/` is the current best accuracy reference: WTQ baseline/audit/joint = 67%/66%/83%; TabFact = 87%/83%/92%.
- `outputs/100_cycle2_clean/` records the next optimization cycle: WTQ = 65%/70%/83%; TabFact = 84%/81%/91%. The second cycle improved WTQ audit but did not improve the joint configuration, so no further unvalidated modification was kept.
- The two supplied knowledge-graph papers contributed bounded Generate-Verify-Refine, local operation verification, execution/evaluator checks, and explicit retry budgets.
- The result contract now includes the first-paper-compatible `idx`, `answer`, `text`, `transpose`, `resort`, `question_id`, `table_id`, `title`, `table`, and `question` fields together with second-paper audit/repair/API metadata.
- Unit tests and compilation checks pass after the final source changes.
