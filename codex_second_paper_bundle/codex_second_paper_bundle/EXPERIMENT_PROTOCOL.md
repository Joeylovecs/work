# Experiment Protocol

## Data
Use the already finalized WTQ and TabFact data format/loaders/evaluators from paper 1.

Do not redesign, reserialize, transpose, reshuffle, or rebuild data.

For WTQ, a processed object may contain one table with parallel `questions`, `answers`, and `ids`. Use the existing paper-1 definition of an evaluation instance and existing flattening logic, if any.

## Debug sample schedule
Use the test set as requested for engineering debugging, gradually:

1. 5 samples
2. 10 samples
3. 20 samples
4. 30 samples
5. at most 40 samples if needed

For every comparison, baseline and audited method must use the exact same sample IDs.

### Reproducible sampling
Either:
- fixed contiguous slice from existing evaluation order, or
- fixed random seed with selected IDs stored in `selected_samples.json`.

Never resample separately for different methods.

## Core methods
- B: DeepSeek-V3.2 + Python baseline
- C: DeepSeek-V3.2 + Python + 3-level semantic audit

Optional later:
- A: Text-only
- D: Adaptive Text/Python + semantic audit

Do not begin with Text x5 + Python x5 majority voting.

## Metrics
Main:
- WTQ existing metric from paper 1
- TabFact existing metric from paper 1

Audit/debug analysis:
- execution success rate
- execution-success-but-wrong count
- semantic exception count
- repair success
- auditor false positive
- repair degradation
- detected silent failure
- missed silent failure
- Global/Operator/Parameter trigger counts
- API calls/tokens/latency

## Error taxonomy
- wrong_column
- missing_filter
- wrong_comparator
- wrong_aggregation
- wrong_entity
- wrong_ranking_or_sort
- answer_type_mismatch
- cardinality_mismatch
- arithmetic_error
- boolean_polarity_error
- execution_failure
- auditor_false_positive
- auditor_false_negative
- repair_failure

## Scientific hygiene
Because these test slices are inspected during development, save them under `debug_test_slice` and do not treat them as an untouched final test estimate.

Never add sample-specific or gold-answer-specific logic.
