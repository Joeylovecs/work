# Second Paper TableQA Runtime

The detailed Chinese guide is [项目说明_中文.md](项目说明_中文.md).

Main entry: `scripts/run_paper1.py`. The fixed first-100 wrappers are under `scripts/*_0_100.sh`; `scripts/run_100_all.sh` runs all six dataset/method combinations sequentially with the default interval `[0,100)`.

The runner preserves the first project's WTQ/TabFact loader, serialization, execution, normalization, evaluator, sample order, and core `result.jsonl` fields. The second-paper additions are Question Intent IR, AST/Pandas/runtime Code Intent IR, Global/Operator/Parameter semantic audits, bounded semantic repair, joint selection, and paired silent-failure metrics.

Recommended result directories:

- `outputs/100_cycle1_clean/`: current best joint accuracy on the 100-sample comparison.
- `outputs/100_cycle2_clean/`: second optimization cycle and failure-analysis reference.

Credentials are read from the process environment or the server-local `.secrets/paratera.env`; secrets are not stored in source, logs, or result files.
