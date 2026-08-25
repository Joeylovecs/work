# Research Method Spec

## Research question
How can a tool-augmented TableQA agent detect semantic logic errors after generated Python code has executed successfully?

## Main concept
Execution-Trust Bias: an agent over-trusts successful execution as evidence that the program faithfully implements the natural-language question.

## Core method
Question Intent IR -> Python execution -> Hybrid Code Intent IR -> 3-level semantic alignment -> Semantic Logic Exception -> targeted repair -> re-execute -> re-audit.

## Three audit levels

### Global
Checks final task/answer contract:
- answer type
- cardinality
- task type
- final return semantics

### Operator
Checks operation graph/sequence:
- filter/select
- aggregation
- comparison
- arithmetic
- groupby
- ranking/sort/top-k
- temporal operations
- boolean composition

### Parameter
Checks concrete grounded arguments:
- columns
- cell/entity values
- constants
- dates/years
- comparators
- target/sort columns
- aliases

## Hybrid evidence priority
1. Deterministic AST evidence
2. Runtime trace evidence
3. Table schema/cell grounding
4. LLM semantic matching only for unresolved language ambiguity

## Exception schema
Each semantic error should record:
- level
- error_type
- expected
- actual
- evidence
- confidence
- repair_hint

## Repair policy
- Max semantic repairs: 2 by default
- Feed concise diagnostics, not full old traces
- Every repair must re-execute and re-audit
- Correct existing programs must not be changed unless there is audit evidence

## Main causal comparison
Same model + same samples + same evaluator + same executor:
- Python baseline
- Python + multi-granularity semantic audit

This comparison is primary. Routing/hybrid reasoning is secondary.
