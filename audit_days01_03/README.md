# Days 1–3 completion audit

**Complete: the delegated implementation and theory deliverables for Days 1–3,
using the documented small finite-grid baseline.** Prepared by Nouman with
ChatGPT assistance following Nouman's report of Parsa's permission to complete
both roles. This record supersedes the earlier provisional preparation status
for deliverable completion. It does not assert Parsa's personal independent
review or that the original two-person review meeting occurred.

## What was completed

| Day | Code and evidence | Completed theory deliverable |
|---|---|---|
| 1 | Runnable example, two allocations, objective and feasibility checks | [Model card](../day01/MODEL_CARD.md), [explicit worked calculations](../day01/WORKED_EXAMPLE.md) |
| 2 | General model, finite differences, hand fixture and direct-consumption HiGHS comparisons | [Gradient, increment, tail-score and exact-oracle derivation](../day02/DERIVATION.md) |
| 3 | Line search, logs, gap stopping, SLSQP and final HiGHS comparison | [Step derivation, certificate proof, explicit iteration and analytic small-grid benchmark](../day03/DERIVATION.md) |

The model card makes wealth versus loss, rank versus wealth, agent order, weights,
probability convention, objective, domain, budget and monotonicity explicit.
The Day 2 proof covers both directions of the increment/consumption equivalence,
where each gradient weight enters, oracle optimality including ties and negative
scores, and the linear hand-fixture value 17/6. Day 3 proves the certificate and
separates welfare optimization error from grid/distribution approximation.

## Separate scalar verification

[check_theory.py](check_theory.py) uses Python Decimal at 50-digit working
precision. It evaluates scalar utility terms from exact rational nodes without
calling the shared model's utilities, objective or gradient. It also computes
first-step gradients and tail scores, bisects the explicit line-search derivative,
and obtains the two-agent pointwise analytic optimum. That allocation is checked
for monotonicity on these three nodes before being used as a global grid benchmark.
Decimal is a high-precision diagnostic, not an interval-arithmetic proof.

| Calculation | Result |
|---|---:|
| Equal-sharing welfare | 1.236217807543917768 |
| One-third / two-thirds welfare | 1.235554019370132683 |
| First step | 0.110112563741513479 |
| Welfare after first step | 1.236430915639813286 |
| Analytic optimal welfare on three nodes | 1.236811579347415741 |
| Absolute difference from saved solver welfare | about 4.4e-16 |

Raw values are in [theory_reference.json](evidence/theory_reference.json), and
all six scalar utility terms per allocation are in
[tiny_example_terms.csv](evidence/tiny_example_terms.csv).

The existing Day 2 numerical suite has 12 gradient cases and 38 oracle/LP
comparisons. Day 3 reaches the 1e-9 tolerance after 291 updates, with final gap
9.597698538699184e-10. Budget residuals remain at floating-point rounding levels.
The original Colab evidence is preserved in `day03/evidence/`.

## Reproduce the completion checks

From the repository root:

```bash
python -m pip install -r requirements.txt
python day01/day1_check.py
python day02/validate_day2.py --out day02/results_completion
python day03/day3_example.py
python audit_days01_03/check_theory.py
```

Day 1's existing script overwrites its generated working-directory outputs.
Use a fresh working directory if retaining several Day 1 runs. Day 2's output
folder is explicit; Day 3 creates a timestamped folder. The scalar audit rewrites
its two generated reference files. Save previous runs before rerunning if needed.
Actual completion-run versions and source hashes are in
[evidence/completion_checks.json](evidence/completion_checks.json).

## Completion scope and remaining limits

[completion_status.json](completion_status.json) records the delegated completion
and the source of authorization. The older review fields in
`day02/review_status.json` remain intact as historical personal-review records;
a new dated section records delegated completion. Original evidence has not been
relabeled or altered to manufacture another person's approval.

The original plan requested independently conducted human calculations, including
an oracle prediction before seeing code. We cannot recreate that history. The
explicit derivations, scalar reference and independent optimization formulations
provide alternative verification under the expanded scope. A review by a separate
researcher remains distinct from these completed assisted deliverables.

The supplied replication code package is unavailable; it has not been run.
Days 1–3 cover the small example, not the full eight-agent Example 2 experiment.
The gap certifies fixed-grid welfare optimization within numerical accuracy,
not continuous-model accuracy. Day 4 and Day 5 are outside this completion record.

Sources: five-day plan pp. 1–3; replication guide sections 3–5, pp. 4–7;
Carlier–Lachapelle, equation (2.5), adjoint (3.5), and Example 2.

## Notebook status correction

The current notebooks and matching scripts now report delegated deliverable
completion after successful checks. All 27 code cells were executed and their
displayed outputs refreshed. Day 1 section 7 no longer prints the obsolete
Parsa-pending message. Day 2/3 new-run summaries use the same completion scope.
Original numerical evidence remains historical and is retained unchanged.
The updated scripts also passed when run from the repository root; see
[evidence/status_fix_checks.json](evidence/status_fix_checks.json).
