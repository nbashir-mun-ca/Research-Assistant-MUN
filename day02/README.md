# Day 2: objective, gradient, and linear oracle

**Status: provisional preparation.** The code checks pass. Parsa's Day 1 model
confirmation, independent calculations, and Day 2 theory review are still pending.

Start with [Risk_Sharing_Day_2.ipynb](Risk_Sharing_Day_2.ipynb). It teaches the
calculations in small cells. The longer validation suite stays in a separate file.

## The ideas in simple terms

Imagine Alice and Bob sharing a bonus whose total amount is uncertain.

| Term | Meaning |
|---|---|
| Agent | A person, household, or company receiving a share |
| Wealth state | One possible amount of total wealth; states are ordered from low to high |
| Consumption | The amount allocated to one agent in one state |
| Utility | The benefit the model assigns to that consumption |
| Pareto weight | How strongly an agent's utility counts in the objective |
| Welfare objective | The score used to compare feasible allocations |
| Gradient | The approximate welfare gain from a tiny increase in one consumption entry |
| Tail score | The sum of gradients from the current wealth state through the highest state |
| Linear oracle | The helper that assigns each wealth increment to a maximum-tail-score agent |

Probabilities weight states; Pareto weights weight agents. Equal Pareto weights
give utilities equal importance, while consumption can differ with preferences.

The model imposes three rules: distribute exactly the available wealth in every
state; give each agent nonnegative consumption; and keep each agent's consumption
constant or increasing as total wealth increases. The last rule is monotonicity.
These are alternative wealth scenarios, not successive payments over time.

## Why add the tail?

Suppose the bonus can be $100, $200, or $300. Equal sharing gives Alice
$50, $100, or $150. Redirecting $1 of the middle-state increment from Bob to Alice
gives Alice $50, $101, or $151 and Bob $50, $99, or $149. The increment carries
through to every higher wealth state, so both affected states count.

For Alice's invented gradients `(4, 2, 1)`, her tail scores are `(7, 3, 1)`:
the current row plus all rows below it in a low-to-high wealth table. Bob has his
own tail scores. The oracle compares them when assigning each wealth increment.
It splits an increment equally among exact ties, then cumulatively adds the
increments to produce a feasible candidate allocation.

The oracle solves the current linear approximation. The optimizer's step size,
repeated updates and convergence certificate belong to Day 3.

## What to read and run

| File | Purpose |
|---|---|
| `Risk_Sharing_Day_2.ipynb` | Short, self-contained learning notebook for the two-agent illustration |
| `day2_example.py` | Generated from the notebook code cells; same calculations |
| `model.py` | General implementation with validated inputs and any number of h1/h2 agents |
| `validate_day2.py` | Scalar objective checks, feasible finite differences, and independent HiGHS LP |
| `config.json` | Runtime input for the general implementation and validation |
| `review_status.json` | Human review record; validation preserves it |
| `DERIVATION.md` | The mathematical reasoning for Parsa's review |
| `evidence/` | Saved reference results, exact inputs, environment and source hashes |

In Colab, upload the learning notebook and choose **Runtime > Run all**. It needs
NumPy and SciPy. If either is missing, use `%pip install numpy scipy` in a separate
cell. Save your notebook explicitly; Colab does not automatically sync to GitHub.

From the repository root:

```bash
python -m pip install -r requirements.txt
python day02/day2_example.py
python day02/validate_day2.py
```

The example writes `risk_sharing_day2/` and `day2_results.zip` relative to the
working directory. Its last saving cell is collapsed in Colab. The validator
uses paths relative to its own script, writing to `day02/results/` by default.
Generated outputs are ignored by Git. To preserve a later run, choose a new folder:

```bash
python day02/validate_day2.py --config day02/config.json --out day02/results_after_review
```

The short notebook has explicit inputs for the original two-agent illustration.
The general validator reads `config.json` and also checks the configured inputs.
If Parsa changes that illustration, update the agreed notebook inputs and config
together. Source functions in the teaching notebook are cross-checked against the
general implementation before committing.

## What the evidence shows

The reference run preserves both Day 1 welfare values. It includes 12 directional
gradient cases, 44 finite-difference measurements, and 38 oracle-LP comparisons.
The largest selected gradient discrepancy is about `1.91e-8`; the largest oracle
value discrepancy is about `4.44e-16`. Numerical tolerances are declared in
`config.json`. Centered gradient checks use a predeclared `1e-5` acceptance step;
boundary checks use a forward `1e-6` step. Other steps remain in the evidence.

HiGHS solves a separately formulated LP using direct consumption variables,
budget equalities and monotonicity inequalities. It does not reuse tail scores.
Compare linear objective values, since exact ties can produce different optimal
allocations. Signed random gradients, negative scores, exact ties and zero wealth
increments are included. The small eight-agent case checks array handling; it is
not the full Example 2 optimization. Diagnostic nonuniform-probability cases are
weighted sums rather than the uniform midpoint replication.

`RUN_STATUS.txt` identifies whether a run passed or failed. A failed rerun can
leave older CSVs in its folder; only use results from a run marked passed.
`run_manifest.json` records source/config hashes and `environment.json` records
actual versions. Dependency ranges in `requirements.txt` are not a full lockfile.

## What remains before Day 2 is complete

1. Parsa confirms the Day 1 specification and independently matches its values.
2. Parsa reviews the gradient and tail-score derivations and predicts a tiny oracle
   example independently.
3. Review the numerical checks and tolerances together, resolve discrepancies,
   and rerun after any agreed changes.
4. Record the actual joint review outcome in `review_status.json`.

Passing the software checks does not automatically close either joint gate.

## Sources

- Five-day research plan, page 2, Day 2.
- Replication guide, pages 4-6, equations (6)-(9).
- Carlier-Lachapelle (2011), Example 2 on page 12 and adjoint formula (3.5) on page 6.

Attachment hashes are in `source_manifest.json`. The separate supplied code
package was unavailable; this is an independently prepared implementation.
