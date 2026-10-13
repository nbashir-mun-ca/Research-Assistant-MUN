# Day 3: improve the allocation, measure what remains

**Status:** the small-run implementation and numerical audit pass. Parsa's
specification, theory review, and independent one-step calculation remain pending.

Start with [the learning notebook](Risk_Sharing_Day_3.ipynb). Each short section
explains one step. The [Python script](day3_example.py) runs the same code cells.

## What changed after Day 2?

Day 2 gave us three tools: the objective scores an allocation; the gradient gives
local guidance; the linear oracle proposes a feasible allocation. Day 3 adds
line search and repeats these tools until the remaining welfare gap is small.

Imagine two people sharing a bonus that can be small, medium, or large. We start
by dividing each possible bonus equally. Different preferences mean another
split can produce higher weighted welfare. Every allocation must distribute the
whole bonus, give nonnegative shares, and keep each person's share from falling
when the total bonus rises. These are alternative scenarios, not payments over time.

## Why not accept the oracle's suggestion immediately?

The oracle optimizes a linear approximation. Actual log utility is curved, with
decreasing extra benefit from additional consumption. The first oracle gives all
wealth to agent 2, but line search selects only part of that proposed change:

$$C_{new}=(1-t)C+tV,\qquad 0\leq t\leq1.$$

Here C is our current allocation and V is the oracle's candidate. A step of 0.1
means 10% of the proposed change. Both endpoints are feasible, so their mixture
is feasible too. Bisection uses 42 interval halvings to locate the best step;
endpoint checks also allow a zero or full step.

The first step is approximately 0.110112564. Welfare rises from
1.236217807544 to 1.236430915640.

## The loop and stopping rule

1. Check feasibility; compute welfare and its gradient.
2. Ask the oracle for a feasible candidate and compute the gap.
3. Stop if the gap is at most 1e-9.
4. Otherwise choose a step, update, check feasibility and welfare, and repeat.

The log records welfare, gap, and upper bound at the same current allocation.
`step_to_next` describes the following update. The iteration cap is 2000 updates;
`iteration_cap` or `stalled` is reported separately and does not pass the final audit.
Tiny negative gaps from rounding are clipped to zero; materially negative gaps fail.

## What does the gap tell us?

$$G(C)=\langle\nabla F(C),V-C\rangle.$$

For the concave objective and exact linear oracle:

$$F(C)\leq F^*\leq F(C)+G(C).$$

Concavity puts welfare below its tangent approximation, and the oracle finds the
best feasible value of that approximation. The gap therefore bounds remaining
welfare improvement on the fixed grid, subject to numerical arithmetic. It is in
welfare units, not a percentage or an allocation-distance bound. It does not
measure the error from using only three wealth states. This argument cannot be
assumed to hold for nonconcave preferences.

## Checked results

| Measurement | Value |
|---|---:|
| Updates | 291 |
| Logged allocations (including start) | 292 |
| Final welfare | 1.2368115793474153 |
| Final gap | 9.597698538699184e-10 |
| Upper bound | 1.236811580307185 |
| Maximum budget residual | 3.3306690738754696e-16 |
| Welfare difference versus SLSQP | 2.220446049250313e-15 |
| Final oracle value difference versus HiGHS | 2.7755575615628914e-17 |

| Total wealth | Agent 1 | Agent 2 |
|---|---:|---:|
| 0.166667 | 0.030636 | 0.136031 |
| 0.500000 | 0.211427 | 0.288573 |
| 0.833333 | 0.403006 | 0.430328 |

SLSQP solves the nonlinear problem using direct consumption variables and explicit
constraints. HiGHS independently checks the final linear oracle value. SLSQP
shares the checked objective and gradient, so these comparisons supplement
Parsa's theoretical review. Close welfare values need not imply equally close shares.

## Run it

From the repository root, with Python 3.12 as the guide's target:

```bash
python -m pip install -r requirements.txt
python day03/day3_example.py
```

Open the notebook in Jupyter from the repository root or day03 folder. In Colab,
upload the notebook and [day02/model.py](../day02/model.py) into the same runtime,
then Run all. No GPU is required. An unrelated uploaded model.py changes the model;
always use the repository's Day 2 file. Inputs and tolerances are explicit in section 2.

The script writes a new `day03/results_<UTC timestamp>/` folder. In Jupyter or
Colab, it writes the results folder below the notebook's working directory.
Save the notebook and results explicitly; Colab does not automatically sync them
with GitHub. Generated result folders are ignored by Git.

## Saved evidence and provenance

`evidence/` preserves the seven files from your completed Colab run: iterations,
allocations, summary, inputs, settings, environment, and model-source hashes.
The upload was named `results_day_2_risk_sharing_RA.zip`, but its contents are
Day 3 results. These original files are preserved without relabeling their values.
Their model hash matches the exact shared `day02/model.py` used here.
New runs additionally hash the local script and notebook when available.

The repository notebook adapts file loading and output paths for the shared Day 2
module. Its optimization and audit calculations are unchanged from your uploaded
Day 3 notebook. The accompanying script is generated from its code cells.

GitHub Actions reruns the small solver and independent audit after relevant
changes. Passing checks establishes computational behavior for this case.

## What remains for the team?

Parsa reviews the specification, oracle derivation, certificate proof, and an
independent hand calculation of one iteration. Record his actual review and rerun
after agreed changes. Computational PASS does not constitute team sign-off.
The full eight-agent Example 2 comparison belongs to Day 4.

Sources: five-day research plan, Day 3, page 3; replication guide, sections
5.4–5.5, pages 6–7; the shared Day 2 implementation. The separate supplied
replication code package was unavailable; this is our independent implementation.
