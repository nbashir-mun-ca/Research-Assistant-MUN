# Day 4 presentation notes

## Your main conclusion

I completed the eight-agent Example 2 replication, including the analytic and
computational tasks under the delegated scope. The numerical solution matches the
analytic benchmark within our declared tolerances, and all feasibility and
validation checks pass. The remaining welfare improvement on the 400-state grid
is bounded by 3.31 × 10^-6, within numerical accuracy.

## A five-part presentation

1. **Question.** How should eight agents share total wealth when their two types
   have different rank-dependent logarithmic preferences? The objective is total
   weighted utility, subject to a budget in every state, nonnegative shares, and
   shares that do not fall as aggregate wealth rises.
2. **Two routes.** I reused the checked Days 1–3 model and conditional-gradient
   method, then derived a closed-form benchmark by reducing the eight agents to
   five identical h1 allocations b and three identical h2 allocations a.
   The budget is 5b+3a=s; positive allocations equalize marginal utility.
3. **Economic result.** Below s≈0.299, each h1 agent receives zero while each h2
   agent receives s/3. Above that level, both groups receive consumption. At s=1,
   all eight receive 0.125. Agents of the same type receive identical allocations.
4. **Numerical evidence.** The 400-state solver takes six updates. Welfare is
   4.146769154342 versus analytic grid welfare 4.146770230933. The actual welfare
   difference, 1.08e-6, is below the certified gap, 3.31e-6. Maximum allocation
   error is 0.002013, below the declared 0.0025 tolerance. The budget residual is
   5.55e-16. Gradient, LP, scalar-root, and small direct-optimizer checks pass.
5. **Scope and next step.** These checks establish the baseline on this finite
   grid. Day 5 should refine the grid and test stability before adapting the
   method to Friedman–Savage preferences or lotteries. The concave certificate
   must not be carried over to a nonconcave objective without a new argument.

## Figures to show

- evidence/allocation_overlay.png: solid numerical and dashed analytic curves,
  with an activation-region close-up. Lines represent consumption PER agent.
- evidence/cumulative_allocations.png: eight colored bands; band thickness is
  individual consumption and dashed boundaries are the analytic comparison.
- evidence/convergence.png: the gap falls below the chosen stopping threshold.

Use evidence/comparison_table.csv for numerical versus analytic midpoint values.
Use evidence/analytic_benchmark_table.csv for clean theoretical examples at exact
wealth levels such as 0.1, 0.5, and 1. These are different tables; label them.

## Questions you should be able to answer

**Why are there only two different curves?** Within each type, preferences and
weights are identical. Strict concavity makes equal allocation within each type
optimal; the eight agents are still modeled explicitly.

**Does equal Pareto weight mean equal consumption?** No. Equal weights count
utilities equally, but different marginal utilities can imply different shares.

**Why do the h1 agents get nothing at low wealth?** At zero h1 consumption,
the next unit benefits h2 more until the marginal utilities meet at the threshold.
This is a feature of the stated preferences, not a general recommendation for
wealth distribution or evidence that one type deserves more.

**Why does the numerical curve have tiny positive values below activation?**
The algorithm moves by convex combinations and stops at a positive gap tolerance.
The maximum leakage is below 0.000544. Strictly positive is not an appropriate
numerical test of the theoretical activation threshold.

**Why can allocation error be 0.002 even when welfare error is tiny?** Welfare
can be relatively flat around its optimum; a small welfare error does not imply
the same numerical magnitude for consumption error. Both are reported separately.

**Does the gap certify the continuous problem?** No. The exact 400-node welfare
exceeds the estimated continuous reference by 4.49e-5. That quadrature difference
is separate from the solver's 1.08e-6 welfare error on the grid.

**Did Parsa independently verify this result?** These theory and code deliverables
were prepared with ChatGPT assistance under the reported delegation. Independent
formulation checks were performed; a separate personal review is not claimed.

## Short closing you can adapt

Day 4 gives us a reproducible eight-agent baseline with an analytic benchmark,
verified constraints, and a finite-grid welfare bound. This is enough to mark the
delegated Day 4 deliverables complete and proceed to the Day 5 stability checks.
