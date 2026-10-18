# Day 5 research memo

**Nouman Bashir | MUN Economics | 16 October 2026**

Prepared with ChatGPT/Codex assistance for the project led technically by Parsa
under Professor Lauzier. This is an assisted replication and audit, not a claim
of Parsa's separate personal verification.

## What was reproduced

The checked baseline solves the printed eight-agent logarithmic Example 2 of
Carlier and Lachapelle. Wealth is uniform on [0,1]; midpoint rank equals wealth
in this example, although code stores both separately. Each agent has Pareto
weight one. Five agents use log(1+sqrt(rank)+consumption), and three use
log(1+rank^(2/3)+consumption). Natural logarithms, probability weights, consumption
nonnegativity, statewise clearing and monotonicity are explicit.

The analytic benchmark assigns each h1 agent
b(s)=max((s+3(s^(2/3)-sqrt(s)))/8,0) and each h2 agent a(s)=(s-5b(s))/3.
The Day 4 derivation proves feasibility and activation near 0.298995684337.
Equal weighted marginal utilities and concavity establish pointwise optimality;
monotonicity makes this answer optimal for the constrained grid problem.

## Evidence from the stress tests

Both the original gap tolerance 5e-6 and a tighter tolerance 1e-7 were tested at
100, 200, 400, 800 and 1600 states. All ten runs reached their requested gap.
Every tightened run met the unchanged allocation tolerance 0.0025. Their maximum
allocation errors ranged from about 0.000115 to 0.000861. At 400 states,
tightening the gap from 5e-6 to 1e-7 reduced maximum allocation error from
0.002013 to 0.000365 and the actual grid welfare error from 1.08e-6 to 8.80e-8.

The original 800-state run had allocation error 0.002501108, just above 0.0025.
Its welfare certificate still passed. This is a recorded allocation-tolerance
miss; no tolerance was enlarged to hide it. Finer grids and smaller welfare gaps
do not imply monotonically decreasing maximum allocation errors.

Day 4's independent gradient, thirteen HiGHS oracle cases, small SLSQP comparison,
scalar roots and rejection checks passed again. Agent permutation preserved
allocations. An alternative five-state case with unequal probabilities,
separate rank and wealth, and unequal Pareto weights agreed with SLSQP.
A harder mixed-preference alternative did not reach 1e-7 within its declared
2000 updates: gap 1.57e-5 and welfare difference from SLSQP 6.27e-6.
It remains explicitly unconverged. A separate zero-update probe also correctly
reported nonconvergence. The solver is therefore verified for the stated
baseline, not universally fast for every allowed parameter choice.

Once the core checks passed, all four Example 5 weights (0.45, 0.48, 0.50, 0.55)
were reproduced at 400 states with gap tolerance 5e-6. The separately derived
pointwise formula is c1=clip(lambda*(1+s^(2/3)+s)-(1-lambda)*(1+sqrt(s)),0,s),
c2=s-c1. Substitution into equal weighted marginals gives this expression;
feasibility on each tested grid is checked before comparison. All four cases
passed the declared Example 5 allocation tolerance 0.006. Analytic-grid
comparative statics confirmed that increasing agent 1's weight raises its
allocation and lowers agent 2's utility. This follows the guide's documented
correction of the source's reversed utility-direction sentence.

## What is proved, and what is numerical evidence

**Analytic result:** the stated Example 2 allocation, activation and monotonicity
follow from the derivation. The Example 5 expression follows from weighted
marginals; its feasible grid evaluations supply exact benchmarks for those grids.

**Finite-grid certificate:** for the concave objective and exact linear oracle,
F(C) <= F* <= F(C)+G(C), subject to floating point accuracy. The gap measures
remaining welfare improvement on that grid. It does not bound allocation
max-error or continuous-distribution approximation error. SLSQP comparisons are
numerical cross-checks, not analytic certificates of its own optimum.

**Numerical evidence:** analytic midpoint welfare minus the integrated continuous
reference decreases from 3.67e-4 at 100 states to 5.48e-6 at 1600 states.
The continuous reference is approximately 4.146725329706; adaptive quadrature's
estimated error is 8.62e-13, not an interval proof. Evaluating all allocations
on the same 20,000-node grid gives a fairer comparison. Interpolation anchors
zero and splits the final wealth increment equally at one. Even that common
grid has midpoint bias, so its welfare may slightly exceed the continuous
reference without violating optimality.

## Handover and priorities

The package includes the source, executed notebook, per-case settings,
allocations, iteration histories, figure, environment and rerun instructions.
Use the tighter Example 2 profile for allocation comparisons and retain both
solver gaps and mesh changes. Before extending difficult parameter cases,
investigate conditional-gradient convergence and compare with constrained
optimization. Parsa should review the assumptions and interpretation before
Friedman–Savage preferences or public lotteries are introduced. The current
certificate must not be transferred to a nonconcave objective.

Sources: five-day plan p.4; replication guide sections 6.1, 6.2 and 7.1;
Carlier–Lachapelle (2011), Examples 2 and 5. Source editions are hashed in the
Day 4 manifest. The separate original-author/supplied solver package was
unavailable; this is an independent implementation of the printed model.
