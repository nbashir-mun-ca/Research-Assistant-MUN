# Dynamic programming alternative

This implements a Bellman solver for the **finite wealth/rank-grid monotone
risk-sharing problem already used in Days 1–5**. It discretizes consumption as
well as wealth. It preserves the original scripts and results.

For the eight-agent Example 2, identical agents are averaged into two groups:
three `h2` agents and five `h1` agents. This is the same concavity-based symmetry
reduction derived in [Day 4](../day04/DERIVATION.md). The DP chooses group totals
and then divides each total equally among that group's agents. Original agent
ordering is preserved.

## Run it

From the repository root, using Python 3.10 or later:

```bash
python -m pip install -r requirements.txt
python -m unittest dynamic_programming.test_solver -v
python -m dynamic_programming.example
python -m dynamic_programming.example --states 400 --refinement 4 --output dynamic_programming/results/refined
python -m dynamic_programming.compare_nouman
```

The example prints welfare, the analytic comparison, feasibility residuals,
the finite-grid duality bound, allocation error, and elapsed solver time.
`--output` also saves `summary.json` and `allocation.csv`; these generated
results are ignored by Git. The commands use the existing dependencies.

`--states` changes the wealth/rank quadrature grid. `--refinement` changes
only the consumption lattice: its step is `1/(2*states*refinement)`.
The default is 400 states and refinement 1. Refinements by integer factors
produce nested consumption lattices, so their optimal welfare cannot decrease
apart from numerical roundoff.

## Bellman state and recurrence

Let `h` be the consumption quantum, and require each supplied wealth value to
satisfy `s_j = h*q_j` for a nonnegative integer `q_j`. Off-lattice wealth inputs
are rejected (a small tolerance accommodates floating-point representation).

With two agents or two groups, a state `x` at wealth node `j` means group 1
receives `h*x` and group 2 receives `h*(q_j-x)`. Both consumptions must be
nondecreasing. A predecessor state `k` at node `j-1` must therefore satisfy

```text
max(0, x - (q_j - q_(j-1))) <= k <= min(x, q_(j-1)).

V_j(x) = R_j(x) + max_k V_(j-1)(k)
V_(-1)(0) = 0, q_(-1) = 0.
```

`R_j` is that node's weighted utility contribution, including its probability.
For the eight-agent case (group 1 is `h2`):

```text
R_j(x) = p_j * [3*log(1 + r_j^(2/3) + h*x/3)
                + 5*log(1 + sqrt(r_j) + h*(q_j-x)/5)].
```

The terminal state with largest `V` determines welfare. Stored predecessor
indices reconstruct the full allocation path. Every feasible path has exactly
one feasible predecessor at each step of its own history; induction on the
Bellman recurrence therefore gives the global optimum **on this lattice**,
subject to floating-point evaluation. This searches whole monotone paths,
rather than optimizing each wealth node independently.

The predecessor interval moves right as `x` increases. A decreasing deque of
predecessor values computes its maximum in amortized constant time per state.
Time and traceback storage are `O(sum_j (q_j+1))`. Without this sliding-window
step, explicitly searching every predecessor would be substantially slower.

For three or more groups, the implementation enumerates integer allocation
vectors and feasible increment vectors. A state is a nonnegative vector
summing to `q_j`; a transition adds a nonnegative vector summing to
`q_j-q_(j-1)`. There are `choose(q_j+g-1,g-1)` states at node `j`. This grows
quickly with the number of groups. Before constructing states, the solver
checks cumulative state and transition counts and refuses runs above explicit
limits. Use coarse lattices for general unequal-weight, multi-agent cases;
the default fine eight-agent run is practical because it has only two groups.
The count limits are safeguards, not byte-level memory guarantees.

## Use with the existing model

```python
from day04.day4_example import ExperimentConfig, RiskSharingModel
from dynamic_programming import solve_log_model

model = RiskSharingModel.from_config(ExperimentConfig(states=400))
result = solve_log_model(model, quantum=1/800)
print(result.objective)
print(result.allocation)  # Same (wealth states, original agents) layout.
print(result.finite_grid_upper_bound)
```

`solve_log_model` accepts the existing `RiskSharingModel`, including nonuniform
ranks and probabilities, zero probabilities, and unequal Pareto weights.
Agents are grouped only if both utility offsets and weights are exactly equal.
Set `group_identical=False` to discretize each individual consumption instead.

These are **different lattices**: a group's total is a multiple of `h`, but an
individual within a group of size `k` receives a multiple of `h/k`. Grouping is
an exact reduction of the continuous-consumption log problem; it does not
claim equivalence to an individual-consumption lattice with the same `h`.

## Accuracy and scope

For this concave log model, the solver also evaluates the existing tail-score
linear oracle at its allocation `C`. With `G = grad F(C) · (oracle - C)`,

```text
F(C) <= optimum with continuous shares on the finite wealth grid <= F(C) + G.
```

The upper bound uses concavity, independently of lattice optimality. This is
the [Day 2](../day02/DERIVATION.md) / [Day 4](../day04/DERIVATION.md) certificate,
evaluated at the DP allocation. It requires no Frank–Wolfe iterations. It is a
numerical bound subject to floating-point roundoff, not interval arithmetic.
Welfare accuracy does not by itself bound allocation error, so the example
reports both against the analytic benchmark.

For two groups, convergence under consumption refinement also has a direct
explanation. Round the first total of any feasible continuous allocation
down to a multiple of `h`; allocate the residual wealth to the second group.
Because wealth differences are integer multiples of `h`, both rounded totals
remain nondecreasing. Each total changes by less than `h`. Continuity of the
stage utilities then makes the optimal lattice welfare approach the optimal
continuous-share welfare as nested lattices are refined on a fixed wealth
grid. This statement does not supply an error rate for general multi-group
lattices.

The DP does **not** integrate over continuous wealth, optimize lotteries, or
establish the intended future Friedman–Savage model. The general
`solve_lattice(wealth, quantum, stage_reward, num_groups)` interface does
accept nonconcave, finite, additive rewards: `stage_reward(j, shares)` receives
a `(candidates, groups)` array and returns one reward per row. Its Bellman
lattice optimum remains valid without concavity. However, symmetry averaging
and the log adapter's gradient upper bound must not be applied to arbitrary
nonconcave utilities. A nonconcave test validates this distinction at the
discrete algorithm level, rather than claiming an economic-model replication.

## Verification

The test suite compares complete feasible-path enumeration with both the
two-group deque solver and the general three-group solver, including
nonconcave rewards, negative rewards, ties, zero wealth, and repeated wealth
nodes. It also checks Example 2's analytic solution, nested refinement, agent
permutation, unequal weights, nonuniform ranks, zero probabilities, feasibility,
the duality gap, resource guards, and a case with binding monotonicity against
an independent direct SLSQP numerical reference.

Recorded local results are in [VALIDATION.md](VALIDATION.md).

## Compare directly with Nouman's saved results

`python -m dynamic_programming.compare_nouman` checks the preserved Day 4 and
Day 5 allocation CSVs and their configurations, not only their summary values.
It refuses comparisons with different wealth nodes, ranks, probabilities,
weights, agent types, or ordering. It independently recomputes log welfare
and the maximum linearized gain from the allocation data.

The default audit runs DP at 100, 200, 400, and 800 wealth states, with nested
consumption refinements 1 and 2; 400 states also uses refinement 4. It compares
with both original and tightened Day 5 evidence and the original Day 4 run.
It independently solves scalar marginal-utility equations, checks feasibility,
and verifies compatibility of the welfare bounds. Finally, it freshly reruns
Nouman's original and tightened 400-state Frank-Wolfe profiles and checks their
reproducibility against the saved allocations.

Results, source hashes, a comparison CSV, and DP allocation CSVs are saved to
`dynamic_programming/results/nouman_comparison/` (ignored by Git). Use `--output`
to choose another directory. The audit reports known allocation-tolerance
misses separately from compatibility of welfare certificates. Agreement means
the methods solve the same specified problem to their reported numerical
accuracies; it does not require bit-identical shares.

Run both test suites with:

```bash
python -m unittest dynamic_programming.test_solver dynamic_programming.test_comparison -v
```
