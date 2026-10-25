# Local validation record

Validated on 22 October 2026, on macOS arm64 with Python 3.11.4, NumPy 2.4.4,
SciPy 1.17.1, and Matplotlib 3.10.9. The existing model and analytic benchmark
were read from repository base commit
`27d020af2320766bafad75c122f27a8d6064e7e1` and left unchanged.

## Independent checks

```text
python3 -m unittest dynamic_programming.test_solver -v
Ran 12 tests
OK
```

The 12 test methods include eight seeded comparisons against complete feasible
path enumeration. They cover two- and three-group Bellman recurrences, including
nonconcave stage rewards. Other checks cover the analytic benchmark, a direct
SLSQP numerical comparison with binding monotonicity, group symmetry, unequal
weights, zero probabilities, grid refinement, input rejection, and size guards.
These are automated checks with AI assistance, not a claim of an independent
personal mathematical review by Parsa.

## Eight-agent Example 2

Both commands keep the original 400 wealth/rank midpoint nodes. Only the
consumption lattice changes:

```bash
python3 -m dynamic_programming.example --states 400
python3 -m dynamic_programming.example --states 400 --refinement 4
```

| Metric | Refinement 1 | Refinement 4 |
|---|---:|---:|
| Group-total consumption quantum | 0.00125 | 0.0003125 |
| Cumulative lattice states | 160,400 | 640,400 |
| DP welfare | 4.146770223639475 | 4.146770230482595 |
| Analytic welfare on the same wealth grid | 4.146770230933288 | 4.146770230933288 |
| Analytic welfare minus DP welfare | 7.2938e-9 | 4.5069e-10 |
| Finite-grid concavity gap | 1.4761e-7 | 6.8272e-8 |
| Maximum per-agent allocation error | 2.0754e-4 | 5.1818e-5 |
| Maximum budget residual | 2.2204e-16 | 2.2204e-16 |
| Minimum consumption increment | 0 | 0 |
| Solver elapsed time in one local run | 0.114 s | 0.434 s |

Times exclude imports and environment setup and are illustrative single runs,
not a controlled performance comparison with Frank–Wolfe. Both reconstructed
allocations pass the existing `AllocationConstraints.check` at its default
tolerance. The analytic welfare lies between the DP welfare and its reported
finite-grid upper bound in both runs.

These results describe the concave log model on the finite wealth/rank grid.
They do not certify a continuous-wealth integral, an arbitrary nonconcave
economic model, or lottery allocations. See [README.md](README.md) for the
lattice definition, Bellman recurrence, and scaling limits.

## Direct comparison with Nouman's evidence

A fresh cross-solver audit on 22 October 2026 used the actual allocation CSVs,
configurations and summaries in `day04/evidence` and `day05/results`. Nouman's
upstream `main` was checked at `27d020af2320766bafad75c122f27a8d6064e7e1`;
the local Day 2-5 sources/evidence were unchanged from that commit. The fork's
published DP implementation was `890d038bfcd456a006e9e93e4b28c1fea0e72366`.

```bash
python3 -m unittest dynamic_programming.test_solver dynamic_programming.test_comparison -v
python3 -m dynamic_programming.compare_nouman
```

All **16 tests passed**. The new audit passed **9 DP runs**, **21 pairwise
comparisons** with **9 preserved evidence sets**, and **2 fresh Frank-Wolfe
reruns**. Wealth grids were 100, 200, 400 and 800 nodes, with consumption
refinements 1 and 2, plus refinement 4 at 400 nodes. The independent benchmark
solves scalar marginal-utility equations and verifies that the resulting
allocation is monotone before using it as a global log-model reference.

### Same-model comparison at 400 wealth states

| Allocation | Welfare | Error from analytic welfare | Maximum share error from analytic |
|---|---:|---:|---:|
| Analytic optimum | 4.146770230933288 | 0 | 0 |
| DP, refinement 1 | 4.146770223639475 | 7.2938e-9 | 2.0754e-4 |
| DP, refinement 4 | 4.146770230482596 | 4.5069e-10 | 5.1818e-5 |
| Nouman, original tolerance 5e-6 | 4.146769154341898 | 1.0766e-6 | 2.0129e-3 |
| Nouman, tightened tolerance 1e-7 | 4.146770142982100 | 8.7951e-8 | 3.6500e-4 |

The default DP welfare exceeds the original saved welfare by 1.0693e-6 and
the tightened welfare by 8.0657e-8. Its maximum difference in an individual
share is 2.0740e-3 against the original allocation and 3.6648e-4 against the
tightened allocation. These differences are consistent with the reported
numerical errors; the outputs are not bit-identical. At 400 nodes the DP
solutions are closer to the analytic optimum than either saved FW profile.

The fresh original FW run reproduced its saved allocation exactly. The fresh
tightened run reproduced welfare exactly and differed in shares by at most
1.89e-14. Every DP budget residual was at most 2.23e-16 and every DP increment
was nonnegative. Independent objective and certificate calculations agreed
with the stored summaries, and all cross-solver welfare bounds were compatible.

### Retained limitation in the older results

Nouman's original 800-state FW run has maximum share error 0.0025011081,
slightly exceeding its declared 0.0025 allocation tolerance. This was already
documented in the Day 5 memo and remains visible in the comparison report.
Its welfare bound remains valid. The tightened 800-state profile passes its
allocation tolerance. The audit does not relabel the original allocation check
as passing merely because the welfare certificates are compatible.

The comparison command writes `comparison.json`, `comparison.csv`, and the DP
allocation CSVs to `dynamic_programming/results/nouman_comparison/`. The JSON
includes input evidence hashes, executed source hashes, software versions,
recomputed feasibility and welfare, per-agent differences, and fresh FW results.
These are newly executed automated checks, not a claim of Parsa's unaided review.

Both methods target the same finite wealth/rank-grid log-utility objective.
DP adds a group-total consumption lattice; FW stops at a specified gradient
gap. Neither agreement of the two methods nor the concavity certificate bounds
continuous-wealth quadrature error or validates the future nonconcave model.
