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
