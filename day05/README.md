# Day 5: stress tests and handover

Open `Risk_Sharing_Day_5.ipynb` in Colab and choose **Runtime > Run all**.
It carries the exact Day 4 solver and Day 5 audit inside the notebook. No GitHub
download or other upload is required. NumPy, SciPy and Matplotlib must be installed.
The supporting-source cell is collapsed; the remaining cells explain each study.
The final cell offers a ZIP download in Colab. Save the executed notebook yourself.

For a local run, extract the complete bundle, open a terminal in its root, and use:

```bash
python -m pip install -r day05/requirements-verified.txt
python day05/day5_audit.py
```

The second command reruns the complete Day 5 audit. Python 3.12 was used for the
verified run. `requirements-verified.txt` pins the numerical libraries used here;
it is not an operating-system or transitive-dependency lockfile.
Use `--out /path/to/new_run` to preserve earlier results. A reused output folder is
overwritten. Only use a run whose `run_status.json` says PASS; a failed run can
leave files from an earlier attempt.

## What is tested

- Example 2: 100, 200, 400, 800 and 1600 midpoint states, with welfare-gap tolerances
  5e-6 (original) and 1e-7 (tight).
- All grids are evaluated on the same 20,000-state grid. Linear interpolation
  anchors (0,0); at wealth 1 each agent receives an equal part of the remaining
  increment. This preserves nonnegativity, monotonicity and the budget between nodes.
- At 400 states, 1e-6 is also tested between the original and tight tolerances.
- The Day 4 independent gradient, LP, SLSQP, scalar-root and rejection checks rerun.
- Agent permutation, unequal probabilities, separate wealth/rank, and changed
  Pareto weights are checked. Direct-consumption SLSQP provides a numerical
  reference for alternative cases; it is not labeled an analytic solution.
- A zero-update probe must report nonconvergence. A harder mixed-preference case
  has a declared 2,000-update budget; its actual convergence status is retained.
- After the core tight-grid checks pass, all four Example 5 weights are tested:
  0.45, 0.48, 0.50 and 0.55. Analytic comparative statics isolate the effect of
  weights from numerical solver noise. Example 5 allocation tolerance is 0.006;
  this is separate from Example 2's 0.0025 tolerance.

## How to read PASS

PASS means that the core tightened Example 2 checks and specified validation
checks pass, and that diagnostics correctly preserve reported limitations.
It does not mean that every stress scenario converged or that the original
profile met every allocation tolerance. `summary.json` records those distinctions.
The original 800-state allocation miss is intentionally retained.

Read `RESEARCH_MEMO.md` for conclusions, limits and next steps. `results/` contains
the checked tables, figure, per-case allocations, iteration histories, settings,
environment and source hashes. Original Days 1–4 evidence is preserved unchanged.
The script reuses `day04/day4_example.py`; it does not modify its mathematics.

No separate personal review by Parsa is claimed. This implements the delegated
computational and memo deliverables; discuss the research interpretation with him.

Sources: five-day plan, p.4; replication guide, sections 6.1, 6.2, 7.1 and
milestones 6–7; Carlier–Lachapelle (2011), Examples 2 and 5. PDF editions are
identified by the unchanged Day 4 source manifest.
