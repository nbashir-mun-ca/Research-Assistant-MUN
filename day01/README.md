# Day 1 reference notes

**Current status (13 October 2026): delegated Day 1 deliverables complete** for the small finite-grid baseline. See the [completion audit](../audit_days01_03/README.md). The older provisional/pending wording below records the original workflow; it does not supersede this dated status. Parsa’s personal independent review is not claimed.

Theory additions: [model card](MODEL_CARD.md) and [worked calculations](WORKED_EXAMPLE.md).

## Original workflow notes (historical)

Start with **Risk_Sharing_Day_1.ipynb**. Its eight numbered cells contain the inputs, allocations, feasibility rules, utilities, results, checks, and export code. The final saving cell is collapsed in Colab. Only NumPy is needed for the calculations.

In Colab: upload the notebook and choose **Runtime → Run all**. Save your notebook in Drive and download `day1_results.zip` from the Files panel. It contains result files, not source code. It does not push changes to GitHub.

From the repository root, run `python -m pip install -r requirements.txt` and `python day01/day1_check.py`. The script has the same calculations as the notebook; keep both in sync when editing. Outputs are written relative to the current working directory and overwritten on rerun. Environment versions are recorded silently in `risk_sharing_day1/environment.json`; the generated requirements record is not a complete dependency lockfile.

The example remains provisional pending Parsa's review. The original notebook is retained in Git history.

## The five workdays
| Day | Your work | Evidence required to finish |
|---|---|---|
| 1 | Environment and repository; arrays, ranks, probabilities; two feasible allocations; tiny objective check | Agreed model card; Parsa's independently matching objective values; budget/domain/monotonicity residuals |
| 2 | General objective, gradient, and reverse cumulative sum linear oracle | Directional finite differences and independently formulated HiGHS LP comparison |
| 3 | Conditional-gradient loop, line search, stopping rule, and gap | Feasible small run compared with independent optimizer; welfare F, gap G, and bound F+G |
| 4 | Eight-agent Example 2 | Allocations and welfare compared with Parsa's analytic benchmark on the same grid; table and overlay plot |
| 5 | Grid refinement, tolerance checks, cleanup, rerun instructions | Reproducible baseline package; separate solver and grid errors; joint review of Parsa's memo |

Move on when the checks pass, not just because the calendar advances. Neural-network work comes after the reference solver is trusted.

## Day 1 specification and remaining decisions
The eventual Example 2 target has eight agents, wealth uniform on [0,1], all Pareto weights one, h1 for agents 3–7 and h2 for agents 1,2,8. Each allocation is nonnegative, each row sums to wealth, and each column is nondecreasing.

For today's **provisional two-agent illustration**, use one h1 agent and one h2 agent, weights (1,1), and three midpoint quadrature nodes. These are not the full eight-agent experiment.

- `w`: wealth nodes, shape `(3,)`.
- `r`: probability ranks, shape `(3,)`; separate from wealth even when numerically equal.
- `p`: integration weights/probabilities, shape `(3,)`.
- `C`: consumptions, shape `(3,2)`; rows are states and columns are agents.
- `r = (j + 0.5)/3`, `w = r`, `p = 1/3`. This follows the guide's midpoint convention. It replaces the earlier informal endpoint example `[0, 0.5, 1]`; do not mix those inputs with these outputs.
- `h1(r,c) = log(1 + sqrt(r) + c)`; `h2(r,c) = log(1 + r**(2/3) + c)`. Log means natural logarithm.
- Equal sharing: `(w/2, w/2)`. Other feasible sharing: `(w/3, 2*w/3)`.
- Finite-grid welfare: `F(C) = sum_j p[j] * (h1(r[j], C[j,0]) + h2(r[j], C[j,1]))`.
- Use absolute tolerance `1e-12` for these small floating-point checks. This is a declared software tolerance, not a continuous-model error bound.

Parsa must confirm the tiny model, agent order, weights, ranks, and allocations, or supply replacements. A midpoint sum approximates the continuous rank integral; we are checking that finite sum, not claiming an exact discrete-atom utility calculation.

## Environment and supplied baseline
The guide requests Python 3.12 and reports tested versions Python 3.12.14, NumPy 2.3.5, SciPy 1.18.1, and Matplotlib 3.11.2. This notebook records the versions actually present rather than assuming Colab matches them. It needs NumPy for today's calculation; SciPy and Matplotlib are recorded for later days. Missing NumPy raises an explicit error. Do not silently treat a different environment as an exact reproduction of the supplied environment.

The separate `replication/` code package was not included among the three PDFs. Ask Parsa for it; the notebook records the supplied baseline as **not run**. If supplied, run its documented commands in that folder and retain its logs. The guide's Linux commands are `python -m venv .venv`, `.venv/bin/python -m pip install -r requirements.txt`, `.venv/bin/python run_replication.py --profile verified --out results`, and `.venv/bin/python test_replication.py`. These are for the supplied package, not this independent Day 1 script.

## Expected results and how to interpret them

| Allocation | Welfare on this three-node grid |
|---|---:|
| Equal sharing | 1.236217807544 |
| One third / two thirds | 1.235554019370 |

Both should be feasible. Probability sum should be 1 within 1e-12. Maximum budget error should be at most 1e-12. Minimum consumption and minimum monotonicity increment should be nonnegative, allowing 1e-12 rounding tolerance. A larger welfare value between these two candidates does not prove global optimality.

For hand calculation, use r=w=(1/6,1/2,5/6). Equal-sharing rows are (1/12,1/12), (1/4,1/4), (5/12,5/12). Other-sharing rows are (1/18,1/9), (1/6,1/3), (5/18,5/9).

For each row j, calculate `A_j = log(1 + sqrt(r_j) + c1_j)` and `B_j = log(1 + r_j**(2/3) + c2_j)`. Then calculate `(A_1+B_1+A_2+B_2+A_3+B_3)/3`. This is the independent calculator exercise to request from Parsa, using the confirmed specification. The CSVs include every weighted term to locate disagreements.

Our scalar and vectorized implementations agree within tolerance, but both were prepared here. **They do not satisfy the requirement for Parsa's independent calculation.** If hand values are rounded, agree a suitable comparison tolerance based on retained digits; do not demand 1e-12 from a six-decimal calculator answer.

## What to send to Parsa at review
- Draft model card and exact inputs in `config.json`.
- Both allocation CSVs and objective/feasibility values in `results.json`.
- Recorded environment and the missing supplied-code-package status.
- Ask him to confirm or replace the illustrative specification, and compare his two independently computed welfare values.
- Record disagreements and resolve them before moving to Day 2.

## Day 1 is finished only when
1. The project environment and repository are set up and commands/versions recorded.
2. The supplied baseline has been run if its code is available; otherwise its unavailability is recorded and raised with Parsa.
3. Parsa's model card and tiny example are saved and agreed.
4. Probabilities, shapes, domain, statewise budget, and monotonicity checks pass for equal sharing and the other allocation.
5. Your objective values match Parsa's independent calculations to an agreed tolerance.
6. The notebook and results are saved outside the temporary Colab runtime and reviewed together.

If steps 3 or 5 are pending, report **implementation ready for review; Day 1 joint gate pending**, not “Day 1 complete.”

## Troubleshooting
- `NameError`: rerun all numbered code cells in order.
- Missing NumPy: run `%pip install numpy` in a separate Colab cell, then rerun all cells.
- An assertion fails: retain the exact error and inputs. Do not loosen tolerances until the cause is understood.
- Results differ: check midpoint versus endpoint grid, natural logarithms, agent order, weights, and probability factors first.
- After a runtime reset: rerun the notebook to regenerate outputs; use your saved notebook and downloaded archive.

## Sources
- `Parsa_and_Nouman_Five_Day_Research_Plan (1).pdf`, Day 1 on page 2; daily roadmap and later gates on pages 1–4.
- `replication_guide.pdf`, environment on printed page 3; wealth, utilities and midpoint convention on page 4.

No supplied baseline or full eight-agent optimization was executed in this Day 1 notebook.
