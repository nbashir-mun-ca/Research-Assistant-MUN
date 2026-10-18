# Research Assistant MUN
Research assistant work at Memorial University: machine learning, code, and research tasks.

## Start here

Open [Day 1 notebook](day01/Risk_Sharing_Day_1.ipynb) for the calculations, explained in eight short steps. In Colab, upload the notebook and select **Runtime → Run all**.

For Day 2, start with the [short learning notebook](day02/Risk_Sharing_Day_2.ipynb)
or its [plain-language explanation](day02/README.md). The full validation code is
separate from the notebook so you can learn one small calculation at a time.

| File | Purpose |
|---|---|
| `day01/Risk_Sharing_Day_1.ipynb` | Main learning and working notebook |
| `day01/day1_check.py` | The same calculations as a Python script |
| `day01/README.md` | Model details, expected results, and review checklist |
| `day02/Risk_Sharing_Day_2.ipynb` | Short Day 2 notebook: objective, gradient, tail scores, oracle, and small checks |
| `day02/day2_example.py` | The same short Day 2 calculations as a script |
| `day02/model.py` | Reusable objective, gradient, feasibility checks, and oracle |
| `day02/validate_day2.py` | Independent finite-difference and HiGHS validation suite |
| `day02/evidence/` | Recorded provisional validation evidence |
| `day03/Risk_Sharing_Day_3.ipynb` | Day 3 learning notebook: line search, loop, certificate, and audit |
| `day03/day3_example.py` | The same Day 3 calculations as a script |
| `day03/README.md` | Simple explanation, results, and run instructions |
| `day03/evidence/` | Original checked Colab results |
| `day04/Risk_Sharing_Day_4.ipynb` | Self-contained eight-agent Example 2 notebook |
| `day04/day4_example.py` | The same Day 4 model, solver, and checks as a script |
| `day04/README.md` | Day 4 explanation, results, and run instructions |
| `day04/evidence/` | Original checked Day 4 Colab tables, diagnostics, and plots |
| `requirements.txt` | NumPy, SciPy, and Matplotlib for Days 1–4 |

Run locally from this folder:

```bash
python -m pip install -r requirements.txt
python day01/day1_check.py
python day02/day2_example.py
python day02/validate_day2.py
python day03/day3_example.py
MPLBACKEND=Agg python day04/day4_example.py
```

Results go into `risk_sharing_day1/` and `day1_results.zip`. Software versions are recorded there without a long printout. These generated files are ignored by Git. Save notebook changes explicitly; Colab does not automatically sync to this repository.

## Days 1–3 completed under delegated scope

The implementation and theory deliverables for the documented small finite-grid
baseline are now complete. The [completion audit](audit_days01_03/README.md)
links the model card, scalar worked calculations, gradient/oracle derivation,
certificate proof, explicit iteration and analytic small-grid comparison.
Nouman reports Parsa authorized completion of both roles with ChatGPT assistance.
Original provisional records remain historical; no personal independent review
by Parsa is asserted. Full Example 2 replication is completed in [Day 4](day04/README.md).

## Day 4 completed under delegated scope

The [Day 4 notebook](day04/Risk_Sharing_Day_4.ipynb) contains the complete
eight-agent replication, analytic derivation, checks, plots, and presentation
notes. It runs in Colab without uploads. The original executed notebook and
eleven result files are preserved. See [Day 4](day04/README.md) for local run
instructions and the finite-grid certificate scope. Day 5 stability evidence is now available in [Day 5](day05/README.md).

## Day 5 audit and handover

Start with the self-contained [Day 5 notebook](day05/Risk_Sharing_Day_5.ipynb).
The audit covers five grids, original and tighter stopping tolerances, a common
20,000-node evaluation, alternative cases and all four Example 5 weights.
Read the [research memo](day05/RESEARCH_MEMO.md) for results and the retained
original-profile allocation miss and mixed-case iteration limit.

```bash
python -m pip install -r day05/requirements-verified.txt
python day05/day5_audit.py
```

The Day 5 audit passes for the stated tightened Example 2 baseline; it does not
claim that every alternative converged or provide a continuous error certificate.
