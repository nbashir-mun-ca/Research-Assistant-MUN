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
| `requirements.txt` | NumPy and SciPy for Days 1 and 2 |

Run locally from this folder:

```bash
python -m pip install -r requirements.txt
python day01/day1_check.py
python day02/day2_example.py
python day02/validate_day2.py
```

Results go into `risk_sharing_day1/` and `day1_results.zip`. Software versions are recorded there without a long printout. These generated files are ignored by Git. Save notebook changes explicitly; Colab does not automatically sync to this repository.

Day 1 status: computational implementation ready for review; Parsa's specification confirmation and independent calculation remain pending. Add future work in `day02/`, `day03/`, and so on as it is implemented.

Day 2 status: provisional implementation and computational checks pass; Parsa's
independent theory review and joint sign-off remain pending. Human review status
is recorded in `day02/review_status.json` and is never rewritten by the validator.
See [Day 2 instructions](day02/README.md) for output paths and review steps.

GitHub Actions runs the Day 2 example and full validation on pushes and pull
requests. Passing CI checks computational behavior; it does not establish team
approval or close the joint research review gate.

