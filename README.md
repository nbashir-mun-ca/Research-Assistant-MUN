# Research Assistant MUN
Research assistant work at Memorial University: machine learning, code, and research tasks.

## Start here

Open [Day 1 notebook](day01/Risk_Sharing_Day_1.ipynb) for the calculations, explained in eight short steps. In Colab, upload the notebook and select **Runtime → Run all**.

| File | Purpose |
|---|---|
| `day01/Risk_Sharing_Day_1.ipynb` | Main learning and working notebook |
| `day01/day1_check.py` | The same calculations as a Python script |
| `day01/README.md` | Model details, expected results, and review checklist |
| `requirements.txt` | Package needed to run Day 1 |

Run locally from this folder:

```bash
python -m pip install -r requirements.txt
python day01/day1_check.py
```

Results go into `risk_sharing_day1/` and `day1_results.zip`. Software versions are recorded there without a long printout. These generated files are ignored by Git. Save notebook changes explicitly; Colab does not automatically sync to this repository.

Day 1 status: computational implementation ready for review; Parsa's specification confirmation and independent calculation remain pending. Add future work in `day02/`, `day03/`, and so on as it is implemented.
