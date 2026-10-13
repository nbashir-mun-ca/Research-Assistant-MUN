# Day 1: explicit tiny-example calculations

Prepared by Nouman with ChatGPT assistance under reported Parsa authorization.
This is a calculator-style derivation, verified by a separate 50-digit scalar
implementation. It is not presented as Parsa’s personal hand calculation.

For every row, substitute rank r and the two consumptions into

$$A_j=\log(1+\sqrt{r_j}+c_{j1}),\quad B_j=\log(1+r_j^{2/3}+c_{j2}).$$

Then average the three row sums: F=(A_0+B_0+A_1+B_1+A_2+B_2)/3.

## Equal sharing

| Rank | h1 utility | h2 utility | Weighted row total |
|---|---:|---:|---:|
| 0.166666667 | 0.399837049454 | 0.326556643112 | 0.242131230855 |
| 0.5 | 0.671467250638 | 0.631250779253 | 0.434239343297 |
| 0.833333333 | 0.845669791162 | 0.833871909012 | 0.559847233391 |

## One third / two thirds

| Rank | h1 utility | h2 utility | Weighted row total |
|---|---:|---:|---:|
| 0.166666667 | 0.381038421603 | 0.346397491665 | 0.242478637756 |
| 0.5 | 0.627954284221 | 0.674623602654 | 0.434192628958 |
| 0.833333333 | 0.784197713187 | 0.892450544781 | 0.558882752656 |

Using full precision before rounding gives **1.236217807543917768** for equal
sharing and **1.235554019370132683** for one third / two thirds. Equal sharing
is better among these two candidates; Day 1 does not establish global optimality.

Both budgets clear algebraically, both columns are positive and increasing,
and probabilities sum to one. Exact fractions are in [MODEL_CARD.md](MODEL_CARD.md).
The scalar reference and existing vectorized Day 1 calculations agree within 1e-15.
Rounding the displayed row terms before averaging may change the last shown digits.

Full precision is saved in
[the row-term CSV](../audit_days01_03/evidence/tiny_example_terms.csv).
Run `python audit_days01_03/check_theory.py` to reproduce these calculations
without calling the shared model’s utility or objective functions.
