# Day 4: full eight-agent Example 2

The delegated Day 4 implementation and analytic deliverables are complete. The
uploaded Colab run passed all seven acceptance gates and thirteen independent
linear-oracle comparisons. Parsa's separate personal review is not claimed.

Start with [the self-contained notebook](Risk_Sharing_Day_4.ipynb). Open it in
Colab and select **Runtime → Run all**. No other uploads are needed. The original
executed notebook, explanations, selective class/function comments, and lambda
comments are preserved.

## Files

| File | Purpose |
|---|---|
| `Risk_Sharing_Day_4.ipynb` | Original executed Colab notebook with all definitions and outputs |
| `day4_example.py` | The same definitions with a guarded local entry point |
| `DERIVATION.md` | Analytic allocation, activation threshold, monotonicity proof, certificate |
| `PRESENTATION_NOTES.md` | Presentation outline and questions |
| `evidence/` | All eleven original Colab result files, including three plots |
| `source_manifest.json` | SHA-256 hashes of uploaded inputs and supplied source editions |

## Model and result

Eight agents share uniform aggregate wealth on [0,1], represented by 400 midpoint
states with probability 1/400 each. Pareto weights are all one. Agents 1, 2, and 8
use h2(r,c) = log(1 + r^(2/3) + c); agents 3–7 use
h1(r,c) = log(1 + sqrt(r) + c). Here rank r equals aggregate wealth s.
Every allocation is nonnegative, state-wise budget balanced, and nondecreasing
in wealth for each agent. Each agent has its own allocation column.

| Measurement | Original Colab run |
|---|---:|
| Solver updates | 6 |
| Numerical welfare | 4.146769154341898 |
| Analytic grid welfare | 4.146770230933288 |
| Actual welfare error | 1.0765913893706625e-6 |
| Certified grid gap | 3.307203389844745e-6 |
| Maximum allocation error | 0.002012878969562523 |
| Allocation tolerance | 0.0025 |
| Maximum budget residual | 5.551115123125783e-16 |
| Analytic activation wealth | 0.29899568433689316 |

Below activation, the analytic allocation gives each h2 agent s/3 and each h1
agent zero. Above activation both types receive consumption. Small positive
numerical h1 shares below activation reflect the stopping tolerance.

The certificate bounds remaining welfare improvement on the fixed 400-state
grid, subject to floating point accuracy. It does not bound continuous-grid
approximation error. Day 5 grid refinement and stability analysis remain separate.

## Run locally

From the repository root:

```bash
python -m pip install -r requirements.txt
MPLBACKEND=Agg python day04/day4_example.py
```

Use `python day04/day4_example.py` for interactive plotting. The script generates
`day04/results/` and `day04/results_download.zip`; these are ignored by Git.
The original uploaded outputs in `evidence/` stay unchanged. A rerun executes the
gradient, thirteen HiGHS oracle cases, small SLSQP comparison, scalar root,
feasibility rejection checks, and seven final acceptance gates.

The Day 4 GitHub Actions workflow runs this same command and uploads the new
results. Local checks and GitHub Actions status should be distinguished: a local
PASS does not establish that the hosted workflow has completed.

## Provenance

The notebook is copied byte for byte from
`Risk_Sharing_Day_4_Standalone(1).ipynb`; evidence is extracted byte for byte from
`day4_results_download.zip`. The script reuses notebook definitions without
changing calculations; its entry point chooses a repository-local output path.
The derivation and presentation notes are also included in the notebook.
Prepared by Nouman with ChatGPT assistance under the reported delegation of both
Day 4 roles. Sources are the supplied five-day plan, replication guide, and
Carlier–Lachapelle paper; hashes identify their editions without adding the PDFs.
