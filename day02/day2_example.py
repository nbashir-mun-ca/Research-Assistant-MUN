"""Day 2 teaching example, generated from the short notebook code cells."""

import numpy as np
from scipy.optimize import linprog

ranks = (np.arange(3) + 0.5) / 3
wealth = ranks.copy()
probabilities = np.full(3, 1 / 3)
pareto_weights = np.array([1.0, 1.0])
consumption = np.column_stack((wealth / 2, wealth / 2))
print("Wealth:", wealth)
print("Equal shares:\n", consumption)

def objective(c):
    utilities = np.column_stack((
        np.log1p(np.sqrt(ranks) + c[:, 0]),
        np.log1p(ranks ** (2 / 3) + c[:, 1]),
    ))
    return float(np.sum(probabilities[:, None] * pareto_weights * utilities))

print("Equal-sharing welfare:", objective(consumption))

def gradient(c):
    denominators = np.column_stack((
        1 + np.sqrt(ranks) + c[:, 0],
        1 + ranks ** (2 / 3) + c[:, 1],
    ))
    return probabilities[:, None] * pareto_weights / denominators

g = gradient(consumption)
print("Gradient:\n", g)

def tail_scores(g):
    return np.cumsum(g[::-1], axis=0)[::-1]

scores = tail_scores(g)
print("Tail scores:\n", scores)

def linear_oracle(g, wealth):
    scores = tail_scores(g)
    wealth_increments = np.diff(np.r_[0.0, wealth])
    winners = scores == scores.max(axis=1, keepdims=True)
    increments = wealth_increments[:, None] * winners / winners.sum(axis=1, keepdims=True)
    return np.cumsum(increments, axis=0)

candidate = linear_oracle(g, wealth)
print("Oracle candidate:\n", candidate)

def check_feasibility(c):
    increments = np.diff(np.vstack((np.zeros((1, 2)), c)), axis=0)
    np.testing.assert_allclose(c.sum(axis=1), wealth, rtol=0, atol=1e-10)
    if c.min() < -1e-12 or increments.min() < -1e-12:
        raise ValueError("Negative consumption or decreasing share")

check_feasibility(consumption)
check_feasibility(candidate)
print("PASS: both allocations satisfy all three rules.")

direction = candidate - consumption
step = 1e-5
check_feasibility(consumption + step * direction)
check_feasibility(consumption - step * direction)
predicted = float(np.sum(g * direction))
measured = (objective(consumption + step * direction) - objective(consumption - step * direction)) / (2 * step)
gradient_error = abs(predicted - measured)
np.testing.assert_allclose(predicted, measured, rtol=1e-6, atol=1e-7)
print("Gradient check error:", gradient_error)

m, n = consumption.shape
budgets = np.zeros((m, m * n))
for j in range(m):
    budgets[j, j * n:(j + 1) * n] = 1
monotonicity = np.zeros(((m - 1) * n, m * n))
for j in range(m - 1):
    for i in range(n):
        monotonicity[j * n + i, j * n + i] = 1
        monotonicity[j * n + i, (j + 1) * n + i] = -1
lp = linprog(-g.ravel(), A_eq=budgets, b_eq=wealth,
             A_ub=monotonicity, b_ub=np.zeros((m - 1) * n),
             bounds=(0, None), method="highs")
if not lp.success:
    raise RuntimeError(lp.message)
check_feasibility(lp.x.reshape(m, n))
oracle_error = abs(float(np.sum(g * candidate)) + lp.fun)
np.testing.assert_allclose(float(np.sum(g * candidate)), -lp.fun, rtol=1e-9, atol=1e-9)
print("Oracle/HiGHS value difference:", oracle_error)
print("Computational checks passed; joint review remains pending.")

#@title Save the example and environment record
import json
import platform
import sys
import zipfile
from pathlib import Path
from importlib.metadata import version

out = Path("risk_sharing_day2")
out.mkdir(exist_ok=True)
summary = {"status": "PROVISIONAL: joint review pending",
           "equal_welfare": objective(consumption), "gradient_error": gradient_error,
           "oracle_lp_value_difference": oracle_error}
(out / "summary.json").write_text(json.dumps(summary, indent=2))
(out / "environment.json").write_text(json.dumps({"python": sys.version,
    "platform": platform.platform(), "packages": {name: version(name) for name in ("numpy", "scipy")}}, indent=2))
np.savetxt(out / "oracle.csv", np.column_stack((wealth, candidate)), delimiter=",",
           header="wealth,agent1,agent2", comments="")
with zipfile.ZipFile("day2_results.zip", "w", zipfile.ZIP_DEFLATED) as archive:
    for name in ("summary.json", "environment.json", "oracle.csv"):
        archive.write(out / name, arcname="risk_sharing_day2/" + name)
print("Saved: risk_sharing_day2/ and day2_results.zip")
