"""Day 1 runnable example; code cells from the learning notebook."""
import numpy as np

TOLERANCE = 1e-12
number_of_states, number_of_agents = 3, 2
ranks = (np.arange(number_of_states, dtype=float) + 0.5) / number_of_states
wealth = ranks.copy()  # Uniform [0,1] ONLY: keep a separate array.
probabilities = np.full(number_of_states, 1.0 / number_of_states)
pareto_weights = np.ones(number_of_agents)

print("Wealth:", wealth)
print("Ranks:", ranks)
print("Probability sum:", probabilities.sum())

allocations = {
    "equal": np.column_stack((wealth / 2, wealth / 2)),
    "one_third_two_thirds": np.column_stack((wealth / 3, 2 * wealth / 3)),
}

for name, consumption in allocations.items():
    print(f"\n{name}")
    print("  Wealth     Agent 1    Agent 2")
    for amount, row in zip(wealth, consumption):
        print(f"  {amount:.6f}   {row[0]:.6f}   {row[1]:.6f}")

def check_feasibility(consumption):
    consumption = np.asarray(consumption, dtype=float)
    if consumption.shape != (number_of_states, number_of_agents) or not np.isfinite(consumption).all():
        raise ValueError(f"Allocation must be finite with shape ({number_of_states}, {number_of_agents}).")
    budget = float(np.max(np.abs(consumption.sum(axis=1) - wealth)))
    minimum = float(consumption.min())
    increment = float(np.diff(consumption, axis=0).min())
    return {
        "maximum_absolute_budget_residual": budget,
        "minimum_consumption": minimum,
        "minimum_monotonicity_increment": increment,
        "nonnegativity_violation": max(0.0, -minimum),
        "monotonicity_violation": max(0.0, -increment),
        "feasible": bool(budget <= TOLERANCE and minimum >= -TOLERANCE and increment >= -TOLERANCE),
    }

def calculate_welfare(consumption):
    utility = np.column_stack((
        np.log1p(np.sqrt(ranks) + consumption[:, 0]),
        np.log1p(ranks ** (2.0 / 3.0) + consumption[:, 1]),
    ))
    contributions = probabilities[:, None] * pareto_weights[None, :] * utility
    return float(contributions.sum()), contributions

import math

assert np.isfinite(wealth).all() and np.isfinite(ranks).all() and np.isfinite(probabilities).all()
assert np.all(wealth >= 0) and np.all(np.diff(wealth) > 0)
assert np.all((ranks > 0) & (ranks < 1)) and np.all(np.diff(ranks) > 0)
assert np.all(probabilities >= 0) and abs(float(probabilities.sum()) - 1.0) <= TOLERANCE
assert pareto_weights.shape == (number_of_agents,) and np.all(pareto_weights > 0)

assert number_of_states >= 2, "Use at least two wealth states."
assert number_of_agents == 2, "This example implements exactly two utility types."
assert wealth.shape == ranks.shape == probabilities.shape == (number_of_states,)

results = {}
for label, consumption in allocations.items():
    check = check_feasibility(consumption)
    assert check["feasible"], (label, check)
    value, terms = calculate_welfare(consumption)
    # Scalar arithmetic cross-check; separate high-precision verification is in audit_days01_03.
    scalar_value = sum(
        float(probabilities[j]) * (
            float(pareto_weights[0]) * math.log(1 + math.sqrt(float(ranks[j])) + float(consumption[j, 0]))
            + float(pareto_weights[1]) * math.log(1 + float(ranks[j]) ** (2 / 3) + float(consumption[j, 1]))
        ) for j in range(number_of_states)
    )
    assert abs(value - scalar_value) <= TOLERANCE
    results[label] = {**check, "welfare": value,
                      "scalar_crosscheck_absolute_error": abs(value - scalar_value)}

print(f"{'Allocation':<25} {'Feasible':<10} {'Budget error':<14} Welfare")
for name, result in results.items():
    print(f"{name:<25} {str(result['feasible']):<10} "
          f"{result['maximum_absolute_budget_residual']:<14.2e} {result['welfare']:.12f}")

bad_budget = allocations["equal"].copy()
bad_budget[1, 0] += 0.01
assert not check_feasibility(bad_budget)["feasible"]

bad_domain = allocations["equal"].copy()
bad_domain[0] = [-0.01, wealth[0] + 0.01]
assert not check_feasibility(bad_domain)["feasible"]

bad_monotone = allocations["equal"].copy()
bad_monotone[-2] = [wealth[-2], 0.0]
bad_monotone[-1] = [0.0, wealth[-1]]
assert not check_feasibility(bad_monotone)["feasible"]

print("PASS: input, feasibility, welfare, and error-detection checks.")
print("Day 1 complete for the documented small finite-grid baseline: code, theory notes, and verification.")

#@title Save results and environment record
import json
import platform
import sys
import zipfile
from pathlib import Path
from importlib.metadata import version, PackageNotFoundError

OUT = Path("risk_sharing_day1")
OUT.mkdir(exist_ok=True)

for label, consumption in allocations.items():
    value, terms = calculate_welfare(consumption)
    np.savetxt(
        OUT / (label + ".csv"),
        np.column_stack((wealth, ranks, probabilities, consumption, terms, terms.sum(axis=1))),
        delimiter=",",
        header="wealth,rank,probability,c1,c2,weighted_h1,weighted_h2,state_welfare",
        comments="", fmt="%.17g",
    )

packages = {}
for name in ["numpy", "scipy", "matplotlib"]:
    try:
        packages[name] = version(name)
    except PackageNotFoundError:
        packages[name] = "NOT INSTALLED"
environment = {"python": sys.version, "platform": platform.platform(), "packages": packages}
environment["matches_requested_python_3_12"] = sys.version_info[:2] == (3, 12)
config = {"status": "COMPLETE: delegated Day 1 deliverables; small finite-grid baseline", "m": number_of_states, "n": number_of_agents,
          "wealth": wealth.tolist(), "ranks": ranks.tolist(), "probabilities": probabilities.tolist(),
          "pareto_weights": pareto_weights.tolist(), "agent_types": ["h1", "h2"],
          "quadrature": "uniform midpoint", "tolerance": TOLERANCE}
(OUT / "config.json").write_text(json.dumps(config, indent=2))
(OUT / "results.json").write_text(json.dumps(results, indent=2))
(OUT / "environment.json").write_text(json.dumps(environment, indent=2))
(OUT / "requirements-recorded.txt").write_text("\n".join(
    name + "==" + value for name, value in packages.items() if value != "NOT INSTALLED") + "\n")
(OUT / "review_status.json").write_text(json.dumps({
    "computational_checks": "passed",
    "model_card": "completed in day01/MODEL_CARD.md",
    "worked_calculations": "completed; separate scalar reference in audit_days01_03",
    "supplied_baseline": "not run: separate code package not supplied",
    "delegated_day1_deliverables": "complete",
    "personal_independent_review": "not claimed",
}, indent=2))

export_names = ["equal.csv", "one_third_two_thirds.csv", "config.json", "results.json",
                "environment.json", "requirements-recorded.txt", "review_status.json"]
with zipfile.ZipFile("day1_results.zip", "w", zipfile.ZIP_DEFLATED) as archive:
    for name in export_names:
        archive.write(OUT / name, arcname="risk_sharing_day1/" + name)
print("Saved: risk_sharing_day1/ and day1_results.zip")