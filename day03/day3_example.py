"""Day 3 runnable example; code cells from the learning notebook."""
import importlib.util, json, sys
from pathlib import Path
import numpy as np

base = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
candidates = [base.parent / "day02/model.py", base / "day02/model.py", base / "model.py"]
model_path = next((p for p in candidates if p.is_file()), None)
if model_path is None:
    raise FileNotFoundError("Upload day02/model.py in Colab or run inside this repository.")
spec = importlib.util.spec_from_file_location("day3_model", model_path)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
Model, feasibility, linear_oracle = module.Model, module.feasibility, module.linear_oracle


r = (np.arange(3, dtype=float) + 0.5) / 3
model = Model(r.copy(), r.copy(), np.full(3, 1/3), np.ones(2), ("h1", "h2"))
settings = dict(gap_tol=1e-9, max_iter=2000, budget_atol=1e-10,
                domain_atol=1e-12, welfare_atol=1e-13)

def check_feasible(c):
    diagnostic = feasibility(c, model.wealth,
                             settings["budget_atol"], settings["domain_atol"])
    if not diagnostic["feasible"]: raise AssertionError(diagnostic)
    return diagnostic

initial = model.equal_sharing()
print("Initial welfare:", model.objective(initial))
print(check_feasible(initial))

def line_search(c, direction):
    def slope(t):
        return float(np.sum(model.gradient(c + t * direction) * direction))
    if slope(0.0) <= 0: return 0.0
    if slope(1.0) >= 0: return 1.0
    left, right = 0.0, 1.0
    for _ in range(42):
        middle = (left + right) / 2
        if slope(middle) > 0: left = middle
        else: right = middle
    return (left + right) / 2

g = model.gradient(initial)
candidate = linear_oracle(g, model.wealth)
direction = candidate - initial
first_step = line_search(initial, direction)
after_first = initial + first_step * direction
print("Oracle allocation:\n", candidate)
print("Initial gap:", float(np.sum(g * direction)))
print("Step:", first_step)
print("Welfare after update:", model.objective(after_first))
print(check_feasible(after_first))

def solve(start):
    c, log = np.array(start, copy=True), []
    for iteration in range(settings["max_iter"] + 1):
        diagnostic = check_feasible(c)
        value, gradient = model.objective(c), model.gradient(c)
        candidate = linear_oracle(gradient, model.wealth)
        check_feasible(candidate)
        direction = candidate - c
        raw_gap = float(np.sum(gradient * direction))
        if raw_gap < -1e-12: raise AssertionError("Negative oracle gap")
        gap = max(0.0, raw_gap)
        log.append(dict(iteration=iteration, welfare=value, raw_gap=raw_gap,
                        gap=gap, upper_bound=value + gap, **diagnostic,
                        step_to_next=None))
        if gap <= settings["gap_tol"]: return c, log, "gap_reached"
        if iteration == settings["max_iter"]: return c, log, "iteration_cap"
        step = line_search(c, direction)
        updated = c + step * direction
        check_feasible(updated)
        if model.objective(updated) < value - settings["welfare_atol"]:
            raise AssertionError("Welfare decreased")
        if step == 0 or np.array_equal(updated, c): return c, log, "stalled"
        log[-1]["step_to_next"] = step
        c = updated

allocation, log, status = solve(initial)
final = log[-1]
print("Status:", status, "| Updates:", final["iteration"])
print("F(C):", final["welfare"])
print("G(C):", final["gap"])
print("F(C) + G(C):", final["upper_bound"])
print("Allocation:\n", allocation)

from scipy.optimize import minimize, linprog

m, n = model.shape
budget_matrix = np.kron(np.eye(m), np.ones((1, n)))
monotone_matrix = np.zeros(((m-1)*n, m*n))
for j in range(m-1):
    for i in range(n):
        row = j*n+i
        monotone_matrix[row, j*n+i] = 1
        monotone_matrix[row, (j+1)*n+i] = -1

constraints = [dict(type="eq", fun=lambda x: budget_matrix @ x - model.wealth,
                    jac=lambda x: budget_matrix),
               dict(type="ineq", fun=lambda x: -monotone_matrix @ x,
                    jac=lambda x: -monotone_matrix)]
reference = minimize(lambda x: -model.objective(x.reshape(m, n)), initial.ravel(),
                     jac=lambda x: -model.gradient(x.reshape(m, n)).ravel(),
                     bounds=[(0, None)]*(m*n), constraints=constraints,
                     method="SLSQP", options=dict(ftol=1e-13, maxiter=1000))
if not reference.success: raise AssertionError(reference.message)
reference_allocation = reference.x.reshape(m, n)
reference_diagnostic = check_feasible(reference_allocation)

final_gradient = model.gradient(allocation)
lp = linprog(-final_gradient.ravel(), A_eq=budget_matrix, b_eq=model.wealth,
             A_ub=monotone_matrix if m > 1 else None,
             b_ub=np.zeros((m-1)*n) if m > 1 else None, bounds=(0, None),
             method="highs", options=dict(primal_feasibility_tolerance=1e-10,
                                          dual_feasibility_tolerance=1e-10))
if not lp.success: raise AssertionError(lp.message)
check_feasible(lp.x.reshape(m, n))
oracle_value = float(np.sum(final_gradient * linear_oracle(final_gradient, model.wealth)))
lp_difference = abs(oracle_value + lp.fun)
reference_welfare = model.objective(reference_allocation)
welfare_difference = abs(final["welfare"] - reference_welfare)
if lp_difference > 1e-9: raise AssertionError("Oracle/LP discrepancy")
if welfare_difference > final["gap"] + 1e-9: raise AssertionError("Solver discrepancy")
if not final["welfare"]-1e-9 <= reference_welfare <= final["upper_bound"]+1e-9:
    raise AssertionError("Reference outside certificate plus numerical allowance")
if status != "gap_reached": raise AssertionError("Solver did not reach gap tolerance")
print("Independent welfare:", reference_welfare)
print("Absolute welfare difference:", welfare_difference)
print("Final oracle/LP value difference:", lp_difference)
print("PASS: small-run computational audit; delegated Day 3 code and theory deliverables complete.")

import csv, hashlib, platform
from datetime import datetime, timezone
from importlib.metadata import version

out = base / ("results_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
out.mkdir()
config = dict(wealth=model.wealth.tolist(), ranks=model.ranks.tolist(),
              probabilities=model.probabilities.tolist(),
              pareto_weights=model.pareto_weights.tolist(), agent_types=list(model.agent_types))
summary = dict(status=status, deliverables_status="complete for the small finite-grid baseline",
               personal_independent_review="not claimed",
               final=final, maximum_budget_residual=max(row["budget_residual"] for row in log),
               minimum_increment_over_run=min(row["minimum_increment_including_initial"] for row in log),
               reference_welfare=reference_welfare, welfare_difference=welfare_difference,
               maximum_allocation_difference=float(np.max(abs(allocation-reference_allocation))),
               final_oracle_lp_difference=float(lp_difference),
               certificate_scope="concave finite-grid optimization; floating-point accuracy")
for name, data in [("summary", summary), ("config_used", config), ("settings", settings)]:
    (out / (name + ".json")).write_text(json.dumps(data, indent=2, allow_nan=False))
with (out / "iterations.csv").open("w", newline="") as stream:
    writer = csv.DictWriter(stream, fieldnames=list(log[0]))
    writer.writeheader(); writer.writerows(log)
np.savetxt(out / "allocations.csv", np.column_stack((model.wealth, model.ranks,
           model.probabilities, allocation, reference_allocation)), delimiter=",",
           header="wealth,rank,probability," + ",".join(
               [f"solver_c{i+1}" for i in range(n)] + [f"reference_c{i+1}" for i in range(n)]), comments="")

environment = dict(python=sys.version, platform=platform.platform(),
                   numpy=version("numpy"), scipy=version("scipy"))
source_hashes = {"model.py": hashlib.sha256(model_path.read_bytes()).hexdigest()}
for source in ("Risk_Sharing_Day_3.ipynb", "day3_example.py"):
    if (base / source).exists():
        source_hashes[source] = hashlib.sha256((base / source).read_bytes()).hexdigest()
(out / "environment.json").write_text(json.dumps(environment, indent=2))
(out / "source_hashes.json").write_text(json.dumps(source_hashes, indent=2))
print("Evidence:", out.resolve())
print("Budget residual:", summary["maximum_budget_residual"])
print("Minimum increment:", summary["minimum_increment_over_run"])
print("Day 3 complete for the small finite-grid baseline. See audit_days01_03 for verification and scope.")