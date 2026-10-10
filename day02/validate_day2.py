"""Independent checks and saved evidence for provisional Day 2 preparation.

Run from any directory: python /path/to/validate_day2.py
Config and output paths default to the directory containing this script.
"""
import argparse
import csv
from datetime import datetime, timezone
from importlib.metadata import version
import hashlib
import json
from pathlib import Path
import platform
import sys

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import lil_matrix

from model import Model, feasibility, linear_oracle, tail_scores

ROOT = Path(__file__).resolve().parent


def require(condition, message):
    # Explicit checks remain active even when Python runs with -O.
    if not condition:
        raise AssertionError(message)


def dump_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def scalar_objective(model, consumption):
    """Independent scalar loops: no model utilities/objective calls."""
    import math
    result = 0.0
    for j in range(model.shape[0]):
        for i, kind in enumerate(model.agent_types):
            r = float(model.ranks[j])
            rank_term = math.sqrt(r) if kind == "h1" else r ** (2 / 3)
            result += float(model.probabilities[j]) * float(model.pareto_weights[i]) * math.log(
                1 + rank_term + float(consumption[j, i]))
    return result


def direct_allocation_lp(gradient, wealth):
    """Reference LP using consumption variables, budgets and monotonicity.

    Deliberately does not use increments, tail_scores or linear_oracle.
    scipy.linprog minimizes, so negate the maximization coefficients.
    """
    m, n = gradient.shape
    equality = lil_matrix((m, m * n))
    for j in range(m):
        equality[j, j * n:(j + 1) * n] = 1.0
    monotonicity = lil_matrix(((m - 1) * n, m * n))
    for j in range(m - 1):
        for i in range(n):
            row = j * n + i
            monotonicity[row, j * n + i] = 1.0
            monotonicity[row, (j + 1) * n + i] = -1.0
    result = linprog(-gradient.ravel(), A_eq=equality.tocsr(), b_eq=wealth,
                     A_ub=monotonicity.tocsr() if m > 1 else None,
                     b_ub=np.zeros((m - 1) * n) if m > 1 else None,
                     bounds=(0, None), method="highs",
                     options={"primal_feasibility_tolerance": 1e-9,
                              "dual_feasibility_tolerance": 1e-9})
    require(result.success, f"HiGHS failed: {result.message}")
    return result.x.reshape(m, n), float(-result.fun)


def random_feasible(model, rng):
    # Positive increment fractions keep centered differences away from boundaries.
    shares = 0.2 + rng.random(model.shape)
    shares /= shares.sum(axis=1, keepdims=True)
    ds = np.diff(np.r_[0.0, model.wealth])
    return np.cumsum(ds[:, None] * shares, axis=0)


def run_checks(config, out):
    tol = config["tolerances"]
    require(all(np.isfinite(v) and v >= 0 for v in tol.values()), "invalid tolerance")
    rng = np.random.default_rng(config["validation_seed"])
    configured_model = Model.from_config(config)
    out.mkdir(parents=True, exist_ok=True)

    def check_feasible(c, s):
        result = feasibility(c, s, tol["budget_atol"], tol["domain_atol"])
        require(result["feasible"], f"infeasible allocation: {result}")
        return result

    fixed = Model([1/6, 1/2, 5/6], [1/6, 1/2, 5/6], [1/3]*3, [1, 1], ("h1", "h2"))
    baseline_rows = []
    for label, c, expected in [
        ("day1_equal", fixed.equal_sharing(), 1.2362178075439176),
        ("day1_one_third_two_thirds", np.column_stack((fixed.wealth/3, 2*fixed.wealth/3)),
         1.2355540193701327),
    ]:
        value = fixed.objective(c)
        diag = check_feasible(c, fixed.wealth)
        scalar_error = abs(value - scalar_objective(fixed, c))
        require(abs(value - expected) <= 1e-12 and scalar_error <= 1e-12, "Day 1 regression failed")
        baseline_rows.append({"case": label, "welfare": value, "saved_day1_welfare": expected,
                              "absolute_difference": abs(value-expected),
                              "scalar_objective_error": scalar_error, **diag})

    models = {
        "configured_input": configured_model,
        "day1": fixed,
        "unequal_pareto": Model(fixed.wealth, fixed.ranks, fixed.probabilities,
                                [0.35, 1.7], ("h1", "h2")),
        # This is a diagnostic weighted sum, not the uniform midpoint replication.
        "separate_rank_wealth_nonuniform_p": Model(
            [0.02, 0.2, 0.65, 1.4], [0.08, 0.3, 0.58, 0.9], [0.1, 0.2, 0.3, 0.4],
            [0.3, 1.4, 0.8], ("h1", "h2", "h1")),
        "eight_agent_small_grid": Model(
            (np.arange(7)+0.5)/7, (np.arange(7)+0.5)/7, [1/7]*7, [1]*8,
            ("h2", "h2", "h1", "h1", "h1", "h1", "h1", "h2")),
    }
    gradient_rows, gradient_cases, objective_checks = [], [], []
    oracle_cases = []
    for label, model in models.items():
        for k in range(2):
            c = model.equal_sharing() if k == 0 else random_feasible(model, rng)
            other = random_feasible(model, rng)
            direction = other - c
            case = f"{label}_{k}"
            check_feasible(c, model.wealth)
            scalar_error = abs(model.objective(c) - scalar_objective(model, c))
            require(scalar_error <= 1e-12, f"weighted objective mismatch: {case}")
            objective_checks.append({"case": case, "scalar_objective_error": scalar_error})
            predicted = float(np.sum(model.gradient(c) * direction))
            case_rows = []
            for step in (1e-3, 1e-4, 1e-5, 1e-6):
                check_feasible(c + step * direction, model.wealth)
                check_feasible(c - step * direction, model.wealth)
                measured = (model.objective(c+step*direction) - model.objective(c-step*direction))/(2*step)
                error = abs(predicted-measured)
                limit = tol["gradient_atol"] + tol["gradient_rtol"]*abs(predicted)
                row = {"case": case, "method": "centered", "step": step,
                       "gradient_prediction": predicted, "finite_difference": measured,
                       "absolute_error": error, "allowed_error": limit, "passed": error <= limit}
                gradient_rows.append(row)
                case_rows.append(row)
            selected = next(row for row in case_rows if row["step"] == 1e-5)
            require(selected["passed"], f"gradient check failed at predeclared step: {case}")
            gradient_cases.append(selected)
            oracle_cases.append((f"utility_gradient_{case}", model.gradient(c), model.wealth))

    # A boundary allocation cannot generally support a centered feasible direction.
    for k in range(2):
        c = np.column_stack((fixed.wealth, np.zeros(3))) if k == 0 else np.column_stack((np.zeros(3), fixed.wealth))
        direction = fixed.equal_sharing() - c
        predicted = float(np.sum(fixed.gradient(c)*direction))
        boundary_rows = []
        for step in (1e-6, 1e-7):
            check_feasible(c + step*direction, fixed.wealth)
            measured = (fixed.objective(c+step*direction)-fixed.objective(c))/step
            error = abs(predicted-measured)
            limit = tol["gradient_atol"] + tol["gradient_rtol"]*abs(predicted)
            row = {"case": f"boundary_{k}", "method": "forward", "step": step,
                   "gradient_prediction": predicted, "finite_difference": measured,
                   "absolute_error": error, "allowed_error": limit, "passed": error <= limit}
            gradient_rows.append(row)
            boundary_rows.append(row)
        selected = boundary_rows[0]  # Fixed before running: forward step 1e-6.
        require(selected["passed"], "boundary directional derivative failed")
        gradient_cases.append(selected)

    # Hand-predicted fixture, with a tail-score tie at the middle state.
    toy_g = np.array([[4., 3.], [1., 2.], [2., 1.]])
    toy_h = np.array([[7., 6.], [3., 3.], [2., 1.]])
    toy_y = np.array([[1/6, 0], [1/3, 1/6], [2/3, 1/6]])
    require(np.array_equal(tail_scores(toy_g), toy_h), "hand tail scores disagree")
    require(np.allclose(linear_oracle(toy_g, fixed.wealth), toy_y, rtol=0, atol=1e-15), "hand oracle disagrees")
    oracle_cases.extend([
        ("hand_example", toy_g, fixed.wealth),
        ("all_scores_negative", np.array([[-4., -3.], [-2., -5.], [-1., -2.]]), fixed.wealth),
        ("identical_agent_ties", np.ones((3, 4)), fixed.wealth),
        ("zero_gradient", np.zeros((3, 2)), fixed.wealth),
        ("zero_and_repeated_wealth", np.array([[1., -1.], [2., 3.], [-3., 1.], [4., -2.]]), np.array([0., .2, .2, .9])),
        ("one_state", np.array([[-2., -1., -3.]]), np.array([.7])),
        ("one_agent", np.array([[1.], [-3.], [2.]]), fixed.wealth),
    ])
    for k in range(20):
        m, n = int(rng.integers(2, 8)), int(rng.integers(2, 6))
        s = np.cumsum(0.02 + rng.random(m))
        s /= s[-1]
        oracle_cases.append((f"random_signed_{k:02d}", rng.normal(size=(m, n)), s))

    oracle_rows, raw_oracle_rows = [], []
    max_budget, min_increment = 0.0, float("inf")
    for label, gradient, wealth in oracle_cases:
        y = linear_oracle(gradient, wealth)
        diag = check_feasible(y, wealth)
        lp_y, lp_value = direct_allocation_lp(gradient, wealth)
        lp_diag = check_feasible(lp_y, wealth)
        value = float(np.sum(gradient*y))
        error = abs(value-lp_value)
        limit = tol["lp_atol"] + tol["lp_rtol"]*abs(lp_value)
        require(error <= limit, f"oracle LP value discrepancy: {label}")
        # Tail scores also match a separately computed explicit suffix sum.
        scores = tail_scores(gradient)
        reference = np.array([np.sum(gradient[j:], axis=0) for j in range(len(wealth))])
        require(np.allclose(scores, reference, rtol=1e-13, atol=1e-13), "tail score discrepancy")
        if label in ("identical_agent_ties", "zero_gradient"):
            require(np.allclose(y, wealth[:, None]/gradient.shape[1], rtol=0, atol=1e-15), "exact ties not shared equally")
        max_budget = max(max_budget, diag["budget_residual"], lp_diag["budget_residual"])
        min_increment = min(min_increment, diag["minimum_increment_including_initial"], lp_diag["minimum_increment_including_initial"])
        oracle_rows.append({"case": label, "states": gradient.shape[0], "agents": gradient.shape[1],
                            "oracle_linear_value": value, "highs_linear_value": lp_value,
                            "absolute_value_difference": error, "allowed_error": limit,
                            "oracle_budget_residual": diag["budget_residual"],
                            "oracle_minimum_increment": diag["minimum_increment_including_initial"],
                            "lp_budget_residual": lp_diag["budget_residual"],
                            "lp_minimum_increment": lp_diag["minimum_increment_including_initial"],
                            "passed": True})
        for j in range(gradient.shape[0]):
            for i in range(gradient.shape[1]):
                raw_oracle_rows.append({"case": label, "state": j, "agent_zero_based": i,
                                        "wealth": float(wealth[j]), "gradient": float(gradient[j,i]),
                                        "tail_score": float(scores[j,i]), "oracle_consumption": float(y[j,i]),
                                        "lp_consumption": float(lp_y[j,i])})

    rejected = []
    candidates = {}
    c = fixed.equal_sharing()
    bad = c.copy(); bad[1,0] += .01; candidates["budget"] = bad
    bad = c.copy(); bad[0] = [-.01, fixed.wealth[0]+.01]; candidates["negative_consumption"] = bad
    bad = c.copy(); bad[1] = [.4,.1]; bad[2] = [.3,fixed.wealth[2]-.3]; candidates["decreasing_consumption"] = bad
    for label, bad in candidates.items():
        require(not feasibility(bad, fixed.wealth)["feasible"], f"invalid allocation accepted: {label}")
        rejected.append(label)
    invalid_calls = {
        "nonfinite_pareto": lambda: Model(fixed.wealth, fixed.ranks, fixed.probabilities, [1,np.nan], ("h1","h2")),
        "probabilities_not_normalized": lambda: Model(fixed.wealth, fixed.ranks, [1,1,1], [1,1], ("h1","h2")),
        "unsupported_agent_type": lambda: Model(fixed.wealth, fixed.ranks, fixed.probabilities, [1,1], ("h1","bad")),
        "wrong_consumption_shape": lambda: fixed.objective(np.zeros((2,2))),
        "nonfinite_consumption": lambda: fixed.gradient(np.full((3,2),np.nan)),
        "nonfinite_oracle_gradient": lambda: linear_oracle(np.full((3,2),np.inf),fixed.wealth),
        "decreasing_wealth": lambda: linear_oracle(np.ones((3,2)),[.1,.7,.3]),
        "wrong_rank_shape": lambda: Model(fixed.wealth,[.1,.2],fixed.probabilities,[1,1],("h1","h2")),
    }
    for label, call in invalid_calls.items():
        try:
            call()
        except ValueError:
            rejected.append(label)
        else:
            raise AssertionError(f"invalid input accepted: {label}")

    configured_c = configured_model.equal_sharing()
    configured_y = linear_oracle(configured_model.gradient(configured_c), configured_model.wealth)
    check_feasible(configured_c, configured_model.wealth)
    check_feasible(configured_y, configured_model.wealth)
    configured_lp_y, configured_lp_value = direct_allocation_lp(configured_model.gradient(configured_c), configured_model.wealth)
    configured_lp_diag = check_feasible(configured_lp_y, configured_model.wealth)
    configured_oracle_diag = check_feasible(configured_y, configured_model.wealth)
    max_budget = max(max_budget, configured_lp_diag["budget_residual"], configured_oracle_diag["budget_residual"])
    min_increment = min(min_increment, configured_lp_diag["minimum_increment_including_initial"],
                        configured_oracle_diag["minimum_increment_including_initial"])
    configured_error = abs(float(np.sum(configured_model.gradient(configured_c)*configured_y))-configured_lp_value)
    require(configured_error <= tol["lp_atol"]+tol["lp_rtol"]*abs(configured_lp_value), "configured oracle LP mismatch")
    summary = {
        "status": "PROVISIONAL: computational checks passed; joint review pending",
        "day1_review": "pending; no Parsa sign-off inferred",
        "day2_joint_gate": "provisional preparation",
        "directional_gradient_cases": len(gradient_cases),
        "finite_difference_measurements": len(gradient_rows),
        "maximum_selected_gradient_absolute_error": max(r["absolute_error"] for r in gradient_cases),
        "centered_acceptance_step": 1e-5, "forward_acceptance_step": 1e-6,
        "oracle_lp_cases": len(oracle_rows)+1,
        "maximum_oracle_lp_absolute_value_difference": max(configured_error, max(r["absolute_value_difference"] for r in oracle_rows)),
        "maximum_oracle_or_lp_budget_residual": max_budget,
        "minimum_oracle_or_lp_increment": min_increment,
        "invalid_inputs_correctly_rejected": len(rejected),
        "configured_model_equal_welfare": configured_model.objective(configured_c),
        "configured_oracle_lp_absolute_difference": configured_error,
        "tolerances": tol,
        "interpretation": "Only objective, gradient, and linear-oracle checks. No nonlinear optimizer or continuous-model error certificate.",
        "random_seed": config["validation_seed"],
    }
    dump_json(out/"summary.json", summary)
    dump_json(out/"config_used.json", config)
    dump_json(out/"objective_checks.json", objective_checks)
    dump_json(out/"rejection_checks.json", rejected)
    dump_json(out/"hand_example.json", {"note": "invented linear coefficients, not a log-utility gradient",
                                       "wealth": fixed.wealth.tolist(), "gradient": toy_g.tolist(),
                                       "tail_scores": toy_h.tolist(), "oracle": toy_y.tolist()})
    write_csv(out/"day1_regression.csv", baseline_rows)
    write_csv(out/"gradient_checks.csv", gradient_rows)
    write_csv(out/"oracle_checks.csv", oracle_rows)
    write_csv(out/"oracle_allocations.csv", raw_oracle_rows)
    np.savetxt(out/"configured_equal.csv", configured_c, delimiter=",")
    np.savetxt(out/"configured_linear_oracle.csv", configured_y, delimiter=",")
    np.savetxt(out/"configured_highs_oracle.csv", configured_lp_y, delimiter=",")
    dump_json(out/"environment.json", {"python": sys.version, "platform": platform.platform(),
                                       "packages": {name:version(name) for name in ("numpy","scipy")}})
    hashes = {name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
              for name in ("model.py","validate_day2.py")}
    dump_json(out/"run_manifest.json", {"created_utc": datetime.now(timezone.utc).isoformat(),
                                       "code_sha256": hashes,
                                       "config_sha256": hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()})
    (out/"RUN_STATUS.txt").write_text("PASSED computational checks. PROVISIONAL; Parsa review pending.\n",encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT/"config.json")
    parser.add_argument("--out", type=Path, default=ROOT/"results")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out/"RUN_STATUS.txt").write_text("RUNNING; previous results are not current.\n",encoding="utf-8")
    try:
        config = json.loads(args.config.read_text(encoding="utf-8"))
        run_checks(config, args.out)
    except Exception as exc:
        (args.out/"RUN_STATUS.txt").write_text(f"FAILED: {type(exc).__name__}: {exc}\n",encoding="utf-8")
        raise


if __name__ == "__main__":
    main()
