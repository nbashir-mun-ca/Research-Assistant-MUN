"""Compare Bellman allocations with Nouman's preserved Day 4/5 evidence."""

import argparse
import csv
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import subprocess

os.environ.setdefault("MPLBACKEND", "Agg")
import numpy as np
import scipy
from scipy.optimize import brentq

from day04.day4_example import (
    AllocationConstraints, ExperimentConfig, FrankWolfeSolver, RiskSharingModel,
)
from dynamic_programming.solver import solve_log_model

ROOT = Path(__file__).resolve().parents[1]
ARITHMETIC_TOLERANCE = 2e-11


@dataclass
class SavedReference:
    directory: Path
    model: RiskSharingModel
    allocation: np.ndarray
    summary: dict
    config: dict
    welfare: float
    gap: float
    source_hashes: dict


def independent_welfare(model, allocation):
    """Evaluate the printed objective without the model's offsets/objective."""
    c = np.asarray(allocation, dtype=float)
    if c.shape != model.shape or not np.isfinite(c).all() or np.any(c < 0):
        raise ValueError("Invalid allocation for independent welfare evaluation")
    return math.fsum(
        float(p) * float(model.weights[i])
        * math.log(1 + float(r)**(.5 if kind == "h1" else 2/3) + float(c[j, i]))
        for j, (p, r) in enumerate(zip(model.probabilities, model.ranks))
        for i, kind in enumerate(model.agent_types))


def independent_gap(model, allocation):
    """Maximize the linear objective by its simplex value, not an oracle call."""
    exponents = np.array([.5 if kind == "h1" else 2/3 for kind in model.agent_types])
    gradient = (model.probabilities[:, None] * model.weights
                / (1 + model.ranks[:, None]**exponents + allocation))
    tails = np.cumsum(gradient[::-1], axis=0)[::-1]
    maximum_linear_value = np.dot(np.diff(np.r_[0., model.wealth]), tails.max(axis=1))
    gap = float(maximum_linear_value - np.sum(gradient * allocation))
    if gap < -ARITHMETIC_TOLERANCE:
        raise AssertionError("Negative independently evaluated concavity gap")
    return max(0., gap)


def require_same_model(expected, actual):
    for field in ("wealth", "ranks", "probabilities", "weights"):
        a, b = getattr(expected, field), getattr(actual, field)
        if a.shape != b.shape or not np.allclose(a, b, rtol=0, atol=2e-15):
            raise ValueError(f"Cannot compare different {field}")
    if expected.agent_types != actual.agent_types:
        raise ValueError("Cannot compare different agent ordering or preferences")


def load_reference(directory):
    directory = Path(directory)
    settings = json.loads((directory / "config.json").read_text())
    config = settings.get("solver", settings)
    if "wealth" in settings:
        model = RiskSharingModel(settings["wealth"], settings["ranks"],
                                 settings["probabilities"], settings["weights"],
                                 tuple(settings["agent_types"]))
    else:
        model = RiskSharingModel.from_config(ExperimentConfig(**config))
    data = np.genfromtxt(directory / "allocations.csv", delimiter=",", names=True)
    prefix = "numerical_agent_" if "numerical_agent_1" in data.dtype.names else "numerical_"
    allocation = np.column_stack([data[f"{prefix}{i+1}"] for i in range(model.shape[1])])
    for field, column in [("wealth", "wealth"), ("ranks", "rank"), ("probabilities", "probability")]:
        if not np.allclose(getattr(model, field), data[column], rtol=0, atol=2e-15):
            raise ValueError(f"Saved CSV and config disagree about {field}")
    AllocationConstraints(model).check(allocation)
    summary = json.loads((directory / "summary.json").read_text())
    final = summary.get("final", summary)
    welfare = independent_welfare(model, allocation)
    gap = independent_gap(model, allocation)
    if abs(welfare - final["welfare"]) > ARITHMETIC_TOLERANCE:
        raise AssertionError("Saved welfare does not match the actual allocation")
    if abs(gap - final["gap"]) > ARITHMETIC_TOLERANCE:
        raise AssertionError("Saved certificate does not match the actual allocation")
    hashes = {name: hashlib.sha256((directory / name).read_bytes()).hexdigest()
              for name in ["allocations.csv", "summary.json", "config.json"]}
    return SavedReference(directory, model, allocation, summary, config, welfare, gap, hashes)


def scalar_reference(model):
    """Solve pointwise marginal equations instead of using the closed formula."""
    rows = []
    for wealth, rank in zip(model.wealth, model.ranks):
        base = np.array([1 + rank**(.5 if kind == "h1" else 2/3)
                         for kind in model.agent_types])
        mu = brentq(lambda x: np.maximum(model.weights/x - base, 0).sum() - wealth,
                    1e-8, float(np.max(model.weights/base)), xtol=1e-14)
        rows.append(np.maximum(model.weights/mu - base, 0))
    result = np.array(rows)
    # Only a pointwise solution that is also monotone is a valid full benchmark.
    AllocationConstraints(model).check(result)
    return result


def compare(states=(100, 200, 400, 800), refinements=(1, 2), rerun_fw=True):
    rows, sources, reruns, saved_allocations = [], {}, [], {}
    for m in states:
        paths = [ROOT / "day05" / "results" / f"grid_{profile}_{m}"
                 for profile in ("original", "tight")]
        if m == 400:
            paths.insert(0, ROOT / "day04" / "evidence")
        references = [load_reference(path) for path in paths]
        model = RiskSharingModel.from_config(ExperimentConfig(states=m))
        exact = scalar_reference(model)
        exact_value = independent_welfare(model, exact)
        for reference in references:
            require_same_model(model, reference.model)
            key = str(reference.directory.relative_to(ROOT))
            error = float(np.max(np.abs(reference.allocation - exact)))
            sources[key] = {"sha256": reference.source_hashes,
                            "welfare_recomputed": reference.welfare,
                            "gap_recomputed": reference.gap,
                            "maximum_allocation_error_to_scalar": error,
                            "allocation_tolerance": reference.config["allocation_tolerance"],
                            "allocation_passes_own_tolerance": error <= reference.config["allocation_tolerance"]}
        selected = sorted(set(refinements) | ({4} if m == 400 else set()))
        previous_value = -np.inf
        for refinement in selected:
            result = solve_log_model(model, 1/(2*m*refinement))
            value = independent_welfare(model, result.allocation)
            gap = independent_gap(model, result.allocation)
            feasibility = AllocationConstraints(model).check(result.allocation)
            assert abs(value - result.objective) <= ARITHMETIC_TOLERANCE
            assert abs(value - result.lattice.objective) <= ARITHMETIC_TOLERANCE
            assert abs(gap - result.duality_gap) <= ARITHMETIC_TOLERANCE
            assert value <= exact_value + ARITHMETIC_TOLERANCE
            assert exact_value <= value + gap + ARITHMETIC_TOLERANCE
            assert value >= previous_value - ARITHMETIC_TOLERANCE
            previous_value = value
            saved_allocations[f"dp_{m}_r{refinement}"] = (model, result.allocation)
            for reference in references:
                difference = value - reference.welfare
                assert value <= reference.welfare + reference.gap + ARITHMETIC_TOLERANCE
                assert reference.welfare <= value + gap + ARITHMETIC_TOLERANCE
                row = {
                    "states": m, "refinement": refinement,
                    "reference": str(reference.directory.relative_to(ROOT)),
                    "dp_welfare": value, "nouman_welfare": reference.welfare,
                    "dp_minus_nouman_welfare": difference,
                    "scalar_reference_welfare": exact_value,
                    "dp_welfare_error": exact_value - value,
                    "nouman_welfare_error": exact_value - reference.welfare,
                    "dp_gap": gap, "nouman_gap": reference.gap,
                    "maximum_dp_vs_nouman_share_difference": float(np.max(np.abs(result.allocation-reference.allocation))),
                    "maximum_dp_vs_scalar_share_error": float(np.max(np.abs(result.allocation-exact))),
                    **feasibility,
                    "compatible_welfare_certificates": True,
                }
                rows.append(row)
    if rerun_fw and 400 in states:
        for profile in ("original", "tight"):
            reference = load_reference(ROOT / "day05" / "results" / f"grid_{profile}_400")
            settings = dict(reference.config)
            settings["agent_types"] = tuple(settings["agent_types"])
            settings["pareto_weights"] = tuple(settings["pareto_weights"])
            config = ExperimentConfig(**settings)
            fresh = FrankWolfeSolver(reference.model, AllocationConstraints(reference.model), config).solve()
            assert fresh.converged
            value = independent_welfare(reference.model, fresh.allocation)
            welfare_difference = abs(value-reference.welfare)
            share_difference = float(np.max(np.abs(fresh.allocation-reference.allocation)))
            # Same-code reproducibility is separate from agreement of different solvers.
            assert welfare_difference <= ARITHMETIC_TOLERANCE
            assert share_difference <= 1e-8
            reruns.append({"profile": profile, "states": 400, "iterations": fresh.final["iteration"],
                           "welfare": value, "gap": fresh.final["gap"],
                           "welfare_difference_from_saved": welfare_difference,
                           "maximum_share_difference_from_saved": share_difference})
    try:
        revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        revision = None
    source_files = ["dynamic_programming/solver.py", "dynamic_programming/compare_nouman.py", "day04/day4_example.py"]
    report = {"status": "PASS", "repository_commit": revision,
              "executed_source_sha256": {name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in source_files},
              "environment": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__},
              "source_evidence": sources, "fresh_frank_wolfe_runs": reruns,
              "comparisons": rows,
              "interpretation": "Same finite wealth/rank-grid log-utility model. Allocations and welfare need not be bit-identical: Bellman consumption discretization and Frank-Wolfe stopping error differ. All compared welfare certificates are compatible.",
              "scope": "Finite-grid concave baseline with floating-point arithmetic, not continuous-wealth, Friedman-Savage, or lottery verification."}
    return report, saved_allocations


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "dynamic_programming/results/nouman_comparison")
    args = parser.parse_args()
    report, allocations = compare()
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "comparison.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    with (args.output / "comparison.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(report["comparisons"][0]))
        writer.writeheader()
        writer.writerows(report["comparisons"])
    for name, (model, allocation) in allocations.items():
        header = "wealth,rank,probability," + ",".join(f"agent_{i+1}" for i in range(model.shape[1]))
        np.savetxt(args.output / f"{name}.csv", np.column_stack((model.wealth, model.ranks, model.probabilities, allocation)), delimiter=",", header=header, comments="")
    print(f"PASS: {len(allocations)} DP runs, {len(report['source_evidence'])} saved evidence sets, "
          f"{len(report['comparisons'])} comparisons, and {len(report['fresh_frank_wolfe_runs'])} fresh FW reruns.")
    for row in report["comparisons"]:
        if row["states"] == 400 and row["refinement"] in (1, 4):
            print(f"400 states, refinement {row['refinement']}, {row['reference']}: "
                  f"welfare difference={row['dp_minus_nouman_welfare']:.6e}, "
                  f"max share difference={row['maximum_dp_vs_nouman_share_difference']:.6e}")
    print("Saved:", args.output)


if __name__ == "__main__":
    main()
