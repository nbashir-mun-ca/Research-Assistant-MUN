"""Compare dynamic programming with the existing Example 2 benchmark."""

import argparse
import json
import os
from pathlib import Path
from time import perf_counter

os.environ.setdefault("MPLBACKEND", "Agg")
import numpy as np

from day04.day4_example import (
    AllocationConstraints, Example2Benchmark, ExperimentConfig, RiskSharingModel,
)
from dynamic_programming.solver import solve_log_model


def run(states=400, refinement=1):
    if states < 1 or refinement < 1:
        raise ValueError("states and refinement must be positive integers")
    model = RiskSharingModel.from_config(ExperimentConfig(states=states))
    # Midpoints are odd multiples of 1/(2m). Refinement changes consumption
    # resolution without changing the existing wealth/rank quadrature grid.
    quantum = 1/(2*states*refinement)
    start = perf_counter()
    result = solve_log_model(model, quantum)
    elapsed = perf_counter() - start
    diagnostics = AllocationConstraints(model).check(result.allocation)
    reference = Example2Benchmark(model).allocation(model.wealth)
    exact = model.objective(reference)
    if (result.objective > exact + 1e-11
            or exact > result.finite_grid_upper_bound + 1e-11):
        raise AssertionError("Analytic welfare lies outside the numerical bounds")
    summary = {
        "wealth_states": states, "consumption_refinement": refinement,
        "quantum_group_total": quantum, "agents": model.shape[1],
        "groups_original_agent_indices": result.groups,
        "algorithm": result.lattice.algorithm,
        "lattice_states_evaluated": result.lattice.states_evaluated,
        "transition_visits": result.lattice.transitions_considered,
        "dp_welfare": result.objective, "analytic_finite_grid_welfare": exact,
        "welfare_difference_from_analytic": exact - result.objective,
        "finite_grid_upper_bound": result.finite_grid_upper_bound,
        "finite_grid_duality_gap": result.duality_gap,
        "maximum_per_agent_allocation_error": float(np.max(abs(result.allocation-reference))),
        "feasibility": diagnostics, "elapsed_seconds": elapsed,
        "scope": "Lattice optimum; bound is for continuous shares on the finite wealth/rank grid. "
                 "No continuous-wealth quadrature certificate or lottery optimization.",
        "numpy_version": np.__version__,
    }
    return model, result, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--states", type=int, default=400)
    parser.add_argument("--refinement", type=int, default=1)
    parser.add_argument("--output", type=Path, help="Optional directory for summary.json and allocation.csv")
    args = parser.parse_args()
    model, result, summary = run(args.states, args.refinement)
    report = json.dumps(summary, indent=2)
    print(report)
    if args.output:
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / "summary.json").write_text(report + "\n")
        header = "wealth,rank,probability," + ",".join(
            f"agent_{i}_{kind}" for i, kind in enumerate(model.agent_types))
        np.savetxt(args.output / "allocation.csv", np.column_stack(
            (model.wealth, model.ranks, model.probabilities, result.allocation)),
            delimiter=",", header=header, comments="")


if __name__ == "__main__":
    main()
