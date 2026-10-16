"""Day 4 notebook definitions, with a local command-line entry point."""

from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile
import csv
import json
import platform
import sys

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy.integrate import quad
from scipy.optimize import Bounds, LinearConstraint, brentq, linprog, minimize

# Optional Colab support is imported once, alongside every other dependency.
try:
    from google.colab import files as colab_files
except ImportError:
    colab_files = None

# Class: The single place to change experiment settings.
@dataclass(frozen=True)
class ExperimentConfig:
    """The single place to change experiment settings."""
    states: int = 400
    gap_tolerance: float = 5e-6
    allocation_tolerance: float = 0.0025
    feasibility_tolerance: float = 1e-10
    roundoff_tolerance: float = 1e-12
    max_updates: int = 20000
    validation_seed: int = 20261009
    output_directory: str = "day4_results"
    agent_types: tuple = ("h2", "h2", "h1", "h1", "h1", "h1", "h1", "h2")
    pareto_weights: tuple = (1.,) * 8

    # Function: Check that the grid size, update limit, tolerances, and output folder are valid.
    def __post_init__(self):
        if not isinstance(self.states, int) or self.states < 1:
            raise ValueError("states must be a positive integer")
        if not isinstance(self.max_updates, int) or self.max_updates < 0:
            raise ValueError("max_updates must be a nonnegative integer")
        tolerances = (self.gap_tolerance, self.allocation_tolerance,
                      self.feasibility_tolerance, self.roundoff_tolerance)
        if not all(np.isfinite(x) and x > 0 for x in tolerances):
            raise ValueError("All tolerances must be finite and positive")
        if not self.output_directory:
            raise ValueError("Choose an output directory")

# Class: Hold model inputs and calculate utility, welfare, and its gradient.
@dataclass(frozen=True)
class RiskSharingModel:
    """Hold model inputs and calculate utility, welfare, and its gradient."""
    wealth: np.ndarray
    ranks: np.ndarray
    probabilities: np.ndarray
    weights: np.ndarray
    agent_types: tuple
    offsets: np.ndarray = field(init=False, repr=False)

    # Function: Validate and protect model inputs, then calculate the utility offsets once.
    def __post_init__(self):
        # Copy and protect inputs so later cells cannot accidentally change them.
        for name in ("wealth", "ranks", "probabilities", "weights"):
            value = np.array(getattr(self, name), dtype=float, copy=True)
            if value.ndim != 1 or not value.size or not np.isfinite(value).all():
                raise ValueError(f"{name} must be a finite, nonempty vector")
            value.setflags(write=False)
            object.__setattr__(self, name, value)
        types = tuple(self.agent_types)
        object.__setattr__(self, "agent_types", types)
        if self.ranks.shape != self.wealth.shape or self.probabilities.shape != self.wealth.shape:
            raise ValueError("One rank and probability are required per wealth state")
        if np.any(self.wealth < 0) or np.any(np.diff(self.wealth) < 0):
            raise ValueError("Wealth must be nonnegative and nondecreasing")
        if np.any((self.ranks <= 0) | (self.ranks >= 1)) or np.any(np.diff(self.ranks) <= 0):
            raise ValueError("Ranks must increase strictly within (0, 1)")
        if np.any(self.probabilities < 0) or not np.isclose(self.probabilities.sum(), 1, rtol=0, atol=1e-12):
            raise ValueError("Probabilities must be nonnegative and sum to one")
        if np.any(self.weights <= 0) or len(types) != self.weights.size:
            raise ValueError("One positive Pareto weight is required per agent")
        if any(t not in ("h1", "h2") for t in types):
            raise ValueError("Each agent type must be h1 or h2")
        powers = np.array([.5 if t == "h1" else 2/3 for t in types])
        offsets = 1 + self.ranks[:, None] ** powers[None, :]
        offsets.setflags(write=False)
        object.__setattr__(self, "offsets", offsets)

    # Function: Build a midpoint wealth grid from settings and return a model object.
    @classmethod
    def from_config(cls, config, states=None):
        """Generate the midpoint grid; no uploaded data are needed."""
        count = config.states if states is None else states
        ranks = (np.arange(count) + .5) / count
        return cls(ranks.copy(), ranks, np.full(count, 1/count),
                   config.pareto_weights, config.agent_types)

    # Function: Return the number of states and agents as a pair: (states, agents).
    @property
    def shape(self):
        return self.wealth.size, self.weights.size  # (states, agents)

    # Function: Check that shares have the right dimensions and are finite and nonnegative.
    def consumption(self, values):
        """Validate an allocation's shape and utility domain."""
        c = np.asarray(values, dtype=float)
        if c.shape != self.shape or not np.isfinite(c).all() or np.any(c < 0):
            raise ValueError(f"Consumption must be finite, nonnegative, and shape {self.shape}")
        return c

    # Function: Return total expected weighted satisfaction as one welfare number.
    def objective(self, consumption):
        c = self.consumption(consumption)
        return float(np.sum(self.probabilities[:, None] * self.weights[None, :]
                            * np.log(self.offsets + c)))

    # Function: Return the extra welfare benefit per unit of consumption at every table entry.
    def gradient(self, consumption):
        c = self.consumption(consumption)
        return self.probabilities[:, None] * self.weights[None, :] / (self.offsets + c)

    # Function: Create a feasible starting table by dividing each state's wealth equally.
    def equal_sharing(self):
        return np.repeat(self.wealth[:, None] / self.shape[1], self.shape[1], axis=1)

# Class: Own budget balance, nonnegativity, monotonicity, and the linear oracle.
class AllocationConstraints:
    """Own budget balance, nonnegativity, monotonicity, and the linear oracle."""
    # Function: Store the wealth vector, required table shape, and feasibility tolerance.
    def __init__(self, model, tolerance=1e-10):
        self.wealth = model.wealth
        self.shape = model.shape
        self.tolerance = tolerance

    # Function: Check budget balance, nonnegative shares, and monotonicity; return residuals.
    def check(self, allocation):
        c = np.asarray(allocation, dtype=float)
        if c.shape != self.shape or not np.isfinite(c).all():
            raise AssertionError("Allocation has an invalid shape or nonfinite values")
        increments = np.diff(np.vstack((np.zeros((1, self.shape[1])), c)), axis=0)
        residuals = {"budget_residual": float(np.max(abs(c.sum(axis=1) - self.wealth))),
                     "minimum_consumption": float(c.min()),
                     "minimum_increment": float(increments.min())}
        if (residuals["budget_residual"] > self.tolerance
                or min(residuals["minimum_consumption"], residuals["minimum_increment"]) < -self.tolerance):
            raise AssertionError(residuals)
        return residuals

    # Function: Find the best feasible allocation for the supplied linear gradient scores.
    def linear_oracle(self, gradient):
        """Allocate wealth increments using tail sums of the gradient."""
        g = np.asarray(gradient, dtype=float)
        if g.shape != self.shape or not np.isfinite(g).all():
            raise ValueError("Gradient must be finite and match the allocation shape")
        scores = np.cumsum(g[::-1], axis=0)[::-1]
        winners = scores == scores.max(axis=1, keepdims=True)
        wealth_increments = np.diff(np.r_[0., self.wealth])
        # Share exact ties equally; cumulative sums reconstruct consumption.
        increments = wealth_increments[:, None] * winners / winners.sum(axis=1, keepdims=True)
        return np.cumsum(increments, axis=0)

# Class: Keep the solution and convergence evidence together.
@dataclass
class SolverResult:
    """Keep the solution and convergence evidence together."""
    allocation: np.ndarray
    history: list
    converged: bool

    # Function: Read the last saved iteration without copying the history.
    @property
    def final(self):
        return self.history[-1]


# Class: Improve a feasible allocation using gradients, an oracle, and line search.
class FrankWolfeSolver:
    """Improve a feasible allocation using gradients, an oracle, and line search."""
    # Function: Connect this solver to its model, feasibility rules, and settings.
    def __init__(self, model, constraints, config):
        self.model = model
        self.constraints = constraints
        self.config = config

    # Function: Use derivative signs and bisection to choose a step between zero and one.
    def _line_search(self, allocation, direction):
        # A positive derivative means a slightly larger step improves welfare.
        def slope(step):
            return float(np.sum(self.model.gradient(allocation + step*direction) * direction))
        if slope(0) <= 0:
            return 0.
        if slope(1) >= 0:
            return 1.
        left, right = 0., 1.
        for _ in range(42):
            middle = (left + right) / 2
            if slope(middle) > 0:
                left = middle
            else:
                right = middle
        return (left + right) / 2

    # Function: Repeat oracle and line-search steps until the welfare gap is small or the run stops.
    def solve(self):
        c, history = self.model.equal_sharing(), []
        for iteration in range(self.config.max_updates + 1):
            residuals = self.constraints.check(c)
            value, gradient = self.model.objective(c), self.model.gradient(c)
            candidate = self.constraints.linear_oracle(gradient)
            self.constraints.check(candidate)
            direction = candidate - c
            gap = float(np.sum(gradient * direction))
            if gap < -self.config.roundoff_tolerance:
                raise AssertionError("An exact linear oracle cannot have a negative gap")
            gap = max(gap, 0.)  # clip harmless floating point rounding only
            record = {"iteration": iteration, "welfare": value, "gap": gap,
                      "upper_bound": value + gap, **residuals, "step": None}
            history.append(record)
            converged = gap <= self.config.gap_tolerance
            if converged or iteration == self.config.max_updates:
                return SolverResult(c, history, converged)
            record["step"] = self._line_search(c, direction)
            updated = c + record["step"] * direction
            self.constraints.check(updated)
            if self.model.objective(updated) < value - self.config.roundoff_tolerance:
                raise AssertionError("Welfare decreased")
            if record["step"] == 0 or np.array_equal(updated, c):
                return SolverResult(c, history, False)
            c = updated

# Class: Supply the analytic answer and a separate scalar-equation check.
class Example2Benchmark:
    """Supply the analytic answer and a separate scalar-equation check."""
    # Function: Identify the two agent groups and reject settings outside this analytic formula.
    def __init__(self, model):
        types = np.array(model.agent_types)
        self.h1 = np.flatnonzero(types == "h1")
        self.h2 = np.flatnonzero(types == "h2")
        if model.shape[1] != 8 or len(self.h1) != 5 or len(self.h2) != 3:
            raise ValueError("Example 2 requires five h1 and three h2 agents")
        if not np.array_equal(model.weights, np.ones(8)):
            raise ValueError("The group formula requires unit Pareto weights")
        if not np.array_equal(model.wealth, model.ranks) or np.any(model.wealth > 1):
            raise ValueError("The group formula requires rank = wealth in [0,1]")
        self.model = model

    # Function: Solve the boundary equation and return the wealth where h1 consumption activates.
    @property
    def activation(self):
        # t = wealth**(1/6) turns the boundary condition into a cubic equation.
        # brentq varies t until this expression equals zero; t**6 gives wealth.
        return brentq(
            lambda t: t**3 + 3*t - 3,  # Input: t. Output: the cubic equation's residual.
            0., 1., xtol=1e-14) ** 6

    # Function: Return the exact per-agent consumption table for specified wealth values.
    def allocation(self, wealth):
        s = np.asarray(wealth, dtype=float)
        if s.ndim != 1 or not np.isfinite(s).all() or np.any((s < 0) | (s > 1)):
            raise ValueError("Benchmark wealth must be a vector in [0,1]")
        b = np.maximum((s + 3*(s**(2/3) - np.sqrt(s))) / 8, 0.)
        a = (s - 5*b) / 3
        result = np.empty((len(s), 8))
        result[:, self.h1], result[:, self.h2] = b[:, None], a[:, None]
        return result

    # Function: Find each state's allocation by solving a separate marginal-utility equation.
    def scalar_allocation(self):
        """Solve marginal utility balance without using the group formula."""
        allocations = []
        for wealth, base in zip(self.model.wealth, self.model.offsets):
            weights = self.model.weights
            # x is a trial common weighted marginal utility. weights/x - base
            # gives each share; maximum(..., 0) prevents negative consumption.
            # The root makes total allocated consumption equal available wealth.
            mu = brentq(
                lambda x: np.maximum(weights/x - base, 0).sum() - wealth,  # Return total shares minus wealth.
                1e-6, float(np.max(weights/base)), xtol=1e-14)
            allocations.append(np.maximum(weights/mu - base, 0))
        return np.array(allocations)

    # Function: Build theoretical predictions at selected wealth values before optimization.
    def prediction_table(self):
        wealth = np.array([0., .1, .2, self.activation, .3, .35, .5, .75, 1.])
        allocation = self.allocation(wealth)
        return [{"wealth": s, "b_each_h1": row[self.h1[0]],
                 "a_each_h2": row[self.h2[0]], "total": row.sum()}
                for s, row in zip(wealth, allocation)]

# Build the budget and monotonicity matrices used by the independent SciPy solvers.
def validation_matrices(model):
    states, agents = model.shape
    budget = np.kron(np.eye(states), np.ones((1, agents)))
    differences = np.eye(states) - np.eye(states, k=-1)
    return budget, np.kron(differences, np.eye(agents))

# Class: Compare the model's gradient with a small numerical change in welfare.
@dataclass
class GradientCheck:
    """Compare the model's gradient with a small numerical change in welfare."""
    config: ExperimentConfig

    # Function: Compare the analytic gradient with a centered finite-difference welfare change.
    def run(self, model, rng):
        c = model.equal_sharing()
        increments = rng.dirichlet(np.ones(8), 6) * np.diff(np.r_[0., model.wealth])[:, None]
        direction = np.cumsum(increments, axis=0) - c
        step = 1e-5
        difference = (model.objective(c + step*direction) - model.objective(c - step*direction))/(2*step)
        error = abs(difference - np.sum(model.gradient(c)*direction))
        assert error < 1e-8
        return {"gradient_absolute_error": float(error)}

# Class: Compare the linear oracle with HiGHS in thirteen small cases.
@dataclass
class OracleCheck:
    """Compare the linear oracle with HiGHS in thirteen small cases."""
    config: ExperimentConfig

    # Function: Compare our linear oracle against HiGHS in ten random and three special cases.
    def run(self, model, rng):
        constraints = AllocationConstraints(model, self.config.feasibility_tolerance)
        budget, monotone = validation_matrices(model)
        c = model.equal_sharing()
        cases = [rng.normal(size=model.shape) for _ in range(10)]
        cases += [-np.ones_like(c), np.zeros_like(c), np.tile(np.arange(6.)[:, None], (1, 8))]
        errors = []
        for gradient in cases:
            independent = linprog(-gradient.ravel(), A_ub=-monotone, b_ub=np.zeros(c.size),
                                  A_eq=budget, b_eq=model.wealth, bounds=(0, None), method="highs")
            assert independent.success, independent.message
            candidate = constraints.linear_oracle(gradient)
            constraints.check(candidate)
            errors.append(abs(float(np.sum(gradient*candidate)) + independent.fun))
        assert max(errors) < 1e-10
        for gradient in cases[-3:]:
            assert np.max(abs(constraints.linear_oracle(gradient) - c)) < 1e-12
        return {"oracle_lp_cases": len(cases), "maximum_oracle_value_error": max(errors)}

# Class: Compare the small-grid solver with SLSQP and the analytic benchmark.
@dataclass
class OptimizerCheck:
    """Compare the small-grid solver with SLSQP and the analytic benchmark."""
    config: ExperimentConfig

    # Function: Compare a small-grid solver result with SLSQP, the analytic answer, and scalar roots.
    def run(self, model):
        constraints = AllocationConstraints(model, self.config.feasibility_tolerance)
        solver = FrankWolfeSolver(model, constraints, self.config).solve()
        assert solver.converged
        budget, monotone = validation_matrices(model)
        # SLSQP expects flat shares x; reshape restores the (states, agents) table.
        # Negative welfare converts maximization into minimization.
        independent = minimize(
            lambda x: -model.objective(x.reshape(model.shape)),  # Input flat shares; return negative welfare.
            model.equal_sharing().ravel(),
            jac=lambda x: -model.gradient(x.reshape(model.shape)).ravel(),  # Return the flat gradient of negative welfare.
            method="SLSQP", constraints=[LinearConstraint(budget, model.wealth, model.wealth),
                                         LinearConstraint(monotone, 0, np.inf)],
            bounds=Bounds(0, np.inf), options={"ftol": 1e-12, "maxiter": 1000})
        assert independent.success, independent.message
        direct = independent.x.reshape(model.shape)
        constraints.check(direct)
        benchmark = Example2Benchmark(model)
        exact = benchmark.allocation(model.wealth)
        direct_error = abs(model.objective(direct) - model.objective(exact))
        solver_error = model.objective(exact) - model.objective(solver.allocation)
        root_error = float(np.max(abs(benchmark.scalar_allocation() - exact)))
        assert direct_error < 1e-9 and root_error < 1e-10
        assert -1e-12 <= solver_error <= solver.final["gap"] + 1e-12
        report = {"small_slsqp_welfare_error": direct_error, "small_solver_welfare_error": solver_error,
                  "small_solver_gap": solver.final["gap"], "scalar_root_allocation_error": root_error}
        return report, direct

# Class: Verify that the feasibility checker rejects three deliberately invalid allocations.
@dataclass
class FeasibilityCheck:
    """Verify that the feasibility checker rejects three deliberately invalid allocations."""
    config: ExperimentConfig

    # Function: Confirm that budget, negative-share, and monotonicity violations are all rejected.
    def run(self, model, direct):
        constraints = AllocationConstraints(model, self.config.feasibility_tolerance)
        invalid = [direct + .01, direct.copy(), direct.copy()]
        invalid[1][0] = [-.01, model.wealth[0]+.01, 0, 0, 0, 0, 0, 0]
        invalid[2][1] = [model.wealth[1], 0, 0, 0, 0, 0, 0, 0]
        rejections = 0
        for allocation in invalid:
            try:
                constraints.check(allocation)
            except AssertionError:
                rejections += 1
        assert rejections == len(invalid)
        return {"feasibility_rejections": rejections}

# Class: Run the four checks and combine their small result dictionaries.
@dataclass
class NumericalValidator:
    """Run the four checks and combine their small result dictionaries."""
    config: ExperimentConfig

    # Function: Run every independent check in a fixed order and combine their result dictionaries.
    def run(self):
        rng = np.random.default_rng(self.config.validation_seed)
        small = RiskSharingModel.from_config(self.config, states=6)
        small = replace(small, ranks=small.ranks**.8,
                        probabilities=np.arange(1., 7.)/21, weights=np.linspace(.7, 1.4, 8))
        report = {"passed": True}
        report.update(GradientCheck(self.config).run(small, rng))
        report.update(OracleCheck(self.config).run(small, rng))
        model = RiskSharingModel.from_config(self.config, states=12)
        optimizer_report, direct = OptimizerCheck(self.config).run(model)
        report.update(optimizer_report)
        report.update(FeasibilityCheck(self.config).run(model, direct))
        return report

# Class: Calculate acceptance checks and distinguish grid error from optimization error.
class ResultAnalysis:
    """Calculate acceptance checks and distinguish grid error from optimization error."""
    # Function: Keep a reference to the experiment whose numerical evidence will be analyzed.
    def __init__(self, experiment):
        self.experiment = experiment

    # Function: Calculate allocation errors, seven acceptance gates, and the continuous reference.
    def evaluate(self):
        e = self.experiment
        model, benchmark, result = e.model, e.benchmark, e.result
        exact_value = model.objective(e.exact)
        welfare_error = exact_value - result.final["welfare"]
        allocation_error = float(np.max(abs(result.allocation - e.exact)))
        symmetry_error = max(float(np.ptp(result.allocation[:, group], axis=1).max())
                             for group in (benchmark.h1, benchmark.h2))
        root_error = float(np.max(abs(benchmark.scalar_allocation() - e.exact)))
        inactive = result.allocation[model.wealth <= benchmark.activation][:, benchmark.h1]
        leakage = float(inactive.max()) if inactive.size else 0.
        tolerance = e.config.roundoff_tolerance
        gates = {"converged": result.converged,
                 "allocation_matches": allocation_error <= e.config.allocation_tolerance,
                 "analytic_inside_certificate": -tolerance <= welfare_error <= result.final["gap"] + tolerance,
                 "scalar_root_matches": root_error < 1e-10,
                 "identical_agents_match": symmetry_error < 1e-12,
                 "welfare_nondecreasing": bool(np.all(np.diff([r["welfare"] for r in result.history]) >= -tolerance)),
                 "inactive_region_matches": leakage <= e.config.allocation_tolerance}
        # Function: Return total analytic utility at one continuous wealth value for integration.
        def continuous_utility(wealth):
            row = benchmark.allocation(np.array([wealth]))[0]
            return (5*np.log1p(np.sqrt(wealth) + row[benchmark.h1[0]])
                    + 3*np.log1p(wealth**(2/3) + row[benchmark.h2[0]]))
        continuous, integration_error = quad(continuous_utility, 0., 1., points=[benchmark.activation],
                                             epsabs=1e-11, epsrel=1e-11, limit=200)
        return {"status": "PASS" if all(gates.values()) else "FAIL",
                "completed_at_utc": datetime.now(timezone.utc).isoformat(),
                "config": asdict(e.config), "iterations": len(result.history)-1,
                "final": result.final, "analytic_grid_welfare": exact_value,
                "welfare_error": welfare_error, "maximum_allocation_error": allocation_error,
                "symmetry_error": symmetry_error, "analytic_feasibility": e.constraints.check(e.exact),
                "scalar_root_allocation_error": root_error, "analytic_activation_wealth": benchmark.activation,
                "maximum_h1_leakage_below_activation": leakage,
                "continuous_welfare_reference": continuous, "quad_estimated_absolute_error": integration_error,
                "midpoint_grid_minus_continuous_reference": exact_value - continuous, "gates": gates,
                "certificate_scope": "Concave optimization on the fixed finite grid; floating point accuracy",
                "day4_deliverables": "COMPLETE" if all(gates.values()) else "INCOMPLETE",
                "review_note": "Both roles delegated; no blind comparison or personal review by Parsa claimed"}

# Class: Display the stored evidence and build the comparison table.
class ResultTables:
    """Display the stored evidence and build the comparison table."""
    # Function: Keep a reference to the experiment whose tables will be displayed.
    def __init__(self, experiment):
        self.experiment = experiment

    # Function: Print selected numeric columns with aligned headings and eight decimal places.
    @staticmethod
    def _print_table(rows, columns):
        print("  ".join(f"{column:>16}" for column in columns))
        for row in rows:
            print("  ".join(f"{row[column]:16.8f}" for column in columns))

    # Function: Build numerical-versus-analytic comparisons at selected midpoint states.
    def comparison_table(self):
        e = self.experiment
        e._require_result()
        wealth, b = e.model.wealth, e.benchmark
        points = [0., .1, .2, b.activation, .35, .5, .75, 1.]
        indices = sorted(set(np.argmin(abs(wealth-point)) for point in points))
        return [{"wealth": wealth[j], "numerical_b_h1": e.result.allocation[j, b.h1[0]],
                 "analytic_b_h1": e.exact[j, b.h1[0]],
                 "numerical_a_h2": e.result.allocation[j, b.h2[0]],
                 "analytic_a_h2": e.exact[j, b.h2[0]],
                 "max_error": float(np.max(abs(e.result.allocation[j]-e.exact[j])))} for j in indices]

    # Function: Display the stored welfare, gap, feasibility checks, and comparison table.
    def show_summary(self):
        e = self.experiment
        e._require_result()
        for label, value in [("Status", e.summary["status"]), ("Solver updates", e.summary["iterations"]),
                             ("Welfare F", e.result.final["welfare"]), ("Gap G", e.result.final["gap"]),
                             ("Upper bound F+G", e.result.final["upper_bound"]),
                             ("Analytic grid welfare", e.summary["analytic_grid_welfare"]),
                             ("Actual welfare error", e.summary["welfare_error"]),
                             ("Maximum allocation error", e.summary["maximum_allocation_error"])]:
            print(f"{label}: {value}")
        print("\nFeasibility:", e.constraints.check(e.result.allocation))
        print("Validation:", e.checks)
        print("\nComparison at selected grid states (b: each h1; a: each h2):")
        self._print_table(self.comparison_table(),
                          ["wealth", "numerical_b_h1", "analytic_b_h1", "numerical_a_h2", "analytic_a_h2"])
        print("\nGrid minus continuous analytic welfare:", e.summary["midpoint_grid_minus_continuous_reference"])

# Class: Write any list of row dictionaries to a CSV file.
@dataclass
class CSVWriter:
    """Write any list of row dictionaries to a CSV file."""
    directory: Path

    # Function: Write column headings and row dictionaries to a CSV text file.
    def write(self, filename, rows):
        with (self.directory / filename).open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

# Class: Build the complete numerical and analytic allocation table.
@dataclass
class AllocationTable:
    """Build the complete numerical and analytic allocation table."""
    experiment: object

    # Function: Construct one complete allocation row per state, including all eight agents.
    def rows(self):
        e = self.experiment
        rows = []
        for j, wealth in enumerate(e.model.wealth):
            row = {"wealth": wealth, "rank": e.model.ranks[j], "probability": e.model.probabilities[j]}
            for prefix, allocation in [("numerical", e.result.allocation), ("analytic", e.exact)]:
                row.update({f"{prefix}_agent_{i+1}": value for i, value in enumerate(allocation[j])})
            row["budget_residual"] = e.result.allocation[j].sum() - wealth
            row["max_allocation_error"] = float(np.max(abs(e.result.allocation[j] - e.exact[j])))
            rows.append(row)
        return rows

# Class: Save JSON diagnostics and reuse the CSV writer for every table.
class ResultFiles:
    """Save JSON diagnostics and reuse the CSV writer for every table."""
    # Function: Connect this exporter to the experiment, shared CSV writer, and allocation table.
    def __init__(self, experiment):
        self.experiment = experiment
        self.csv = CSVWriter(experiment.output)
        self.allocations = AllocationTable(experiment)

    # Function: Save four JSON diagnostics files and four reusable CSV tables.
    def save_results(self):
        e = self.experiment
        e._require_result()
        e.output.mkdir(parents=True, exist_ok=True)
        environment = {"python": sys.version, "platform": platform.platform(), "numpy": np.__version__,
                       "scipy": scipy.__version__, "matplotlib": matplotlib.__version__}
        for filename, data in [("summary.json", e.summary), ("checks.json", e.checks),
                               ("environment.json", environment), ("config.json", asdict(e.config))]:
            (e.output / filename).write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
        for filename, rows in [("analytic_benchmark_table.csv", e.predictions),
                               ("comparison_table.csv", e.tables.comparison_table()),
                               ("convergence.csv", e.result.history), ("allocations.csv", self.allocations.rows())]:
            self.csv.write(filename, rows)
        print("Saved tables and diagnostics to:", e.output.resolve())

# Class: Bundle only this experiment's expected files and offer a Colab download.
@dataclass
class ResultDownload:
    """Bundle only this experiment's expected files and offer a Colab download."""
    experiment: object

    # Function: Check for all eleven output files, ZIP them, and offer a download in Colab.
    def download(self):
        e = self.experiment
        e._require_result()
        expected = ["summary.json", "checks.json", "environment.json", "config.json", "allocations.csv",
                    "analytic_benchmark_table.csv", "comparison_table.csv", "convergence.csv",
                    "allocation_overlay.png", "cumulative_allocations.png", "convergence.png"]
        if not all((e.output / name).is_file() for name in expected):
            raise RuntimeError("Run experiment.files.save_results() and experiment.plots.show() first")
        archive = e.output.resolve().with_name(e.output.name + "_download.zip")
        with ZipFile(archive, "w", compression=ZIP_DEFLATED) as bundle:
            for name in expected:
                bundle.write(e.output / name, arcname=name)
        print("Results ZIP:", archive)
        if colab_files is not None:
            colab_files.download(archive)
        return archive

# Class: Build, save, and display each plot once from the stored experiment.
class ResultPlots:
    """Build, save, and display each plot once from the stored experiment."""
    # Function: Keep a reference to the solved experiment used by every plot.
    def __init__(self, experiment):
        self.experiment = experiment

    # Function: Draw numerical and analytic group curves, plus a zoom around activation.
    def _allocation_overlay(self):
        e = self.experiment
        wealth, allocation, benchmark = e.model.wealth, e.result.allocation, e.benchmark
        figure, axes = plt.subplots(1, 2, figsize=(12, 4.5))
        for ax in axes:
            groups = [(benchmark.h1, "h1: agents 3–7", "#1976b9"),
                      (benchmark.h2, "h2: agents 1, 2, 8", "#d16d28")]
            for group, label, color in groups:
                ax.plot(wealth, allocation[:, group[0]], color=color, label=label+" numerical")
                ax.plot(wealth, e.exact[:, group[0]], "--", color=color, label=label+" analytic")
            ax.axvline(benchmark.activation, color="gray", ls=":", label="Activation ≈ 0.299")
            ax.set(xlabel="Total wealth", ylabel="Consumption per agent")
            ax.grid(alpha=.2)
        axes[0].set_title("Numerical and analytic allocations")
        axes[0].legend(fontsize=8)
        axes[1].set(xlim=(.24, .38), ylim=(-.001, .13), title="Activation region")
        return figure

    # Function: Draw every agent's share as a band with analytic cumulative boundaries.
    def _cumulative_allocations(self):
        e = self.experiment
        figure, ax = plt.subplots(figsize=(9, 4.5))
        ax.stackplot(e.model.wealth, e.result.allocation.T,
                     labels=[f"Agent {i+1}" for i in range(e.model.shape[1])], alpha=.8)
        for boundary in np.cumsum(e.exact, axis=1).T:
            ax.plot(e.model.wealth, boundary, "k--", lw=.65)
        ax.set(xlabel="Total wealth", ylabel="Cumulative consumption", title="All eight agents")
        ax.legend(loc="upper left", ncol=4, fontsize=8)
        return figure

    # Function: Plot the welfare gap per iteration and the stopping tolerance.
    def _convergence(self):
        e = self.experiment
        figure, ax = plt.subplots(figsize=(8, 4))
        ax.semilogy([r["iteration"] for r in e.result.history],
                    [r["gap"] for r in e.result.history], "o-")
        ax.axhline(e.config.gap_tolerance, ls="--", color="#d16d28", label="Stopping tolerance")
        ax.set(xlabel="Iteration", ylabel="Finite-grid welfare gap", title="Convergence certificate")
        ax.legend()
        ax.grid(alpha=.2)
        return figure

    # Function: Build each figure once, save it, optionally display it, and close it.
    def show(self, display=True):
        e = self.experiment
        e._require_result()
        e.output.mkdir(parents=True, exist_ok=True)
        plot_builders = [(self._allocation_overlay, "allocation_overlay.png"),
                         (self._cumulative_allocations, "cumulative_allocations.png"),
                         (self._convergence, "convergence.png")]
        # Create one figure at a time, then save, display, and close it.
        for build, filename in plot_builders:
            figure = build()
            figure.tight_layout()
            figure.savefig(e.output / filename, dpi=180)
            if display:
                plt.show()
            plt.close(figure)

# Class: Coordinate the components; delegate analysis, tables, files, and plots.
class Day4Experiment:
    """Coordinate the components; delegate analysis, tables, files, and plots."""
    # Function: Create the model and connect each solver, checking, reporting, and export object.
    def __init__(self, config):
        self.config = config
        self.model = RiskSharingModel.from_config(config)
        self.constraints = AllocationConstraints(self.model, config.feasibility_tolerance)
        self.benchmark = Example2Benchmark(self.model)
        self.solver = FrankWolfeSolver(self.model, self.constraints, config)
        self.output = Path(config.output_directory)
        self.result = None
        # These objects share this experiment, keeping one source of results.
        self.analysis = ResultAnalysis(self)
        self.tables = ResultTables(self)
        self.files = ResultFiles(self)
        self.downloads = ResultDownload(self)
        self.plots = ResultPlots(self)

    # Function: Create predictions, validate, solve once, analyze results, and enforce acceptance.
    def run(self):
        self.result = None
        self.predictions = self.benchmark.prediction_table()
        self.checks = NumericalValidator(self.config).run()
        self.result = self.solver.solve()
        self.exact = self.benchmark.allocation(self.model.wealth)
        self.summary = self.analysis.evaluate()
        if not all(self.summary["gates"].values()):
            raise AssertionError(self.summary["gates"])
        return self

    # Function: Stop with a clear message when reporting is requested before a solver run.
    def _require_result(self):
        if self.result is None:
            raise RuntimeError("Call experiment.run() before viewing or saving results")

# Run the complete notebook workflow locally without opening Colab.
def main():
    config = ExperimentConfig(output_directory=str(Path(__file__).resolve().parent / "results"))
    experiment = Day4Experiment(config).run()
    experiment.tables.show_summary()
    experiment.files.save_results()
    experiment.plots.show()
    experiment.downloads.download()


if __name__ == "__main__":
    main()
