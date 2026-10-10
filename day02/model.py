"""Provisional concave log risk-sharing model; no optimization loop."""
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class Model:
    wealth: np.ndarray
    ranks: np.ndarray
    probabilities: np.ndarray
    pareto_weights: np.ndarray
    agent_types: tuple[str, ...]

    def __post_init__(self):
        for name in ("wealth", "ranks", "probabilities", "pareto_weights"):
            value = np.array(getattr(self, name), dtype=float, copy=True)
            if value.ndim != 1 or not value.size or not np.isfinite(value).all():
                raise ValueError(f"{name} must be a finite nonempty vector")
            value.setflags(write=False)
            object.__setattr__(self, name, value)
        object.__setattr__(self, "agent_types", tuple(self.agent_types))
        if self.ranks.shape != self.wealth.shape or self.probabilities.shape != self.wealth.shape:
            raise ValueError("wealth, ranks, and probabilities must have the same shape")
        if np.any(self.wealth < 0) or np.any(np.diff(self.wealth) < 0):
            raise ValueError("wealth must be nonnegative and nondecreasing")
        if np.any((self.ranks <= 0) | (self.ranks >= 1)) or np.any(np.diff(self.ranks) <= 0):
            raise ValueError("ranks must be strictly increasing in (0,1)")
        if np.any(self.probabilities < 0) or not np.isclose(self.probabilities.sum(), 1, rtol=0, atol=1e-12):
            raise ValueError("probabilities must be nonnegative and sum to one")
        if np.any(self.pareto_weights <= 0):
            raise ValueError("Pareto weights must be positive")
        if len(self.agent_types) != self.pareto_weights.size or any(t not in ("h1", "h2") for t in self.agent_types):
            raise ValueError("one h1/h2 type is required per Pareto weight")

    @classmethod
    def from_config(cls, config):
        return cls(config["wealth"], config["ranks"], config["probabilities"],
                   config["pareto_weights"], tuple(config["agent_types"]))

    @property
    def shape(self):
        return self.wealth.size, self.pareto_weights.size

    @property
    def offsets(self):
        exponents = np.array([0.5 if t == "h1" else 2 / 3 for t in self.agent_types])
        return 1 + self.ranks[:, None] ** exponents[None, :]

    def _consumption(self, consumption):
        c = np.asarray(consumption, dtype=float)
        if c.shape != self.shape or not np.isfinite(c).all():
            raise ValueError(f"consumption must be finite with shape {self.shape}")
        if np.any(c < 0):
            raise ValueError("consumption must be nonnegative")
        return c

    def utilities(self, consumption):
        """Return unweighted utilities, one entry per state and agent."""
        c = self._consumption(consumption)
        return np.log(self.offsets + c)

    def objective(self, consumption):
        return float(np.sum(self.probabilities[:, None] * self.pareto_weights[None, :]
                            * self.utilities(consumption)))

    def gradient(self, consumption):
        """Derivative with respect to direct consumption, not increments."""
        c = self._consumption(consumption)
        return self.probabilities[:, None] * self.pareto_weights[None, :] / (self.offsets + c)

    def equal_sharing(self):
        return np.repeat(self.wealth[:, None] / self.shape[1], self.shape[1], axis=1)


def feasibility(consumption, wealth, budget_atol=1e-10, domain_atol=1e-12):
    c, s = np.asarray(consumption, dtype=float), np.asarray(wealth, dtype=float)
    if c.ndim != 2 or not c.shape[1] or s.ndim != 1 or c.shape[0] != s.size or not s.size:
        raise ValueError("expected consumption (states,agents) and wealth (states,)")
    if not np.isfinite(c).all() or not np.isfinite(s).all():
        raise ValueError("allocation and wealth must be finite")
    if np.any(s < 0) or np.any(np.diff(s) < 0):
        raise ValueError("wealth must be nonnegative and nondecreasing")
    increments = np.diff(np.vstack((np.zeros((1, c.shape[1])), c)), axis=0)
    budget = float(np.max(np.abs(c.sum(axis=1) - s)))
    minimum = float(c.min())
    min_increment = float(increments.min())
    return {"budget_residual": budget, "minimum_consumption": minimum,
            "minimum_increment_including_initial": min_increment,
            "feasible": bool(budget <= budget_atol and minimum >= -domain_atol
                             and min_increment >= -domain_atol)}


def tail_scores(gradient):
    g = np.asarray(gradient, dtype=float)
    if g.ndim != 2 or not g.size or not np.isfinite(g).all():
        raise ValueError("gradient must be a finite nonempty matrix")
    return np.cumsum(g[::-1], axis=0)[::-1]


def linear_oracle(gradient, wealth):
    """Maximize <gradient,y> over nonnegative, monotone, statewise clearing y.

    Equal splits apply only to exact score ties. An approximate tie rule could
    sacrifice exact linear optimality and would require separate error accounting.
    All agents remain explicit, including when every score is negative.
    """
    scores = tail_scores(gradient)
    s = np.asarray(wealth, dtype=float)
    if s.shape != (scores.shape[0],) or not np.isfinite(s).all():
        raise ValueError("wealth must be a finite vector matching the state count")
    if np.any(s < 0) or np.any(np.diff(s) < 0):
        raise ValueError("wealth must be nonnegative and nondecreasing")
    ds = np.diff(np.r_[0.0, s])
    winners = scores == scores.max(axis=1, keepdims=True)
    increments = ds[:, None] * winners / winners.sum(axis=1, keepdims=True)
    return np.cumsum(increments, axis=0)
