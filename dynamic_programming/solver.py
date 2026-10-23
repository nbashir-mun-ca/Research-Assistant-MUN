"""Bellman optimization over monotone cumulative allocations.

The lattice solver accepts arbitrary additive, finite stage rewards. The log
adapter uses the existing model and optionally averages identical agents, an
exact reduction for that concave objective. See README.md for certificate scope.
"""

from collections import deque
from dataclasses import dataclass
from math import comb
from numbers import Integral
from typing import Callable

import numpy as np


@dataclass
class LatticeResult:
    allocation: np.ndarray
    objective: float
    quantum: float
    states_evaluated: int
    transitions_considered: int
    algorithm: str


@dataclass
class LogModelResult:
    allocation: np.ndarray
    objective: float
    finite_grid_upper_bound: float
    duality_gap: float
    groups: tuple
    lattice: LatticeResult


def _compositions(total, count):
    """Yield nonnegative integer vectors summing to total, in stable order."""
    if count == 1:
        yield (total,)
    else:
        for first in range(total + 1):
            for rest in _compositions(total - first, count - 1):
                yield (first,) + rest


def _positive_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise ValueError(f"{name} must be a positive integer")


def _prepare(wealth, quantum, groups, max_states, max_transitions):
    _positive_integer(groups, "num_groups")
    _positive_integer(max_states, "max_states")
    _positive_integer(max_transitions, "max_transitions")
    if groups > 32:
        raise ValueError("At most 32 groups are supported; aggregate identical log agents")
    s = np.asarray(wealth, dtype=float)
    if s.ndim != 1 or not s.size or not np.isfinite(s).all():
        raise ValueError("wealth must be a finite nonempty vector")
    if np.any(s < 0) or np.any(np.diff(s) < 0):
        raise ValueError("wealth must be nonnegative and nondecreasing")
    if not np.isscalar(quantum) or not np.isfinite(quantum) or quantum <= 0:
        raise ValueError("quantum must be finite and positive")
    with np.errstate(over="ignore", invalid="ignore"):
        scaled = s / quantum
    # Refuse unsafe integer conversion and excessively large lattices up front.
    if not np.isfinite(scaled).all() or np.max(scaled) > 2**31 - 1:
        raise ValueError("Wealth/quantum is too large; use a coarser quantum")
    rounded = np.rint(scaled)
    if not np.allclose(scaled, rounded, rtol=0, atol=1e-8):
        raise ValueError("Every wealth value must be an integer multiple of quantum")
    units = rounded.astype(np.int64)
    counts = [comb(int(q) + groups - 1, groups - 1) for q in units]
    cells = sum(counts)
    if cells > max_states:
        raise ValueError(f"Lattice needs {cells:,} stored states; limit is {max_states:,}. "
                         "Use a coarser quantum or aggregate identical log agents.")
    work = 0
    previous_units, previous_count = 0, 1
    for q, count in zip(units, counts):
        if groups <= 2:
            work += count  # Sliding-window visits, rather than all Bellman edges.
        else:
            work += previous_count * comb(int(q) - previous_units + groups - 1,
                                          groups - 1)
        previous_units, previous_count = int(q), count
    if work > max_transitions:
        raise ValueError(f"Lattice needs {work:,} transition visits; limit is "
                         f"{max_transitions:,}. Use a coarser quantum.")
    return units, cells, work


def solve_lattice(wealth, quantum, stage_reward: Callable, num_groups=2, *,
                  max_states=2_000_000, max_transitions=20_000_000):
    """Maximize sum_j stage_reward(j, allocation[j]) on a monotone lattice.

    stage_reward(j, shares) receives a (candidates, num_groups) array of wealth
    shares and returns one finite reward per candidate, including probabilities
    and weights. Rewards may be nonconcave. Budgets are exact on the supplied
    lattice; wealth nodes are never silently rounded to an unrelated grid.
    Limits count cumulative stored states and transition visits, not bytes.
    """
    units, cells, work = _prepare(wealth, quantum, num_groups, max_states,
                                  max_transitions)
    quantum = float(quantum)
    previous = np.array([0.])
    previous_units = 0
    previous_states = [(0,) * num_groups]
    parents, stage_states = [], []
    for j, q_raw in enumerate(units):
        q = int(q_raw)
        delta = q - previous_units
        if num_groups == 2:
            x = np.arange(q + 1)
            states = np.column_stack((x, q - x))
            best = np.empty(q + 1)
            parent = np.empty(q + 1, dtype=np.int64)
            candidates = deque()
            # Feasible predecessor k lies in [max(0,x-delta), min(x,q_prev)].
            # Both bounds move right. Keep decreasing values in a deque.
            for current in range(q + 1):
                if current <= previous_units:
                    while candidates and previous[candidates[-1]] < previous[current]:
                        candidates.pop()
                    candidates.append(current)
                lower = max(0, current - delta)
                while candidates[0] < lower:
                    candidates.popleft()
                parent[current] = candidates[0]
                best[current] = previous[candidates[0]]
        else:
            states = list(_compositions(q, num_groups))
            index = {state: i for i, state in enumerate(states)}
            best = np.full(len(states), -np.inf)
            parent = np.full(len(states), -1, dtype=np.int64)
            increments = list(_compositions(delta, num_groups))
            for k, predecessor in enumerate(previous_states):
                for increment in increments:
                    target = tuple(a + b for a, b in zip(predecessor, increment))
                    i = index[target]
                    if previous[k] > best[i]:
                        best[i], parent[i] = previous[k], k
            states = np.asarray(states, dtype=np.int64)
            previous_states = states
        reward = np.asarray(stage_reward(j, states * quantum), dtype=float)
        if reward.shape != best.shape or not np.isfinite(reward).all():
            raise ValueError("stage_reward must return one finite value per candidate")
        previous = best + reward
        if not np.isfinite(previous).all():
            raise ValueError("Accumulated rewards overflowed; rescale the objective")
        parents.append(parent)
        stage_states.append(states)
        previous_units = q

    chosen = int(np.argmax(previous))
    objective = float(previous[chosen])
    allocation = np.empty((len(units), num_groups))
    for j in range(len(units) - 1, -1, -1):
        allocation[j] = stage_states[j][chosen] * quantum
        chosen = int(parents[j][chosen])
    algorithm = "sliding-window Bellman" if num_groups == 2 else "enumerated Bellman"
    return LatticeResult(allocation, objective, quantum, cells, work, algorithm)


def solve_log_model(model, quantum, *, group_identical=True, **limits):
    """Solve day04.RiskSharingModel on a lattice of agent/group total shares.

    Only agents with exactly equal utility offsets AND Pareto weights are
    grouped. Concavity justifies equal shares within each such group. Passing
    group_identical=False instead discretizes every individual agent's share.
    The reported upper bound concerns the existing finite wealth/rank grid,
    with continuous consumption, and is subject to floating-point roundoff.
    """
    groups = []
    for agent in range(model.shape[1]):
        matching = next((group for group in groups if group_identical
                         and model.weights[agent] == model.weights[group[0]]
                         and np.array_equal(model.offsets[:, agent],
                                            model.offsets[:, group[0]])), None)
        if matching is None:
            groups.append([agent])
        else:
            matching.append(agent)
    sizes = np.array([len(group) for group in groups])
    representatives = np.array([group[0] for group in groups])

    def reward(j, totals):
        return model.probabilities[j] * np.sum(
            sizes * model.weights[representatives]
            * np.log(model.offsets[j, representatives] + totals / sizes), axis=1)

    lattice = solve_lattice(model.wealth, quantum, reward, len(groups), **limits)
    allocation = np.empty(model.shape)
    for g, group in enumerate(groups):
        allocation[:, group] = lattice.allocation[:, g, None] / len(group)
    value = model.objective(allocation)
    # Concavity plus a feasible linear oracle supplies an upper bound without
    # needing to run Frank-Wolfe. The oracle assigns each wealth increment to
    # the agent with the largest tail sum of marginal utilities.
    gradient = model.gradient(allocation)
    tails = np.flip(np.cumsum(np.flip(gradient, axis=0), axis=0), axis=0)
    increments = np.zeros(model.shape)
    increments[np.arange(model.shape[0]), np.argmax(tails, axis=1)] = np.diff(
        np.r_[0., model.wealth])
    vertex = np.cumsum(increments, axis=0)
    gap = float(np.sum(gradient * (vertex - allocation)))
    if gap < -1e-10:
        raise ArithmeticError("Negative duality gap beyond floating-point tolerance")
    gap = max(0., gap)
    return LogModelResult(allocation, value, value + gap, gap,
                          tuple(tuple(g) for g in groups), lattice)
