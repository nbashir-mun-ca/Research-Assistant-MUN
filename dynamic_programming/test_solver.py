"""Independent tiny exhaustive checks and existing-model benchmarks."""

from itertools import product
import os
import unittest

os.environ.setdefault("MPLBACKEND", "Agg")
import numpy as np
from scipy.optimize import Bounds, LinearConstraint, minimize

from day04.day4_example import (
    AllocationConstraints, Example2Benchmark, ExperimentConfig, RiskSharingModel,
    validation_matrices,
)
from dynamic_programming.solver import solve_lattice, solve_log_model


def exhaustive(wealth, quantum, groups, reward):
    """Enumerate entire paths, without a Bellman recurrence or shared helpers."""
    stages = []
    for wealth_value in wealth:
        q = round(wealth_value / quantum)
        stages.append([np.array(x) * quantum for x in product(range(q + 1), repeat=groups)
                       if sum(x) == q])
    best = -np.inf
    for path in product(*stages):
        c = np.array(path)
        if np.any(np.diff(c, axis=0) < -1e-12):
            continue
        value = sum(float(reward(j, row[None, :])[0]) for j, row in enumerate(c))
        best = max(best, value)
    return best


class LatticeTests(unittest.TestCase):
    def test_two_agent_nonconcave_rewards_and_binding_monotonicity(self):
        wealth = np.array([.2, .3, .5])
        targets = np.array([.2, 0., .5])

        def reward(j, x):
            return -(x[:, 0] - targets[j])**2 + .003 * np.cos(70*x[:, 0])

        result = solve_lattice(wealth, .1, reward)
        self.assertAlmostEqual(result.objective, exhaustive(wealth, .1, 2, reward))
        np.testing.assert_allclose(result.allocation.sum(axis=1), wealth, atol=1e-14)
        self.assertTrue(np.all(np.diff(result.allocation, axis=0) >= -1e-14))
        pointwise = [int(np.argmax(reward(j, np.column_stack(
            (np.arange(q + 1)*.1, wealth[j] - np.arange(q + 1)*.1)))))*.1
                     for j, q in enumerate([2, 3, 5])]
        self.assertLess(pointwise[1], pointwise[0])

    def test_random_rewards_match_complete_path_enumeration(self):
        rng = np.random.default_rng(20261022)
        for groups in (2, 3):
            for _ in range(4):
                wealth = np.array([0., .2, .2, .4])
                coefficients = rng.normal(size=(4, groups, 3))

                def reward(j, x):
                    a, b, c = coefficients[j].T
                    return np.sum(a*x + b*x*x + c*np.cos(17*x), axis=1)

                actual = solve_lattice(wealth, .1, reward, groups)
                self.assertAlmostEqual(actual.objective, exhaustive(wealth, .1, groups, reward))

    def test_one_group_and_negative_rewards(self):
        reward = lambda j, x: -np.sum(1 + x*x, axis=1)
        actual = solve_lattice([0., .2, .2, .5], .1, reward, 1)
        np.testing.assert_allclose(actual.allocation[:, 0], [0., .2, .2, .5])
        self.assertAlmostEqual(actual.objective, -4.33)

    def test_zero_reward_ties_are_deterministic(self):
        reward = lambda j, x: np.zeros(len(x))
        for groups in (2, 3):
            a = solve_lattice([.1, .3], .1, reward, groups)
            b = solve_lattice([.1, .3], .1, reward, groups)
            np.testing.assert_array_equal(a.allocation, b.allocation)

    def test_invalid_grid_and_rewards(self):
        reward = lambda j, x: np.zeros(len(x))
        for wealth, quantum in (([.15], .1), ([.2, .1], .1), ([-.1], .1),
                                ([], .1), ([np.nan], .1), ([.1], 0),
                                ([.1], np.inf), ([1.], 1e-300)):
            with self.subTest(wealth=wealth, quantum=quantum), self.assertRaises(ValueError):
                solve_lattice(wealth, quantum, reward)
        for bad in (lambda j, x: 0., lambda j, x: np.full(len(x), np.nan)):
            with self.assertRaises(ValueError):
                solve_lattice([.1], .1, bad)

    def test_resource_limits_before_evaluating_rewards(self):
        def forbidden(j, x):
            self.fail("The resource check must run before constructing states")
        with self.assertRaisesRegex(ValueError, "stored states"):
            solve_lattice([1.], .001, forbidden, 8)
        with self.assertRaisesRegex(ValueError, "transition visits"):
            solve_lattice([.2, .4], .1, forbidden, 3, max_transitions=1)


class LogModelTests(unittest.TestCase):
    def test_eight_agent_example2_and_duality_certificate(self):
        model = RiskSharingModel.from_config(ExperimentConfig(states=40))
        result = solve_log_model(model, 1/80)
        AllocationConstraints(model).check(result.allocation)
        self.assertEqual(result.groups, ((0, 1, 7), (2, 3, 4, 5, 6)))
        reference = Example2Benchmark(model).allocation(model.wealth)
        optimum = model.objective(reference)
        self.assertLessEqual(result.objective, optimum + 1e-12)
        self.assertLessEqual(optimum, result.finite_grid_upper_bound + 1e-12)
        self.assertAlmostEqual(result.objective, result.lattice.objective)
        oracle = AllocationConstraints(model).linear_oracle(model.gradient(result.allocation))
        expected_gap = np.sum(model.gradient(result.allocation)*(oracle-result.allocation))
        self.assertAlmostEqual(result.duality_gap, expected_gap)

    def test_refining_a_nested_lattice_cannot_reduce_welfare(self):
        model = RiskSharingModel.from_config(ExperimentConfig(states=30))
        coarse = solve_log_model(model, 1/60)
        fine = solve_log_model(model, 1/240)
        self.assertGreaterEqual(fine.objective, coarse.objective - 1e-12)
        optimum = model.objective(Example2Benchmark(model).allocation(model.wealth))
        self.assertLess(optimum - fine.objective, 1e-6)

    def test_agent_permutation_preserves_solution(self):
        model = RiskSharingModel.from_config(ExperimentConfig(states=20))
        order = np.array([3, 7, 0, 5, 1, 2, 6, 4])
        permuted = RiskSharingModel(model.wealth, model.ranks, model.probabilities,
                                   model.weights[order], tuple(model.agent_types[i] for i in order))
        a = solve_log_model(model, 1/40)
        b = solve_log_model(permuted, 1/40)
        np.testing.assert_allclose(a.allocation, b.allocation[:, np.argsort(order)], atol=1e-13)

    def test_unequal_weights_nonuniform_ranks_zero_probability(self):
        model = RiskSharingModel([0., .2, .4], [.1, .4, .9], [0., .3, .7],
                                 [.7, 1.4, .9], ("h1", "h2", "h1"))
        result = solve_log_model(model, .1)
        self.assertEqual(len(result.groups), 3)  # h1 agents have unequal weights.
        AllocationConstraints(model).check(result.allocation)
        reward = lambda j, x: model.probabilities[j] * np.sum(
            model.weights * np.log(model.offsets[j] + x), axis=1)
        self.assertAlmostEqual(result.objective, exhaustive(model.wealth, .1, 3, reward))

    def test_grouped_and_individual_lattices_have_distinct_meanings(self):
        model = RiskSharingModel([.1], [.5], [1.], [1., 1.], ("h1", "h1"))
        grouped = solve_log_model(model, .1)
        separate = solve_log_model(model, .1, group_identical=False)
        np.testing.assert_allclose(grouped.allocation, [[.05, .05]])
        self.assertGreater(grouped.objective, separate.objective)
        self.assertEqual(len(separate.groups), 2)

    def test_log_case_with_binding_constraints_against_direct_slsqp(self):
        model = RiskSharingModel([.2, .21, .22], [.01, .2, .9], [.2, .3, .5],
                                 [1., 1.], ("h1", "h2"))
        pointwise = np.clip((model.wealth + model.ranks**(2/3)
                             - np.sqrt(model.ranks))/2, 0, model.wealth)
        self.assertTrue(np.any(np.diff(pointwise) < 0))
        budget, monotone = validation_matrices(model)
        reference = minimize(
            lambda x: -model.objective(x.reshape(model.shape)),
            model.equal_sharing().ravel(),
            jac=lambda x: -model.gradient(x.reshape(model.shape)).ravel(),
            bounds=Bounds(0, np.inf),
            constraints=[LinearConstraint(budget, model.wealth, model.wealth),
                         LinearConstraint(monotone, 0, np.inf)],
            method="SLSQP", options={"ftol": 1e-13, "maxiter": 1000})
        self.assertTrue(reference.success, reference.message)
        result = solve_log_model(model, .001)
        AllocationConstraints(model).check(result.allocation)
        optimum = -reference.fun  # Numerical reference, not a proof of optimality.
        self.assertLessEqual(result.objective, optimum + 1e-11)
        self.assertLess(optimum - result.objective, 1e-6)
        self.assertLessEqual(optimum, result.finite_grid_upper_bound + 1e-11)


if __name__ == "__main__":
    unittest.main()
