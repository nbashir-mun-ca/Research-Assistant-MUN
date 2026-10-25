"""Check that the cross-solver audit rejects comparisons of different models."""

from dataclasses import replace
import unittest

import numpy as np

from day04.day4_example import ExperimentConfig, RiskSharingModel
from dynamic_programming.compare_nouman import (
    ROOT, independent_gap, independent_welfare, load_reference,
    require_same_model, scalar_reference,
)


class ComparisonTests(unittest.TestCase):
    def test_independent_evaluation_includes_weights_and_probabilities(self):
        model = RiskSharingModel([.2, .4], [.1, .8], [.3, .7], [2., 1.], ("h1", "h2"))
        allocation = model.equal_sharing()
        self.assertAlmostEqual(independent_welfare(model, allocation), model.objective(allocation))
        from day04.day4_example import AllocationConstraints
        gradient = model.gradient(allocation)
        oracle = AllocationConstraints(model).linear_oracle(gradient)
        expected = float(np.sum(gradient*(oracle-allocation)))
        self.assertAlmostEqual(independent_gap(model, allocation), expected)

    def test_model_mismatch_is_rejected(self):
        model = RiskSharingModel.from_config(ExperimentConfig(states=4))
        for field, values in [("wealth", model.wealth+.001),
                              ("ranks", model.ranks+.001),
                              ("probabilities", np.array([.1, .2, .3, .4])),
                              ("weights", model.weights*2)]:
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, field):
                require_same_model(model, replace(model, **{field: values}))
        changed_types = tuple(reversed(model.agent_types))
        with self.assertRaisesRegex(ValueError, "preferences"):
            require_same_model(model, replace(model, agent_types=changed_types))

    def test_saved_day4_and_day5_allocations_match_reported_objectives(self):
        day4 = load_reference(ROOT / "day04/evidence")
        day5 = load_reference(ROOT / "day05/results/grid_original_400")
        require_same_model(day4.model, day5.model)
        np.testing.assert_allclose(day4.allocation, day5.allocation, rtol=0, atol=2e-15)
        self.assertAlmostEqual(day4.welfare, day5.welfare)

    def test_scalar_equations_supply_a_feasible_global_example2_benchmark(self):
        model = RiskSharingModel.from_config(ExperimentConfig(states=40))
        allocation = scalar_reference(model)
        self.assertLess(independent_gap(model, allocation), 1e-11)


if __name__ == "__main__":
    unittest.main()
