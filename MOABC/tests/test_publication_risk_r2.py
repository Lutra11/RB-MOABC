import unittest

import numpy as np


class ProjectionTests(unittest.TestCase):
    def test_projection_is_sequential_from_previous_release(self):
        from MOABC.experiments.risk_r2 import project_release_plan

        projected = project_release_plan(
            np.array([[9000., 0., 9000.]]),
            minimum=np.array([0.]), maximum=np.array([10000.]),
            ramp=np.array([1000.]), previous=np.array([5000.]))
        np.testing.assert_allclose(projected, [[6000., 5000., 6000.]])

    def test_plan_diagnostics_do_not_hide_original_violation(self):
        from MOABC.experiments.risk_r2 import plan_diagnostics

        diagnostics = plan_diagnostics(
            np.array([[12000., 0.]]), np.array([1000.]),
            np.array([10000.]), np.array([1000.]), np.array([5000.]))
        self.assertGreater(diagnostics['planned_bound_violation'], 0)
        self.assertGreater(diagnostics['planned_ramp_violation'], 0)

    def test_rolling_problem_projects_from_last_executed_release(self):
        from MOABC.experiments.risk_r2 import problem_for_r2
        from MOABC.simulation import build_default_network

        network = build_default_network()
        previous = np.array([10000., 11000., 12000., 13000., 14000.])
        problem = problem_for_r2(network, np.zeros((5, 2, 3)), None, 'P01',
                                 previous_release=previous)
        submitted = np.full((5, 2), 1e9)
        projected = problem.decode(problem.clip(submitted.ravel()))
        ramps = np.array([network.reservoirs[name].max_ramp for name in
                          ('Wudongde', 'Baihetan', 'Xiluodu', 'Xiangjiaba', 'TGD')])
        np.testing.assert_array_less(projected[:, 0], previous + ramps + 1e-9)
        np.testing.assert_array_less(previous - ramps - 1e-9, projected[:, 0])


class PhysicalDiagnosticTests(unittest.TestCase):
    def test_water_shortage_is_not_mislabeled_as_forced_spill(self):
        from MOABC.experiments.risk_r2 import NAMES, simulate_r2
        from MOABC.simulation import build_default_network

        network = build_default_network()
        initial = np.array([network.reservoirs[name].S_min for name in NAMES])
        plan = np.array([[network.reservoirs[name].min_release] for name in NAMES])
        result = simulate_r2(network, initial, np.zeros((5, 1, 3)), plan)
        self.assertGreater(result.water_curtailment, 0)
        self.assertEqual(result.hard_physical_violation, 0)

    def test_forced_spill_above_capacity_is_hard_physical_violation(self):
        from MOABC.experiments.risk_r2 import NAMES, simulate_r2
        from MOABC.simulation import build_default_network

        network = build_default_network()
        initial = np.array([network.reservoirs[name].S_max for name in NAMES])
        plan = np.array([[network.reservoirs[name].min_release] for name in NAMES])
        inflow = np.full((5, 1, 3), 1e8)
        result = simulate_r2(network, initial, inflow, plan)
        self.assertGreater(result.forced_spill_capacity_excess, 0)
        self.assertGreater(result.hard_physical_violation, 0)


class PressureMatchingTests(unittest.TestCase):
    def test_downstream_linear_stress_is_fixed_and_event_only(self):
        from MOABC.experiments.risk_r2 import apply_stress_pattern

        base = np.ones((5, 4))
        stressed = apply_stress_pattern(base, event_days=2, factor=3.,
                                        pattern='downstream_linear')
        expected_scale = np.array([1., 1.5, 2., 2.5, 3.])
        np.testing.assert_allclose(stressed[:, :2],
                                   np.repeat(expected_scale[:, None], 2, axis=1))
        np.testing.assert_allclose(stressed[:, 2:], 1.)
        np.testing.assert_allclose(base, 1.)

    def test_unknown_stress_pattern_is_rejected(self):
        from MOABC.experiments.risk_r2 import apply_stress_pattern

        with self.assertRaises(ValueError):
            apply_stress_pattern(np.ones((5, 2)), 1, 2., 'event_tuned')

    def test_tail_peak_ratio_cvar_is_continuous(self):
        from MOABC.experiments.risk_r2 import tail_peak_ratio_cvar95

        q = np.array([[90., 100., 110., 120.], [95., 101., 111., 121.]])
        value = tail_peak_ratio_cvar95(q, 100.)
        self.assertGreater(value, 1.2)
        self.assertLess(value, 1.22)

    def test_pressure_selection_rejects_physical_violation(self):
        from MOABC.experiments.risk_r2 import select_pressure_factor

        rows = [
            dict(factor=2., tail_peak_ratio_cvar95=1.10,
                 plan_constraint_violation=0., hard_physical_violation=.01),
            dict(factor=2.5, tail_peak_ratio_cvar95=1.18,
                 plan_constraint_violation=0., hard_physical_violation=0.),
        ]
        selected = select_pressure_factor(rows, 'tail_peak_ratio_cvar95', 1.02, 1.20)
        self.assertEqual(selected['matched_factor'], 2.5)

    def test_pressure_selection_reports_unmatched(self):
        from MOABC.experiments.risk_r2 import select_pressure_factor

        selected = select_pressure_factor([
            dict(factor=1., tail_peak_ratio_cvar95=.8,
                 plan_constraint_violation=0., hard_physical_violation=0.)
        ], 'tail_peak_ratio_cvar95', 1.02, 1.20)
        self.assertIsNone(selected['matched_factor'])
        self.assertEqual(selected['status'], 'unmatched')


class LagGuidanceTests(unittest.TestCase):
    def test_upstream_guidance_precedes_downstream_guidance(self):
        from MOABC.experiments.risk_r2 import lag_aware_guidance
        from MOABC.simulation import build_default_network

        network = build_default_network()
        importance = np.zeros(20)
        importance[14] = 1.
        guidance, responses = lag_aware_guidance(network, importance)
        self.assertEqual(guidance.shape, (5, 20))
        self.assertTrue(np.isfinite(guidance).all())
        self.assertTrue((guidance >= 0).all())
        self.assertLessEqual(int(np.argmax(guidance[0])), int(np.argmax(guidance[-1])))
        self.assertGreater(int(np.argmax(responses[0])), int(np.argmax(responses[-1])))


if __name__ == '__main__':
    unittest.main()
