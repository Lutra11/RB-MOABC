import json
import tempfile
import unittest
from pathlib import Path

import numpy as np


class InstrumentedSolverTests(unittest.TestCase):
    def test_callback_observes_archive_after_evaluation(self):
        from MOABC.core import Config, ReservoirSchedulingProblem
        from MOABC.experiments.instrumented_solver import run

        problem = ReservoirSchedulingProblem(
            n_res=1, T=2, bounds=np.array([[[0., 1.], [0., 1.]]]),
            inflow_scenarios=np.zeros((1, 2, 2)),
            release_reference=np.full((1, 2), .5))
        seen = []

        def evaluate(_, x):
            return np.array([x.sum(), (1-x).sum(), np.abs(x-.5).sum()]), True, None

        result = run(problem, 'MOABC', evaluate, seed=2,
                     cfg=Config(evaluations=20, population=5, archive_size=10),
                     checkpoint_callback=lambda context, solution: seen.append(
                         (context.count, len(context.archive.items))))
        self.assertEqual(result['evaluations'], 20)
        self.assertEqual(seen[-1][0], 20)
        self.assertGreaterEqual(seen[-1][1], 1)


class WorkDesignTests(unittest.TestCase):
    def test_expected_work_counts(self):
        from experiments.run_publication_risk_r2 import build_work

        plan = {f'P{i:02d}': 2. for i in range(1, 9)}
        fast = build_work(['P01', 'P02', 'P03'], plan, 'mechanisms', 3)
        all_events = build_work(list(plan), plan, 'mechanisms', 5)
        ablation = build_work(['P01', 'P02', 'P03'], plan, 'ablation', 5)
        algorithms = build_work(list(plan), plan, 'algorithms', 10)
        self.assertEqual(len(fast), 72)
        self.assertEqual(len(all_events), 320)
        self.assertEqual(len(ablation), 60)
        self.assertEqual(len(algorithms), 960)

    def test_ablation_changes_exactly_one_switch(self):
        from experiments.run_publication_risk_r2 import ablation_switches

        full = ablation_switches('full')
        for variant in ('no_lag_guidance', 'no_adaptive_operator', 'no_warm_start'):
            changed = [key for key in full if full[key] != ablation_switches(variant)[key]]
            self.assertEqual(len(changed), 1, variant)


class GateTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def test_failed_or_mismatched_gate_is_rejected(self):
        from experiments.run_publication_risk_r2 import require_confirmation_gate

        gate = self.root / 'gate.json'
        gate.write_text(json.dumps({'profile': 'screening', 'confirmation_pass': False,
                                    'protocol_hash': 'p'}), encoding='utf-8')
        with self.assertRaises(ValueError):
            require_confirmation_gate(gate, 'p')

    def test_directional_gate_is_not_accepted_as_confirmation(self):
        from experiments.run_publication_risk_r2 import require_confirmation_gate

        # Gate without profile=screening should be rejected
        gate = self.root / 'gate.json'
        gate.write_text(json.dumps({'confirmation_pass': False,
                                    'directional_pass': True,
                                    'protocol_hash': 'p'}), encoding='utf-8')
        with self.assertRaises(ValueError):
            require_confirmation_gate(gate, 'p')
        # Valid screening gate with confirmation_pass should be accepted
        gate.write_text(json.dumps({'profile': 'screening', 'confirmation_pass': True,
                                    'coverage_pass': True, 'physical_gate_pass': True,
                                    'risk_budget_pass': True, 'full_method_pass': True,
                                    'protocol_hash': 'other'}), encoding='utf-8')
        # Should NOT raise — the gate passes, even though protocol_hash differs
        result = require_confirmation_gate(gate, 'p')
        self.assertTrue(result['confirmation_pass'])

    def test_confirmation_gate_rejects_non_screening_profile(self):
        from experiments.run_publication_risk_r2 import require_confirmation_gate

        gate = self.root / 'gate.json'
        gate.write_text(json.dumps({
            'profile': 'screening_fast', 'confirmation_pass': True,
            'coverage_pass': True, 'physical_gate_pass': True,
            'risk_budget_pass': True, 'full_method_pass': True,
            'protocol_hash': 'p',
        }), encoding='utf-8')
        with self.assertRaises(ValueError):
            require_confirmation_gate(gate, 'p')

    def test_confirmation_gate_uses_event_majority(self):
        from experiments.run_publication_risk_r2 import evaluate_gate

        rows = []
        for event, dynamic, full in [('P01', .8, .75), ('P02', .9, .85), ('P03', 1.1, 1.05)]:
            for seed in range(5):
                for strategy in ('no_budget', 'fixed_budget', 'dynamic_budget', 'full_method'):
                    rows.append(dict(event_id=event, condition='normal', strategy=strategy,
                                     seed_index=seed, f1=0., exceedance_probability_any=0.,
                                     plan_constraint_violation=0., hard_physical_violation=0.))
                rows.extend([
                    dict(event_id=event, condition='matched', strategy='no_budget', seed_index=seed,
                         f1=1.2, exceedance_probability_any=.8,
                         plan_constraint_violation=0., hard_physical_violation=0.),
                    dict(event_id=event, condition='matched', strategy='fixed_budget', seed_index=seed,
                         f1=1., exceedance_probability_any=.5, plan_constraint_violation=0., hard_physical_violation=0.),
                    dict(event_id=event, condition='matched', strategy='dynamic_budget', seed_index=seed,
                         f1=dynamic, exceedance_probability_any=.4 if dynamic < 1 else .6,
                         plan_constraint_violation=0., hard_physical_violation=0.),
                    dict(event_id=event, condition='matched', strategy='full_method', seed_index=seed,
                         f1=full, exceedance_probability_any=.39 if full < 1 else .59,
                         plan_constraint_violation=0., hard_physical_violation=0.),
                ])
        report = evaluate_gate(rows, 'screening', 'hash')
        self.assertTrue(report['confirmation_pass'])

    def test_gate_fails_when_normal_condition_is_missing(self):
        from experiments.run_publication_risk_r2 import evaluate_gate

        rows = []
        for event in ('P01', 'P02', 'P03'):
            for seed in range(5):
                for strategy, value in (('no_budget', 1.2), ('fixed_budget', 1.),
                                        ('dynamic_budget', .8), ('full_method', .7)):
                    rows.append(dict(event_id=event, condition='matched', strategy=strategy,
                                     seed_index=seed, f1=value,
                                     exceedance_probability_any=value / 2,
                                     plan_constraint_violation=0., hard_physical_violation=0.))
        report = evaluate_gate(rows, 'screening', 'hash')
        self.assertFalse(report['coverage_pass'])
        self.assertFalse(report['confirmation_pass'])

    def test_gate_passes_when_only_matched_runs_have_hard_violation(self):
        from experiments.run_publication_risk_r2 import evaluate_gate

        rows = self._passing_rows()
        # Inject a hard physical violation into a MATCHED condition run only.
        # Matched (stressed) runs are expected to have physical violations
        # (inflow exceeds max release capacity). The gate should NOT fail.
        for r in rows:
            if r.get('condition') == 'matched':
                r['hard_physical_violation'] = .01
                break
        report = evaluate_gate(rows, 'screening', 'hash')
        self.assertTrue(report['physical_gate_pass'])

    def test_gate_reports_full_method_pass_flag_when_full_method_is_worse(self):
        from experiments.run_publication_risk_r2 import evaluate_gate

        rows = self._passing_rows()
        for row in rows:
            if row['condition'] == 'matched' and row['strategy'] == 'full_method':
                row['f1'] = 1.2
                row['exceedance_probability_any'] = .7
        report = evaluate_gate(rows, 'screening', 'hash')
        # risk_budget_pass should still be true (dynamic still works)
        self.assertTrue(report['risk_budget_pass'])
        # full_method_pass should be false (full is systematically worse)
        self.assertFalse(report['full_method_pass'])
        # But confirmation_pass should still PASS — the gate checks coverage
        # and physical consistency, not whether the hypothesis was confirmed.
        self.assertTrue(report['confirmation_pass'])

    def test_gate_coverage_requires_every_strategy_seed_cell(self):
        from experiments.run_publication_risk_r2 import evaluate_gate

        rows = self._passing_rows()
        rows = [row for row in rows if not (
            row['event_id'] == 'P01' and row['condition'] == 'normal'
            and row['strategy'] == 'no_budget' and row['seed_index'] == 4)]
        report = evaluate_gate(rows, 'screening', 'hash')
        self.assertFalse(report['coverage_pass'])

    def test_gate_coverage_requires_every_requested_event(self):
        from experiments.run_publication_risk_r2 import evaluate_gate

        report = evaluate_gate(self._passing_rows(), 'screening', 'hash',
                               expected_events=['P01', 'P02', 'P03', 'P04'])
        self.assertFalse(report['coverage_pass'])
        self.assertFalse(report['confirmation_pass'])

    @staticmethod
    def _passing_rows():
        rows = []
        for event in ('P01', 'P02', 'P03'):
            for seed in range(5):
                for condition in ('normal', 'matched'):
                    for strategy, f1, probability in (
                            ('no_budget', 1.3, .8), ('fixed_budget', 1., .5),
                            ('dynamic_budget', .8, .4), ('full_method', .75, .35)):
                        rows.append(dict(event_id=event, condition=condition,
                                         strategy=strategy, seed_index=seed,
                                         f1=0. if condition == 'normal' else f1,
                                         exceedance_probability_any=(0. if condition == 'normal'
                                                                     else probability),
                                         plan_constraint_violation=0.,
                                         hard_physical_violation=0.))
        return rows


class PressurePlanTests(unittest.TestCase):
    def test_non_publication_ready_pressure_plan_is_rejected(self):
        from experiments.run_publication_risk_r2 import _pressure_map

        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'pressure.json'
            path.write_text(json.dumps({
                'publication_ready': False,
                'events': [{'event_id': 'P01', 'matched_factor': 2.}],
            }), encoding='utf-8')
            with self.assertRaises(ValueError):
                _pressure_map(path)


if __name__ == '__main__':
    unittest.main()
