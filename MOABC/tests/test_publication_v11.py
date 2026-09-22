"""Regression checks for the isolated v11 protocol; no real experiments."""
import sys
import unittest
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from MOABC.experiments.legacy_v11 import (cvar, chance_envelope, allocate_budget,
    fit_variability, scenario_rain, simulate, metrics, event_steps, select_factor,
    seed_for, audit_calibration, evaluate_prefix, runoff, problem_for, event_cube)
from MOABC.simulation import build_default_network


class ProtocolTests(unittest.TestCase):
    def test_cvar_fractional_tail(self):
        self.assertAlmostEqual(cvar(np.array([0., 1., 2., 3.]), .625), 8 / 3)

    def test_control_chance_majorant(self):
        q = np.array([[80., 95., 100., 110.]])
        proxy = chance_envelope(q, 100., .1)
        self.assertTrue(np.all(proxy >= (q > 100)))
        np.testing.assert_allclose(proxy, [[0., .5, 1., 1.]])

    def test_budget_is_event_not_daily(self):
        remaining = .05
        spent = []
        for t in range(12):
            b = allocate_budget(np.full((12 - t, 20), 50000.), remaining, 'dynamic', 56700.)
            self.assertAlmostEqual(b.sum(), remaining)
            spent.append(b[0])
            remaining -= b[0]
        self.assertAlmostEqual(sum(spent), .05)

    def test_common_seeds_and_stress_draws(self):
        self.assertEqual(seed_for('P01', 0, 'optimizer'), seed_for('P01', 0, 'optimizer'))
        model = (np.full(5, .5), np.eye(5) * .1)
        rain = np.ones((5, 12))
        a = scenario_rain(rain, model, 15, 7)
        b = scenario_rain(rain * 3, model, 15, 7)
        np.testing.assert_allclose(b, a * 3)
        self.assertTrue(np.all(a >= 0))

    def test_event_duration_is_calendar_duration(self):
        self.assertEqual(event_steps({'start': '2023-07-27', 'end': '2023-08-12'}, 5), 22)

    def test_factor_selection_never_fakes_match(self):
        self.assertIsNone(select_factor([(1., 0.), (6., 0.)], .2, .8))
        self.assertEqual(select_factor([(1., 0.), (2., .4), (3., .9)], .2, .8), 2.)

    def test_mass_balance_and_actual_ramps(self):
        net = build_default_network()
        pars = list(net.reservoirs.values())
        initial = np.array([p.S_flood for p in pars])
        inflow = np.ones((5, 8, 4)) * 8000
        plan = np.ones((5, 8)) * 10000
        result = simulate(net, initial, inflow, plan)
        before = np.concatenate([np.repeat(initial[:, None, None], 4, axis=2), result.storage[:, :-1]], axis=1)
        np.testing.assert_allclose(result.storage - before, (result.inflow - result.release) * .000864, atol=1e-10)
        self.assertTrue(np.isfinite(metrics(net, result, 56700.)['f1']))

    def test_dry_event_reports_deficit(self):
        net = build_default_network()
        initial = np.array([p.S_min for p in net.reservoirs.values()])
        result = simulate(net, initial, np.zeros((5, 3, 2)), np.ones((5, 3)) * 10000)
        self.assertGreater(result.physical_violation, 0)
        self.assertTrue(np.all(result.storage >= initial[:, None, None] - 1e-10))

    def test_raw_risk_not_penalty(self):
        net = build_default_network()
        initial = np.array([p.S_flood for p in net.reservoirs.values()])
        result = simulate(net, initial, np.ones((5, 3, 2)), np.ones((5, 3)) * 500)
        self.assertEqual(metrics(net, result, 1e9)['f1'], 0.)

    def test_transfer_not_validated(self):
        params = {n: {'runoff_coeff': .2, 'recession': .9} for n in build_default_network().reservoirs}
        audit = audit_calibration(params, {})
        self.assertFalse(audit['publication_ready'])

    def test_integer_inputs_and_nonfinite_initial(self):
        net = build_default_network()
        initial = np.array([43,120,60,38,172])
        a = simulate(net, initial, np.ones((5,3,2),int)*1000, np.ones((5,3),int)*1000)
        b = simulate(net, initial.astype(float), np.ones((5,3,2))*1000, np.ones((5,3))*1000)
        np.testing.assert_allclose(a.storage,b.storage)
        with self.assertRaises(ValueError):
            simulate(net, initial*np.nan, np.ones((5,3,2)), np.ones((5,3)))
        generated = scenario_rain(np.ones((5,4),int),(np.ones(5)*.5,np.eye(5)*.1),5,7)
        self.assertEqual(generated.dtype, np.float64)

    def test_past_violation_not_future_penalty(self):
        net = build_default_network()
        initial = np.array([p.S_min for p in net.reservoirs.values()])
        cube = np.full((5,3,3), 10000.)
        cube[:,0] = 0
        prefix = np.zeros((5,1))
        _, feasible, detail = evaluate_prefix(net,initial,cube,prefix,np.ones((5,2))*1000,None)
        self.assertGreater(detail['result'].physical_violation, 0)
        self.assertLess(detail['violation'], detail['result'].physical_violation)

    def test_actual_first_release_ramp(self):
        net=build_default_network()
        pars=list(net.reservoirs.values())
        result=simulate(net,np.array([p.S_flood for p in pars]),np.full((5,2,3),30000.),np.full((5,2),100000.))
        for k,p in enumerate(pars):
            self.assertTrue(np.all(result.controlled[k,0] <= p.min_release+p.max_ramp+1e-9))

    def test_archive_protection_and_mechanism_isolation(self):
        from MOABC.core.operators import reconstruct
        net=build_default_network()
        cube=np.ones((5,3,4))*1000
        p=problem_for(net,cube,np.ones(3)*.01,'synthetic',search_guidance=False)
        self.assertIsNone(p.risk_budget)
        x=p.random_solution(np.random.default_rng(1))
        old=x.copy()
        reconstruct(p,x,np.random.default_rng(2))
        np.testing.assert_array_equal(x,old)
        p=problem_for(net,cube,np.ones(3)*.01,'synthetic',search_guidance=True)
        self.assertEqual(p.risk_budget.shape,(5,3))

    def test_synthetic_runner_all_solvers_and_rolling(self):
        from experiments.run_publication_v11 import run_one, Profile, ALGORITHMS
        from types import SimpleNamespace
        import pandas as pd
        import tempfile
        import contextlib
        import io
        profile=Profile(5,7,2,24,5,1,1)
        event=pd.Series(dict(event_id='SYNTHETIC_UNIT',event_type='independent'))
        cube=np.full((5,3,5),8000.)
        heldout=np.full((5,3,7),8500.)
        with tempfile.TemporaryDirectory(prefix='v11_test_') as folder, contextlib.redirect_stdout(io.StringIO()):
            args=SimpleNamespace(stage='mechanisms',profile='smoke',safe_q=56700.,chance_margin=.05)
            for strategy in ['no_budget','fixed_budget','dynamic_budget','full_method']:
                row=run_one(args,event,1.,'normal',strategy,'MOABC',0,profile,cube,heldout,'test',Path(folder))
                self.assertEqual(row['schedule'].shape,(5,3))
                self.assertEqual(row['evaluations'],72)
                self.assertAlmostEqual(row['epsilon_sum'],0 if strategy=='no_budget' else .05)
                self.assertEqual(row['n_random_scenarios'],6)
            args.stage='algorithms'
            for algorithm in ALGORITHMS:
                row=run_one(args,event,1.,'normal','full_method',algorithm,0,profile,cube,heldout,'test',Path(folder))
                self.assertEqual(row['evaluations'],24)
                self.assertEqual(row['schedule'].shape,(5,3))

    def test_loader_combines_separate_training_and_test_forcing(self):
        from experiments.run_publication_v11 import load_data
        import json
        import tempfile
        import pandas as pd
        names=list(build_default_network().reservoirs)
        with tempfile.TemporaryDirectory(prefix='v11_loader_') as folder:
            root=Path(folder)
            train=pd.DataFrame([(d,n,1.,1.,True) for d in pd.date_range('2007-01-01',periods=45) for n in names],
                               columns=['date','window','precip_mm','window_area_km2','approximate_window'])
            test=pd.DataFrame([(d,n,2.,1.) for d in pd.date_range('2020-01-01',periods=45) for n in names],
                              columns=['date','window','precip_mm','window_area_km2'])
            train.to_csv(root/'gpm_train_daily_precip.csv',index=False)
            test.to_csv(root/'gpm_daily_precipitation.csv',index=False)
            pd.DataFrame([dict(event_id='P01',start='2020-01-01',end='2020-01-02',event_type='independent',window='TGD')]).to_csv(root/'publication_events_v3.csv',index=False)
            params={n:dict(runoff_coeff=.2,recession=.9) for n in names}
            sources={str(i):dict(window=n,area_km2=1.,training_common_dates=10) for i,n in enumerate(names)}
            (root/'calibrated_params_v2.json').write_text(json.dumps(params),encoding='utf-8')
            (root/'calibrated_parameters.json').write_text(json.dumps(sources),encoding='utf-8')
            forcing,_,_,_,hashes=load_data(root)
            self.assertEqual(forcing.time.min(),pd.Timestamp('2007-01-01'))
            self.assertEqual(forcing.time.max(),pd.Timestamp('2020-02-14'))
            self.assertEqual(set(forcing.source_period),{'training','test'})
            self.assertIn('gpm_train_daily_precip.csv',hashes)

    def test_event_cube_can_scale_dataframe_backed_rainfall(self):
        import pandas as pd
        names=list(build_default_network().reservoirs)
        forcing=pd.DataFrame([(d,n,1.) for d in pd.date_range('2019-12-01',periods=50) for n in names],
                             columns=['time','window','precip_mm'])
        params={n:dict(runoff_coeff=.2,recession=.9) for n in names}
        model=(np.ones(5)*.5,np.eye(5)*.01)
        event=pd.Series(dict(event_id='P01',start='2020-01-01',end='2020-01-02'))
        cube=event_cube(forcing,params,model,event,5,2.,5,7)
        self.assertEqual(cube.shape,(5,7,5))
        self.assertTrue(np.isfinite(cube).all())


if __name__ == '__main__':
    unittest.main(verbosity=2)
