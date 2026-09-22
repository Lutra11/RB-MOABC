"""Regression tests for the corrected publication experiment chain.

The file is executable directly so it does not require pytest.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from MOABC.core.metrics import hypervolume_3d
from MOABC.data.gpm_pipeline import PrecipitationScenarioModel
from MOABC.risk import RiskBudgetAllocator, RiskBudgetConfig
from MOABC.simulation import MuskingumRouter, ReservoirSimulator, build_default_network


def test_storage_definitions():
    network = build_default_network()
    assert network.reservoirs["TGD"].S_flood == 171.5
    assert network.reservoirs["TGD"].S_max == 393.0
    assert network.reservoirs["Xiangjiaba"].S_flood < network.reservoirs["Xiangjiaba"].S_max


def test_muskingum_constant_flow_and_coefficients():
    for dt in (3.0, 24.0):
        for K, x in ((15.0, 0.2), (12.0, 0.25), (60.0, 0.15)):
            router = MuskingumRouter(K, x, dt)
            assert min(router.C0, router.C1, router.C2) >= 0
            assert abs(router.C0 + router.C1 + router.C2 - 1.0) < 1e-12
            routed = router.route(np.full(20, 1234.0), Q_out_init=1234.0)
            assert np.allclose(routed, 1234.0)


def test_scenario_reproducibility_and_nonnegativity():
    model = PrecipitationScenarioModel(
        windows=("a", "b"), rho=np.array([0.6, 0.4]),
        innovation_cov=np.array([[0.04, 0.015], [0.015, 0.03]]),
    )
    base = np.full((2, 12), 10.0)
    one = model.generate(base, 50, seed=7)
    two = model.generate(base, 50, seed=7)
    assert one.shape == (2, 12, 50)
    assert np.all(one >= 0) and np.allclose(one, two)
    assert not np.allclose(one[:, :, 0], one[:, :, 1])


def test_risk_budget_boole_and_consequence():
    mean = np.full((5, 8), 1000.0)
    spread = np.full((5, 8), 100.0)
    allocator = RiskBudgetAllocator(RiskBudgetConfig(scope="per_time"))
    eps = allocator.allocate(mean, spread, consequence=np.array([1, 1.2, 1.5, 2, 3]))
    assert np.allclose(eps.sum(axis=0), 0.05)
    assert np.all(eps[-1] < eps[0])


def test_nonanticipative_release_and_mass_balance():
    network = build_default_network()
    simulator = ReservoirSimulator(network, dt=24.0)
    initial = np.array([(network.reservoirs[n].S_flood + network.reservoirs[n].S_max) / 2
                        for n in simulator.topo_order])
    inflow = np.full((5, 5, 4), 3000.0)
    inflow[:, :, 1:] *= np.array([1.1, 1.2, 1.3])[None, None, :]
    plan = np.array([[network.reservoirs[n].min_release] * 5 for n in simulator.topo_order])
    result = simulator.simulate(initial, inflow, plan)
    # With available water, the scheduled/controlled release is common to all scenarios.
    assert np.allclose(result.controlled_release_scen, plan[:, :, None])
    previous = np.concatenate([initial[:, None, None].repeat(4, axis=2),
                               result.storage_scen[:, :-1, :]], axis=1)
    balance = previous + (result.inflow_scen - result.release_scen) * simulator.vol_factor
    assert np.allclose(balance, result.storage_scen, atol=1e-8)


def test_hypervolume_known_cube():
    assert abs(hypervolume_3d(np.array([[0.0, 0.0, 0.0]]), np.ones(3)) - 1.0) < 1e-12


if __name__ == "__main__":
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
