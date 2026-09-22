# -*- coding: utf-8 -*-
"""六项修复的可重复专项回归测试。"""
import numpy as np
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from MOABC.forecast import ForecastCalibrator
from MOABC.risk import RiskBudgetAllocator, RiskBudgetConfig
from MOABC.simulation import build_default_network, ReservoirSimulator
from MOABC.data.observed_flow import load_or_build_yichang_data


def main():
    rng = np.random.default_rng(7)

    # 1. EMOS 拟合 + ECC 逐时刻保秩
    raw = rng.normal(100, 10, (30, 4, 8))
    obs = raw.mean(axis=2) + rng.normal(2, 5, (30, 4))
    cal = ForecastCalibrator(rng)
    cal.fit(raw, obs)
    pred = cal.predict(raw[0])
    assert cal.fitted and len(cal.emos.params) == 4
    for t in range(4):
        assert np.array_equal(np.argsort(raw[0, t]), np.argsort(pred["ens"][t]))

    # 2. joint 风险预算严格守住系统总额；收紧系数生效
    mu = np.ones((5, 12)) * 100
    sigma = np.ones((5, 12)) * 10
    eps = RiskBudgetAllocator().allocate(mu, sigma)
    assert np.isclose(eps.sum(), 0.05)
    eps_half = RiskBudgetAllocator(
        RiskBudgetConfig(boole_correction=0.5)
    ).allocate(mu, sigma)
    assert np.isclose(eps_half.sum(), 0.025)

    # 3/4. 情景独立路由和溢流必须能传到控制断面
    network = build_default_network()
    sim = ReservoirSimulator(network, dt=3.0)
    T, n_scen = 6, 3
    initial = np.array([
        network.reservoirs[name].S_max * 0.6 for name in sim.topo_order
    ])
    initial[-1] = network.reservoirs["TGD"].S_max
    release = np.array([
        [network.reservoirs[name].min_release] * T
        for name in sim.topo_order
    ], dtype=float)
    inflow = np.full((5, T, n_scen), 3000.0)
    inflow[-1, :, 1] = 100000.0
    result = sim.simulate(initial, inflow, release)
    assert result.control_Q_scen.shape == (1, T, n_scen)
    assert result.storage_scen.shape == (5, T, n_scen)
    assert result.release_scen.shape == (5, T, n_scen)
    assert result.control_Q_scen[0].std() > 0.0

    # 5. 经验 CVaR 的尾部值应不小于 VaR
    losses = np.array([0.0, 1.0, 2.0, 10.0])
    assert RiskBudgetAllocator.compute_cvar(losses, 0.75) >= np.percentile(losses, 75)

    # 6. 论文流程禁止无意中回退到合成观测
    try:
        load_or_build_yichang_data(require_observed=True)
    except FileNotFoundError:
        pass
    else:
        raise AssertionError("synthetic data was allowed in observed-only mode")

    print("repair regression: PASS")


if __name__ == "__main__":
    main()
