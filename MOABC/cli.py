# -*- coding: utf-8 -*-
"""
MOABC — CLI 入口
================
用法:
    python -m MOABC.cli                    # 默认快速测试
    python -m MOABC.cli --evals 6000       # 正式运行
    python -m MOABC.cli --method NSGA-II  -- NSGA-II baseline
"""
import argparse
import sys
import os
from pathlib import Path
import numpy as np
import pandas as pd

# 确保项目根目录在 path 中
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from MOABC.simulation import build_default_network, ReservoirSimulator
from MOABC.core import ReservoirSchedulingProblem, Config, solve
from MOABC.data.observed_flow import build_inflow_data, ReservoirInflowData
from MOABC.risk import RiskBudgetAllocator, RiskBudgetConfig


GPM_WINDOW_BY_RESERVOIR = {
    "Wudongde": "Wudongde",
    "Baihetan": "Baihetan",
    "Xiluodu": "Xiluodu",
    "Xiangjiaba": "Xiangjiaba",
    "TGD": "TGD",
}


def load_gpm_inflow_data(path, reservoir_names, T, start_date=None):
    """Load long-format GPM-derived inflow and return one rolling slice."""
    frame = pd.read_csv(path, parse_dates=["date"])
    required = {"date", "window", "inflow_m3s"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"GPM inflow file is missing columns: {sorted(missing)}")
    frame = frame[frame["window"].isin(GPM_WINDOW_BY_RESERVOIR.values())].copy()
    pivot = frame.pivot_table(index="date", columns="window", values="inflow_m3s", aggfunc="mean")
    pivot = pivot.sort_index()
    windows = [GPM_WINDOW_BY_RESERVOIR[name] for name in reservoir_names]
    if any(w not in pivot.columns for w in windows):
        raise ValueError(f"GPM inflow file lacks all reservoir windows: {windows}")
    pivot = pivot[windows].dropna()
    if start_date is not None:
        start = pd.Timestamp(start_date)
        eligible = pivot.index[pivot.index >= start]
        if len(eligible) == 0:
            raise ValueError(f"No GPM inflow dates available after {start_date}")
        pivot = pivot.loc[eligible[0]:]
    if len(pivot) < T:
        raise ValueError(f"Need {T} consecutive dates, but only {len(pivot)} are available")
    dates = pivot.index[:T]
    if not np.all(np.diff(dates.values).astype("timedelta64[D]") == np.timedelta64(1, "D")):
        raise ValueError("GPM inflow slice contains missing dates; do not fill silently")
    base = pivot.iloc[:T].to_numpy(dtype=float).T
    return ReservoirInflowData(
        dates=dates.to_numpy(),
        inflow=base,
        reservoir_names=list(reservoir_names),
        source="gpm_precipitation_to_inflow",
    )


def make_evaluate_fn(network, sim):
    """创建三目标评估函数"""
    def evaluate(problem, x):
        R = problem.decode(x)
        result = sim.simulate(
            initial_storage=problem.initial_storage,
            inflow_scenarios=problem.inflow_scenarios,
            release_decisions=R,
        )
        # f1: CVaR 下游超限风险
        safe = problem.safe_Q[0]
        # 多情景超限率
        if result.control_Q_scen is not None and result.n_scen > 1:
            exceed = np.maximum(0.0, (result.control_Q_scen[0] - safe) / safe)
            # CVaR_alpha: 超过 VaR_alpha 的尾部均值
            from MOABC.risk import RiskBudgetAllocator
            losses = exceed.mean(axis=0)  # (n_scen,) 每情景的时间均值
            f1 = RiskBudgetAllocator.compute_cvar(losses, alpha=problem.alpha)
        else:
            exceed = np.maximum(0.0, (result.control_Q[0] - safe) / safe)
            f1 = np.mean(exceed)
        # f2: 弃水 + 蓄水损失
        spill_total = np.sum(result.spill) * problem.dt
        storage_deficit = np.sum(np.maximum(0,
            problem.target_storage - result.storage[:, -1]))
        w = problem.weight
        f2 = w['spill'] * spill_total + w['storage_deficit'] * storage_deficit
        # f3: 运行波动
        dR = np.diff(R, axis=1, prepend=R[:, :1])
        f3 = w['ramp'] * np.sum(np.abs(dR))
        risk_violation = 0.0
        if problem.risk_budget is not None and problem.chance_threshold is not None:
            exceed_prob = np.mean(
                problem.inflow_scenarios > problem.chance_threshold[:, :, None],
                axis=2,
            )
            risk_violation = float(np.maximum(
                exceed_prob - problem.risk_budget, 0.0
            ).sum())
            f1 += risk_violation
        feasible = result.feasible and risk_violation <= 1e-10
        return np.array([f1, f2, f3]), feasible, result
    return evaluate


def build_problem(
    n_scen=10,
    T=24,
    input_mode="controlled",
    gpm_inflow_path=None,
    start_date=None,
    use_risk_budget=True,
):
    """构建默认调度问题"""
    network = build_default_network()
    sim = ReservoirSimulator(network, dt=3.0)
    n_res = 5

    initial_storage = np.array([
        network.reservoirs[name].S_max * 0.6
        for name in sim.topo_order
    ])

    # Controlled case-study input. In the paper, the base series is replaced
    # by GPM-derived precipitation-to-inflow forcing; this fallback remains
    # available only for solver regression tests and examples.
    if input_mode == "gpm":
        if not gpm_inflow_path:
            raise ValueError("--gpm-inflow is required when --input-mode gpm")
        inflow_data = load_gpm_inflow_data(
            gpm_inflow_path, sim.topo_order, T, start_date=start_date
        )
    elif input_mode == "controlled":
        inflow_data = build_inflow_data(
            start_year=2007, end_year=2007,
            reservoir_names=sim.topo_order,
        )
        base = inflow_data.inflow[:, :T]
        inflow_data = type(inflow_data)(
            dates=inflow_data.dates[:T], inflow=base,
            reservoir_names=inflow_data.reservoir_names,
            source="controlled_example",
        )
    else:
        raise ValueError("input_mode must be 'controlled' or 'gpm'")
    inflow_scenarios = inflow_data.to_perturbation_scenarios(
        n_members=n_scen, spread=0.15, temporal_rho=0.7,
        common_weight=0.5, rng=np.random.default_rng(42),
    )

    bounds = np.zeros((n_res, T, 2))
    for i, name in enumerate(sim.topo_order):
        p = network.reservoirs[name]
        bounds[i, :, 0] = p.min_release
        bounds[i, :, 1] = p.max_release

    target_storage = np.array([
        network.reservoirs[name].S_max
        for name in sim.topo_order
    ])

    risk_budget = None
    chance_threshold = None
    if use_risk_budget:
        forecast_mean = inflow_scenarios.mean(axis=2)
        forecast_std = inflow_scenarios.std(axis=2, ddof=1)
        allocator = RiskBudgetAllocator(RiskBudgetConfig(scope="joint"))
        risk_budget = allocator.allocate(forecast_mean, forecast_std)
        risk_safe_q = np.array([
            network.reservoirs[name].max_release for name in sim.topo_order
        ], dtype=float)
        chance_threshold = allocator.chance_constraint_q(
            forecast_mean, forecast_std, risk_safe_q, risk_budget
        )

    problem = ReservoirSchedulingProblem(
        n_res=n_res, T=T, dt=3.0, n_obj=3, n_scen=n_scen,
        bounds=bounds, initial_storage=initial_storage,
        inflow_scenarios=inflow_scenarios,
        safe_Q=np.array([56700.0]),
        target_storage=target_storage,
        risk_budget=risk_budget,
        chance_threshold=chance_threshold,
    )
    return problem, network, sim


def main():
    parser = argparse.ArgumentParser(description='MOABC 水库调度求解器')
    parser.add_argument('--evals', type=int, default=200, help='函数评估次数')
    parser.add_argument('--pop', type=int, default=20, help='种群规模')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--T', type=int, default=24, help='滚动时域步数')
    parser.add_argument('--scen', type=int, default=10, help='情景数')
    parser.add_argument('--method', type=str, default='MOABC', help='算法')
    parser.add_argument('--input-mode', choices=['controlled', 'gpm'], default='controlled')
    parser.add_argument('--gpm-inflow', type=str, default=None, help='GPM-derived long-format inflow CSV')
    parser.add_argument('--start-date', type=str, default=None, help='first date of rolling slice')
    parser.add_argument('--no-risk-budget', action='store_true', help='disable dynamic chance-constraint penalty')
    args = parser.parse_args()

    print("=" * 60)
    print("MOABC 水库群滚动调度求解器")
    print(f"输入模式: {args.input_mode}")
    print("=" * 60)

    problem, network, sim = build_problem(
        n_scen=args.scen,
        T=args.T,
        input_mode=args.input_mode,
        gpm_inflow_path=args.gpm_inflow,
        start_date=args.start_date,
        use_risk_budget=not args.no_risk_budget,
    )
    eval_fn = make_evaluate_fn(network, sim)

    cfg = Config(
        evaluations=args.evals,
        population=args.pop,
        archive_size=max(40, args.pop * 2),
        limit=15,
    )

    print(f"\n水库数: {problem.n_res}")
    print(f"滚动时域: {problem.T} 步 ({problem.T*3}h)")
    print(f"情景数: {problem.n_scen}")
    print(f"决策变量维度: {problem.dim}")
    print(f"算法: {args.method}")
    print(f"评估预算: {cfg.evaluations}")
    print(f"\n运行中...")

    result = solve(problem, eval_fn, seed=args.seed, cfg=cfg)

    print(f"\n{'='*60}")
    print(f"求解完成")
    print(f"{'='*60}")
    print(f"评估次数: {result['evaluations']}")
    print(f"运行时间: {result['seconds']:.2f}s")
    print(f"Pareto 解数: {len(result['front'])}")
    print(f"世代数: {result['diagnostics']['generations']}")
    print(f"侦察蜂次数: {result['diagnostics']['scouts']}")

    if result['front']:
        print(f"\nPareto 前沿 (f1=超限, f2=弃水+损失, f3=波动):")
        for i, f in enumerate(result['front'][:10]):
            print(f"  解{i+1}: f1={f[0]:.6f}, f2={f[1]:.4f}, f3={f[2]:.4f}")

    print(f"\n✅ 完成")


if __name__ == '__main__':
    main()
