# -*- coding: utf-8 -*-
"""
四算法对比测试: MOABC vs NSGA-II vs MOEA/D vs MOPSO

所有方法共享:
  - 相同的水库网络和参数
  - 相同的入库情景
  - 相同的约束投影修复
  - 相同的函数评估预算
  - 相同的随机种子

输出: 各算法的 Pareto 前沿、运行时间、HV/IGD 对比
"""
import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from MOABC.simulation import build_default_network, ReservoirSimulator
from MOABC.core import ReservoirSchedulingProblem, Config, run


def make_evaluate_fn(network, sim):
    """三目标评估函数"""
    def evaluate(problem, x):
        R = problem.decode(x)
        result = sim.simulate(
            initial_storage=problem.initial_storage,
            inflow_scenarios=problem.inflow_scenarios,
            release_decisions=R,
        )
        # f1: 下游超限风险
        safe = problem.safe_Q[0]
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
        return np.array([f1, f2, f3]), result.feasible, result
    return evaluate


def hypervolume_2d(F, ref):
    """2D 超体积 (简化, 取 f1-f2 投影)"""
    if len(F) == 0:
        return 0.0
    # 取前两维
    pts = F[:, :2]
    pts = pts[pts[:, 0] <= ref[0]]
    pts = pts[pts[:, 1] <= ref[1]]
    if len(pts) == 0:
        return 0.0
    pts = pts[np.argsort(pts[:, 0])]
    hv = 0.0
    prev_y = ref[1]
    for i in range(len(pts)):
        if i < len(pts) - 1:
            x_span = pts[i + 1, 0] - pts[i, 0]
        else:
            x_span = ref[0] - pts[i, 0]
        y_span = prev_y - pts[i, 1]
        hv += max(0, x_span) * max(0, y_span)
        prev_y = min(prev_y, pts[i, 1])
    return hv


def main():
    print("=" * 70)
    print("六算法对比测试: MOABC vs NSGA-II vs NSGA-III vs MOEA/D vs MOPSO vs SPEA2")
    print("=" * 70)

    # 构建问题
    network = build_default_network()
    sim = ReservoirSimulator(network, dt=3.0)
    n_res, T, n_scen = 5, 24, 10

    initial_storage = np.array([
        network.reservoirs[name].S_max * 0.6
        for name in sim.topo_order
    ])

    np.random.seed(42)
    base_inflow = np.array([2000, 2500, 3000, 3500, 5000])
    inflow_scenarios = np.zeros((n_res, T, n_scen))
    for s in range(n_scen):
        for i in range(n_res):
            peak_t = T // 2
            for t in range(T):
                base = base_inflow[i] * (1.0 + 1.5 * np.exp(-((t - peak_t) / 5.0) ** 2))
                noise = 0.15 * np.random.randn()
                inflow_scenarios[i, t, s] = max(100, base * (1 + noise))

    bounds = np.zeros((n_res, T, 2))
    for i, name in enumerate(sim.topo_order):
        p = network.reservoirs[name]
        bounds[i, :, 0] = p.min_release
        bounds[i, :, 1] = p.max_release

    target_storage = np.array([
        network.reservoirs[name].S_max
        for name in sim.topo_order
    ])

    problem = ReservoirSchedulingProblem(
        n_res=n_res, T=T, dt=3.0, n_obj=3, n_scen=n_scen,
        bounds=bounds, initial_storage=initial_storage,
        inflow_scenarios=inflow_scenarios,
        safe_Q=np.array([56700.0]),
        target_storage=target_storage,
    )
    eval_fn = make_evaluate_fn(network, sim)

    # 统一配置: 公平对比
    cfg = Config(
        evaluations=500,
        population=20,
        archive_size=40,
        limit=15,
    )

    methods = ['MOABC', 'NSGA-II', 'NSGA-III', 'MOEA/D', 'MOPSO', 'SPEA2']
    results = {}

    print(f"\n配置: evals={cfg.evaluations}, pop={cfg.population}\n")

    for method in methods:
        print(f"运行 {method}...")
        res = run(problem, method, eval_fn, seed=42, cfg=cfg)
        results[method] = res
        print(f"  → {len(res['front'])} Pareto 解, "
              f"{res['seconds']:.2f}s, "
              f"evals={res['evaluations']}")

    # HV 对比 (2D 投影: f1-f2)
    print(f"\n{'='*70}")
    print(f"{'算法':<12} {'Pareto解数':>10} {'时间(s)':>10} {'评估次数':>10} {'HV(f1,f2)':>12}")
    print(f"{'-'*70}")

    # 参考点: 取各目标最大值 + 小余量, 确保 HV > 0
    all_f = []
    for m in methods:
        all_f.extend(results[m]['front'])
    all_f = np.array(all_f) if all_f else np.zeros((1, 3))
    ref_point = all_f.max(axis=0) + 1e-6

    for method in methods:
        f_arr = np.array(results[method]['front']) if results[method]['front'] else np.zeros((0, 3))
        hv = hypervolume_2d(f_arr, ref_point)
        print(f"{method:<12} {len(f_arr):>10} "
              f"{results[method]['seconds']:>10.2f} "
              f"{results[method]['evaluations']:>10} "
              f"{hv:>12.4f}")

    print(f"\n{'='*70}")
    print("✅ 六算法对比测试完成")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
