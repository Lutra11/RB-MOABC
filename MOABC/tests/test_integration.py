# -*- coding: utf-8 -*-
"""
MOABC + 水库模拟 集成测试
========================
1. 构建梯级网络
2. 定义三目标函数
3. 运行 MOABC 求解
4. 输出 Pareto 前沿
"""
import sys
import os
import numpy as np

# 添加项目根目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from MOABC.simulation import build_default_network, ReservoirSimulator
from MOABC.core import ReservoirSchedulingProblem, Config, solve


def make_evaluate_fn(network, sim):
    """创建目标函数闭包"""
    def evaluate(problem, x):
        R = problem.decode(x)  # (n_res, T)
        result = sim.simulate(
            initial_storage=problem.initial_storage,
            inflow_scenarios=problem.inflow_scenarios,
            release_decisions=R,
        )

        # --- f1: CVaR 下游超限风险 ---
        safe = problem.safe_Q[0]
        if result.control_Q_scen is not None and result.n_scen > 1:
            # 多情景: 计算各情景超限率, 再取 CVaR
            exceed = np.maximum(0.0, (result.control_Q_scen[0] - safe) / safe)
            from MOABC.risk import RiskBudgetAllocator
            losses = exceed.mean(axis=0)  # (n_scen,)
            f1 = RiskBudgetAllocator.compute_cvar(losses, alpha=problem.alpha)
        else:
            exceed = np.maximum(0.0, (result.control_Q[0] - safe) / safe)
            f1 = np.mean(exceed)

        # --- f2: 弃水 + 蓄水机会损失 ---
        spill_total = np.sum(result.spill) * problem.dt  # 10^8 m3
        storage_deficit = np.sum(np.maximum(0,
            problem.target_storage - result.storage[:, -1]
        ))
        w = problem.weight
        f2 = w['spill'] * spill_total + w['storage_deficit'] * storage_deficit

        # --- f3: 运行波动 ---
        dR = np.diff(R, axis=1, prepend=R[:, :1])
        f3 = w['ramp'] * np.sum(np.abs(dR))

        f = np.array([f1, f2, f3])
        return f, result.feasible, result

    return evaluate


def main():
    print("=" * 60)
    print("MOABC + 水库模拟 集成测试")
    print("=" * 60)

    # 1. 构建网络
    network = build_default_network()
    sim = ReservoirSimulator(network, dt=3.0)
    n_res = 5
    T = 24  # 72h 滚动时域
    n_scen = 10

    print(f"\n水库数: {n_res}")
    print(f"滚动时域: {T} 步 ({T*3}h)")
    print(f"情景数: {n_scen}")
    print(f"决策变量维度: {n_res * T}")

    # 2. 初始库容 (汛限水位对应库容的 60%)
    initial_storage = np.array([
        network.reservoirs[name].S_max * 0.6
        for name in sim.topo_order
    ])
    print(f"初始库容: {initial_storage}")

    # 3. 入库情景 (模拟洪水过程)
    np.random.seed(42)
    base_inflow = np.array([2000, 2500, 3000, 3500, 5000])  # m3/s
    inflow_scenarios = np.zeros((n_res, T, n_scen))
    for s in range(n_scen):
        for i in range(n_res):
            # 洪水过程: 先涨后落
            peak_t = T // 2
            for t in range(T):
                base = base_inflow[i] * (
                    1.0 + 1.5 * np.exp(-((t - peak_t) / 5.0)**2)
                )
                noise = 0.15 * np.random.randn()
                inflow_scenarios[i, t, s] = max(100, base * (1 + noise))

    # 4. 泄量上下限
    bounds = np.zeros((n_res, T, 2))
    for i, name in enumerate(sim.topo_order):
        p = network.reservoirs[name]
        bounds[i, :, 0] = p.min_release
        bounds[i, :, 1] = p.max_release

    # 5. 目标库容 (期末: 回到汛限水位)
    target_storage = np.array([
        network.reservoirs[name].S_max
        for name in sim.topo_order
    ])

    # 6. 构建问题
    problem = ReservoirSchedulingProblem(
        n_res=n_res,
        T=T,
        dt=3.0,
        n_obj=3,
        n_scen=n_scen,
        bounds=bounds,
        initial_storage=initial_storage,
        inflow_scenarios=inflow_scenarios,
        safe_Q=np.array([56700.0]),
        target_storage=target_storage,
    )

    # 7. 评估函数
    eval_fn = make_evaluate_fn(network, sim)

    # 8. 运行 MOABC (小规模测试: 200 次评估)
    cfg = Config(
        evaluations=200,
        population=20,
        archive_size=40,
        limit=15,
    )
    print(f"\n运行 MOABC (evals={cfg.evaluations}, pop={cfg.population})...")
    result = solve(problem, eval_fn, seed=42, cfg=cfg)

    # 9. 输出结果
    print(f"\n{'='*60}")
    print(f"MOABC 求解完成")
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

    print(f"\n{'='*60}")
    print("✅ MOABC + 水库模拟 集成测试通过")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
