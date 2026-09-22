# -*- coding: utf-8 -*-
"""
邻域搜索算子
============
EAMS-ABC 原版 (离散): swap/insert/machine_change/gap_insert
水库调度版 (连续): Gaussian/local_search/crossover

算子设计:
  0: Gaussian 扰动 — 大范围探索
  1: 局部搜索 — 小步精调
  2: 算术交叉 — 与档案精英混合
  3: 情景引导 — 朝低风险方向扰动
"""
import numpy as np


def move(
    problem,
    x: np.ndarray,
    rng: np.random.Generator,
    operator: int = 0,
    alpha: float = 0.5,
    guided: bool = True,
    risk_sensitivity: np.ndarray = None,
) -> np.ndarray:
    """
    邻域搜索: 生成新候选解

    参数:
        x: 当前解 (1D 向量)
        operator: 0=Gaussian, 1=local, 2=crossover, 3=scenario_guided
        alpha: 步长系数
        guided: 是否使用风险感知引导
        risk_sensitivity: (n_res, T) 各变量对风险的敏感度
    """
    R = problem.decode(x)
    n_res, T = R.shape

    if operator == 0:
        # Gaussian 扰动
        sigma = 0.1 * (problem.bounds[:, :, 1] - problem.bounds[:, :, 0])
        R_new = R + sigma * rng.standard_normal(R.shape)

    elif operator == 1:
        # 局部搜索: 随机选 1-3 个时步微调
        R_new = R.copy()
        n_changes = rng.integers(1, 4)
        for _ in range(n_changes):
            i = int(rng.integers(n_res))
            t = int(rng.integers(T))
            delta = 0.05 * (problem.bounds[i, t, 1] - problem.bounds[i, t, 0])
            R_new[i, t] += float(rng.normal(0, delta))

    elif operator == 2:
        # 算术交叉: x_new = alpha*x + (1-alpha)*random_peer
        R_peer = problem.decode(
            problem.random_solution(rng)
        )
        R_new = alpha * R + (1 - alpha) * R_peer

    else:
        # Risk-aware temporal-transfer operator.  It moves part of a planned
        # release from a high-risk interval to an earlier interval, creating
        # flood-storage space while suppressing coincident downstream peaks.
        # Unlike the former variance-scaled Gaussian mutation, this operator
        # has an explicit hydrological direction and approximately conserves
        # release volume within the horizon.
        if risk_sensitivity is not None:
            sensitivity = np.asarray(risk_sensitivity, dtype=float)
            probability = np.maximum(sensitivity.ravel(), 1e-12)
            probability /= probability.sum()
            flat = int(rng.choice(n_res * T, p=probability))
            i, t = divmod(flat, T)
            R_new = R.copy()
            span = problem.bounds[i, t, 1] - problem.bounds[i, t, 0]
            delta = float(rng.uniform(0.03, 0.12) * span)
            if t > 0:
                lead = max(0, t - int(rng.integers(1, min(4, t) + 1)))
                transferable = min(
                    delta,
                    R_new[i, t] - problem.bounds[i, t, 0],
                    problem.bounds[i, lead, 1] - R_new[i, lead],
                )
                R_new[i, t] -= max(transferable, 0.0)
                R_new[i, lead] += max(transferable, 0.0)
            else:
                R_new[i, t] -= min(delta, R_new[i, t] - problem.bounds[i, t, 0])
        else:
            sigma = 0.1 * (problem.bounds[:, :, 1] - problem.bounds[:, :, 0])
            R_new = R + sigma * rng.standard_normal(R.shape)

    # 裁剪到可行域
    R_new = np.clip(R_new, problem.bounds[:, :, 0], problem.bounds[:, :, 1])
    return problem.encode(R_new)


def reconstruct(
    problem,
    x: np.ndarray,
    rng: np.random.Generator,
    ratio: float = 0.2,
    alpha: float = 0.5,
) -> np.ndarray:
    """
    侦察蜂重建: 大步长扰动部分变量
    EAMS-ABC 原版: 选 critical 操作排列
    水库版: 选高风险时段大扰动
    """
    R = problem.decode(x)
    n_res, T = R.shape
    count = max(1, int(n_res * T * ratio))

    # 随机选位置扰动
    for _ in range(count):
        i = int(rng.integers(n_res))
        t = int(rng.integers(T))
        lo = problem.bounds[i, t, 0]
        hi = problem.bounds[i, t, 1]
        R[i, t] = rng.uniform(lo, hi)

    return problem.encode(R)


def crossover(
    problem,
    a: np.ndarray,
    b: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    SBX-like 交叉 (连续变量版)
    EAMS-ABC 原版: POX crossover (离散)
    """
    R_a = problem.decode(a)
    R_b = problem.decode(b)
    mask = rng.random(R_a.shape) < 0.5
    R_child = np.where(mask, R_a, R_b)

    # 小概率变异
    mutate = rng.random(R_a.shape) < 1.0 / R_a.size
    R_child[mutate] = rng.uniform(
        problem.bounds[:, :, 0][mutate],
        problem.bounds[:, :, 1][mutate],
    )

    return problem.encode(R_child)
