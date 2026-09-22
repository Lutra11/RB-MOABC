# -*- coding: utf-8 -*-
"""
初始化: 生成可行泄量方案
========================
EAMS-ABC 原版: 随机排列 + 机器分配
水库调度版: 均值泄量 + 扰动 + 情景引导
"""
import numpy as np


def initialize(
    problem,
    rng: np.random.Generator,
    index: int = 0,
) -> np.ndarray:
    """
    生成初始泄量方案

    策略 (按 index 切换):
      0: 均值泄量 (入库均值)
      1: 均值 + 高斯扰动
      2: 上游多泄/下游少泄 (梯度策略)
      3: 完全随机
    """
    reference = getattr(problem, "release_reference", None)
    R_mean = (np.asarray(reference, dtype=float).copy() if reference is not None
              else np.mean(problem.inflow_scenarios, axis=2))

    if index % 4 == 0:
        R = R_mean.copy()
    elif index % 4 == 1:
        R = R_mean + 0.1 * R_mean * rng.standard_normal(R_mean.shape)
    elif index % 4 == 2:
        # 梯度策略: 上游库多泄, 下游库少泄
        R = R_mean.copy()
        for i in range(problem.n_res - 1):
            R[i] *= 1.1
        R[-1] *= 0.9
    else:
        R = rng.uniform(
            problem.bounds[:, :, 0],
            problem.bounds[:, :, 1],
        )

    # 裁剪到可行域
    R = np.clip(R, problem.bounds[:, :, 0], problem.bounds[:, :, 1])
    return problem.encode(R)
