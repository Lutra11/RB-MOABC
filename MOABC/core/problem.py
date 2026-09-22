# -*- coding: utf-8 -*-
"""
水库调度问题定义
================
将 EAMS-ABC 的 FJSP Problem 替换为水库调度问题.

EAMS-ABC Problem:
  - 决策: operation_sequence + machine_assignment (整数)
  - 解码: earliest feasible insertion
  - 目标: makespan, energy

水库调度 Problem:
  - 决策: release_schedule R[n_res, T] (连续, m3/s)
  - 解码: ReservoirSimulator.simulate()
  - 目标: f1=CVaR(下游超限), f2=弃水+蓄水损失, f3=运行波动
"""
from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np
from typing import Optional


@dataclass
class ReservoirSchedulingProblem:
    """
    水库群滚动调度问题

    属性:
        n_res: 水库数
        T: 滚动时域步数
        dt: 时间步长 (h)
        n_obj: 目标数 (默认 3)
        n_scen: 情景数 (集合预报成员数)
        bounds: (n_res, T, 2) 每步泄量的 [min, max]
        initial_storage: (n_res,) 初始库容
        inflow_scenarios: (n_res, T, n_scen) 入库流量情景
        safe_Q: (n_ctrl,) 控制断面安全流量
        target_storage: (n_res,) 期末目标库容
        weight: 目标权重
    """
    n_res: int
    T: int
    dt: float = 3.0
    n_obj: int = 3
    n_scen: int = 10
    bounds: np.ndarray = None        # (n_res, T, 2)
    initial_storage: np.ndarray = None  # (n_res,)
    inflow_scenarios: np.ndarray = None  # (n_res, T, n_scen)
    safe_Q: np.ndarray = None         # (n_ctrl,)
    target_storage: np.ndarray = None  # (n_res,)
    risk_budget: np.ndarray = None     # (n_res, T) dynamic epsilon_{k,t}
    chance_threshold: np.ndarray = None  # (n_res, T) chance threshold
    release_reference: np.ndarray = None # (n_res,T) routed/cumulative initialization guide
    risk_mode: str = "dynamic"
    forcing_type: str = "historical_forcing"
    stress_factor: float = 1.0
    event_id: str = ""
    weight: dict = field(default_factory=lambda: {
        'spill': 1.0, 'storage_deficit': 1.0,
        'ramp': 0.5, 'eco': 0.3,
    })
    alpha: float = 0.95  # CVaR 置信水平
    name: str = 'Jinsha_5res'

    @property
    def dim(self) -> int:
        """决策变量维度"""
        return self.n_res * self.T

    def encode(self, R: np.ndarray) -> np.ndarray:
        """将 (n_res, T) 泄量矩阵展平为 1D 向量"""
        return R.flatten()

    def decode(self, x: np.ndarray) -> np.ndarray:
        """将 1D 向量还原为 (n_res, T) 泄量矩阵"""
        return x.reshape(self.n_res, self.T)

    def clip(self, x: np.ndarray) -> np.ndarray:
        """将决策变量裁剪到可行域"""
        R = self.decode(x)
        R = np.clip(R, self.bounds[:, :, 0], self.bounds[:, :, 1])
        return self.encode(R)

    def random_solution(self, rng: np.random.Generator) -> np.ndarray:
        """生成随机可行解"""
        R = rng.uniform(
            self.bounds[:, :, 0], self.bounds[:, :, 1]
        )
        return self.encode(R)
