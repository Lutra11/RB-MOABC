# -*- coding: utf-8 -*-
"""
动态风险预算分配模块
====================
将系统级失效概率 ε_sys 分解为各水库 k、各时步 t 的动态预算 ε_k,t。

核心公式:
    ε_k,t = ε_sys * w_k,t / Σ_k Σ_t w_k,t   (joint 模式，默认)

其中权重 w_k,t 由预报不确定性 (μ, σ) 和事件连续性 g 驱动:
    w_k,t = (1 + α_u * |μ_k,t - μ_ref| / σ_ref) * (1 + α_s * σ_k,t / σ_ref)
            * (1 + α_g * g_k,t)

当预报均值偏离参考值越大、预报离散度越大、事件连续性越强时,
风险优先级越高，因此分配到的 ε 越小（更严格的允许超限概率）。

机会约束:
    P[Q_k,t ≤ Q_safe_k] ≥ 1 - ε_k,t

跨库协调 (Boole 上界):
    joint 模式下 Σ_k Σ_t ε_k,t ≤ ε_sys；per_time 模式下每个 t 的
    Σ_k ε_k,t ≤ ε_sys。

参考:
    Rockafellar & Uryasev (2002). Conditional value-at-risk for general
    loss distributions. J. Banking Finance, 26(7), 1443-1471.
    Chen et al. (2020). A multi-objective risk management model for
    real-time flood control optimal operation of a parallel reservoir
    system. J. Hydrology, 586, 124897.
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Literal
from scipy.stats import norm


@dataclass
class RiskBudgetConfig:
    """风险预算分配配置"""
    epsilon_sys: float = 0.05       # 系统级失效概率 (5%)
    alpha_u: float = 1.0            # 均值偏离权重系数
    alpha_s: float = 1.0            # 离散度权重系数
    alpha_g: float = 0.5            # 事件连续性权重系数
    alpha_c: float = 1.0            # 下游后果/重要性权重系数
    epsilon_min: float = 1e-4       # 单库单时步最小预算
    epsilon_max: float = 0.15       # 单库单时步最大预算 (不超过15%)
    use_boole: bool = True          # 启用 Boole 风险预算解释
    scope: Literal["joint", "per_time"] = "joint"
    # joint: Σ_(k,t) ε_k,t ≤ ε_sys；per_time: 每个 t 的库间和 ≤ ε_sys
    boole_correction: float = 1.0   # <1 时收紧系统预算


class RiskBudgetAllocator:
    """
    动态风险预算分配器

    用法:
        allocator = RiskBudgetAllocator(config)
        eps_kt = allocator.allocate(
            forecast_mean=mu,       # (n_res, T)
            forecast_std=sigma,     # (n_res, T)
            continuity=g,           # (n_res, T) 事件连续性指标
            mu_ref=mu_ref,          # (n_res,) 参考均值
            sigma_ref=sigma_ref,    # (n_res,) 参考离散度
        )
    """

    def __init__(self, config: RiskBudgetConfig = None):
        self.config = config or RiskBudgetConfig()

    def allocate(
        self,
        forecast_mean: np.ndarray,   # (n_res, T) 校准后预报均值
        forecast_std: np.ndarray,    # (n_res, T) 校准后预报标准差
        continuity: np.ndarray = None,  # (n_res, T) 事件连续性 [0, 1]
        mu_ref: np.ndarray = None,    # (n_res,) 各库参考均值
        sigma_ref: np.ndarray = None, # (n_res,) 各库参考离散度
        consequence: np.ndarray = None, # (n_res,) or (n_res,T), larger=stricter
    ) -> np.ndarray:
        """
        分配动态风险预算

        返回: (n_res, T) 各库各时步的 ε_k,t
        """
        cfg = self.config
        n_res, T = forecast_mean.shape

        if mu_ref is None:
            mu_ref = np.mean(forecast_mean, axis=1)
        if sigma_ref is None:
            sigma_ref = np.mean(forecast_std, axis=1) + 1e-10
        if continuity is None:
            continuity = np.ones((n_res, T))
        if consequence is None:
            consequence = np.ones((n_res, T))
        consequence = np.asarray(consequence, dtype=float)
        if consequence.ndim == 1:
            consequence = np.repeat(consequence[:, None], T, axis=1)
        if consequence.shape != (n_res, T) or np.any(consequence <= 0):
            raise ValueError("consequence must be positive with shape (n_res,) or (n_res,T)")

        mu_ref = np.maximum(np.abs(mu_ref), 1e-10)
        sigma_ref = np.maximum(sigma_ref, 1e-10)

        if cfg.scope not in ("joint", "per_time"):
            raise ValueError("scope must be 'joint' or 'per_time'")
        if not (0.0 < cfg.boole_correction <= 1.0):
            raise ValueError("boole_correction must be in (0, 1]")

        # 计算风险优先级。风险越高，允许的 ε 越小（约束越严格）。
        # 这里用相对预报偏离、离散度和连续性构造 priority，随后取其倒数分配预算。
        w = np.zeros((n_res, T))
        for k in range(n_res):
            for t in range(T):
                # 均值偏离因子
                u_factor = 1.0 + cfg.alpha_u * abs(
                    forecast_mean[k, t] - mu_ref[k]
                ) / mu_ref[k]
                # 离散度因子
                s_factor = 1.0 + cfg.alpha_s * forecast_std[k, t] / sigma_ref[k]
                # 事件连续性因子
                g_factor = 1.0 + cfg.alpha_g * continuity[k, t]
                c_factor = 1.0 + cfg.alpha_c * (consequence[k, t] / consequence.mean() - 1.0)
                c_factor = max(c_factor, 1e-6)

                priority = u_factor * s_factor * g_factor * c_factor
                w[k, t] = 1.0 / max(priority, 1e-12)

        total_budget = cfg.epsilon_sys * cfg.boole_correction
        if cfg.scope == "per_time":
            # 每个时刻单独使用一个系统预算。
            eps = np.zeros((n_res, T))
            for t in range(T):
                eps[:, t] = self._bounded_allocate(
                    w[:, t], total_budget, cfg.epsilon_min, cfg.epsilon_max
                )
        else:
            # 整个滚动时域共享一个系统预算。
            eps = self._bounded_allocate(
                w, total_budget, cfg.epsilon_min, cfg.epsilon_max
            ).reshape(n_res, T)

        return eps

    @staticmethod
    def _bounded_allocate(
        priority: np.ndarray, total: float, lower: float, upper: float
    ) -> np.ndarray:
        """按 priority 分配总额，并在上下限裁剪后保持总和不变。"""
        p = np.asarray(priority, dtype=float).ravel()
        if np.any(~np.isfinite(p)) or np.all(p <= 0):
            p = np.ones_like(p)
        if total < len(p) * lower - 1e-12 or total > len(p) * upper + 1e-12:
            raise ValueError("total budget is incompatible with epsilon bounds")

        free = np.ones(len(p), dtype=bool)
        out = np.zeros(len(p), dtype=float)
        remaining = float(total)
        while np.any(free):
            scale = remaining / p[free].sum()
            proposed = p[free] * scale
            low = proposed < lower
            high = proposed > upper
            free_idx = np.flatnonzero(free)
            if not np.any(low | high):
                out[free_idx] = proposed
                break
            if np.any(low):
                idx = free_idx[low]
                out[idx] = lower
                remaining -= lower * len(idx)
                free[idx] = False
            if np.any(high):
                idx = free_idx[high & ~low]
                out[idx] = upper
                remaining -= upper * len(idx)
                free[idx] = False
        return out

    def chance_constraint_q(
        self,
        forecast_mean: np.ndarray,   # (n_res, T) μ
        forecast_std: np.ndarray,    # (n_res, T) σ
        safe_Q: np.ndarray,          # (n_res,) 或 (n_res, T) 安全流量
        eps_kt: np.ndarray,          # (n_res, T) 风险预算
    ) -> np.ndarray:
        """
        计算机会约束对应的流量阈值

        P[Q ≤ Q_thresh] ≥ 1 - ε  ⟺  Q_thresh = μ + Φ^{-1}(1-ε) * σ

        返回: (n_res, T) 流量阈值 Q_thresh
        """
        if safe_Q.ndim == 1:
            safe_Q = safe_Q[:, np.newaxis].repeat(
                forecast_mean.shape[1], axis=1
            )

        # Φ^{-1}(1 - ε_kt)
        z = norm.ppf(1.0 - eps_kt)
        Q_thresh = forecast_mean + z * forecast_std

        # 不超过安全流量
        Q_thresh = np.minimum(Q_thresh, safe_Q)

        return Q_thresh

    @staticmethod
    def compute_cvar(
        losses: np.ndarray,   # (n_scen,) 各情景的损失
        alpha: float = 0.95,   # 置信水平
    ) -> float:
        """
        计算 CVaR (Conditional Value at Risk)

        CVaR_α = E[Loss | Loss > VaR_α]
               = (1/(1-α)) * ∫_{VaR_α}^∞ x * f(x) dx

        参数:
            losses: (n_scen,) 各情景损失值
            alpha: 置信水平 (0.95 = 95%)

        返回: CVaR 值
        """
        losses = np.asarray(losses, dtype=float).ravel()
        losses = losses[np.isfinite(losses)]
        if len(losses) == 0:
            return 0.0
        if not 0.0 <= alpha < 1.0:
            raise ValueError("alpha must satisfy 0 <= alpha < 1")
        sorted_losses = np.sort(losses)
        tail_mass = (1.0 - alpha) * len(sorted_losses)
        if tail_mass <= 0:
            return float(sorted_losses[-1])
        full = int(np.floor(tail_mass))
        fraction = tail_mass - full
        total = float(sorted_losses[-full:].sum()) if full else 0.0
        if fraction > 1e-12 and full < len(sorted_losses):
            total += fraction * sorted_losses[-full - 1]
        return total / tail_mass

    @staticmethod
    def compute_continuity(
        forecast_std: np.ndarray,  # (n_res, T) 预报标准差
        window: int = 3,
    ) -> np.ndarray:
        """
        计算事件连续性指标

        连续性 = 预报高不确定性持续的天数占比
        当连续多个时步 σ > σ_threshold 时, 认为是连续事件

        返回: (n_res, T) 连续性指标 [0, 1]
        """
        n_res, T = forecast_std.shape
        cont = np.zeros((n_res, T))

        for k in range(n_res):
            sigma = forecast_std[k]
            threshold = np.mean(sigma) + 0.5 * np.std(sigma)
            high = (sigma > threshold).astype(float)

            # 滑动窗口: 连续高不确定性
            for t in range(T):
                t_start = max(0, t - window + 1)
                cont[k, t] = np.mean(high[t_start:t + 1])

        return cont
