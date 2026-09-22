# -*- coding: utf-8 -*-
"""
集合预报校准模块
================
EMOS (Ensemble Model Output Statistics) 偏差订正
ECC (Ensemble Copula Coupling) 保秩重排

EMOS: 对每个变量、每个提前期，用训练期数据拟合
    y ~ N(a + b * x_bar, c + d * s^2)
其中 x_bar 为集合均值, s 为集合标准差, a/b/c/d 为回归参数。

ECC: 将 EMOS 订正后的边缘分布采样, 按原始集合的秩结构重排,
保持成员间的时空相关性。

参考:
    Gneiting et al. (2005). Calibrated probabilistic forecasting using
    ensemble model output statistics and minimum CRPS estimation.
    Mon. Weather Rev., 133(5), 1098-1118.
    Schefzik et al. (2013). Uncertainty quantification in complex
    scattering networks. Statistical Science, 28(3), 375-396.
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Optional


# ===========================================================================
# 1. EMOS 偏差订正
# ===========================================================================

@dataclass
class EMOSParams:
    """单个变量/提前期的 EMOS 参数"""
    a: float = 0.0   # 截距
    b: float = 1.0   # 均值系数
    c: float = 1.0   # 方差截距 (>= 0)
    d: float = 1.0   # 方差扩散系数 (>= 0)
    fitted: bool = False


class EMOSCalibrator:
    """
    EMOS 偏差订正器

    用法:
        cal = EMOSCalibrator()
        cal.fit(ens_mean_train, ens_std_train, obs_train)  # 训练
        corrected_mean, corrected_std = cal.predict(ens_mean, ens_std)  # 预测
    """

    def __init__(self, max_iter: int = 500, tol: float = 1e-6):
        self.max_iter = max_iter
        self.tol = tol
        self.params: dict[str, EMOSParams] = {}  # key = "var_step"

    def _crps_cost(
        self, a: float, b: float, c: float, d: float,
        ens_mean: np.ndarray, ens_std: np.ndarray, obs: np.ndarray,
    ) -> float:
        """
        计算 CRPS 代价函数 (负的对数似然近似)

        CRPS = sigma * [z * (2*Phi(z) - 1) + 2*phi(z) - 1/sqrt(pi)]
        其中 z = (obs - mu) / sigma, mu = a + b*ens_mean, sigma = sqrt(c + d*ens_std^2)
        """
        mu = a + b * ens_mean
        var = np.maximum(c + d * ens_std ** 2, 1e-10)
        sigma = np.sqrt(var)
        z = (obs - mu) / sigma
        # 正态分布 CDF/PDF 近似
        from scipy.stats import norm
        Phi = norm.cdf(z)
        phi = norm.pdf(z)
        crps = sigma * (z * (2 * Phi - 1) + 2 * phi - 1.0 / np.sqrt(np.pi))
        return float(np.mean(crps))

    def fit(
        self,
        ens_mean: np.ndarray,   # (N,) 训练期集合均值
        ens_std: np.ndarray,    # (N,) 训练期集合标准差
        obs: np.ndarray,         # (N,) 训练期观测值
        var_step_key: str = "default",
    ) -> EMOSParams:
        """
        拟合 EMOS 参数 (最小化 CRPS)

        使用 Nelder-Mead 优化 (无需梯度)
        """
        from scipy.optimize import minimize

        def cost(theta):
            a, b, c, d = theta
            # 约束: c >= 0, d >= 0
            if c < 0 or d < 0:
                return 1e10
            return self._crps_cost(a, b, c, d, ens_mean, ens_std, obs)

        x0 = np.array([0.0, 1.0, 1.0, 1.0])
        res = minimize(cost, x0, method='Nelder-Mead',
                       options={'maxiter': self.max_iter, 'xatol': self.tol})
        a, b, c, d = res.x
        c = max(c, 0.0)
        d = max(d, 0.0)
        params = EMOSParams(a=a, b=b, c=c, d=d, fitted=True)
        self.params[var_step_key] = params
        return params

    def predict(
        self,
        ens_mean: np.ndarray,   # (S,) or scalar
        ens_std: np.ndarray,    # (S,) or scalar
        var_step_key: str = "default",
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        预测订正后的均值和标准差

        返回:
            mu: 订正后均值 = a + b * ens_mean
            sigma: 订正后标准差 = sqrt(c + d * ens_std^2)
        """
        p = self.params.get(var_step_key)
        if p is None or not p.fitted:
            # 未训练: 返回原始值
            return ens_mean, ens_std
        mu = p.a + p.b * np.asarray(ens_mean)
        sigma = np.sqrt(np.maximum(p.c + p.d * np.asarray(ens_std) ** 2, 1e-10))
        return mu, sigma

    def fit_all_steps(
        self,
        ens_data: np.ndarray,   # (N, T, S) 训练期集合预报
        obs_data: np.ndarray,   # (N, T) 训练期观测
        var_name: str = "default",
    ) -> dict[str, EMOSParams]:
        """
        对所有提前期分别拟合 EMOS 参数

        ens_data: (N_samples, T_steps, S_members)
        obs_data: (N_samples, T_steps)
        """
        N, T, S = ens_data.shape
        for t in range(T):
            ens_mean_t = ens_data[:, t, :].mean(axis=1)
            ens_std_t = ens_data[:, t, :].std(axis=1, ddof=1)
            obs_t = obs_data[:, t]
            key = f"{var_name}_step{t}"
            self.fit(ens_mean_t, ens_std_t, obs_t, key)
        return self.params


# ===========================================================================
# 2. ECC 保秩重排
# ===========================================================================

class ECCReorderer:
    """
    ECC (Ensemble Copula Coupling) 保秩重排

    将 EMOS 订正后从边缘分布采样的成员, 按原始集合预报的秩结构重排,
    保持成员间的时空相关性。

    步骤:
    1. 从 EMOS 订正后的正态分布 N(mu, sigma^2) 采样 S 个成员
    2. 计算原始集合预报各成员的秩
    3. 将采样成员排序后按原始秩重排

    参考: Schefzik et al. (2013)
    """

    def __init__(self, rng: Optional[np.random.Generator] = None):
        self.rng = rng or np.random.default_rng()

    def reorder(
        self,
        raw_ens: np.ndarray,      # (S,) or (T, S) 原始集合预报
        corrected_samples: np.ndarray,  # (S,) or (T, S) EMOS订正后采样
    ) -> np.ndarray:
        """
        保秩重排

        参数:
            raw_ens: 原始集合预报 (用于提取秩结构)
            corrected_samples: EMOS订正后的随机采样

        返回:
            reordered: (S,) or (T, S) 重排后的成员
        """
        if raw_ens.ndim == 1:
            return self._reorder_1d(raw_ens, corrected_samples)

        # (T, S) → 逐时步重排
        T, S = raw_ens.shape
        result = np.zeros_like(corrected_samples)
        for t in range(T):
            result[t] = self._reorder_1d(raw_ens[t], corrected_samples[t])
        return result

    def _reorder_1d(self, raw: np.ndarray, samples: np.ndarray) -> np.ndarray:
        """单时步保秩重排"""
        S = len(raw)
        # 原始集合的秩
        ranks = np.argsort(np.argsort(raw))
        # 采样排序
        sorted_samples = np.sort(samples)
        # 按原始秩重排
        return sorted_samples[ranks]


# ===========================================================================
# 3. 统一校准管道
# ===========================================================================

class ForecastCalibrator:
    """
    集合预报校准管道: EMOS + ECC

    用法:
        cal = ForecastCalibrator()
        # 训练
        cal.fit(train_ens, train_obs)
        # 预测: 返回校准后的集合预报
        calibrated = cal.predict(raw_ens)
    """

    def __init__(self, rng: Optional[np.random.Generator] = None):
        self.emos = EMOSCalibrator()
        self.ecc = ECCReorderer(rng)
        self.fitted = False

    def fit(
        self,
        train_ens: np.ndarray,   # (N, T, S) 训练期集合预报
        train_obs: np.ndarray,   # (N, T) 训练期观测
        var_name: str = "tp",
    ) -> None:
        """训练 EMOS 参数"""
        self.emos.fit_all_steps(train_ens, train_obs, var_name)
        self.fitted = True

    def predict(
        self,
        raw_ens: np.ndarray,    # (T, S) 原始集合预报
        var_name: str = "tp",
    ) -> dict:
        """
        校准预测

        返回:
            dict with:
            - 'ens': (T, S) 校准后集合预报 (ECC重排)
            - 'mean': (T,) 校准后均值
            - 'std': (T,) 校准后标准差
            - 'raw_mean': (T,) 原始均值
            - 'raw_std': (T,) 原始标准差
        """
        T, S = raw_ens.shape
        raw_mean = raw_ens.mean(axis=1)
        raw_std = raw_ens.std(axis=1, ddof=1)

        corrected_mean = np.zeros(T)
        corrected_std = np.zeros(T)
        corrected_samples = np.zeros((T, S))

        for t in range(T):
            key = f"{var_name}_step{t}"
            mu, sigma = self.emos.predict(raw_mean[t], raw_std[t], key)
            corrected_mean[t] = mu
            corrected_std[t] = sigma
            # 从订正后分布采样 S 个成员
            corrected_samples[t] = self.rng_from_emos(mu, sigma, S)

        # ECC 保秩重排
        reordered = self.ecc.reorder(raw_ens, corrected_samples)

        return {
            'ens': reordered,
            'mean': corrected_mean,
            'std': corrected_std,
            'raw_mean': raw_mean,
            'raw_std': raw_std,
        }

    def rng_from_emos(self, mu: float, sigma: float, n: int) -> np.ndarray:
        """从 N(mu, sigma^2) 采样 n 个成员"""
        return self.ecc.rng.normal(mu, sigma, n)


# ===========================================================================
# 4. 验证指标
# ===========================================================================

def crps_normal(mu: np.ndarray, sigma: np.ndarray, obs: np.ndarray) -> np.ndarray:
    """
    计算正态分布预测的 CRPS

    CRPS = sigma * [z*(2*Phi(z)-1) + 2*phi(z) - 1/sqrt(pi)]
    z = (obs - mu) / sigma
    """
    from scipy.stats import norm
    z = (obs - mu) / np.maximum(sigma, 1e-10)
    Phi = norm.cdf(z)
    phi = norm.pdf(z)
    return sigma * (z * (2 * Phi - 1) + 2 * phi - 1.0 / np.sqrt(np.pi))


def rmse(pred: np.ndarray, obs: np.ndarray) -> float:
    """均方根误差"""
    return float(np.sqrt(np.mean((pred - obs) ** 2)))


def bias(pred: np.ndarray, obs: np.ndarray) -> float:
    """偏差 (均值差)"""
    return float(np.mean(pred - obs))


def prediction_interval_coverage(
    mu: np.ndarray, sigma: np.ndarray, obs: np.ndarray, level: float = 0.95
) -> float:
    """
    预测区间覆盖率 (PI coverage)

    level=0.95 → 95% 预测区间
    """
    from scipy.stats import norm
    z = norm.ppf(1 - (1 - level) / 2)
    lower = mu - z * sigma
    upper = mu + z * sigma
    inside = (obs >= lower) & (obs <= upper)
    return float(np.mean(inside))


def rank_histogram(ens: np.ndarray, obs: np.ndarray) -> np.ndarray:
    """
    秩直方图 (Rank Histogram)

    ens: (T, S) 集合预报
    obs: (T,) 观测值

    返回: (S+1,) 各秩的频次
    """
    T, S = ens.shape
    ranks = np.zeros(T, dtype=int)
    for t in range(T):
        # 观测值在集合中的排名
        ranks[t] = np.searchsorted(np.sort(ens[t]), obs[t])
    hist = np.bincount(ranks, minlength=S + 1)
    return hist
