# -*- coding: utf-8 -*-
"""
实测流量数据模块
==================
提供控制断面和梯级水库入库流量的流量参考数据接口。

数据来源:
1. CDR2 (China Daily River Discharge Records) — Zenodo 20152832
   覆盖 1990-2024, 310 个站点的测站约束卫星扩展重建日流量
2. 长江水利委员会水文年鉴 — 宜昌站 1877-至今

当前状态:
- CDR² 文件可通过 load_cdr2_gauge() 读取
- CDR² 是测站约束的卫星扩展重建日流量, 不是连续原始实测序列
- 合成数据仅用于代码测试和流程验证, 不能进入论文结果

重要: 论文最终结果必须使用真实观测数据, 合成数据仅用于代码验证。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional
from pathlib import Path


# ===========================================================================
# 1. 宜昌站流量数据
# ===========================================================================

@dataclass
class YichangFlowData:
    """宜昌站流量数据"""
    dates: np.ndarray         # (N,) 日期序列
    discharge: np.ndarray     # (N,) 日平均流量 m3/s
    source: str = "synthetic"  # 数据来源标记
    note: str = ""

    @property
    def n_days(self) -> int:
        return len(self.dates)

    def to_flood_season(self) -> "YichangFlowData":
        """提取汛期 (5-10月) 数据"""
        months = pd.DatetimeIndex(self.dates).month
        mask = (months >= 5) & (months <= 10)
        return YichangFlowData(
            dates=self.dates[mask],
            discharge=self.discharge[mask],
            source=self.source,
            note=f"{self.note} (flood season only)",
        )

    def get_period(self, start: str, end: str) -> "YichangFlowData":
        """提取指定时间段"""
        dt = pd.DatetimeIndex(self.dates)
        mask = (dt >= start) & (dt <= end)
        return YichangFlowData(
            dates=self.dates[mask],
            discharge=self.discharge[mask],
            source=self.source,
            note=f"{self.note} ({start} to {end})",
        )


def load_or_build_yichang_data(
    start_year: int = 2007,
    end_year: int = 2024,
    real_data_path: Optional[str] = None,
    require_observed: bool = False,
) -> YichangFlowData:
    """
    加载宜昌站实测流量数据

    如果有真实数据文件则加载, 否则构建合成数据。
    require_observed=True 时禁止回退到合成数据，用于论文实验前的安全检查。

    参数:
        start_year: 起始年
        end_year: 结束年
        real_data_path: 真实数据文件路径 (CSV/NetCDF)

    返回: YichangFlowData
    """
    if real_data_path and Path(real_data_path).exists():
        return _load_real_yichang(real_data_path)
    if require_observed:
        raise FileNotFoundError(
            "Real Yichang discharge data is required but no valid real_data_path was provided."
        )
    else:
        return _build_synthetic_yichang(start_year, end_year)


def _load_real_yichang(path: str) -> YichangFlowData:
    """从通用 CSV 文件加载流量数据。

    支持项目内部的 ``date,discharge`` 格式以及 CDR² 的
    ``Date,W,Q`` 格式。CDR² 的来源标签明确标记为 reconstructed。
    """
    df = pd.read_csv(path)
    if {'Date', 'Q'}.issubset(df.columns):
        dates = pd.to_datetime(df['Date']).values
        discharge = df['Q'].to_numpy(dtype=float)
        source = "cdr2_reconstructed"
    elif {'date', 'discharge'}.issubset(df.columns):
        dates = pd.to_datetime(df['date']).values
        discharge = df['discharge'].to_numpy(dtype=float)
        source = "observed"
    else:
        raise ValueError(
            f"Unsupported discharge columns in {path}; expected "
            "Date,Q (CDR²) or date,discharge."
        )
    return YichangFlowData(
        dates=dates,
        discharge=discharge,
        source=source,
        note=f"Loaded from {path}",
    )


def load_cdr2_gauge(
    path: str,
    start_year: Optional[int] = None,
    end_year: Optional[int] = None,
    flood_season: bool = False,
) -> YichangFlowData:
    """加载一个 CDR² 重建测站序列。

    CDR² 每个站点的 CSV 记录是卫星扩展的有效观测日期，通常不是
    1990--2024 的连续逐日序列。因此这里不插值、不伪造缺测日，
    直接保留有效记录供事件匹配和外部验证使用。
    """
    data = _load_real_yichang(path)
    if data.source != "cdr2_reconstructed":
        raise ValueError(f"{path} is not a CDR² Date,Q file")
    if start_year is not None or end_year is not None:
        start = f"{start_year or 1900}-01-01"
        end = f"{end_year or 2100}-12-31"
        data = data.get_period(start, end)
    if flood_season:
        data = data.to_flood_season()
    return data


def _build_synthetic_yichang(start_year: int, end_year: int) -> YichangFlowData:
    """
    构建宜昌站合成流量数据

    基于历史统计特征:
    - 多年平均流量: 14300 m3/s
    - 汛期 (6-9月) 均值: ~25000 m3/s
    - 历史最大: 70800 m3/s (1870)
    - 设计洪水 (千年一遇): 56700 m3/s (宜昌安全流量)
    - 典型洪水过程: 涨水7天 + 洪峰3天 + 退水10天

    合成方法:
    - 基流: 正弦年周期
    - 洪水事件: 对数正态分布随机采样
    - 噪声: AR(1) 自回归
    """
    dates = pd.date_range(
        f"{start_year}-01-01", f"{end_year}-12-31", freq="D"
    ).values
    N = len(dates)
    months = np.array(pd.DatetimeIndex(dates).month)
    days = np.array(pd.DatetimeIndex(dates).dayofyear)

    # 年周期基流
    base_flow = 14300.0
    seasonal = 12000.0 * np.sin(2 * np.pi * (days - 152) / 365.0)
    base = base_flow + seasonal
    base = np.maximum(base, 3000.0)

    # AR(1) 噪声
    rng = np.random.default_rng(42)
    noise = np.zeros(N)
    rho = 0.7  # 自相关系数
    sigma = 0.15 * base
    for t in range(1, N):
        noise[t] = rho * noise[t-1] + np.sqrt(1 - rho**2) * sigma[t] * rng.standard_normal()

    discharge = np.maximum(base + noise, 500.0).astype(float)

    # 注入洪水事件 (每年汛期 1-3 次)
    years_arr = np.array(pd.DatetimeIndex(dates).year)
    for year in range(start_year, end_year + 1):
        year_mask = (years_arr == year)
        year_idx = np.where(year_mask)[0]
        if len(year_idx) == 0:
            continue

        # 汛期日期 (6-9月)
        flood_mask = (months[year_idx] >= 6) & (months[year_idx] <= 9)
        flood_idx = year_idx[flood_mask]
        if len(flood_idx) == 0:
            continue

        # 随机 1-3 次洪水事件
        n_events = rng.integers(1, 4)
        for _ in range(n_events):
            peak_day = rng.choice(flood_idx)
            peak_q = rng.uniform(35000, 55000)

            # 涨水7天 + 洪峰3天 + 退水10天
            for d in range(-7, 11):
                idx = peak_day + d
                if idx < 0 or idx >= N:
                    continue
                if d < 0:
                    # 涨水阶段
                    factor = np.exp(d / 3.0)
                    discharge[idx] = max(discharge[idx], base[idx] + (peak_q - base[idx]) * factor)
                elif d <= 3:
                    # 洪峰阶段
                    discharge[idx] = max(discharge[idx], peak_q * rng.uniform(0.9, 1.1))
                else:
                    # 退水阶段
                    factor = np.exp(-(d - 3) / 4.0)
                    discharge[idx] = max(discharge[idx], base[idx] + (peak_q - base[idx]) * factor)

    return YichangFlowData(
        dates=dates,
        discharge=discharge,
        source="synthetic",
        note="Synthetic data based on Yichang historical statistics "
             "(mean=14300, max=70800). Replace with CDR2 real data.",
    )


# ===========================================================================
# 2. 水库入库流量数据
# ===========================================================================

@dataclass
class ReservoirInflowData:
    """水库入库流量数据"""
    dates: np.ndarray              # (N,) 日期序列
    inflow: np.ndarray             # (n_res, N) 各水库日平均入库 m3/s
    reservoir_names: list          # 水库名称列表
    source: str = "synthetic"

    def get_flood_season(self) -> "ReservoirInflowData":
        """提取汛期数据"""
        months = pd.DatetimeIndex(self.dates).month
        mask = (months >= 5) & (months <= 10)
        return ReservoirInflowData(
            dates=self.dates[mask],
            inflow=self.inflow[:, mask],
            reservoir_names=self.reservoir_names,
            source=self.source,
        )

    def to_perturbation_scenarios(
        self, n_members: int = 51, spread: float = 0.15,
        temporal_rho: float = 0.7, common_weight: float = 0.5,
        rng: Optional[np.random.Generator] = None,
    ) -> np.ndarray:
        """将基准入流转换为 GPM 驱动的扰动情景格式。

        该函数只生成情景，不把结果称为气象集合预报。扰动使用
        对数乘性误差以保证非负，并包含时间相关项和跨库共同项。

        参数:
            n_members: 集合成员数
            spread: 相对误差尺度
            temporal_rho: 误差的一阶时间相关系数
            common_weight: 跨库共同误差比例

        返回: (n_res, N, n_members) 集合流量
        """
        if not 0 <= temporal_rho < 1:
            raise ValueError("temporal_rho must be in [0, 1)")
        if not 0 <= common_weight <= 1:
            raise ValueError("common_weight must be in [0, 1]")
        rng = rng or np.random.default_rng()
        n_res, N = self.inflow.shape
        ens = np.zeros((n_res, N, n_members))
        for s in range(n_members):
            common = np.zeros(N)
            idiosyncratic = np.zeros((n_res, N))
            for t in range(1, N):
                common[t] = (temporal_rho * common[t - 1]
                             + np.sqrt(1 - temporal_rho**2) * rng.standard_normal())
                idiosyncratic[:, t] = (temporal_rho * idiosyncratic[:, t - 1]
                                       + np.sqrt(1 - temporal_rho**2)
                                       * rng.standard_normal(n_res))
            z = common_weight * common[None, :] + np.sqrt(1 - common_weight**2) * idiosyncratic
            # Lognormal multiplier: positive and approximately spread-scaled.
            multiplier = np.exp(spread * z - 0.5 * spread**2)
            ens[:, :, s] = np.maximum(self.inflow * multiplier, 0.0)
        return ens

    def to_ensemble(self, n_members: int = 51, spread: float = 0.15,
                    rng: Optional[np.random.Generator] = None) -> np.ndarray:
        """兼容旧接口；新论文代码应使用 to_perturbation_scenarios。"""
        return self.to_perturbation_scenarios(
            n_members=n_members, spread=spread, rng=rng
        )


def build_inflow_data(
    start_year: int = 2007,
    end_year: int = 2024,
    reservoir_names: Optional[list] = None,
) -> ReservoirInflowData:
    """
    构建梯级水库入库流量数据

    基于各水库多年平均流量统计:
    - 乌东德: 3830 m3/s (控制面积 40.61×10^4 km²)
    - 白鹤滩: 4170 m3/s (控制面积 43.03×10^4 km²)
    - 溪洛渡: 4620 m3/s (控制面积 45.44×10^4 km²)
    - 向家坝: 4620 m3/s (控制面积 45.88×10^4 km²)
    - 三峡: 14300 m3/s (控制面积 100×10^4 km²)
    """
    if reservoir_names is None:
        reservoir_names = ['Wudongde', 'Baihetan', 'Xiluodu', 'Xiangjiaba', 'TGD']

    base_flows = {
        'Wudongde': 3830.0,
        'Baihetan': 4170.0,
        'Xiluodu': 4620.0,
        'Xiangjiaba': 4620.0,
        'TGD': 14300.0,
    }

    dates = pd.date_range(
        f"{start_year}-01-01", f"{end_year}-12-31", freq="D"
    ).values
    N = len(dates)
    n_res = len(reservoir_names)
    days = np.array(pd.DatetimeIndex(dates).dayofyear)

    inflow = np.zeros((n_res, N))
    rng = np.random.default_rng(42)

    for i, name in enumerate(reservoir_names):
        base = base_flows.get(name, 4000.0)
        # 年周期
        seasonal = 0.8 * base * np.sin(2 * np.pi * (days - 152) / 365.0)
        base_series = np.maximum(base + seasonal, 200.0)

        # AR(1) 噪声
        noise = np.zeros(N)
        rho = 0.7
        sigma = 0.15 * base_series
        for t in range(1, N):
            noise[t] = rho * noise[t-1] + np.sqrt(1 - rho**2) * sigma[t] * rng.standard_normal()

        inflow[i] = np.maximum(base_series + noise, 100.0)

        # 注入洪水事件
        months_arr = np.array(pd.DatetimeIndex(dates).month)
        years_arr = np.array(pd.DatetimeIndex(dates).year)
        for year in range(start_year, end_year + 1):
            year_mask = (years_arr == year)
            flood_mask = year_mask & (months_arr >= 6) & (months_arr <= 9)
            flood_idx = np.where(flood_mask)[0]
            if len(flood_idx) == 0:
                continue

            n_events = rng.integers(1, 4)
            for _ in range(n_events):
                peak_day = rng.choice(flood_idx)
                peak_q = base * rng.uniform(2.0, 4.0)

                for d in range(-7, 11):
                    idx = peak_day + d
                    if idx < 0 or idx >= N:
                        continue
                    if d < 0:
                        factor = np.exp(d / 3.0)
                        inflow[i, idx] = max(inflow[i, idx], base_series[idx] + (peak_q - base_series[idx]) * factor)
                    elif d <= 3:
                        inflow[i, idx] = max(inflow[i, idx], peak_q * rng.uniform(0.9, 1.1))
                    else:
                        factor = np.exp(-(d - 3) / 4.0)
                        inflow[i, idx] = max(inflow[i, idx], base_series[idx] + (peak_q - base_series[idx]) * factor)

    return ReservoirInflowData(
        dates=dates,
        inflow=inflow,
        reservoir_names=reservoir_names,
        source="synthetic",
    )


# ===========================================================================
# 3. 数据下载指引
# ===========================================================================

DATA_SOURCES = {
    "CDR2": {
        "name": "China Daily River Discharge Records (CDR2)",
        "url": "https://zenodo.org/records/20152832",
        "coverage": "1990-2024, 310 gauges",
        "format": "CSV",
        "note": "Gauge coverage must be checked from the metadata before "
                "assigning a station to the Yichang control section.",
    },
    "CWRC": {
        "name": "Changjiang Water Resources Commission",
        "url": "http://www.cjw.gov.cn/",
        "coverage": "Yangtze basin, full record",
        "format": "Yearbook (printed)",
        "note": "Official Chinese source. "
                "Data access requires institutional cooperation.",
    },
}
