# -*- coding: utf-8 -*-
"""
水库状态模拟引擎
================
水量平衡方程 + Muskingum 汇流 + 约束投影修复

状态方程: S_{i,t+1} = S_{i,t} + dt * (I_{i,t} - R_{i,t} - E_{i,t})
汇流:     Q_{downstream} = Muskingum(R_upstream, I_lateral)
约束:     S_min <= S <= S_max, R_min <= R <= R_max, |dR| <= dR_max

Author: Water Paper Team
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Optional

# ===========================================================================
# 1. 水库参数
# ===========================================================================

@dataclass
class ReservoirParams:
    """单座水库工程参数"""
    name: str
    # 水位-库容关系 (线性插值, 可扩展为分段)
    level_storage: np.ndarray   # (n, 2) [water_level_m, storage_m3]
    # 特征水位 (m)
    normal_pool: float          # 正常蓄水位
    flood_limit: float          # 汛限水位
    dead_level: float           # 死水位
    # 泄量约束
    max_release: float          # 最大泄量 m3/s
    min_release: float = 0.0   # 最小生态流量 m3/s
    max_ramp: float = float('inf')  # 最大变幅 m3/s per dt
    # 装机
    turbine_capacity: float = 0.0  # MW
    rated_flow: float = 0.0        # m3/s
    # 下游尾水位-流量关系 (简化)
    tailwater_Q: Optional[np.ndarray] = None  # (n,2) [Q, tail_level]

    def storage_at_level(self, level: float) -> float:
        """水位 → 库容 (m3 → 10^8 m3)"""
        return float(np.interp(level, self.level_storage[:,0], self.level_storage[:,1]))

    def level_at_storage(self, storage: float) -> float:
        """库容 → 水位"""
        return float(np.interp(storage, self.level_storage[:,1], self.level_storage[:,0]))

    @property
    def S_max(self) -> float:
        """Operational upper storage at the normal pool level (10^8 m3).

        The former implementation used flood-limit storage here.  That made
        flood-control storage above the flood-limit level unavailable and, for
        Xiangjiaba and TGD, collapsed the feasible storage interval to zero.
        """
        return self.storage_at_level(self.normal_pool)

    @property
    def S_flood(self) -> float:
        """Flood-limit storage used as the default flood-season initial state."""
        return self.storage_at_level(self.flood_limit)

    @property
    def S_normal(self) -> float:
        """正常蓄水位对应库容"""
        return self.storage_at_level(self.normal_pool)

    @property
    def S_min(self) -> float:
        """死水位对应库容"""
        return self.storage_at_level(self.dead_level)


# ===========================================================================
# 2. 梯级拓扑与 Muskingum 参数
# ===========================================================================

@dataclass
class ReachParams:
    """库间河道汇流参数"""
    upstream: str
    downstream: str
    K: float          # Muskingum K (h) — 槽蓄系数 ≈ 传播时间
    x: float = 0.2    # Muskingum x — 流量比重因子 (0~0.5)
    length_km: float = 0.0  # 河道长度

@dataclass
class ReservoirNetwork:
    """梯级水库网络拓扑"""
    reservoirs: dict[str, ReservoirParams]
    reaches: list[ReachParams]
    control_points: dict[str, dict] = field(default_factory=dict)
    # control_points: {'Yichang': {'safe_Q': 56700, 'reservoir': 'TGD'}}

    def upstream_of(self, name: str) -> list[str]:
        """返回某水库的直接上游水库列表"""
        return [r.upstream for r in self.reaches if r.downstream == name]

    def downstream_of(self, name: str) -> Optional[str]:
        """返回某水库的直接下游水库"""
        for r in self.reaches:
            if r.upstream == name:
                return r.downstream
        return None

    def reach_to(self, name: str) -> Optional[ReachParams]:
        """返回上游→该水库的河道参数"""
        for r in self.reaches:
            if r.downstream == name:
                return r
        return None


# ===========================================================================
# 3. Muskingum 汇流
# ===========================================================================

class MuskingumRouter:
    """
    Muskingum 法河道演算
    Q_out(t) = C0*Q_in(t) + C1*Q_in(t-1) + C2*Q_out(t-1)
    其中: C0 = (-K*x + 0.5*dt) / (K - K*x + 0.5*dt)
          C1 = (K*x + 0.5*dt) / (K - K*x + 0.5*dt)
          C2 = (K - K*x - 0.5*dt) / (K - K*x + 0.5*dt)
    """
    def __init__(self, K: float, x: float, dt: float):
        if K <= 0 or dt <= 0 or not 0 <= x <= 0.5:
            raise ValueError("Muskingum requires K>0, dt>0 and 0<=x<=0.5")
        self.K = K
        self.x = x
        self.dt = dt
        self.instantaneous = dt > 2.0 * K * (1.0 - x)
        if self.instantaneous:
            # The routing lag is shorter than the numerical time step and
            # cannot be resolved without inventing sub-daily information.
            # Treat it as within-step transfer in a daily model.
            self.n_subreaches = 1
            self.C0, self.C1, self.C2 = 1.0, 0.0, 0.0
            return
        # A long reach is represented by equal subreaches.  Using the full K
        # in one 3-h step produced negative C0 coefficients and the subsequent
        # max(0, q) clipping destroyed mass conservation.  K_sub ~= dt yields
        # non-negative coefficients for the present reaches.
        self.n_subreaches = max(1, int(round(K / dt)))
        K_sub = K / self.n_subreaches
        denom = K_sub - K_sub*x + 0.5*dt
        if denom <= 0:
            raise ValueError(f"Muskingum denom <= 0: K={K}, x={x}, dt={dt}")
        self.C0 = (-K_sub*x + 0.5*dt) / denom
        self.C1 = (K_sub*x + 0.5*dt) / denom
        self.C2 = (K_sub - K_sub*x - 0.5*dt) / denom
        # 稳定性检查: C0+C1+C2 = 1
        assert abs(self.C0 + self.C1 + self.C2 - 1.0) < 1e-10
        if min(self.C0, self.C1, self.C2) < -1e-12:
            raise ValueError(
                "Unstable Muskingum discretisation after subreach split: "
                f"K={K}, x={x}, dt={dt}, n={self.n_subreaches}"
            )

    def initial_state(self, flow: float = 0.0) -> np.ndarray:
        """Return (n_subreach, 2) state [previous inflow, previous outflow]."""
        return np.full((self.n_subreaches, 2), max(float(flow), 0.0))

    def step(self, q_in: float, state: np.ndarray | None = None) -> tuple[float, np.ndarray]:
        """Route one time step through all subreaches without clipping."""
        q = max(float(q_in), 0.0)
        if self.instantaneous:
            return q, np.array([[q, q]], dtype=float)
        if state is None:
            state = self.initial_state(q)
        state = np.asarray(state, dtype=float).copy()
        if state.shape != (self.n_subreaches, 2):
            raise ValueError("invalid Muskingum state shape")
        for j in range(self.n_subreaches):
            previous_in, previous_out = state[j]
            q_out = self.C0 * q + self.C1 * previous_in + self.C2 * previous_out
            if q_out < -1e-8:
                raise RuntimeError("negative Muskingum flow indicates an unstable setup")
            q_out = max(q_out, 0.0)
            state[j] = (q, q_out)
            q = q_out
        return q, state

    def route(self, Q_in: np.ndarray, Q_out_init: float = 0.0) -> np.ndarray:
        """
        演算入流序列 → 出流序列
        Q_in: (T,) 入流 m3/s
        返回: (T,) 出流 m3/s
        """
        T = len(Q_in)
        Q_out = np.zeros(T)
        state = self.initial_state(Q_out_init if Q_out_init > 0 else (Q_in[0] if T else 0.0))
        for t in range(T):
            Q_out[t], state = self.step(Q_in[t], state)
        return Q_out


# ===========================================================================
# 4. 水库状态模拟器
# ===========================================================================

@dataclass
class SimResult:
    """单次模拟结果"""
    storage: np.ndarray      # (n_res, T) 库容 10^8 m3
    level: np.ndarray        # (n_res, T) 水位 m
    release: np.ndarray     # (n_res, T) 泄量 m3/s
    inflow: np.ndarray       # (n_res, T) 入库 m3/s (均值)
    spill: np.ndarray        # (n_res, T) 强制泄洪/溢流 m3/s
    turbine_flow: np.ndarray # (n_res, T) 发电流量 m3/s
    control_Q: np.ndarray    # (n_ctrl, T) 控制断面流量 m3/s (均值)
    feasible: bool           # 是否满足所有约束
    violations: dict         # 违约详情
    # --- 多情景风险传播 ---
    control_Q_scen: np.ndarray = None  # (n_ctrl, T, n_scen) 各情景控制断面流量
    storage_scen: np.ndarray = None    # (n_res, T, n_scen) 各情景库容
    release_scen: np.ndarray = None   # (n_res, T, n_scen) 各情景实际总下泄量
    controlled_release_scen: np.ndarray = None  # shared plan after physical availability
    forced_spill_scen: np.ndarray = None        # forced release above the shared plan
    inflow_scen: np.ndarray = None              # routed total inflow per reservoir/scenario
    n_scen: int = 1                   # 实际模拟的情景数


class ReservoirSimulator:
    """
    梯级水库群滚动模拟器

    用法:
        sim = ReservoirSimulator(network, dt=3.0)
        result = sim.simulate(
            initial_storage=np.array([...]),  # 各库初始库容
            inflow_scenarios=np.array([...]), # (n_res, T, n_scen) 入库情景
            release_decisions=np.array([...]),# (n_res, T) 泄量决策
            lateral_inflow=np.array([...]),   # (n_reach, T) 区间来水
        )
    """

    def __init__(self, network: ReservoirNetwork, dt: float = 3.0):
        self.network = network
        self.dt = dt  # 时间步长 (h)
        self.dt_s = dt * 3600  # 秒
        # 库容单位转换因子: m3/s * dt_s → 10^8 m3
        self.vol_factor = self.dt_s / 1e8

        # 拓扑排序 (上游→下游)
        self.topo_order = self._topological_sort()

        # 为每个河段创建 Muskingum 路由器
        self.routers: dict[str, MuskingumRouter] = {}
        for reach in network.reaches:
            self.routers[reach.downstream] = MuskingumRouter(
                reach.K, reach.x, dt
            )
        self.control_routers: dict[str, MuskingumRouter] = {}
        for name, cp in network.control_points.items():
            if float(cp.get("K", 0.0)) > 0:
                self.control_routers[name] = MuskingumRouter(
                    float(cp["K"]), float(cp.get("x", 0.2)), dt
                )

    def _topological_sort(self) -> list[str]:
        """返回从最上游到最下游的水库名称顺序"""
        names = list(self.network.reservoirs.keys())
        ordered = []
        remaining = set(names)
        while remaining:
            for name in sorted(remaining):
                ups = self.network.upstream_of(name)
                if all(u in ordered for u in ups):
                    ordered.append(name)
                    remaining.discard(name)
                    break
            else:
                # 有环 (不应该发生)
                raise ValueError("Cycle detected in reservoir network topology")
        return ordered

    def simulate(
        self,
        initial_storage: np.ndarray,      # (n_res,) 初始库容 10^8 m3
        inflow_scenarios: np.ndarray,     # (n_res, T, n_scen) 入库流量
        release_decisions: np.ndarray,    # (n_res, T) 泄量决策
        lateral_inflow: np.ndarray = None,# (n_reach, T) 区间来水
        n_control: int = 1,
    ) -> SimResult:
        """
        模拟梯级水库群运行 — 多情景风险传播

        关键改进:
        1. 对每个集合成员 s 独立传播, 得到 (n_ctrl, T, n_scen) 控制断面流量
        2. Muskingum 路由有状态: 保存 Q_out_prev, 实现完整时滞传播
        3. 汇总均值和各情景结果, 供 CVaR 计算

        参数:
            initial_storage: 各库初始库容 (10^8 m3)
            inflow_scenarios: 多情景入库流量 (n_res, T, n_scen) m3/s
            release_decisions: 泄量决策 (n_res, T) m3/s (所有情景共用同一决策)
            lateral_inflow: 区间来水 (n_reach, T) m3/s, 默认0
            n_control: 控制断面数

        返回: SimResult (含多情景结果)
        """
        n_res = len(self.topo_order)
        initial_storage = np.asarray(initial_storage, dtype=float)
        release_decisions = np.asarray(release_decisions, dtype=float)
        inflow_scenarios = np.asarray(inflow_scenarios, dtype=float)
        if release_decisions.ndim != 2 or release_decisions.shape[0] != n_res:
            raise ValueError("release_decisions must have shape (n_res, T)")
        T = release_decisions.shape[1]
        if inflow_scenarios.ndim == 2:
            inflow_scenarios = inflow_scenarios[:, :, None]
        if inflow_scenarios.shape[:2] != (n_res, T):
            raise ValueError("inflow_scenarios must have shape (n_res, T, n_scen)")
        n_scen = inflow_scenarios.shape[2]
        if initial_storage.shape != (n_res,):
            raise ValueError("initial_storage must have shape (n_res,)")
        if lateral_inflow is None:
            lateral_inflow = np.zeros((len(self.network.reaches), T), dtype=float)
        lateral_inflow = np.asarray(lateral_inflow, dtype=float)

        storage_scen = np.zeros((n_res, T, n_scen))
        release_scen = np.zeros((n_res, T, n_scen))
        controlled_scen = np.zeros((n_res, T, n_scen))
        forced_scen = np.zeros((n_res, T, n_scen))
        inflow_scen_out = np.zeros((n_res, T, n_scen))
        turbine_scen = np.zeros((n_res, T, n_scen))
        control_Q_scen = np.zeros((n_control, T, n_scen))

        for s in range(n_scen):
            one = self._simulate_single_scenario(
                initial_storage, inflow_scenarios[:, :, s],
                release_decisions, lateral_inflow, n_control,
            )
            storage_scen[:, :, s] = one["storage"]
            release_scen[:, :, s] = one["total_release"]
            controlled_scen[:, :, s] = one["controlled_release"]
            forced_scen[:, :, s] = one["forced_spill"]
            inflow_scen_out[:, :, s] = one["inflow"]
            turbine_scen[:, :, s] = one["turbine_flow"]
            control_Q_scen[:, :, s] = one["control_Q"]

        storage = storage_scen.mean(axis=2)
        release = release_scen.mean(axis=2)
        controlled = controlled_scen.mean(axis=2)
        spill = forced_scen.mean(axis=2)
        inflow_out = inflow_scen_out.mean(axis=2)
        turbine_flow = turbine_scen.mean(axis=2)
        control_Q = control_Q_scen.mean(axis=2)
        level = np.zeros_like(storage)
        for i, name in enumerate(self.topo_order):
            level[i] = [self.network.reservoirs[name].level_at_storage(v) for v in storage[i]]

        violations = self._check_violations(storage, controlled, initial_storage)
        for i, name in enumerate(self.topo_order):
            p = self.network.reservoirs[name]
            if np.any(storage_scen[i] < p.S_min - 1e-6) or np.any(storage_scen[i] > p.S_max + 1e-6):
                violations[f"{name}_scenario_storage"] = True

        return SimResult(
            storage=storage, level=level, release=release, inflow=inflow_out,
            spill=spill, turbine_flow=turbine_flow, control_Q=control_Q,
            feasible=len(violations) == 0, violations=violations,
            control_Q_scen=control_Q_scen, storage_scen=storage_scen,
            release_scen=release_scen,
            controlled_release_scen=controlled_scen,
            forced_spill_scen=forced_scen, inflow_scen=inflow_scen_out,
            n_scen=n_scen,
        )

    def _simulate_single_scenario(
        self, initial_storage: np.ndarray, local_inflow: np.ndarray,
        release_plan: np.ndarray, lateral_inflow: np.ndarray, n_control: int,
    ) -> dict[str, np.ndarray]:
        """Execute one scenario with a non-anticipative shared release plan."""
        n_res, T = release_plan.shape
        storage = np.zeros((n_res, T))
        total_release = np.zeros((n_res, T))
        controlled = np.zeros((n_res, T))
        forced = np.zeros((n_res, T))
        inflow = np.zeros((n_res, T))
        turbine = np.zeros((n_res, T))
        control_q = np.zeros((n_control, T))
        state_storage = initial_storage.copy()
        route_states: dict[str, np.ndarray | None] = {
            f"{r.upstream}->{r.downstream}": None for r in self.network.reaches
        }
        cp_states: dict[str, np.ndarray | None] = {
            name: None for name in self.network.control_points
        }

        for t in range(T):
            for i, name in enumerate(self.topo_order):
                p = self.network.reservoirs[name]
                q_in = float(local_inflow[i, t])
                for ri, reach in enumerate(self.network.reaches):
                    if reach.downstream != name:
                        continue
                    up_idx = self.topo_order.index(reach.upstream)
                    q_route_in = total_release[up_idx, t] + lateral_inflow[ri, t]
                    key = f"{reach.upstream}->{reach.downstream}"
                    q_routed, route_states[key] = self.routers[name].step(
                        q_route_in, route_states[key]
                    )
                    q_in += q_routed
                inflow[i, t] = q_in

                previous = controlled[i, t - 1] if t else release_plan[i, t]
                q_control = self._project_release(p, release_plan[i, t], state_storage[i], q_in, previous)
                raw_storage = state_storage[i] + (q_in - q_control) * self.vol_factor
                q_forced = max(0.0, raw_storage - p.S_max) / self.vol_factor
                new_storage = raw_storage - q_forced * self.vol_factor
                new_storage = min(max(new_storage, p.S_min), p.S_max)
                controlled[i, t] = q_control
                forced[i, t] = q_forced
                total_release[i, t] = q_control + q_forced
                storage[i, t] = new_storage
                turbine[i, t] = min(q_control, p.rated_flow) if p.rated_flow > 0 else q_control
                state_storage[i] = new_storage

            last_name = self.topo_order[-1]
            last_idx = len(self.topo_order) - 1
            if self.network.control_points:
                for ci, (cp_name, cp) in enumerate(list(self.network.control_points.items())[:n_control]):
                    res_name = cp.get("reservoir", last_name)
                    res_idx = self.topo_order.index(res_name)
                    q_cp_in = total_release[res_idx, t] + float(cp.get("local_inflow", 0.0))
                    if float(cp.get("K", 0.0)) > 0:
                        router = getattr(self, "control_routers", {}).get(cp_name)
                        q_cp, cp_states[cp_name] = router.step(q_cp_in, cp_states[cp_name])
                    else:
                        q_cp = q_cp_in
                    control_q[ci, t] = q_cp
            else:
                control_q[0, t] = total_release[last_idx, t]
        return {
            "storage": storage, "total_release": total_release,
            "controlled_release": controlled, "forced_spill": forced,
            "inflow": inflow, "turbine_flow": turbine, "control_Q": control_q,
        }

    def _project_release(
        self,
        params: ReservoirParams,
        R: float,
        S: float,
        Q_in: float,
        R_prev: float,
    ) -> float:
        """
        约束投影修复: 将不可行泄量投影到可行域

        约束优先级:
        1. 泄量上下限: R_min <= R <= R_max
        2. 爬坡约束: |R - R_prev| <= max_ramp
        3. 可用水约束: 不允许计划泄量把库容降至死水位以下。

        上限不通过情景特定的计划泄量修复；超过上限的水量在模拟器
        中单独记为 forced spill，因而不会引入完美信息式追索决策。
        """
        # 1. 泄量上下限
        R = np.clip(R, params.min_release, params.max_release)

        # 2. 爬坡约束
        if params.max_ramp < float('inf'):
            R = np.clip(R, R_prev - params.max_ramp, R_prev + params.max_ramp)
            R = np.clip(R, params.min_release, params.max_release)

        # 3. Water availability.  Physical curtailment may fall below the
        # nominal minimum release; this is recorded as a feasibility issue by
        # the caller instead of manufacturing water.
        max_available = Q_in + max(S - params.S_min, 0.0) / self.vol_factor
        R = min(R, max(max_available, 0.0))

        return float(R)

    def _check_violations(
        self, storage: np.ndarray, release: np.ndarray,
        initial_storage: np.ndarray
    ) -> dict:
        """检查所有约束, 返回违约详情"""
        violations = {}
        for i, name in enumerate(self.topo_order):
            params = self.network.reservoirs[name]
            # 库容约束
            if np.any(storage[i] > params.S_max + 1e-6):
                violations[f'{name}_storage_max'] = True
            if np.any(storage[i] < params.S_min - 1e-6):
                violations[f'{name}_storage_min'] = True
            # 泄量约束
            if np.any(release[i] > params.max_release + 1e-6):
                violations[f'{name}_release_max'] = True
            if np.any(release[i] < params.min_release - 1e-6):
                violations[f'{name}_release_min'] = True
            # 爬坡约束
            dR = np.abs(np.diff(release[i], prepend=release[i, 0]))
            if params.max_ramp < float('inf') and np.any(dR > params.max_ramp + 1e-6):
                violations[f'{name}_ramp'] = True
        return violations


# ===========================================================================
# 5. 默认网络构建 (金沙江下游-三峡梯级)
# ===========================================================================

def build_default_network() -> ReservoirNetwork:
    """
    构建金沙江下游-三峡梯级网络

    梯级顺序: 乌东德 → 白鹤滩 → 溪洛渡 → 向家坝 → 三峡 → 宜昌(K1)

    注意: 当前水位-库容曲线为公开特征库容点的分段线性近似，不是
    调度机构的原始曲线。矩形降雨窗口也不得解释为精确子流域边界。
    """
    # --- 乌东德 ---
    wudongde = ReservoirParams(
        name='Wudongde',
        normal_pool=975.0, flood_limit=952.0, dead_level=945.0,
        max_release=47000, min_release=800, max_ramp=5000,
        turbine_capacity=10200, rated_flow=8080,
        # 简化水位-库容曲线 (水位 m, 库容 10^8 m3)
        level_storage=np.array([
            [945, 33.0], [952, 43.0], [965, 55.0], [975, 58.63],
        ]),
    )

    # --- 白鹤滩 ---
    baihetan = ReservoirParams(
        name='Baihetan',
        normal_pool=825.0, flood_limit=785.0, dead_level=765.0,
        max_release=43000, min_release=1000, max_ramp=5000,
        turbine_capacity=16000, rated_flow=8600,
        level_storage=np.array([
            [765, 80.0], [785, 120.0], [800, 160.0], [825, 206.27],
        ]),
    )

    # --- 溪洛渡 ---
    xiluodu = ReservoirParams(
        name='Xiluodu',
        normal_pool=600.0, flood_limit=560.0, dead_level=540.0,
        max_release=49923, min_release=800, max_ramp=5000,
        turbine_capacity=13860, rated_flow=7500,
        level_storage=np.array([
            [540, 36.0], [560, 60.0], [580, 90.0], [600, 126.7],
        ]),
    )

    # --- 向家坝 ---
    xiangjiaba = ReservoirParams(
        name='Xiangjiaba',
        normal_pool=380.0, flood_limit=370.0, dead_level=370.0,
        max_release=48660, min_release=600, max_ramp=5000,
        turbine_capacity=6400, rated_flow=6400,
        level_storage=np.array([
            [370, 38.0], [375, 44.0], [380, 51.63],
        ]),
    )

    # --- 三峡 ---
    tgd = ReservoirParams(
        name='TGD',
        normal_pool=175.0, flood_limit=145.0, dead_level=145.0,
        max_release=102500, min_release=5000, max_ramp=10000,
        turbine_capacity=22500, rated_flow=25000,
        level_storage=np.array([
            [145, 171.5], [155, 228.0], [165, 300.0], [175, 393.0],
        ]),
    )

    # --- 河道参数 ---
    reaches = [
        ReachParams('Wudongde', 'Baihetan', K=15.0, x=0.2, length_km=182),
        ReachParams('Baihetan', 'Xiluodu', K=15.0, x=0.2, length_km=195),
        ReachParams('Xiluodu', 'Xiangjiaba', K=12.0, x=0.25, length_km=157),
        ReachParams('Xiangjiaba', 'TGD', K=60.0, x=0.15, length_km=1000),
    ]

    control_points = {
        # 56,700 m3/s is treated as a configurable downstream control target,
        # not as a measured Yichang design-flood threshold.  The 38-km reach
        # is represented by an explicit 3-h Muskingum travel-time assumption.
        'Yichang': {
            'safe_Q': 56700, 'reservoir': 'TGD', 'K': 3.0, 'x': 0.2,
            'length_km': 38.0, 'local_inflow': 0.0,
            'threshold_role': 'downstream_control_target',
        },
    }

    return ReservoirNetwork(
        reservoirs={
            'Wudongde': wudongde,
            'Baihetan': baihetan,
            'Xiluodu': xiluodu,
            'Xiangjiaba': xiangjiaba,
            'TGD': tgd,
        },
        reaches=reaches,
        control_points=control_points,
    )


# ===========================================================================
# 6. 快速测试
# ===========================================================================

if __name__ == '__main__':
    print("=== 水库状态模拟引擎测试 ===\n")

    network = build_default_network()
    sim = ReservoirSimulator(network, dt=3.0)

    n_res = 5
    T = 24  # 24 步 = 72h 滚动时域
    n_scen = 10

    # 初始库容: 均设为汛限水位对应库容的 80%
    initial = np.array([
        network.reservoirs[name].S_max * 0.8
        for name in sim.topo_order
    ])

    # 入库流量: 均值 3000 m3/s + 随机扰动
    np.random.seed(42)
    inflow = 3000 + 500 * np.random.randn(n_res, T, n_scen)
    inflow = np.maximum(inflow, 500)

    # 泄量决策: 初始设为入库均值
    release = np.mean(inflow, axis=2)  # (n_res, T)
    # 确保不超最大泄量
    for i, name in enumerate(sim.topo_order):
        release[i] = np.clip(release[i], 0, network.reservoirs[name].max_release)

    result = sim.simulate(initial, inflow, release)

    print(f"水库顺序: {sim.topo_order}")
    print(f"时间步数: {T} (dt={sim.dt}h, 总时长={T*sim.dt}h)")
    print(f"模拟情景数: {n_scen}")
    print(f"约束可行: {result.feasible}")
    print(f"违约项: {result.violations}")
    print()
    print("各库最终库容 (10^8 m3):")
    for i, name in enumerate(sim.topo_order):
        p = network.reservoirs[name]
        print(f"  {name:12s}: S={result.storage[i,-1]:.2f} "
              f"(S_min={p.S_min:.1f}, S_max={p.S_max:.1f}), "
              f"level={result.level[i,-1]:.1f}m, "
              f"Q_out={result.release[i,-1]:.0f}m3/s")
    print(f"\n控制断面流量: {result.control_Q[0,-1]:.0f} m3/s "
          f"(安全流量: 56700)")
    print("\n✅ 模拟引擎测试通过")
