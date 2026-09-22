# -*- coding: utf-8 -*-
"""
MOABC 求解器
============
从 EAMS-ABC/solver.py 迁移, 改造为:
  1. 连续变量 (原为离散)
  2. 三目标 (原为两目标)
  3. 约束投影修复 (原为解码时隐式保证)
  4. 风险感知邻域 (原为 criticality 引导)
  5. 滚动温启动 (原为无状态)

EAMS-ABC 核心保留:
  - 自适应算子选择 (Q-learning)
  - 精英档案 → 种群反馈
  - 非支配排序 + 拥挤度
  - 侦察蜂机制
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
import time
import numpy as np
from typing import Callable, Optional

from .pareto import dominates, ranks, crowding, Archive, environmental_selection
from .operators import move, reconstruct, crossover
from .initialization import initialize


# ===========================================================================
# 1. 配置
# ===========================================================================

@dataclass
class Config:
    """MOABC 算法参数"""
    evaluations: int = 6000       # 总函数评估次数
    population: int = 40         # 蜂群规模
    archive_size: int = 80       # 精英档案容量
    limit: int = 30              # 侦察蜂触发阈值
    alpha: float = 0.5           # 邻域步长
    rho: float = 0.2             # Q-learning 学习率
    pmin: float = 0.05           # 算子最小概率
    reconstruction_ratio: float = 0.2  # 侦察蜂重建比例
    guided: bool = True          # 风险感知引导
    adaptive: bool = True        # 自适应算子选择
    rebuild: bool = True         # 侦察蜂重建
    elite_exchange: bool = True  # 精英交换
    warm_start: bool = True      # 滚动温启动
    n_operators: int = 4         # 邻域算子数
    risk_guidance_strength: float = 0.75  # 风险敏感邻域的相对步长放大系数
    feasibility_first: bool = True       # 不可行解优先按约束可行性筛选


# ===========================================================================
# 2. 解与评估
# ===========================================================================

@dataclass
class Solution:
    """一个调度解"""
    x: np.ndarray          # 决策变量 (1D)
    f: np.ndarray           # 目标值 (n_obj,)
    feasible: bool = True
    sim_result: object = None  # SimResult (可选, 用于后处理)
    trials: int = 0


class RunContext:
    """
    运行上下文: 管理评估、档案、统计
    对应 EAMS-ABC RunContext
    """
    def __init__(
        self,
        problem,
        seed: int,
        cfg: Config,
        evaluate_fn: Callable,
        warm_x: Optional[np.ndarray] = None,
    ):
        self.p = problem
        self.rng = np.random.default_rng(seed)
        self.cfg = cfg
        self.archive = Archive(cfg.archive_size, feasibility_first=cfg.feasibility_first)
        self.count = 0
        self.history = []
        self.evaluate_fn = evaluate_fn
        self.warm_x = warm_x

        # 将动态预算转换为搜索敏感度：预算越小，说明该库/时段越关键。
        # 该量只用于分配搜索资源，不替代目标函数或风险约束。
        budget = getattr(problem, 'risk_budget', None)
        if budget is not None:
            b = np.asarray(budget, dtype=float)
            inv = 1.0 / np.maximum(b, 1e-8)
            self.risk_sensitivity = inv / (np.mean(inv) + 1e-12)
        else:
            self.risk_sensitivity = None

        # 检查点: 在 21 个等间距点记录 Pareto 前沿
        self.checkpoints = set(
            np.linspace(cfg.population, cfg.evaluations, 21, dtype=int).tolist()
        )

    def evaluate(self, x: np.ndarray) -> Solution:
        """评估一个解"""
        if self.count >= self.cfg.evaluations:
            return None
        # 约束投影修复
        x = self.p.clip(x)
        # 目标函数
        f, feasible, sim_result = self.evaluate_fn(self.p, x)
        sol = Solution(x=x, f=f, feasible=feasible, sim_result=sim_result)
        self.count += 1
        self.archive.add(sol)
        if self.count in self.checkpoints:
            self.history.append(
                dict(evaluations=self.count,
                     front=[s.f.tolist() for s in self.archive.items])
            )
        return sol

    def initial_population(self) -> list[Solution]:
        """生成初始种群"""
        pop = []
        # 温启动: 第一个解用上一时刻的解
        if self.cfg.warm_start and self.warm_x is not None:
            sol = self.evaluate(self.warm_x.copy())
            if sol is not None:
                pop.append(sol)
        # 其余用多样化策略
        while len(pop) < self.cfg.population:
            idx = len(pop)
            x = initialize(self.p, self.rng, idx)
            sol = self.evaluate(x)
            if sol is not None:
                pop.append(sol)
        return pop


# ===========================================================================
# 3. ABC 主循环
# ===========================================================================

def abc(ctx: RunContext) -> dict:
    """
    ABC 主循环 — 直接迁移自 EAMS-ABC/solver.py

    结构:
      while evaluations < budget:
        1. 雇佣蜂: 每个位置生成一个邻域解
        2. 观察蜂: 按适应度选择位置, 再生成邻域解
        3. 侦察蜂: 超过 limit 的位置重建
        4. 精英交换: 从档案中取精英与种群交叉
        5. 环境选择: 非支配排序+拥挤度
    """
    p = ctx.p
    rng = ctx.rng
    cfg = ctx.cfg
    pop = ctx.initial_population()
    trials = np.zeros(len(pop), dtype=int)

    # 自适应算子 Q 值
    Q = np.ones(cfg.n_operators) * 0.1
    uses = np.zeros(cfg.n_operators, dtype=int)
    gains = np.zeros(cfg.n_operators)
    scout_count = 0
    generation = 0
    op_history = []
    exchange_calls = 0

    while ctx.count < cfg.evaluations:
        # 非支配排序 → 权重
        F = np.array([s.f for s in pop])
        r = ranks(F)
        weights = 1.0 / (1.0 + r)
        weights = weights / weights.sum()

        # 观察蜂: 按权重选择位置
        indices = list(range(len(pop))) + rng.choice(
            len(pop), len(pop), p=weights
        ).tolist()

        candidates = []
        for position, i in enumerate(indices):
            if ctx.count >= cfg.evaluations:
                break
            old = pop[i]

            # 侦察蜂
            if trials[i] >= cfg.limit:
                x_new = reconstruct(p, old.x, rng, cfg.reconstruction_ratio, cfg.alpha)
                sol = ctx.evaluate(x_new)
                if sol is None:
                    break
                pop[i] = sol
                candidates.append(sol)
                trials[i] = 0
                scout_count += 1
                continue

            # 精英交换: 从档案取精英交叉
            if cfg.elite_exchange and position >= len(pop) and rng.random() < 0.75:
                if len(ctx.archive.items) > 0:
                    peer = ctx.archive.items[
                        int(rng.integers(len(ctx.archive.items)))
                    ]
                    x_new = crossover(p, old.x, peer.x, rng)
                    sol = ctx.evaluate(x_new)
                    if sol is None:
                        break
                    candidates.append(sol)
                    exchange_calls += 1
                    continue

            # 选择邻域算子
            if cfg.adaptive:
                probs = cfg.pmin + (1 - cfg.n_operators * cfg.pmin) * (Q + 1e-9) / (
                    Q.sum() + cfg.n_operators * 1e-9
                )
            else:
                probs = np.ones(cfg.n_operators) / cfg.n_operators

            op = int(rng.choice(cfg.n_operators, p=probs))
            uses[op] += 1

            x_new = move(
                p, old.x, rng, op, cfg.alpha, cfg.guided,
                risk_sensitivity= ctx.risk_sensitivity,
            )
            sol = ctx.evaluate(x_new)
            if sol is None:
                break
            candidates.append(sol)

            # 支配关系判断
            dom = dominates(sol.f, old.f)

            # Q-learning 奖励
            improvement = np.maximum(
                0.0, (old.f - sol.f) / (np.abs(old.f) + 1e-12)
            )
            # 奖励覆盖全部目标，并同时奖励从不可行到可行的转换。
            reward = 0.25 * float(improvement.mean())
            if dom:
                reward += 0.5
            elif sol.feasible and not old.feasible:
                reward += 0.25
            reward = float(reward)
            Q[op] = (1 - cfg.rho) * Q[op] + cfg.rho * reward
            gains[op] += reward

            # 接受准则
            feasibility_changed = sol.feasible != old.feasible
            if cfg.feasibility_first and feasibility_changed:
                accept = sol.feasible
            else:
                accept = dom
            if (not feasibility_changed or not cfg.feasibility_first) and \
                    not dom and not dominates(old.f, sol.f):
                # 不可比较 → 覆盖全部目标的随机标量化，避免遗漏 f3。
                w = rng.dirichlet(np.ones(len(old.f)))
                old_scale = np.maximum(np.abs(old.f), 1e-12)
                new_ratio = sol.f / old_scale
                accept = bool(np.dot(w, new_ratio) < 1.0)

            if accept:
                pop[i] = sol
                trials[i] = 0
            else:
                trials[i] += 1

        generation += 1

        # 精英反馈: 档案 → 种群
        if cfg.elite_exchange and candidates:
            unique = {}
            for s in pop + candidates:
                key = s.x.tobytes()
                if key not in unique or dominates(s.f, unique[key].f):
                    unique[key] = s
            pool = list(unique.values())
            if len(pool) >= len(pop):
                pop = environmental_selection(pool, len(pop), cfg.feasibility_first)
                # 重置 trials
                trials = np.zeros(len(pop), dtype=int)

        if generation % 10 == 0:
            op_history.append([ctx.count, *Q.tolist()])

    return dict(
        operator_uses=uses.tolist(),
        operator_reward=gains.tolist(),
        scouts=scout_count,
        operator_history=op_history,
        exchange_calls=exchange_calls,
        generations=generation,
    )


# ===========================================================================
# 4. 统一入口
# ===========================================================================

def solve(
    problem,
    evaluate_fn: Callable,
    seed: int = 42,
    cfg: Config = None,
    warm_x: Optional[np.ndarray] = None,
) -> dict:
    """
    求解水库调度问题

    参数:
        problem: ReservoirSchedulingProblem
        evaluate_fn: (problem, x) -> (f, feasible, sim_result)
        seed: 随机种子
        cfg: 算法配置
        warm_x: 上一时刻解 (温启动)

    返回:
        dict: method, seed, evaluations, seconds, front, solutions, diagnostics
    """
    cfg = cfg or Config()
    ctx = RunContext(problem, seed, cfg, evaluate_fn, warm_x)

    # 预热 (不计时)
    warm = ctx.evaluate(problem.random_solution(ctx.rng))
    ranks(np.array([warm.f, warm.f]))

    start = time.perf_counter()
    info = abc(ctx)
    elapsed = time.perf_counter() - start

    return dict(
        method='MOABC',
        seed=seed,
        instance=problem.name,
        config=asdict(cfg),
        evaluations=ctx.count,
        seconds=elapsed,
        front=[s.f.tolist() for s in ctx.archive.items],
        history=ctx.history,
        diagnostics=info,
        solutions=[dict(x=s.x.tolist(), f=s.f.tolist()) for s in ctx.archive.items],
    )


def run(
    problem,
    method: str,
    evaluate_fn: Callable,
    seed: int = 42,
    cfg: Config = None,
    warm_x: Optional[np.ndarray] = None,
) -> dict:
    """
    统一入口: 支持 MOABC / NSGA-II / MOEA/D / MOPSO

    所有方法共享同一 RunContext, Archive, evaluate() 机制,
    确保公平对比 (相同评估预算、约束修复、情景集合).

    Args:
        problem: ReservoirSchedulingProblem
        method: 'MOABC' | 'Proposed' | 'NSGA-II' | 'MA-NSGA-II' | 'MOEA/D' | 'MOPSO'
        evaluate_fn: (problem, x) -> (f, feasible, sim_result)
        seed: 随机种子
        cfg: 算法配置
        warm_x: 上一时刻解 (温启动, 仅 MOABC)

    Returns:
        dict: method, seed, evaluations, seconds, front, solutions, diagnostics
    """
    cfg = cfg or Config()
    ctx = RunContext(problem, seed, cfg, evaluate_fn, warm_x)

    # 预热 (不计时, 所有方法一致)
    warm = ctx.evaluate(problem.random_solution(ctx.rng))
    ranks(np.array([warm.f, warm.f]))

    start = time.perf_counter()

    method_key = method.lower().replace('-', '').replace('/', '').replace('_', '')
    if method in ('MOABC', 'Proposed'):
        info = abc(ctx)
    elif method_key == 'nsgaii':
        from ..baselines.nsga2 import solve as _solve
        info = _solve(ctx, memetic=False)
    elif method_key == 'mansgaii':
        from ..baselines.nsga2 import solve as _solve
        info = _solve(ctx, memetic=True)
    elif method_key == 'nsgaiii':
        from ..baselines.nsga3 import solve as _solve
        info = _solve(ctx)
    elif method_key == 'moead':
        from ..baselines.moead import solve as _solve
        info = _solve(ctx)
    elif method_key == 'mopso':
        from ..baselines.mopso import solve as _solve
        info = _solve(ctx)
    elif method_key == 'spea2':
        from ..baselines.spea2 import solve as _solve
        info = _solve(ctx)
    else:
        raise ValueError(f'Unknown method: {method}')

    elapsed = time.perf_counter() - start

    return dict(
        method=method,
        seed=seed,
        instance=problem.name,
        config=asdict(cfg),
        evaluations=ctx.count,
        seconds=elapsed,
        front=[s.f.tolist() for s in ctx.archive.items],
        history=ctx.history,
        diagnostics=info,
        solutions=[dict(x=s.x.tolist(), f=s.f.tolist()) for s in ctx.archive.items],
    )
