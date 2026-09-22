# -*- coding: utf-8 -*-
"""
Pareto 支配与非支配排序
=======================
直接迁移自 EAMS-ABC/pareto.py, 支持任意维目标.
改动: ranks() 支持任意目标维度 (原版硬编码 2 维).
"""
import numpy as np


def dominates(a: np.ndarray, b: np.ndarray, eps: float = 1e-10) -> bool:
    """a 是否支配 b (越小越优)"""
    return bool(np.all(a <= b + eps) and np.any(a < b - eps))


def ranks(F: np.ndarray) -> np.ndarray:
    """
    快速非支配排序 (Python, 无 numba 依赖)
    F: (n, n_obj) 目标值矩阵
    返回: (n,) rank 数组, 0=最优前沿
    """
    n = len(F)
    rank = np.full(n, -1, dtype=np.int64)
    remaining = n
    level = 0
    while remaining:
        selected = []
        for i in range(n):
            if rank[i] >= 0:
                continue
            dominated = False
            for j in range(n):
                if rank[j] >= 0:
                    continue
                if np.all(F[j] <= F[i] + 1e-10) and np.any(F[j] < F[i] - 1e-10):
                    dominated = True
                    break
            if not dominated:
                selected.append(i)
        for i in selected:
            rank[i] = level
        remaining -= len(selected)
        level += 1
    return rank


def crowding(F: np.ndarray) -> np.ndarray:
    """
    拥挤度距离
    F: (n, n_obj)
    返回: (n,) 拥挤度, 越大越好
    """
    n, n_obj = F.shape
    out = np.zeros(n)
    if n <= 2:
        return np.full(n, np.inf)
    for k in range(n_obj):
        ids = np.argsort(F[:, k])
        out[ids[0]] = np.inf
        out[ids[-1]] = np.inf
        span = F[ids[-1], k] - F[ids[0], k]
        if span > 0:
            out[ids[1:-1]] += (F[ids[2:], k] - F[ids[:-2], k]) / span
    return out


class Archive:
    """
    精英外部档案
    - add: 添加解, 剔除被支配的, 超容量时按拥挤度删除
    - 支持 2~5 维目标
    """
    def __init__(self, capacity: int = 80, feasibility_first: bool = True):
        self.capacity = capacity
        self.feasibility_first = feasibility_first
        self.items: list = []
        self.F = np.empty((0, 0))

    def add(self, solution) -> bool:
        """添加解, 返回是否进入档案"""
        f = np.asarray(solution.f)
        if len(self.items) == 0:
            self.items.append(solution)
            self.F = f.reshape(1, -1)
            return True

        # Feasibility-first archive: once a feasible solution exists,
        # infeasible solutions cannot enter; a feasible solution removes all
        # infeasible incumbents.  The previous archive silently ignored the
        # ``feasible`` flag, so the advertised feasibility-first mechanism had
        # no effect.
        existing_feasible = np.array([bool(getattr(x, "feasible", True)) for x in self.items])
        candidate_feasible = bool(getattr(solution, "feasible", True))
        if self.feasibility_first and existing_feasible.any() and not candidate_feasible:
            return False
        if self.feasibility_first and candidate_feasible and not existing_feasible.all():
            self.items = [x for x, ok in zip(self.items, existing_feasible) if ok]
            self.F = np.array([x.f for x in self.items]) if self.items else np.empty((0, len(f)))
            if not self.items:
                self.items.append(solution)
                self.F = f.reshape(1, -1)
                return True

        # 检查是否被现有解支配
        if np.any(np.all(self.F <= f + 1e-10, axis=1)):
            return False

        # 剔除被新解支配的
        keep = ~np.all(f <= self.F + 1e-10, axis=1)
        self.items = [x for x, k in zip(self.items, keep) if k]
        self.F = self.F[keep] if len(self.F) > 0 else np.empty((0, 0))

        self.items.append(solution)
        self.F = np.array([x.f for x in self.items])

        # 超容量 → 删拥挤度最小的
        if len(self.items) > self.capacity:
            cr = crowding(self.F)
            ix = int(np.argmin(cr))
            self.items.pop(ix)
            self.F = np.delete(self.F, ix, axis=0)

        return any(x is solution for x in self.items)


def environmental_selection(items: list, n: int, feasibility_first: bool = True) -> list:
    """
    环境选择: 非支配排序 + 拥挤度
    """
    if len(items) <= n:
        return items
    feasible = [x for x in items if bool(getattr(x, "feasible", True))]
    infeasible = [x for x in items if not bool(getattr(x, "feasible", True))]
    if feasibility_first and len(feasible) >= n:
        items = feasible
    elif feasibility_first and feasible:
        # Keep every feasible solution, then fill only if necessary.  Without
        # an explicit violation magnitude, infeasible points are ranked by
        # objectives as a secondary fallback and never displace feasible ones.
        remain = environmental_selection(infeasible, n - len(feasible), False)
        return feasible + remain
    F = np.array([x.f for x in items])
    r = ranks(F)
    selected = []
    for level in range(int(r.max()) + 1):
        ids = np.flatnonzero(r == level)
        if len(selected) + len(ids) <= n:
            selected.extend(ids)
        else:
            cr = crowding(F[ids])
            n_remain = n - len(selected)
            selected.extend(ids[np.argsort(-cr, kind='stable')[:n_remain]])
            break
    return [items[i] for i in selected]
