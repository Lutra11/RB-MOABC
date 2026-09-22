# -*- coding: utf-8 -*-
"""
NSGA-II baseline for multi-objective reservoir scheduling.

Migrated from EAMS-ABC/work-content/code/other-method/nsga2.py
Changes for continuous variables:
  - Discrete crossover (POX) -> SBX-like arithmetic crossover
  - Discrete mutation -> Gaussian polynomial mutation
  - Uses the same RunContext / Archive / evaluate() as MOABC

Reference: Deb et al. (2002) "A Fast and Elitist Multiobjective Genetic Algorithm: NSGA-II"
"""
import numpy as np

from ..core.pareto import ranks, crowding, dominates, environmental_selection
from ..core.operators import crossover as sbx_crossover, move as local_move


def solve(ctx, memetic=False):
    """
    NSGA-II main loop.

    Shared infrastructure (identical to MOABC):
      - ctx.evaluate(x) -> Solution  (counts toward budget)
      - ctx.archive                     (elite external archive)
      - ctx.cfg.evaluations             (function evaluation budget)
      - Same constraint projection, same scenarios

    NSGA-II specific:
      - Binary tournament selection on (rank, -crowding)
      - SBX crossover + polynomial mutation
      - Environmental selection via non-dominated sorting + crowding

    Args:
        ctx: RunContext (shared with MOABC)
        memetic: if True, add one local search step per child (MA-NSGA-II variant)

    Returns:
        dict: diagnostics
    """
    p = ctx.p
    rng = ctx.rng
    cfg = ctx.cfg
    pop = ctx.initial_population()
    local_calls = 0

    while ctx.count < cfg.evaluations:
        F = np.array([s.f for s in pop])
        rank = ranks(F)
        cd = np.zeros(len(pop))
        for level in np.unique(rank):
            ids = np.flatnonzero(rank == level)
            cd[ids] = crowding(F[ids])

        def tournament():
            a, b = rng.integers(len(pop), size=2)
            return pop[a if (rank[a], -cd[a]) < (rank[b], -cd[b]) else b]

        children = []
        while len(children) < len(pop) and ctx.count < cfg.evaluations:
            parent_a = tournament()
            parent_b = tournament()
            x_child = sbx_crossover(p, parent_a.x, parent_b.x, rng)
            sol = ctx.evaluate(x_child)
            if sol is None:
                break
            children.append(sol)

            # Memetic variant: one local search per child (counts in budget)
            if memetic and ctx.count < cfg.evaluations:
                x_local = local_move(
                    p, sol.x, rng,
                    operator=int(rng.integers(4)),
                    alpha=cfg.alpha,
                    guided=cfg.guided,
                )
                sol_local = ctx.evaluate(x_local)
                if sol_local is not None:
                    local_calls += 1
                    if dominates(sol_local.f, sol.f) or (
                        not dominates(sol.f, sol_local.f) and rng.random() < 0.5
                    ):
                        sol = sol_local
                    children[-1] = sol

        pop = environmental_selection(pop + children, cfg.population, cfg.feasibility_first)

    return dict(local_calls=local_calls, generations=-1)
