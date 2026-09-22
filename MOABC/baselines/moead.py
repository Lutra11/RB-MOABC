# -*- coding: utf-8 -*-
"""
MOEA/D baseline for multi-objective reservoir scheduling.

Migrated from EAMS-ABC/work-content/code/other-method/moead.py
Changes for continuous variables:
  - Discrete crossover -> SBX-like arithmetic crossover
  - Weight vectors for 3 objectives (original was 2)
  - Tchebycheff decomposition adapted for n_obj dimensions

Reference: Zhang & Li (2007) "MOEA/D: A Multiobjective Evolutionary Algorithm
Based on Decomposition"
"""
import numpy as np

from ..core.operators import crossover as sbx_crossover


def _generate_weights(n_obj, n, rng):
    """
    Generate n weight vectors for n_obj objectives.
    Uses Dirichlet distribution for uniform coverage of the simplex.
    """
    if n_obj == 2:
        w = np.column_stack([np.linspace(0, 1, n), np.linspace(1, 0, n)])
    elif n_obj == 3:
        # Simple grid for 3 objectives
        points = []
        steps = int(np.sqrt(n))
        for i in range(steps + 1):
            for j in range(steps + 1 - i):
                k = steps - i - j
                w1 = i / steps
                w2 = j / steps
                w3 = k / steps
                if w1 + w2 + w3 > 0:
                    points.append([w1, w2, w3])
        w = np.array(points[:n])
        # Pad if not enough points
        while len(w) < n:
            w = np.vstack([w, rng.dirichlet(np.ones(n_obj))])
    else:
        w = np.array([rng.dirichlet(np.ones(n_obj)) for _ in range(n)])
    return np.maximum(w, 0.001)


def solve(ctx):
    """
    MOEA/D main loop.

    Uses Tchebycheff decomposition:
        min  max_j  w_j * |f_j - z_j*|

    Shared infrastructure (identical to MOABC):
      - ctx.evaluate(x) -> Solution
      - ctx.archive
      - ctx.cfg.evaluations

    MOEA/D specific:
      - Decomposition into n scalar subproblems
      - Neighborhood-based replacement (T=10)
      - Ideal point z* updated online

    Returns:
        dict: diagnostics
    """
    p = ctx.p
    rng = ctx.rng
    cfg = ctx.cfg
    n = cfg.population
    n_obj = p.n_obj

    pop = ctx.initial_population()
    F = np.array([s.f for s in pop])

    # Weight vectors
    w = _generate_weights(n_obj, n, rng)
    # Neighborhood (T nearest weight vectors)
    T = min(10, n)
    dist = np.sqrt(((w[:, None, :] - w[None, :, :]) ** 2).sum(axis=2))
    neighbors = np.argsort(dist, axis=1)[:, :T]

    # Ideal point
    ideal = F.min(axis=0).copy()
    scale = np.maximum(F.max(axis=0) - ideal, 1.0)

    while ctx.count < cfg.evaluations:
        for i in rng.permutation(n):
            if ctx.count >= cfg.evaluations:
                break
            # Select parents from neighborhood (90%) or global (10%)
            pool = neighbors[i] if rng.random() < 0.9 else np.arange(n)
            a, b = rng.choice(pool, size=2, replace=False)
            x_child = sbx_crossover(p, pop[a].x, pop[b].x, rng)
            sol = ctx.evaluate(x_child)
            if sol is None:
                break
            ideal = np.minimum(ideal, sol.f)

            # Tchebycheff replacement
            replaced = 0
            for j in rng.permutation(pool):
                old_val = np.max(w[j] * np.abs(pop[j].f - ideal) / scale)
                new_val = np.max(w[j] * np.abs(sol.f - ideal) / scale)
                if new_val <= old_val:
                    pop[j] = sol
                    replaced += 1
                if replaced >= 2:
                    break

    return dict(
        neighborhood=T,
        replacement_limit=2,
        n_weights=len(w),
        generations=-1,
    )
