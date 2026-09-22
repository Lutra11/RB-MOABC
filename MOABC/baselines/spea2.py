# -*- coding: utf-8 -*-
"""
SPEA2: Improving the Strength Pareto Evolutionary Algorithm.

Reference: Zitzler, Laumanns & Thiele (2001) "SPEA2: Improving the Strength
Pareto Evolutionary Algorithm for Multiobjective Optimization." TIK-Report 103.

Key difference from NSGA-II:
  - Uses STRENGTH-based fitness: each individual's fitness depends on how many
    others it dominates and how many dominate it (not just rank level).
  - Environmental selection: nearest-neighbor density to break ties and trim
    archive (not crowding distance).
  - External archive maintained separately from mating pool.

Shared infrastructure (identical to MOABC):
  - ctx.evaluate(x) / ctx.archive / ctx.cfg.evaluations
  - Same constraint projection, scenarios, evaluation budget.
"""
import numpy as np

from ..core.pareto import dominates
from ..core.operators import crossover as sbx_crossover


def _raw_fitness(F):
    """
    SPEA2 strength + raw fitness.
    S(i) = |{j : i dominates j}|
    R(i) = sum of S(j) for all j that dominate i
    raw(i) = R(i)  (lower is better; 0 = non-dominated)
    """
    n = len(F)
    strength = np.zeros(n, dtype=int)
    for i in range(n):
        for j in range(n):
            if i != j and dominates(F[i], F[j]):
                strength[i] += 1
    raw = np.zeros(n, dtype=float)
    for i in range(n):
        for j in range(n):
            if i != j and dominates(F[j], F[i]):
                raw[i] += strength[j]
    return raw, strength


def _density(F, k):
    """
    SPEA2 density: k-nearest-neighbor density estimate.
    D(i) = 1 / (sigma_k + 2), where sigma_k = distance to k-th nearest neighbor.
    """
    n = len(F)
    if n == 0:
        return np.zeros(0)
    k = min(k, n - 1) if n > 1 else 0
    if k == 0:
        return np.zeros(n)
    dists = np.sqrt(((F[:, None, :] - F[None, :, :]) ** 2).sum(axis=2))
    np.fill_diagonal(dists, np.inf)
    sorted_d = np.sort(dists, axis=1)
    sigma_k = sorted_d[:, k] if k < n else np.full(n, np.inf)
    return 1.0 / (sigma_k + 2.0)


def _spea2_fitness(F):
    """F_total = raw + density (lower is better)."""
    raw, _ = _raw_fitness(F)
    density = _density(F, k=max(1, int(np.sqrt(len(F)))))
    return raw + density


def _environmental_selection(pop, F, archive, archive_F, max_size):
    """
    SPEA2 environmental selection:
    1. Copy non-dominated (raw=0) to archive
    2. Remove dominated from archive
    3. If too large: truncate by nearest-neighbor density
    4. If too small: add best dominated individuals
    """
    # Combine population and archive
    all_F = np.vstack([F, archive_F]) if len(archive_F) > 0 else F
    all_pop = list(pop) + list(archive)

    fit = _spea2_fitness(all_F)
    # Non-dominated = raw fitness 0
    nondom_idx = np.flatnonzero(fit < 1.0)  # raw + density < 1 => raw=0, density<1

    if len(nondom_idx) > max_size:
        # Truncate by density (remove those with smallest nearest-neighbor dist)
        sub_F = all_F[nondom_idx]
        sub_dists = np.sqrt(
            ((sub_F[:, None, :] - sub_F[None, :, :]) ** 2).sum(axis=2))
        np.fill_diagonal(sub_dists, np.inf)
        selected = list(range(len(nondom_idx)))
        while len(selected) > max_size:
            # Find individual with minimum distance to nearest neighbor
            min_dists = np.min(sub_dists[np.ix_(selected, selected)], axis=1)
            remove_idx = int(np.argmin(min_dists))
            selected.pop(remove_idx)
        nondom_idx = nondom_idx[np.array(selected)]
    elif len(nondom_idx) < max_size:
        # Add best (lowest fitness) dominated individuals
        dom_idx = np.flatnonzero(fit >= 1.0)
        if len(dom_idx) > 0:
            sorted_dom = dom_idx[np.argsort(fit[dom_idx])]
            n_add = min(max_size - len(nondom_idx), len(sorted_dom))
            nondom_idx = np.append(nondom_idx, sorted_dom[:n_add])

    return [all_pop[i] for i in nondom_idx]


def _binary_tournament(fitness, rng):
    """Tournament: pick the one with lower SPEA2 fitness."""
    a, b = rng.integers(len(fitness), size=2)
    return a if fitness[a] < fitness[b] else b


def solve(ctx):
    """
    SPEA2 main loop.

    Shared infrastructure (identical to MOABC):
      - ctx.evaluate(x) -> Solution  (counts toward budget)
      - ctx.archive                     (elite external archive)
      - ctx.cfg.evaluations             (function evaluation budget)

    SPEA2 specific:
      - Strength-based fitness assignment
      - k-NN density estimation
      - External archive with truncation

    Returns:
        dict: diagnostics
    """
    p = ctx.p
    rng = ctx.rng
    cfg = ctx.cfg
    n = cfg.population
    archive_max = min(n, cfg.archive_size)

    pop = ctx.initial_population()
    archive = []  # SPEA2's own archive (separate from ctx.archive)

    generation = 0

    while ctx.count < cfg.evaluations:
        pop_F = np.array([s.f for s in pop])

        if len(archive) > 0:
            archive_F = np.array([s.f for s in archive])
        else:
            archive_F = np.empty((0, pop_F.shape[1]))

        # Environmental selection -> update archive
        archive = _environmental_selection(
            pop, pop_F, archive, archive_F, archive_max)

        if len(archive) == 0:
            archive = list(pop)
            archive_F = pop_F

        # Fitness for mating pool (combine pop + archive)
        combined = list(pop) + list(archive)
        combined_F = np.vstack([pop_F, np.array([s.f for s in archive])])
        combined_fit = _spea2_fitness(combined_F)

        # Mating via binary tournament
        children = []
        while len(children) < n and ctx.count < cfg.evaluations:
            a = _binary_tournament(combined_fit, rng)
            b = _binary_tournament(combined_fit, rng)
            x_child = sbx_crossover(
                p, combined[a].x, combined[b].x, rng)
            sol = ctx.evaluate(x_child)
            if sol is None:
                break
            children.append(sol)

        pop = children if children else pop
        generation += 1

    return dict(
        generations=generation,
        archive_size=len(archive),
    )
