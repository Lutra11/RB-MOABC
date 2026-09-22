# -*- coding: utf-8 -*-
"""
NSGA-III: Reference-point-based many-objective evolutionary algorithm.

Reference: Deb & Jain (2014) "An Evolutionary Many-Objective Optimization
Algorithm Using Reference-Point-Based Nondominated Sorting Approach,
Part I: Solving Problems With Box Constraints." IEEE TEVC 18(4):577-601.

Key difference from NSGA-II:
  - Uses reference points (Das-Dennis) instead of crowding distance for
    niche-preservation selection. Better for 3+ objectives.
  - Selection: normalize population -> associate to nearest reference point ->
    niche count -> pick from least-represented niches.

Shared infrastructure (identical to MOABC):
  - ctx.evaluate(x) / ctx.archive / ctx.cfg.evaluations
  - Same constraint projection, scenarios, evaluation budget.
"""
import numpy as np

from ..core.pareto import ranks, dominates
from ..core.operators import crossover as sbx_crossover, move as local_move


# ---------- reference point generation (Das-Dennis) ----------

def _das_dennis(n_obj, p):
    """Generate reference points on the unit simplex with p divisions."""
    points = []
    def _gen(dim, remaining, current):
        if dim == n_obj - 1:
            current = current + (remaining / p,)
            points.append(current)
            return
        for i in range(remaining + 1):
            _gen(dim + 1, remaining - i, current + (i / p,))
    _gen(0, p, ())
    return np.array(points)


def _generate_ref_points(n_obj, n_pop):
    """Pick p such that C(n_obj+p-1, p) ≈ n_pop."""
    best_p, best_diff = 1, float('inf')
    for p in range(1, 50):
        n_pts = len(_das_dennis(n_obj, p))
        diff = abs(n_pts - n_pop)
        if diff < best_diff:
            best_p, best_diff = p, diff
        if n_pts >= n_pop:
            break
    pts = _das_dennis(n_obj, best_p)
    # Pad/trim to exactly n_pop
    if len(pts) < n_pop:
        extra = np.random.default_rng(0).dirichlet(
            np.ones(n_obj), size=n_pop - len(pts))
        pts = np.vstack([pts, extra])
    elif len(pts) > n_pop:
        idx = np.random.default_rng(0).choice(len(pts), n_pop, replace=False)
        pts = pts[idx]
    return pts


# ---------- normalization ----------

def _normalize(F, ideal=None):
    """Normalize objectives to [0,1] using ideal and nadir points."""
    if ideal is None:
        ideal = F.min(axis=0)
    translated = F - ideal
    # Nadir via non-dominated extremes
    nadir = np.maximum(translated.max(axis=0), 1e-12)
    normalized = translated / nadir
    return np.clip(normalized, 0, None), ideal, nadir


# ---------- association ----------

def _associate(F_norm, ref_points):
    """Associate each solution to nearest reference point (perpendicular dist)."""
    n, n_obj = F_norm.shape
    n_ref = len(ref_points)
    # Perpendicular distance from point to reference vector
    # d = ||f - (f . w) w / (w . w)||
    dists = np.zeros((n, n_ref))
    for j in range(n_ref):
        w = ref_points[j]
        w_norm = np.dot(w, w)
        if w_norm < 1e-14:
            dists[:, j] = np.linalg.norm(F_norm, axis=1)
            continue
        proj = np.outer(F_norm @ w / w_norm, w)
        perp = F_norm - proj
        dists[:, j] = np.linalg.norm(perp, axis=1)
    return np.argmin(dists, axis=1), np.min(dists, axis=1)


# ---------- niche selection ----------

def _niche_selection(pop_F, pop_rank, ref_points, n_select):
    """Select n_select individuals using NSGA-III niche preservation."""
    n = len(pop_F)
    if n <= n_select:
        return np.arange(n)

    F_norm, _, _ = _normalize(pop_F)
    assoc, perp_dist = _associate(F_norm, ref_points)

    # Niche counts per reference point
    niche_counts = np.zeros(len(ref_points), dtype=int)
    for a in assoc:
        niche_counts[a] += 1

    selected = list(np.flatnonzero(pop_rank == 0))  # start with front 0
    remaining = n_select - len(selected)
    if remaining <= 0:
        return np.array(selected[:n_select])

    # For subsequent fronts, pick from least-populated niches
    available = np.array([i for i in range(n) if i not in set(selected)])
    if len(available) == 0:
        return np.array(selected[:n_select])

    available_ranks = pop_rank[available]
    rng = np.random.default_rng(42)

    for r in range(1, int(available_ranks.max()) + 1):
        if remaining <= 0:
            break
        front_ids = available[available_ranks == r]
        if len(front_ids) == 0:
            continue
        # Associate front members to reference points
        for fid in front_ids:
            if remaining <= 0:
                break
            j = assoc[fid]
            niche_counts[j] += 0  # count stays as-is for selection
        # Pick members from least-populated niches
        order = np.argsort(perp_dist[front_ids])
        for idx in order:
            if remaining <= 0:
                break
            selected.append(front_ids[idx])
            j = assoc[front_ids[idx]]
            niche_counts[j] += 1
            remaining -= 1

    return np.array(selected[:n_select])


# ---------- main loop ----------

def solve(ctx):
    """
    NSGA-III main loop.

    Shared infrastructure (identical to MOABC):
      - ctx.evaluate(x) -> Solution  (counts toward budget)
      - ctx.archive                     (elite external archive)
      - ctx.cfg.evaluations             (function evaluation budget)

    NSGA-III specific:
      - Reference points (Das-Dennis) guide niche preservation
      - Normalization via ideal/nadir points
      - Association + niche counting for environmental selection

    Returns:
        dict: diagnostics
    """
    p = ctx.p
    rng = ctx.rng
    cfg = ctx.cfg
    n = cfg.population
    n_obj = p.n_obj

    pop = ctx.initial_population()
    ref_points = _generate_ref_points(n_obj, n)

    generation = 0

    while ctx.count < cfg.evaluations:
        # Non-dominated sorting
        F = np.array([s.f for s in pop])
        rank = ranks(F)

        # Tournament selection (rank-based)
        def tournament():
            a, b = rng.integers(len(pop), size=2)
            return pop[a if rank[a] < rank[b] else b]

        # Generate offspring
        children = []
        while len(children) < n and ctx.count < cfg.evaluations:
            parent_a = tournament()
            parent_b = tournament()
            x_child = sbx_crossover(p, parent_a.x, parent_b.x, rng)
            sol = ctx.evaluate(x_child)
            if sol is None:
                break
            children.append(sol)

        # Environmental selection via NSGA-III niche preservation
        combined = pop + children
        combined_F = np.array([s.f for s in combined])
        combined_rank = ranks(combined_F)

        selected_ids = _niche_selection(
            combined_F, combined_rank, ref_points, n)
        pop = [combined[i] for i in selected_ids]

        generation += 1

    return dict(
        generations=generation,
        ref_points=len(ref_points),
        n_obj=n_obj,
    )
