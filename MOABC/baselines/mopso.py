# -*- coding: utf-8 -*-
"""
MOPSO baseline for multi-objective reservoir scheduling.

Newly written (no EAMS-ABC equivalent).
Adapted from Coello et al. (2004) "Handling Multiple Objectives with Particle
Swarm Optimization" and Reyes-Sierra & Coello (2006) MOPSO survey.

Key design choices for reservoir scheduling:
  - Continuous variable optimization (natural fit for PSO)
  - External archive = leader selection pool (same Archive class as MOABC)
  - Adaptive grid or crowding for leader selection
  - Turbulence/mutation operator to prevent premature convergence
  - Same RunContext / evaluate() for fair comparison

Reference: Coello et al. (2004) IEEE Trans. Evolutionary Computation 8(3):256-279
"""
import numpy as np

from ..core.pareto import ranks, crowding, dominates


def solve(ctx):
    """
    MOPSO main loop.

    Shared infrastructure (identical to MOABC):
      - ctx.evaluate(x) -> Solution  (counts toward budget)
      - ctx.archive                     (elite external archive)
      - ctx.cfg.evaluations             (function evaluation budget)

    MOPSO specific:
      - Velocity update: v = w*v + c1*r1*(pbest - x) + c2*r2*(leader - x)
      - Position update: x = x + v
      - Leader from archive (crowding-based selection)
      - Turbulence: random mutation with probability p_m

    Returns:
        dict: diagnostics
    """
    p = ctx.p
    rng = ctx.rng
    cfg = ctx.cfg
    n = cfg.population

    # PSO parameters
    w_inertia = 0.4       # inertia weight
    c1 = 1.5              # cognitive coefficient (pbest pull)
    c2 = 1.5              # social coefficient (leader pull)
    p_mutation = 0.1      # turbulence probability
    v_max_frac = 0.2      # max velocity = 20% of variable range

    # Initialize swarm
    pop = ctx.initial_population()
    velocities = []
    for s in pop:
        v = rng.uniform(-1, 1, size=s.x.shape) * 0.1 * (
            p.bounds[:, :, 1].flatten() - p.bounds[:, :, 0].flatten()
        )
        velocities.append(v)

    # Personal best = initial positions
    pbest = [s for s in pop]
    pbest_f = [s.f.copy() for s in pop]

    generation = 0
    turbulence_count = 0

    while ctx.count < cfg.evaluations:
        # Compute crowding for leader selection
        archive_items = ctx.archive.items
        if len(archive_items) == 0:
            continue

        archive_F = np.array([s.f for s in archive_items])
        if len(archive_items) > 1:
            cr = crowding(archive_F)
            # Handle inf (boundary points) — assign max finite value
            finite_cr = cr[np.isfinite(cr)]
            if len(finite_cr) == 0:
                cr_prob = np.ones(len(archive_items)) / len(archive_items)
            else:
                max_finite = finite_cr.max()
                cr = np.where(np.isfinite(cr), cr, max_finite)
                cr = np.maximum(cr, 0)
                total = cr.sum()
                cr_prob = cr / total if total > 0 else np.ones(len(archive_items)) / len(archive_items)
        else:
            cr_prob = np.array([1.0])

        for i in range(n):
            if ctx.count >= cfg.evaluations:
                break

            # Select leader from archive (crowding-based)
            leader_idx = int(rng.choice(
                len(archive_items), p=cr_prob
            ))
            leader = archive_items[leader_idx]

            # Velocity update
            r1 = rng.random(pop[i].x.shape)
            r2 = rng.random(pop[i].x.shape)
            cognitive = c1 * r1 * (pbest[i].x - pop[i].x)
            social = c2 * r2 * (leader.x - pop[i].x)
            velocities[i] = w_inertia * velocities[i] + cognitive + social

            # Velocity clamp
            var_range = (
                p.bounds[:, :, 1].flatten() - p.bounds[:, :, 0].flatten()
            )
            v_max = v_max_frac * var_range
            velocities[i] = np.clip(velocities[i], -v_max, v_max)

            # Position update
            x_new = pop[i].x + velocities[i]
            # Clip to bounds
            x_new = p.clip(x_new)

            # Turbulence/mutation
            if rng.random() < p_mutation:
                n_mutate = max(1, int(len(x_new) * 0.1))
                mut_idx = rng.choice(len(x_new), size=n_mutate, replace=False)
                flat_lo = p.bounds[:, :, 0].flatten()
                flat_hi = p.bounds[:, :, 1].flatten()
                x_new[mut_idx] = rng.uniform(
                    flat_lo[mut_idx], flat_hi[mut_idx]
                )
                turbulence_count += 1

            # Evaluate
            sol = ctx.evaluate(x_new)
            if sol is None:
                break

            # Update personal best
            if dominates(sol.f, pbest_f[i]):
                pbest[i] = sol
                pbest_f[i] = sol.f.copy()
            elif not dominates(pbest_f[i], sol.f):
                # Non-dominated -> keep with 50% probability
                if rng.random() < 0.5:
                    pbest[i] = sol
                    pbest_f[i] = sol.f.copy()

            pop[i] = sol

        generation += 1

    return dict(
        generations=generation,
        turbulence=turbulence_count,
        w_inertia=w_inertia,
        c1=c1,
        c2=c2,
    )
