"""Rolling-horizon execution utilities.

The optimizer solves a finite horizon, but only the first release decision is
executed. The resulting reservoir state is then used to initialize the next
optimization window. This module keeps that outer control loop separate from
the MOABC implementation so it can be tested independently.
"""
from __future__ import annotations

import numpy as np


def select_compromise(solutions: list[dict]) -> np.ndarray:
    """Select the normalized-sum compromise from an archive of solutions."""
    if not solutions:
        raise ValueError("solver returned no solutions")
    f = np.asarray([s["f"] for s in solutions], dtype=float)
    lo = f.min(axis=0)
    scale = np.maximum(f.max(axis=0) - lo, 1e-12)
    score = ((f - lo) / scale).sum(axis=1)
    return np.asarray(solutions[int(np.argmin(score))]["x"], dtype=float)


def _select_compromise_index(solutions: list[dict]) -> int:
    """Return the archive index of the normalized-sum compromise."""
    if not solutions:
        raise ValueError("solver returned no solutions")
    f = np.asarray([s["f"] for s in solutions], dtype=float)
    lo = f.min(axis=0)
    scale = np.maximum(f.max(axis=0) - lo, 1e-12)
    score = ((f - lo) / scale).sum(axis=1)
    return int(np.argmin(score))


def run_rolling_horizon(
    inflow_scenarios: np.ndarray,
    initial_storage: np.ndarray,
    problem_factory,
    evaluate_fn,
    solve_fn,
    simulator,
    horizon: int,
    step: int = 1,
    seed: int = 42,
    cfg=None,
):
    """Run a receding-horizon simulation over a scenario cube.

    ``problem_factory`` receives ``(scenario_slice, storage, warm_x)`` and
    must return a ``ReservoirSchedulingProblem``. ``evaluate_fn`` must be the
    direct solver callback with signature ``(problem, x) -> (f, feasible,
    sim_result)``. The function returns one record per executed interval.
    """
    cube = np.asarray(inflow_scenarios, dtype=float)
    if cube.ndim != 3:
        raise ValueError("inflow_scenarios must have shape (n_res, T, n_scen)")
    n_res, total_T, _ = cube.shape
    if horizon <= 0 or horizon > total_T:
        raise ValueError("horizon must be within the scenario time range")
    if step <= 0:
        raise ValueError("step must be positive")

    storage = np.asarray(initial_storage, dtype=float).copy()
    warm_x = None
    records = []
    for origin in range(0, total_T - horizon + 1, step):
        window = cube[:, origin:origin + horizon, :]
        problem = problem_factory(window, storage.copy(), warm_x)
        solved = solve_fn(problem, evaluate_fn,
                          seed=seed + origin, cfg=cfg, warm_x=warm_x)
        selected_idx = _select_compromise_index(solved["solutions"])
        selected = solved["solutions"][selected_idx]
        x = np.asarray(selected["x"], dtype=float)
        release = problem.decode(x)

        one_step = simulator.simulate(
            initial_storage=storage,
            inflow_scenarios=window[:, :1, :],
            release_decisions=release[:, :1],
        )
        storage = one_step.storage_scen[:, 0, :].mean(axis=1)
        next_schedule = np.concatenate([release[:, 1:], release[:, -1:]], axis=1)
        warm_x = problem.encode(next_schedule)
        records.append({
            "origin": origin,
            "objective": selected["f"],
            "selected_release": release[:, 0].copy(),
            "storage_after": storage.copy(),
            "solver": solved,
        })
    return {"records": records, "final_storage": storage, "warm_x": warm_x}
