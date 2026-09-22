"""Instrumented solver entry point used only by Risk Protocol R2."""
from __future__ import annotations

from dataclasses import asdict
import time
from typing import Callable, Optional

import numpy as np

from MOABC.core.solver import Config, RunContext, abc
from MOABC.core.pareto import ranks


class InstrumentedRunContext(RunContext):
    def __init__(self, *args, checkpoint_callback=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.checkpoint_callback = checkpoint_callback
        guidance = getattr(self.p, 'search_guidance', None)
        if guidance is not None:
            guidance = np.asarray(guidance, dtype=float)
            if guidance.shape != (self.p.n_res, self.p.T):
                raise ValueError('Search guidance shape does not match problem')
            mean = guidance.mean()
            self.risk_sensitivity = guidance / mean if mean > 0 else np.ones_like(guidance)

    def evaluate(self, x):
        solution = super().evaluate(x)
        if solution is not None and self.checkpoint_callback is not None:
            self.checkpoint_callback(self, solution)
        return solution


def run(problem, method: str, evaluate_fn: Callable, seed: int = 42,
        cfg: Config | None = None, warm_x: Optional[np.ndarray] = None,
        checkpoint_callback=None) -> dict:
    cfg = cfg or Config()
    ctx = InstrumentedRunContext(problem, seed, cfg, evaluate_fn, warm_x,
                                 checkpoint_callback=checkpoint_callback)
    warm = ctx.evaluate(problem.random_solution(ctx.rng))
    ranks(np.array([warm.f, warm.f]))
    start = time.perf_counter()
    key = method.lower().replace('-', '').replace('/', '').replace('_', '')
    if method in ('MOABC', 'Proposed'):
        info = abc(ctx)
    elif key == 'nsgaii':
        from MOABC.baselines.nsga2 import solve
        info = solve(ctx, memetic=False)
    elif key == 'mansgaii':
        from MOABC.baselines.nsga2 import solve
        info = solve(ctx, memetic=True)
    elif key == 'nsgaiii':
        from MOABC.baselines.nsga3 import solve
        info = solve(ctx)
    elif key == 'moead':
        from MOABC.baselines.moead import solve
        info = solve(ctx)
    elif key == 'mopso':
        from MOABC.baselines.mopso import solve
        info = solve(ctx)
    elif key == 'spea2':
        from MOABC.baselines.spea2 import solve
        info = solve(ctx)
    else:
        raise ValueError(f'Unknown method: {method}')
    elapsed = time.perf_counter() - start
    return {
        'method': method, 'seed': seed, 'instance': problem.name,
        'config': asdict(cfg), 'evaluations': ctx.count, 'seconds': elapsed,
        'front': [solution.f.tolist() for solution in ctx.archive.items],
        'history': ctx.history, 'diagnostics': info,
        'solutions': [{'x': solution.x.tolist(), 'f': solution.f.tolist()}
                      for solution in ctx.archive.items],
    }


__all__ = ['Config', 'InstrumentedRunContext', 'run']
