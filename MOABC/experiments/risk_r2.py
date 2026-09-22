"""Scientifically isolated Risk Protocol R2 model interfaces.

The legacy ``publication_v11`` module remains untouched.  R2 projects planned
releases before simulation, keeps operational and physical diagnostics
separate, matches stress factors with a continuous tail-risk metric, and maps
downstream risk importance back through Muskingum travel time.
"""
from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd

from MOABC.core import ReservoirSchedulingProblem
from MOABC.experiments.legacy_v11 import (
    AREAS, NAMES, SAFE_Q, allocate_budget, audit_calibration, chance_envelope,
    cvar, event_steps, fit_variability, rainfall_matrix, runoff,
    scenario_rain, score_objectives, seed_for,
)
from MOABC.simulation.reservoir_sim import MuskingumRouter

VERSION = 'risk-r2.1-strict'

STRESS_PATTERNS = {
    # Kept only for explicit sensitivity/diagnostic comparisons.
    'uniform': np.ones(5, dtype=float),
    # A fixed, event-independent gradient that activates downstream risk
    # without unrealistically amplifying every upstream forcing window alike.
    'downstream_linear': np.array([0., .25, .5, .75, 1.], dtype=float),
}


def apply_stress_pattern(base: np.ndarray, event_days: int, factor: float,
                         pattern: str = 'downstream_linear') -> np.ndarray:
    """Apply a declared synthetic pressure pattern to event dates only."""
    rainfall = np.asarray(base, dtype=float)
    if rainfall.ndim != 2 or rainfall.shape[0] != len(NAMES):
        raise ValueError('Rainfall must be a five-window matrix')
    if pattern not in STRESS_PATTERNS:
        raise ValueError(f'Unknown stress pattern: {pattern}')
    if not np.isfinite(factor) or factor < 1:
        raise ValueError('Stress factor must be finite and >=1')
    if not 0 <= event_days <= rainfall.shape[1]:
        raise ValueError('Invalid event-day count')
    stressed = rainfall.copy()
    multiplier = 1. + (float(factor) - 1.) * STRESS_PATTERNS[pattern]
    stressed[:, :event_days] *= multiplier[:, None]
    return stressed


def event_cube_r2(forcing, params, model, event, tail, factor, count, seed,
                  pattern: str = 'downstream_linear'):
    """Build runoff scenarios under a fixed, auditable spatial stress pattern."""
    start = pd.Timestamp(event['start'])
    antecedent = rainfall_matrix(forcing, start - pd.Timedelta(days=30), 30)
    initial_q = runoff(antecedent[:, :, None], params)[:, -1, 0]
    base = rainfall_matrix(forcing, start, event_steps(event, tail))
    stressed = apply_stress_pattern(base, event_steps(event, 0), factor, pattern)
    return runoff(scenario_rain(stressed, model, count, seed), params, initial_q)


def project_release_plan(plan: np.ndarray, minimum: np.ndarray,
                         maximum: np.ndarray, ramp: np.ndarray,
                         previous: np.ndarray) -> np.ndarray:
    """Project bounds and ramps sequentially, including the first decision."""
    values = np.asarray(plan, dtype=float).copy()
    minimum = np.asarray(minimum, dtype=float)
    maximum = np.asarray(maximum, dtype=float)
    ramp = np.asarray(ramp, dtype=float)
    prior = np.asarray(previous, dtype=float).copy()
    if values.ndim != 2 or values.shape[0] != len(minimum):
        raise ValueError('Plan and reservoir parameter dimensions disagree')
    if not all(np.isfinite(x).all() for x in (values, minimum, maximum, ramp, prior)):
        raise ValueError('Projection inputs must be finite')
    if np.any(minimum > maximum) or np.any(ramp < 0):
        raise ValueError('Invalid release bounds or ramp')
    for t in range(values.shape[1]):
        lower = np.maximum(minimum, prior - ramp)
        upper = np.minimum(maximum, prior + ramp)
        values[:, t] = np.clip(values[:, t], lower, upper)
        prior = values[:, t]
    return values


def plan_diagnostics(plan: np.ndarray, minimum: np.ndarray,
                     maximum: np.ndarray, ramp: np.ndarray,
                     previous: np.ndarray) -> dict[str, Any]:
    """Diagnose the submitted plan before any projection can hide violations."""
    values = np.asarray(plan, dtype=float)
    scale = np.maximum(np.asarray(maximum, dtype=float), 1e-12)[:, None]
    bound = np.maximum(np.asarray(minimum)[:, None] - values, 0)
    bound = np.maximum(bound, np.maximum(values - np.asarray(maximum)[:, None], 0)) / scale
    prior = np.concatenate([np.asarray(previous)[:, None], values[:, :-1]], axis=1)
    ramp_excess = np.maximum(np.abs(values - prior) - np.asarray(ramp)[:, None], 0) / scale
    bound_time = bound.max(axis=0) if values.shape[1] else np.zeros(0)
    ramp_time = ramp_excess.max(axis=0) if values.shape[1] else np.zeros(0)
    return {
        'planned_bound_violation': float(bound.max(initial=0.)),
        'planned_ramp_violation': float(ramp_excess.max(initial=0.)),
        'plan_constraint_violation': float(max(bound.max(initial=0.), ramp_excess.max(initial=0.))),
        'planned_bound_by_time': bound_time,
        'planned_ramp_by_time': ramp_time,
    }


@dataclass
class RiskR2Problem(ReservoirSchedulingProblem):
    previous_release: np.ndarray = None
    max_ramp: np.ndarray = None
    search_guidance: np.ndarray = None

    def decode(self, x):
        return np.asarray(x, dtype=float).reshape(self.n_res, self.T).copy()

    def clip(self, x):
        plan = self.decode(x)
        projected = project_release_plan(
            plan, self.bounds[:, 0, 0], self.bounds[:, 0, 1],
            self.max_ramp, self.previous_release)
        return projected.ravel()


def _route_zero_state(router: MuskingumRouter, flow: np.ndarray,
                      state: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    for j in range(router.n_subreaches):
        out = router.C0 * flow + router.C1 * state[j, 0] + router.C2 * state[j, 1]
        state[j, 0], state[j, 1] = flow, out
        flow = out
    return flow, state


def route_impulse_responses(network, horizon: int) -> np.ndarray:
    """Return release-to-control zero-state impulse responses for five reservoirs."""
    if horizon < 1:
        raise ValueError('Positive horizon required')
    reach_by_upstream = {r.upstream: r for r in network.reaches}
    cp = list(network.control_points.values())[0]
    responses = np.zeros((len(NAMES), horizon))
    for source in range(len(NAMES)):
        routers = []
        for position in range(source, len(NAMES) - 1):
            reach = reach_by_upstream[NAMES[position]]
            routers.append(MuskingumRouter(reach.K, reach.x, 24.))
        if cp.get('K', 0) > 0:
            routers.append(MuskingumRouter(cp['K'], cp.get('x', .2), 24.))
        states = [np.zeros((router.n_subreaches, 2, 1)) for router in routers]
        for t in range(horizon):
            signal = np.array([1. if t == 0 else 0.])
            for j, router in enumerate(routers):
                signal, states[j] = _route_zero_state(router, signal, states[j])
            responses[source, t] = max(float(signal[0]), 0.)
        total = responses[source].sum()
        if total > 0:
            responses[source] /= total
    return responses


def lag_aware_guidance(network, downstream_importance: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Back-map downstream time importance to release decision time."""
    importance = np.asarray(downstream_importance, dtype=float).ravel()
    if not len(importance) or not np.isfinite(importance).all() or np.any(importance < 0):
        raise ValueError('Downstream importance must be finite and nonnegative')
    responses = route_impulse_responses(network, len(importance))
    guidance = np.zeros_like(responses)
    for k, response in enumerate(responses):
        for decision_time in range(len(importance)):
            remaining = len(importance) - decision_time
            guidance[k, decision_time] = np.dot(response[:remaining], importance[decision_time:])
        peak = guidance[k].max(initial=0.)
        if peak > 0:
            guidance[k] /= peak
    return guidance, responses


def tail_peak_ratio_cvar95(q: np.ndarray, threshold: float) -> float:
    q = np.asarray(q, dtype=float)
    if q.ndim != 2 or threshold <= 0 or not np.isfinite(q).all():
        raise ValueError('Expected finite time-by-scenario flows and a positive threshold')
    return cvar(q.max(axis=0) / threshold, .95)


def select_pressure_factor(rows: list[dict[str, Any]], metric: str,
                           low: float, high: float,
                           tolerance: float = 1e-9) -> dict[str, Any]:
    if not low < high or tolerance < 0:
        raise ValueError('Invalid pressure selection interval or tolerance')
    eligible = [row for row in rows
                if low <= float(row[metric]) <= high
                and float(row['plan_constraint_violation']) <= tolerance
                and float(row['hard_physical_violation']) <= tolerance]
    if not eligible:
        return {'matched_factor': None, 'status': 'unmatched', 'metric': metric,
                'target_band': [low, high], 'eligible_count': 0}
    midpoint = (low + high) / 2
    best = min(eligible, key=lambda row: (abs(float(row[metric]) - midpoint),
                                          float(row['factor'])))
    return {'matched_factor': float(best['factor']), 'status': 'matched',
            'metric': metric, 'target_band': [low, high],
            'matched_metric': float(best[metric]), 'eligible_count': len(eligible)}


def _route_legacy_state(router, flow, state):
    if state is None:
        state = np.broadcast_to(flow, (router.n_subreaches, 2, len(flow))).copy()
    for j in range(router.n_subreaches):
        out = router.C0 * flow + router.C1 * state[j, 0] + router.C2 * state[j, 1]
        state[j, 0], state[j, 1] = flow, out
        flow = out
    return flow, state


def simulate_r2(network, initial, local, plan):
    """Scenario simulation with projected plans and cause-specific diagnostics."""
    initial, local, submitted = [np.asarray(x, dtype=float) for x in (initial, local, plan)]
    if initial.shape != (5,) or local.ndim != 3 or submitted.ndim != 2:
        raise ValueError('Invalid initial state, inflow cube, or plan')
    if local.shape[:2] != submitted.shape or local.shape[0] != 5:
        raise ValueError('Inflow/plan dimensions do not agree')
    if not np.isfinite(local).all() or not np.isfinite(submitted).all() or np.any(local < 0):
        raise ValueError('Non-finite or negative simulation input')
    pars = [network.reservoirs[name] for name in NAMES]
    minimum_release = np.array([p.min_release for p in pars])
    maximum_release = np.array([p.max_release for p in pars])
    ramp = np.array([p.max_ramp for p in pars])
    previous_plan = minimum_release.copy()
    planned = plan_diagnostics(submitted, minimum_release, maximum_release, ramp, previous_plan)
    projected = project_release_plan(submitted, minimum_release, maximum_release, ramp, previous_plan)
    storage_min = np.array([p.S_min for p in pars])
    storage_max = np.array([p.S_max for p in pars])
    if np.any(initial < storage_min) or np.any(initial > storage_max):
        raise ValueError('Initial storage outside physical range')

    storage, release, controlled, forced, total_in = [np.zeros_like(local) for _ in range(5)]
    curtailment = np.zeros_like(local)
    actual_ramp = np.zeros_like(local)
    capacity_excess = np.zeros_like(local)
    T, scenarios = local.shape[1:]
    state = np.repeat(initial[:, None], scenarios, axis=1)
    previous_actual = minimum_release[:, None] * np.ones((5, scenarios))
    reaches = [(NAMES.index(r.upstream), NAMES.index(r.downstream),
                MuskingumRouter(r.K, r.x, 24.)) for r in network.reaches]
    route_states = [None] * len(reaches)
    cp = list(network.control_points.values())[0]
    cp_router = MuskingumRouter(cp['K'], cp.get('x', .2), 24.) if cp.get('K', 0) > 0 else None
    cp_state = None
    outlet = np.zeros((T, scenarios))
    for t in range(T):
        for k, reservoir in enumerate(pars):
            inflow = local[k, t].copy()
            for j, (upstream, downstream, router) in enumerate(reaches):
                if downstream == k:
                    routed, route_states[j] = _route_legacy_state(
                        router, release[upstream, t], route_states[j])
                    inflow += routed
            command = np.full(scenarios, projected[k, t])
            available = np.maximum(inflow + (state[k] - reservoir.S_min) / .000864, 0.)
            actual = np.minimum(command, available)
            curtailment[k, t] = np.maximum(command - actual, 0.) / maximum_release[k]
            actual_ramp[k, t] = np.maximum(
                np.abs(actual - previous_actual[k]) - ramp[k], 0.) / maximum_release[k]
            raw_storage = state[k] + (inflow - actual) * .000864
            overflow = np.maximum(raw_storage - reservoir.S_max, 0.) / .000864
            state[k] = raw_storage - overflow * .000864
            capacity_excess[k, t] = np.maximum(actual + overflow - maximum_release[k], 0.) / maximum_release[k]
            total_in[k, t], storage[k, t] = inflow, state[k]
            controlled[k, t], forced[k, t], release[k, t] = actual, overflow, actual + overflow
            previous_actual[k] = actual
        downstream = release[NAMES.index(cp.get('reservoir', 'TGD')), t] + float(cp.get('local_inflow', 0.))
        if cp_router is not None:
            downstream, cp_state = _route_legacy_state(cp_router, downstream, cp_state)
        outlet[t] = downstream

    operational_by_time = np.maximum(curtailment.max(axis=0), actual_ramp.max(axis=0)).max(axis=1)
    hard_by_time = capacity_excess.max(axis=0).max(axis=1)
    plan_by_time = np.maximum(planned['planned_bound_by_time'], planned['planned_ramp_by_time'])
    return SimpleNamespace(
        storage=storage, release=release, controlled=controlled, forced=forced,
        inflow=total_in, q=outlet, projected_plan=projected,
        planned_bound_violation=planned['planned_bound_violation'],
        planned_ramp_violation=planned['planned_ramp_violation'],
        plan_constraint_violation=planned['plan_constraint_violation'],
        water_curtailment=float(curtailment.max(initial=0.)),
        actual_ramp_deviation=float(actual_ramp.max(initial=0.)),
        forced_spill_capacity_excess=float(capacity_excess.max(initial=0.)),
        operational_deviation=float(max(curtailment.max(initial=0.), actual_ramp.max(initial=0.))),
        hard_physical_violation=float(capacity_excess.max(initial=0.)),
        physical_violation=float(max(planned['plan_constraint_violation'],
                                     curtailment.max(initial=0.), actual_ramp.max(initial=0.),
                                     capacity_excess.max(initial=0.))),
        plan_violation_by_time=plan_by_time,
        operational_deviation_by_time=operational_by_time,
        hard_physical_violation_by_time=hard_by_time,
        curtailment_scen=curtailment, actual_ramp_deviation_scen=actual_ramp,
        forced_spill_capacity_excess_scen=capacity_excess,
    )


def benchmark_r2(network, cube, previous_release=None):
    pars = [network.reservoirs[name] for name in NAMES]
    schedule = np.cumsum(np.asarray(cube).mean(axis=2), axis=0)
    for t in range(1, schedule.shape[1]):
        schedule[:, t] = .65 * schedule[:, t - 1] + .35 * schedule[:, t]
    previous = (np.array([p.min_release for p in pars]) if previous_release is None
                else np.asarray(previous_release, dtype=float))
    return project_release_plan(
        schedule, np.array([p.min_release for p in pars]),
        np.array([p.max_release for p in pars]),
        np.array([p.max_ramp for p in pars]),
        previous)


def metrics_r2(network, result, threshold, start=0):
    pars = [network.reservoirs[name] for name in NAMES]
    q = result.q[start:, 1:]
    if q.shape[1] < 1:
        raise ValueError('At least one random scenario is required')
    loss = np.maximum(q / threshold - 1, 0).mean(axis=0)
    target = np.array([p.S_flood for p in pars])[:, None]
    forced = result.forced[:, start:, 1:].sum(axis=(0, 1)) * .000864
    deficit = np.maximum(target - result.storage[:, -1, 1:], 0).sum(axis=0)
    previous = np.array([p.min_release for p in pars])[:, None, None]
    change = np.diff(result.controlled, axis=1,
                     prepend=np.repeat(previous, result.controlled.shape[2], axis=2))
    ramp = np.abs(change[:, start:, 1:]) / np.array([p.max_release for p in pars])[:, None, None]
    successes = int(np.any(q > threshold, axis=0).sum())
    n = q.shape[1]
    from scipy.stats import beta
    low = 0. if successes == 0 else float(beta.ppf(.025, successes, n - successes + 1))
    high = 1. if successes == n else float(beta.ppf(.975, successes + 1, n - successes))
    return dict(
        f1=cvar(loss), f2=float(np.mean(forced + deficit)), f3=float(ramp.mean()),
        tail_peak_ratio_cvar95=tail_peak_ratio_cvar95(q, threshold),
        exceedance_probability_any=successes / n,
        probability_ci_low=low, probability_ci_high=high,
        max_q_safe_ratio=float(q.max() / threshold),
        minimum_safety_margin_m3s=float(threshold - q.max()),
        max_control_q_m3s=float(q.max()),
        expected_forced_spill_1e8m3=float(forced.mean()),
        plan_constraint_violation=float(result.plan_constraint_violation),
        water_curtailment=float(result.water_curtailment),
        actual_ramp_deviation=float(result.actual_ramp_deviation),
        operational_deviation=float(result.operational_deviation),
        hard_physical_violation=float(result.hard_physical_violation),
        physical_violation=float(result.physical_violation),
        n_random_scenarios=n,
    )


def search_metrics_r2(network, result, threshold, start=0):
    pars = [network.reservoirs[name] for name in NAMES]
    q = result.q[start:, 1:]
    f1 = cvar(np.maximum(q / threshold - 1, 0).mean(axis=0))
    target = np.array([p.S_flood for p in pars])[:, None]
    f2 = np.mean(result.forced[:, start:, 1:].sum(axis=(0, 1)) * .000864
                 + np.maximum(target - result.storage[:, -1, 1:], 0).sum(axis=0))
    previous = np.array([p.min_release for p in pars])[:, None, None]
    change = np.diff(result.controlled, axis=1,
                     prepend=np.repeat(previous, result.controlled.shape[2], axis=2))
    f3 = np.mean(np.abs(change[:, start:, 1:])
                 / np.array([p.max_release for p in pars])[:, None, None])
    return np.array([f1, f2, f3])


def problem_for_r2(network, cube, epsilon, event_id, search_guidance=False,
                   previous_release=None):
    pars = [network.reservoirs[name] for name in NAMES]
    bounds = np.array([[[p.min_release, p.max_release]] * cube.shape[1] for p in pars])
    importance = np.ones(cube.shape[1]) if epsilon is None else 1 / np.maximum(epsilon, 1e-12)
    guidance, _ = lag_aware_guidance(network, importance)
    previous = (np.array([p.min_release for p in pars]) if previous_release is None
                else np.asarray(previous_release, dtype=float))
    return RiskR2Problem(
        n_res=5, T=cube.shape[1], dt=24., n_scen=cube.shape[2], bounds=bounds,
        initial_storage=np.array([p.S_flood for p in pars]), inflow_scenarios=cube,
        target_storage=np.array([p.S_flood for p in pars]),
        risk_budget=None if epsilon is None else np.asarray(epsilon),
        release_reference=benchmark_r2(network, cube, previous), safe_Q=np.array([SAFE_Q]),
        event_id=event_id, previous_release=previous,
        max_ramp=np.array([p.max_ramp for p in pars]),
        search_guidance=guidance if search_guidance else np.ones_like(guidance))


def evaluate_prefix_r2(network, initial, cube, prefix, plan, epsilon,
                       threshold=SAFE_Q, margin=.05):
    origin = prefix.shape[1]
    full_plan = np.concatenate([prefix, plan], axis=1)
    used = cube[:, :full_plan.shape[1]].copy()
    used[:, :origin] = used[:, :origin, :1]
    result = simulate_r2(network, initial, used, full_plan)
    raw = search_metrics_r2(network, result, threshold, origin)
    chance = 0. if epsilon is None else float(np.maximum(
        chance_envelope(result.q[origin:, 1:], threshold, margin).mean(axis=1) - epsilon, 0).max())
    future_plan = float(result.plan_violation_by_time[origin:].max(initial=0.))
    future_hard = float(result.hard_physical_violation_by_time[origin:].max(initial=0.))
    future_operational = float(result.operational_deviation_by_time[origin:].max(initial=0.))
    violation = max(chance, future_plan, future_hard, future_operational)
    return score_objectives(raw, violation), violation <= 1e-10, {
        'raw': raw, 'violation': violation, 'chance_violation': chance,
        'plan_violation': future_plan, 'hard_physical_violation': future_hard,
        'operational_deviation': future_operational, 'result': result,
    }
