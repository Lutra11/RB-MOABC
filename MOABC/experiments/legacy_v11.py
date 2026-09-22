"""Auditable retrospective protocol. Legacy v10 runs are deliberately untouched.

Local runoff parameters are TRANSFERRED hypotheses, not validated local inflows.
Chance constraints are finite-sample constraints, not population certificates.
The configured downstream point is an outlet proxy, not an observed Yichang gauge.
"""
from __future__ import annotations
from dataclasses import dataclass
from types import SimpleNamespace
import hashlib
import json
import numpy as np
import pandas as pd
from MOABC.simulation import build_default_network
from MOABC.simulation.reservoir_sim import MuskingumRouter
from MOABC.core import ReservoirSchedulingProblem

NAMES = ('Wudongde', 'Baihetan', 'Xiluodu', 'Xiangjiaba', 'TGD')
AREAS = np.diff([0., 406100., 430300., 454400., 458800., 1005500.])
SAFE_Q = 56700.  # Declared research threshold, NOT a verified gauge safety limit.
VERSION = 'v11.0'


class PublicationProblem(ReservoirSchedulingProblem):
    def decode(self, x):
        # Legacy reconstruction mutates its decoded array. Protect stored archive x.
        return np.asarray(x, dtype=float).reshape(self.n_res, self.T).copy()


def seed_for(event, index, purpose):
    # Deliberately independent of strategy, algorithm AND stress factor.
    return int.from_bytes(hashlib.sha256(f'{event}|{index}|{purpose}'.encode()).digest()[:4], 'little')


def cvar(values, alpha=.95):
    x = np.sort(np.asarray(values, float).ravel())[::-1]
    if not len(x) or not np.isfinite(x).all() or not 0 <= alpha < 1:
        raise ValueError('Invalid CVaR input')
    mass = (1 - alpha) * len(x)
    whole = int(np.floor(mass))
    remainder = mass - whole
    return float((x[:whole].sum() + (remainder * x[whole] if whole < len(x) else 0)) / mass)


def chance_envelope(q, threshold, margin=.05):
    """Continuous conservative upper bound on 1[Q>threshold], sample by sample."""
    if threshold <= 0 or not 0 < margin < 1:
        raise ValueError('Positive threshold and margin in (0,1) required')
    return np.clip(1 + (np.asarray(q) - threshold) / (margin * threshold), 0, 1)


def allocate_budget(q, remaining, mode, threshold):
    if remaining < -1e-12 or len(q) == 0:
        raise ValueError('Invalid remaining event budget')
    if mode == 'fixed':
        weights = np.ones(len(q))
    elif mode == 'dynamic':
        ratio = np.asarray(q) / threshold
        severity = np.maximum(ratio.mean(axis=1) - .8, 0)
        spread = ratio.std(axis=1)
        continuity = np.array([severity[max(0, t-2):t+1].mean() for t in range(len(q))])
        weights = 1 + severity + spread + continuity
    else:
        raise ValueError(mode)
    return max(remaining, 0.) * weights / weights.sum()


def event_steps(event, tail):
    if tail < 0:
        raise ValueError('Negative tail')
    n = (pd.Timestamp(event['end']) - pd.Timestamp(event['start'])).days + 1 + tail
    if n <= tail:
        raise ValueError('Invalid event interval')
    return n


def select_factor(samples, low, high):
    candidates = [(abs(p - (low + high) / 2), f) for f, p in samples if low <= p <= high]
    return min(candidates)[1] if candidates else None


def audit_calibration(params, sources):
    rows = []
    for k, name in enumerate(NAMES):
        p = params[name]
        if not 0 < p['runoff_coeff'] <= 1 or not 0 <= p['recession'] < 1:
            raise ValueError(f'Invalid runoff parameters: {name}')
        source = [(station, s) for station, s in sources.items() if s.get('window') == name]
        station, s = source[0] if source else ('unknown', {})
        rows.append(dict(window=name, source_station=station,
                         source_area_km2=s.get('area_km2'), target_incremental_area_km2=float(AREAS[k]),
                         source_matches=int(s.get('training_common_dates', 0)),
                         runoff_coeff=float(p['runoff_coeff']), recession=float(p['recession']),
                         status='exploratory_parameter_transfer',
                         station_topology_verified=False))
    return dict(publication_ready=False, parameters=rows, blockers=[
        'Cumulative-area calibration does not validate incremental-area parameter transfer.',
        'Station-to-catchment topology and independent local-flow validation are unverified.',
        'Storage curves/routing are approximations; downstream threshold is a research setting.'])


def fit_variability(forcing, cutoff='2015-12-31'):
    frame = forcing[(forcing['time'] <= pd.Timestamp(cutoff)) &
                    (forcing['time'] >= pd.Timestamp('2007-01-01'))].pivot(
        index='time', columns='window', values='precip_mm').sort_index().reindex(columns=NAMES)
    if len(frame) < 40 or frame.isna().any().any():
        raise ValueError('Incomplete training windows')
    # Do NOT connect October to next May as adjacent daily AR pairs.
    adjacent = np.diff(frame.index.values).astype('timedelta64[D]').astype(int) == 1
    x = np.log1p(frame.to_numpy(float))
    a, b = x[:-1][adjacent], x[1:][adjacent]
    if len(a) < 30:
        raise ValueError('Too few contiguous daily training pairs')
    ac, bc = a-a.mean(axis=0), b-b.mean(axis=0)
    rho = np.clip((ac*bc).sum(axis=0) / np.maximum((ac*ac).sum(axis=0), 1e-12), 0, .95)
    cov = np.cov(b-a*rho, rowvar=False) * .35**2
    ev, u = np.linalg.eigh(cov)
    cov = (u * np.maximum(ev, 1e-8)) @ u.T
    return rho, cov


def scenario_rain(base, model, count, seed):
    base = np.asarray(base, dtype=float)
    if count < 2 or not np.isfinite(base).all() or np.any(base < 0):
        raise ValueError('Invalid precipitation/scenario count')
    rho, cov = model
    stationary = cov / (1 - np.outer(rho, rho))
    rng = np.random.default_rng(seed)
    eta = rng.multivariate_normal(np.zeros(len(rho)), stationary, size=count-1).T
    chol = np.linalg.cholesky(cov)
    out = np.repeat(np.asarray(base)[:, :, None], count, axis=2)
    for t in range(base.shape[1]):
        eta = rho[:, None] * eta + chol @ rng.normal(size=(len(rho), count-1))
        out[:, t, 1:] *= np.exp(eta - .5*np.diag(stationary)[:, None])
    return out


def rainfall_matrix(forcing, start, steps):
    dates = pd.date_range(start, periods=steps)
    frame = forcing[forcing.time.isin(dates)].pivot(index='time', columns='window', values='precip_mm')
    frame = frame.reindex(index=dates, columns=NAMES)
    x = frame.to_numpy(float).T
    if not np.isfinite(x).all() or np.any(x < 0) or np.any(x > 10000):
        raise ValueError(f'Invalid or missing rainfall for {start}, {steps} days; do not zero-fill')
    return x.copy()


def runoff(rain, parameters, initial=None):
    rain = np.asarray(rain, dtype=float)
    q = np.zeros_like(rain)
    prev = np.zeros((5, rain.shape[2])) if initial is None else np.repeat(initial[:, None], rain.shape[2], axis=1)
    c = np.array([parameters[n]['runoff_coeff'] for n in NAMES])[:, None]
    r = np.array([parameters[n]['recession'] for n in NAMES])[:, None]
    for t in range(rain.shape[1]):
        prev = r*prev + (1-r)*c*rain[:, t]*AREAS[:, None]*1000/86400
        q[:, t] = prev
    return q


def event_cube(forcing, params, model, event, tail, factor, count, seed):
    if not np.isfinite(factor) or factor < 1:
        raise ValueError('Stress factor must be finite and >=1')
    start = pd.Timestamp(event['start'])
    antecedent = rainfall_matrix(forcing, start-pd.Timedelta(days=30), 30)
    initial_q = runoff(antecedent[:, :, None], params)[:, -1, 0]
    base = rainfall_matrix(forcing, start, event_steps(event, tail))
    # Stress the event dates only; retain the original recession-tail rainfall.
    base[:, :event_steps(event, 0)] *= factor
    return runoff(scenario_rain(base, model, count, seed), params, initial_q)


def _route(router, flow, state):
    if state is None:
        state = np.broadcast_to(flow, (router.n_subreaches, 2, len(flow))).copy()
    for j in range(router.n_subreaches):
        out = router.C0*flow + router.C1*state[j, 0] + router.C2*state[j, 1]
        state[j, 0], state[j, 1] = flow, out
        flow = out
    return flow, state


def simulate(network, initial, local, plan):
    """Vectorized over scenarios, explicit physical curtailment and emergency spill.

    Prefix replay preserves reach memory, storage AND previous actual release.
    Short reaches retain legacy within-daily-step transfer, stated in the protocol.
    """
    initial, local, plan = [np.asarray(x, dtype=float) for x in (initial, local, plan)]
    if initial.shape != (5,) or not np.isfinite(initial).all():
        raise ValueError('Invalid initial storage')
    if local.ndim != 3 or plan.ndim != 2 or local.shape[:2] != plan.shape or local.shape[0] != 5:
        raise ValueError('Inflow/plan dimensions do not agree')
    if not np.isfinite(local).all() or not np.isfinite(plan).all() or np.any(local < 0):
        raise ValueError('Non-finite or negative input')
    pars = [network.reservoirs[n] for n in NAMES]
    minimum = np.array([p.S_min for p in pars])
    maximum = np.array([p.S_max for p in pars])
    if np.any(initial < minimum) or np.any(initial > maximum):
        raise ValueError('Initial storage outside physical range')
    storage, release, controlled, forced, total_in = [np.zeros_like(local) for _ in range(5)]
    T, ns = local.shape[1:]
    state = np.repeat(initial[:, None], ns, axis=1)
    # Explicit common initial-release assumption, not chosen from the first candidate.
    previous = np.array([p.min_release for p in pars])[:, None] * np.ones((5, ns))
    reaches = [(NAMES.index(r.upstream), NAMES.index(r.downstream), MuskingumRouter(r.K, r.x, 24.)) for r in network.reaches]
    route_states = [None]*len(reaches)
    cp = list(network.control_points.values())[0]
    cp_router = MuskingumRouter(cp['K'], cp.get('x', .2), 24.) if cp.get('K', 0) > 0 else None
    cp_state = None
    q = np.zeros((T, ns))
    violation = 0.
    violations_by_time = np.zeros(T)
    for t in range(T):
        for k, p in enumerate(pars):
            inc = local[k, t].copy()
            for j, (up, down, router) in enumerate(reaches):
                if down == k:
                    routed, route_states[j] = _route(router, release[up, t], route_states[j])
                    inc += routed
            command = np.clip(plan[k, t], p.min_release, p.max_release)
            actual = np.clip(command, previous[k]-p.max_ramp, previous[k]+p.max_ramp)
            actual = np.clip(actual, p.min_release, p.max_release)
            available = np.maximum(inc+(state[k]-p.S_min)/.000864, 0.)
            actual = np.minimum(actual, available)
            raw = state[k]+(inc-actual)*.000864
            overflow = np.maximum(raw-p.S_max, 0)/.000864
            state[k] = raw-overflow*.000864
            step_violation = max(float(np.maximum(p.min_release-actual, 0).max()/p.max_release),
                            float(np.maximum(np.abs(actual-previous[k])-p.max_ramp, 0).max()/p.max_release),
                            float(np.maximum(actual+overflow-p.max_release, 0).max()/p.max_release))
            violations_by_time[t] = max(violations_by_time[t], step_violation)
            violation = max(violation, step_violation)
            total_in[k, t], storage[k, t] = inc, state[k]
            controlled[k, t], forced[k, t], release[k, t] = actual, overflow, actual+overflow
            previous[k] = actual
        downstream = release[NAMES.index(cp.get('reservoir', 'TGD')), t]+float(cp.get('local_inflow', 0.))
        if cp_router is not None:
            downstream, cp_state = _route(cp_router, downstream, cp_state)
        q[t] = downstream
    return SimpleNamespace(storage=storage, release=release, controlled=controlled, forced=forced,
                           inflow=total_in, q=q, physical_violation=violation,
                           violations_by_time=violations_by_time)


def metrics(network, result, threshold, start=0):
    pars = [network.reservoirs[n] for n in NAMES]
    q = result.q[start:, 1:]  # member 0 is the historical centre, not a random draw.
    loss = np.maximum(q/threshold-1, 0).mean(axis=0)
    target = np.array([p.S_flood for p in pars])[:, None]
    forced = result.forced[:, start:, 1:].sum(axis=(0, 1))*.000864
    deficit = np.maximum(target-result.storage[:, -1, 1:], 0).sum(axis=0)
    initial_release = np.array([p.min_release for p in pars])[:, None, None]
    initial_release = np.repeat(initial_release, result.controlled.shape[2], axis=2)
    change = np.diff(result.controlled, axis=1, prepend=initial_release)
    ramp = np.abs(change[:, start:, 1:]) / np.array([p.max_release for p in pars])[:, None, None]
    n = q.shape[1]
    successes = int(np.any(q > threshold, axis=0).sum())
    # Exact Clopper-Pearson interval; reports sampling uncertainty, not a guarantee.
    from scipy.stats import beta
    ci_low = 0. if successes == 0 else float(beta.ppf(.025, successes, n-successes+1))
    ci_high = 1. if successes == n else float(beta.ppf(.975, successes+1, n-successes))
    return dict(f1=cvar(loss), f2=float(np.mean(forced+deficit)), f3=float(ramp.mean()),
                exceedance_probability_any=successes/n, probability_ci_low=ci_low, probability_ci_high=ci_high,
                max_q_safe_ratio=float(q.max()/threshold), minimum_safety_margin_m3s=float(threshold-q.max()),
                max_control_q_m3s=float(q.max()), expected_forced_spill_1e8m3=float(forced.mean()),
                physical_violation=float(result.physical_violation), n_random_scenarios=n)


def search_metrics(network, result, threshold, start=0):
    """Same raw objectives without confidence interval overhead inside optimizer."""
    pars = [network.reservoirs[n] for n in NAMES]
    q = result.q[start:, 1:]
    f1 = cvar(np.maximum(q/threshold-1, 0).mean(axis=0))
    target = np.array([p.S_flood for p in pars])[:, None]
    f2 = np.mean(result.forced[:, start:, 1:].sum(axis=(0, 1))*.000864 +
                 np.maximum(target-result.storage[:, -1, 1:], 0).sum(axis=0))
    previous = np.array([p.min_release for p in pars])[:, None, None]
    previous = np.repeat(previous, result.controlled.shape[2], axis=2)
    ramp = np.diff(result.controlled, axis=1, prepend=previous)[:, start:, 1:]
    f3 = np.mean(np.abs(ramp)/np.array([p.max_release for p in pars])[:, None, None])
    return np.array([f1, f2, f3])


def benchmark(network, cube):
    pars = [network.reservoirs[n] for n in NAMES]
    schedule = np.cumsum(cube.mean(axis=2), axis=0)
    for t in range(1, schedule.shape[1]):
        schedule[:, t] = .65*schedule[:, t-1]+.35*schedule[:, t]
    return np.clip(schedule, np.array([p.min_release for p in pars])[:, None],
                   np.array([p.max_release for p in pars])[:, None])


def problem_for(network, cube, epsilon, event_id, search_guidance=False):
    pars = [network.reservoirs[n] for n in NAMES]
    bounds = np.array([[[p.min_release, p.max_release]]*cube.shape[1] for p in pars])
    return PublicationProblem(n_res=5, T=cube.shape[1], dt=24., n_scen=cube.shape[2],
        bounds=bounds, initial_storage=np.array([p.S_flood for p in pars]),
        inflow_scenarios=cube, target_storage=np.array([p.S_flood for p in pars]),
        risk_budget=None if epsilon is None or not search_guidance else np.tile(epsilon, (5, 1)),
        release_reference=benchmark(network, cube), safe_Q=np.array([SAFE_Q]), event_id=event_id)


def score_objectives(raw, violation):
    # Feasible scores <1; ALL infeasible scores >=2. Report raw f, never this score.
    # Among infeasible solutions use scalar normalized violation first.
    bounded = raw/(1+raw)
    return bounded if violation <= 1e-10 else np.full(3, 2+violation)


def evaluate_prefix(network, initial, cube, prefix, plan, epsilon, threshold=SAFE_Q, margin=.05):
    origin = prefix.shape[1]
    full_plan = np.concatenate([prefix, plan], axis=1)
    used = cube[:, :full_plan.shape[1]].copy()
    # Assimilate only the already realized history, not future noisy members.
    used[:, :origin] = used[:, :origin, :1]
    result = simulate(network, initial, used, full_plan)
    raw = search_metrics(network, result, threshold, origin)
    chance = 0. if epsilon is None else float(np.maximum(
        chance_envelope(result.q[origin:, 1:], threshold, margin).mean(axis=1)-epsilon, 0).max())
    # Historical failures remain in full-event reporting, not future-plan ranking.
    violation = max(chance, float(result.violations_by_time[origin:].max()))
    return score_objectives(raw, violation), violation <= 1e-10, dict(raw=raw, violation=violation, result=result)
