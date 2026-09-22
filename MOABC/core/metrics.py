"""Publication-grade common-reference Pareto indicators.

All algorithms for one event must be evaluated against the same pooled
non-dominated reference set, normalization bounds and hypervolume reference
point.  Per-front references, used by the legacy v8 runner, are incomparable.
"""
from __future__ import annotations

import numpy as np


def nondominated(points: np.ndarray) -> np.ndarray:
    values = np.asarray(points, dtype=float)
    if values.ndim != 2 or len(values) == 0:
        return np.empty((0, values.shape[1] if values.ndim == 2 else 0))
    finite = values[np.all(np.isfinite(values), axis=1)]
    keep = np.ones(len(finite), dtype=bool)
    for i in range(len(finite)):
        if not keep[i]:
            continue
        dominated = np.all(finite <= finite[i] + 1e-12, axis=1) & np.any(
            finite < finite[i] - 1e-12, axis=1
        )
        dominated[i] = False
        if dominated.any():
            keep[i] = False
    return np.unique(finite[keep], axis=0)


def normalize_with_pool(fronts: list[np.ndarray]) -> tuple[list[np.ndarray], np.ndarray, np.ndarray]:
    pool = np.vstack([np.asarray(f, dtype=float) for f in fronts if len(f)])
    ideal = np.min(pool, axis=0)
    nadir = np.max(pool, axis=0)
    span = np.maximum(nadir - ideal, 1e-12)
    return [(np.asarray(f, dtype=float) - ideal) / span for f in fronts], ideal, nadir


def hypervolume_3d(points: np.ndarray, reference: np.ndarray | None = None) -> float:
    """Exact dominated hypervolume for a minimization front in three dimensions."""
    ref = np.asarray(reference if reference is not None else [1.1, 1.1, 1.1], dtype=float)
    p = nondominated(np.asarray(points, dtype=float))
    p = p[np.all(p < ref, axis=1)]
    if len(p) == 0:
        return 0.0

    def area_2d(yz: np.ndarray) -> float:
        ys = np.unique(yz[:, 0])
        area = 0.0
        for j, y in enumerate(ys):
            next_y = ys[j + 1] if j + 1 < len(ys) else ref[1]
            active = yz[yz[:, 0] <= y + 1e-12]
            if next_y > y and len(active):
                area += (next_y - y) * max(ref[2] - float(active[:, 1].min()), 0.0)
        return area

    xs = np.unique(p[:, 0])
    volume = 0.0
    for i, x in enumerate(xs):
        next_x = xs[i + 1] if i + 1 < len(xs) else ref[0]
        active = p[p[:, 0] <= x + 1e-12, 1:]
        if next_x > x and len(active):
            volume += (next_x - x) * area_2d(active)
    return float(volume)


def igd(front: np.ndarray, reference_front: np.ndarray) -> float:
    candidate = np.asarray(front, dtype=float)
    reference = np.asarray(reference_front, dtype=float)
    if len(candidate) == 0 or len(reference) == 0:
        return float("inf")
    distance = np.sqrt(((reference[:, None, :] - candidate[None, :, :]) ** 2).sum(axis=2))
    return float(distance.min(axis=1).mean())


def common_reference_indicators(
    fronts: list[np.ndarray], reference_point: np.ndarray | None = None,
) -> tuple[list[dict[str, float]], np.ndarray, np.ndarray, np.ndarray]:
    normalized, ideal, nadir = normalize_with_pool(fronts)
    reference_front = nondominated(np.vstack(normalized))
    metrics = [
        {"hv": hypervolume_3d(front, reference_point), "igd": igd(front, reference_front)}
        for front in normalized
    ]
    return metrics, reference_front, ideal, nadir
