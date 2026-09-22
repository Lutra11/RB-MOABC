"""Sensitivity variants for approximate GPM rainfall forcing windows."""
from __future__ import annotations

from dataclasses import replace
from typing import Iterable

from .gpm_pipeline import RainfallWindow


def make_window_sensitivity_variants(
    windows: Iterable[RainfallWindow],
    delta_deg: float = 0.2,
) -> dict[str, list[RainfallWindow]]:
    """Return original, contracted and expanded windows.

    No variant claims to be a hydrological catchment. The returned names are
    intended for the precipitation-forcing sensitivity experiment.
    """
    if delta_deg <= 0:
        raise ValueError("delta_deg must be positive")
    original = list(windows)
    def shifted(window: RainfallWindow, sign: float) -> RainfallWindow:
        return replace(
            window,
            lat_min=window.lat_min - sign * delta_deg,
            lat_max=window.lat_max + sign * delta_deg,
            lon_min=window.lon_min - sign * delta_deg,
            lon_max=window.lon_max + sign * delta_deg,
        )
    return {
        "original": original,
        "contracted": [shifted(w, -1.0) for w in original],
        "expanded": [shifted(w, 1.0) for w in original],
    }
