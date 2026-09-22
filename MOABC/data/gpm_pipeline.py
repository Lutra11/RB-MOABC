"""GPM precipitation to inflow utilities.

This module implements the reproducible data path used by the current paper:

    GPM half-hourly rate -> window-mean precipitation -> daily precipitation
    -> simple lumped rainfall-runoff conversion -> inflow scenarios

The configured rectangular windows are explicitly treated as rainfall forcing
windows, not as exact hydrological catchments.  A future GeoJSON boundary can
be supplied to :func:`extract_window_precipitation` without changing the rest
of the pipeline.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Iterable, Mapping, Optional

import numpy as np
import pandas as pd


DATE_RE = re.compile(r"(\d{8})$")


@dataclass(frozen=True)
class RainfallWindow:
    name: str
    lat_min: float
    lat_max: float
    lon_min: float
    lon_max: float
    area_km2: Optional[float] = None
    approximate: bool = True

    @property
    def bbox_area_km2(self) -> float:
        # Area of the forcing window, not a claimed basin area.
        lat = np.deg2rad((self.lat_min + self.lat_max) / 2.0)
        return ((self.lat_max - self.lat_min) * 111.32
                * (self.lon_max - self.lon_min) * 111.32 * np.cos(lat))

    @property
    def forcing_area_km2(self) -> float:
        return float(self.area_km2 or self.bbox_area_km2)


def discover_gpm_files(directory: str | Path, start: str | None = None,
                       end: str | None = None) -> list[Path]:
    """Return date-sorted GPM files whose dates fall within the interval."""
    start_dt = pd.Timestamp(start) if start else None
    end_dt = pd.Timestamp(end) if end else None
    found: list[tuple[pd.Timestamp, Path]] = []
    candidates = list(Path(directory).glob("*.nc")) + list(Path(directory).glob("*.nc4"))
    for path in candidates:
        match = DATE_RE.search(path.stem)
        if not match:
            continue
        date = pd.to_datetime(match.group(1), format="%Y%m%d")
        if start_dt is not None and date < start_dt:
            continue
        if end_dt is not None and date > end_dt:
            continue
        found.append((date, path))
    return [path for _, path in sorted(found)]


def _window_mask(lat: np.ndarray, lon: np.ndarray, window: RainfallWindow) -> np.ndarray:
    return ((lat >= window.lat_min) & (lat <= window.lat_max)[:, None]
            if lat.ndim == 1 else np.ones_like(lat, dtype=bool))


def extract_window_precipitation(
    files: Iterable[str | Path],
    windows: Iterable[RainfallWindow],
) -> pd.DataFrame:
    """Extract daily window precipitation totals in millimetres.

    GPM values are precipitation rates in mm/hr. Each file contains 48
    half-hour records. Longitude/latitude grid-cell area is approximated by
    cosine latitude weighting. Missing files are not filled silently.
    """
    import xarray as xr

    windows = list(windows)
    records: list[dict[str, float | str]] = []
    skipped = 0
    for file in files:
        file = Path(file)
        match = DATE_RE.search(file.stem)
        if not match:
            continue
        date = pd.to_datetime(match.group(1), format="%Y%m%d")
        try:
            ds = xr.open_dataset(file, decode_times=False)
        except Exception:
            skipped += 1
            continue
        with ds:
            if "precipitation" not in ds:
                skipped += 1
                continue
            da = ds["precipitation"]
            lat = np.asarray(ds["lat"].values, dtype=float)
            lon = np.asarray(ds["lon"].values, dtype=float)
            if len(lat) == 0 or len(lon) == 0 or lat[0] == 0:
                skipped += 1
                continue
            values = np.asarray(da.values, dtype=float)
            if values.ndim != 3:
                raise ValueError(f"{file}: expected (time, lon, lat), got {values.shape}")
            # Convert to (time, lat, lon) for intuitive masking.
            values = np.transpose(values, (0, 2, 1))
            for window in windows:
                lat_mask = (lat >= window.lat_min) & (lat <= window.lat_max)
                lon_mask = (lon >= window.lon_min) & (lon <= window.lon_max)
                if not lat_mask.any() or not lon_mask.any():
                    skipped += 1
                    continue
                sub = values[:, lat_mask][:, :, lon_mask]
                # Cosine-latitude weighted spatial mean.
                # weights has shape (n_lat,); broadcast to (n_lat, n_lon)
                # so that nansum in numerator and denominator are consistent.
                w_1d = np.cos(np.deg2rad(lat[lat_mask]))
                w_2d = np.broadcast_to(w_1d[:, None], sub.shape[1:])
                masked = np.where(np.isnan(sub), 0.0, sub)
                w_masked = np.where(np.isnan(sub), 0.0, w_2d)
                spatial_rate = (np.nansum(masked * w_masked, axis=(1, 2))
                                / np.nansum(w_masked, axis=(1, 2)).clip(min=1e-12))
                units = str(da.attrs.get("units", "")).lower().replace(" ", "")
                if len(spatial_rate) == 1 and units in {"mm/day", "mmd-1"}:
                    daily_mm = float(spatial_rate[0])
                elif len(spatial_rate) == 48 and ("mm/hr" in units or "mmh-1" in units):
                    daily_mm = float(np.nansum(spatial_rate) * 0.5)
                else:
                    raise ValueError(
                        f"{file}: unsupported temporal structure/time units "
                        f"({len(spatial_rate)} records, units={da.attrs.get('units')!r})"
                    )
                records.append({"date": date, "window": window.name,
                                "precip_mm": max(0.0, daily_mm),
                                "window_area_km2": window.forcing_area_km2,
                                "approximate_window": window.approximate})
    if not records:
        return pd.DataFrame(columns=["date", "window", "precip_mm",
                                     "window_area_km2", "approximate_window"])
    if skipped > 0:
        import warnings
        warnings.warn(f"extract_window_precipitation: skipped {skipped} corrupted/empty files")
    return pd.DataFrame(records).sort_values(["date", "window"]).reset_index(drop=True)


def extract_window_precipitation_3h(
    files: Iterable[str | Path],
    windows: Iterable[RainfallWindow],
    minimum_valid_fraction: float = 0.95,
) -> pd.DataFrame:
    """Extract genuine 3-hour precipitation totals from half-hourly IMERG.

    Each daily file must contain 48 half-hourly precipitation-rate fields.
    Six consecutive fields are integrated into one 3-hour total.  The
    function rejects fill-valued or incomplete files instead of emitting an
    all-missing placeholder.  Rectangles are labelled ``forcing_window`` to
    prevent their interpretation as delineated catchments.
    """
    import xarray as xr

    windows = list(windows)
    records: list[dict] = []
    errors: list[str] = []
    for raw_file in files:
        file = Path(raw_file)
        match = DATE_RE.search(file.stem)
        if not match:
            continue
        date = pd.to_datetime(match.group(1), format="%Y%m%d")
        try:
            ds = xr.open_dataset(file, decode_times=False)
            with ds:
                if "precipitation" not in ds:
                    raise ValueError("missing precipitation variable")
                values = np.asarray(ds["precipitation"].values, dtype=float)
                lat = np.asarray(ds["lat"].values, dtype=float)
                lon = np.asarray(ds["lon"].values, dtype=float)
                if values.shape != (48, len(lon), len(lat)):
                    raise ValueError(f"expected (48, lon, lat), got {values.shape}")
                values = np.transpose(values, (0, 2, 1))
                values[(~np.isfinite(values)) | (np.abs(values) > 1e20)] = np.nan
                if np.isfinite(values).mean() < minimum_valid_fraction:
                    raise ValueError("insufficient valid precipitation cells")
                for window in windows:
                    lat_mask = (lat >= window.lat_min) & (lat <= window.lat_max)
                    lon_mask = (lon >= window.lon_min) & (lon <= window.lon_max)
                    if not lat_mask.any() or not lon_mask.any():
                        raise ValueError(f"forcing window outside grid: {window.name}")
                    sub = values[:, lat_mask][:, :, lon_mask]
                    weights = np.broadcast_to(
                        np.cos(np.deg2rad(lat[lat_mask]))[:, None], sub.shape[1:]
                    )
                    valid = np.isfinite(sub)
                    spatial_rate = np.sum(np.where(valid, sub, 0.0) * weights, axis=(1, 2)) / np.maximum(
                        np.sum(np.where(valid, weights, 0.0), axis=(1, 2)), 1e-12
                    )
                    totals = spatial_rate.reshape(8, 6).sum(axis=1) * 0.5
                    for block, total in enumerate(totals):
                        records.append({
                            "time": date + pd.Timedelta(hours=3 * block),
                            "date": date,
                            "step_3h": block,
                            "window": window.name,
                            "forcing_window": True,
                            "precip_mm": max(float(total), 0.0),
                            "forcing_area_km2": window.forcing_area_km2,
                            "approximate_window": window.approximate,
                            "source_file": file.name,
                        })
        except Exception as exc:
            errors.append(f"{file.name}: {exc}")
    if errors:
        preview = "; ".join(errors[:5])
        raise ValueError(f"Rejected {len(errors)} invalid GPM files. First errors: {preview}")
    columns = ["time", "date", "step_3h", "window", "forcing_window",
               "precip_mm", "forcing_area_km2", "approximate_window", "source_file"]
    return pd.DataFrame(records, columns=columns).sort_values(["time", "window"]).reset_index(drop=True)


@dataclass
class PrecipitationScenarioModel:
    """Correlated, temporally persistent multiplicative precipitation errors.

    This is a scenario generator fitted from IMERG temporal variability, not
    a meteorological ensemble forecast.  The distinction is persisted in
    ``provenance`` and should also be retained in tables and figure captions.
    """
    windows: tuple[str, ...]
    rho: np.ndarray
    innovation_cov: np.ndarray
    uncertainty_scale: float = 0.35
    provenance: str = "IMERG-derived stochastic precipitation scenarios"

    @classmethod
    def fit(cls, precipitation_3h: pd.DataFrame, windows: Iterable[str],
            uncertainty_scale: float = 0.35) -> "PrecipitationScenarioModel":
        names = tuple(windows)
        frame = (precipitation_3h.pivot_table(index="time", columns="window",
                                              values="precip_mm", aggfunc="mean")
                 .sort_index().reindex(columns=names).fillna(0.0))
        if len(frame) < 40:
            raise ValueError("at least 40 three-hour steps are required to fit scenarios")
        x = np.log1p(frame.to_numpy(dtype=float))
        rho = np.zeros(len(names))
        residual = np.zeros((len(x) - 1, len(names)))
        for k in range(len(names)):
            a, b = x[:-1, k], x[1:, k]
            denom = float(np.dot(a - a.mean(), a - a.mean()))
            rho[k] = 0.0 if denom <= 1e-12 else np.dot(a - a.mean(), b - b.mean()) / denom
            rho[k] = np.clip(rho[k], 0.0, 0.95)
            residual[:, k] = b - rho[k] * a
        scale = np.std(residual, axis=0, ddof=1)
        scale = np.maximum(scale, 0.05)
        corr = np.corrcoef(residual, rowvar=False)
        corr = np.nan_to_num(corr, nan=0.0)
        np.fill_diagonal(corr, 1.0)
        # Eigenvalue clipping makes the empirical matrix safe for simulation.
        eigval, eigvec = np.linalg.eigh((corr + corr.T) / 2.0)
        corr = eigvec @ np.diag(np.maximum(eigval, 1e-6)) @ eigvec.T
        cov = corr * np.outer(scale, scale) * uncertainty_scale ** 2
        return cls(names, rho, cov, uncertainty_scale)

    def generate(self, base_precip: np.ndarray, n_scen: int, seed: int) -> np.ndarray:
        """Return (n_window, T, n_scen) non-negative scenarios."""
        base = np.maximum(np.asarray(base_precip, dtype=float), 0.0)
        if base.shape[0] != len(self.windows) or n_scen < 2:
            raise ValueError("base shape or n_scen is invalid")
        rng = np.random.default_rng(seed)
        n_window, horizon = base.shape
        out = np.zeros((n_window, horizon, n_scen))
        # Member 0 is the unperturbed historical forcing and serves as the
        # realized path for retrospective rolling-state updates.  Remaining
        # members represent decision-time uncertainty around that forcing.
        out[:, :, 0] = base
        chol = np.linalg.cholesky(self.innovation_cov + np.eye(n_window) * 1e-10)
        stationary_var = np.diag(self.innovation_cov) / np.maximum(1.0 - self.rho ** 2, 1e-6)
        for s in range(1, n_scen):
            eta = np.zeros(n_window)
            for t in range(horizon):
                eta = self.rho * eta + chol @ rng.normal(size=n_window)
                multiplier = np.exp(eta - 0.5 * stationary_var)
                out[:, t, s] = base[:, t] * multiplier
        return out


def simulate_lumped_runoff_timestep(
    precip_mm: np.ndarray, runoff_coeff: float, recession_daily: float,
    area_km2: float, dt_hours: float = 3.0, initial_q: float = 0.0,
) -> np.ndarray:
    """Run the calibrated lumped response at a sub-daily time step."""
    if not 0 < runoff_coeff <= 1 or not 0 <= recession_daily < 1:
        raise ValueError("invalid runoff or recession parameter")
    if area_km2 <= 0 or dt_hours <= 0:
        raise ValueError("area and time step must be positive")
    recession = recession_daily ** (dt_hours / 24.0)
    p = np.maximum(np.asarray(precip_mm, dtype=float), 0.0)
    q = np.zeros_like(p)
    previous = float(initial_q)
    for i, value in enumerate(p):
        pulse = runoff_coeff * value * area_km2 * 1000.0 / (dt_hours * 3600.0)
        previous = recession * previous + (1.0 - recession) * pulse
        q[i] = previous
    return q


def identify_rainfall_events(
    daily: pd.DataFrame,
    dry_gap_hours: int = 12,
    threshold_quantile: float = 0.90,
    min_total_mm: float = 20.0,
) -> pd.DataFrame:
    """Identify candidate events from daily window precipitation.

    The event catalogue is deliberately data-driven. It must be intersected
    with CDR² valid dates before being used as a discharge-validation catalogue.
    """
    if daily.empty:
        return pd.DataFrame()
    rows = []
    for window, group in daily.groupby("window"):
        s = group.set_index("date")["precip_mm"].sort_index().asfreq("D", fill_value=0.0)
        threshold = max(float(s.quantile(threshold_quantile)), min_total_mm / 3.0)
        wet = s >= threshold
        # A daily product cannot resolve 12 h; dry_gap_hours is retained in the
        # API and represented conservatively as one dry day for daily totals.
        groups = (wet.astype(int).diff().fillna(wet.astype(int)).ne(0)).cumsum()
        for _, idx in wet[wet].groupby(groups[wet]).groups.items():
            part = s.loc[idx]
            if part.sum() < min_total_mm:
                continue
            rows.append({"window": window, "start": part.index.min(),
                         "end": part.index.max(), "duration_days": len(part),
                         "total_precip_mm": float(part.sum()),
                         "peak_precip_mm": float(part.max())})
    if not rows:
        return pd.DataFrame(columns=["window", "start", "end",
                                     "duration_days", "total_precip_mm",
                                     "peak_precip_mm"])
    return pd.DataFrame(rows).sort_values(["start", "window"]).reset_index(drop=True)


def rainfall_to_inflow(
    daily: pd.DataFrame,
    windows: Mapping[str, RainfallWindow],
    runoff_coeff: float = 0.35,
    recession: float = 0.70,
    allow_approximate_area: bool = False,
) -> pd.DataFrame:
    """Convert daily rainfall to a transparent lumped inflow estimate (m3/s).

    This is a controlled-case hydrological conversion, not a replacement for
    a calibrated rainfall-runoff model. Parameters must be fitted only on the
    training period when used for publication results.
    """
    if not 0 < runoff_coeff <= 1 or not 0 <= recession < 1:
        raise ValueError("runoff_coeff must be in (0,1], recession in [0,1)")
    if not allow_approximate_area:
        missing = [name for name, window in windows.items()
                   if window.area_km2 is None]
        if missing:
            raise ValueError(
                "Hydrological area is required for inflow conversion; "
                f"missing for {missing}. Supply calibrated/DEM areas or "
                "explicitly enable allow_approximate_area for debugging."
            )
    out = daily.copy()
    out["inflow_m3s"] = 0.0
    for name, group in out.groupby("window"):
        area = (windows[name].area_km2 if windows[name].area_km2 is not None
                else windows[name].bbox_area_km2)
        ordered = group.sort_values("date")
        q = simulate_lumped_runoff(
            ordered["precip_mm"].to_numpy(dtype=float),
            runoff_coeff=runoff_coeff,
            recession=recession,
            area_km2=area,
        )
        out.loc[ordered.index, "inflow_m3s"] = q
    return out.sort_values(["date", "window"]).reset_index(drop=True)


def simulate_lumped_runoff(
    precip_mm: np.ndarray,
    runoff_coeff: float,
    recession: float,
    area_km2: float,
    initial_q: float = 0.0,
) -> np.ndarray:
    """Simulate the shared daily lumped rainfall--runoff equation.

    The same stateful implementation is used by calibration, validation and
    bbox sensitivity analysis.  ``precip_mm`` must be a chronologically
    ordered, gap-free (or explicitly gap-filled) daily sequence.
    """
    if not 0 < runoff_coeff <= 1:
        raise ValueError("runoff_coeff must be in (0, 1]")
    if not 0 <= recession < 1:
        raise ValueError("recession must be in [0, 1)")
    if area_km2 <= 0:
        raise ValueError("area_km2 must be positive")
    p = np.maximum(np.asarray(precip_mm, dtype=float), 0.0)
    q = np.zeros_like(p, dtype=float)
    q_prev = float(initial_q)
    for i, value in enumerate(p):
        runoff_q = runoff_coeff * value * area_km2 * 1000.0 / 86400.0
        q_prev = recession * q_prev + (1.0 - recession) * runoff_q
        q[i] = q_prev
    return q


def simulate_lumped_runoff_with_dates(
    dates: Iterable[pd.Timestamp],
    precip_mm: np.ndarray,
    runoff_coeff: float,
    recession: float,
    area_km2: float,
    initial_q: float = 0.0,
) -> np.ndarray:
    """Simulate daily runoff while respecting gaps in the date sequence.

    The ordinary daily recurrence assumes adjacent observations.  The formal
    experiments use May--October GPM windows for multiple years, so the state
    must decay across missing non-flood-season days instead of carrying the
    previous October flow directly into the next May.
    """
    if not 0 < runoff_coeff <= 1:
        raise ValueError("runoff_coeff must be in (0, 1]")
    if not 0 <= recession < 1:
        raise ValueError("recession must be in [0, 1)")
    if area_km2 <= 0:
        raise ValueError("area_km2 must be positive")

    dates = pd.to_datetime(list(dates))
    p = np.maximum(np.asarray(precip_mm, dtype=float), 0.0)
    if len(dates) != len(p):
        raise ValueError("dates and precip_mm must have the same length")
    if len(p) == 0:
        return np.zeros(0, dtype=float)

    q = np.zeros_like(p, dtype=float)
    q_prev = float(initial_q)
    prev_date = None
    for i, (date, value) in enumerate(zip(dates, p)):
        if prev_date is not None:
            gap_days = max(int((date - prev_date).days), 1)
            if gap_days > 1:
                q_prev *= recession ** (gap_days - 1)
        runoff_q = runoff_coeff * value * area_km2 * 1000.0 / 86400.0
        q_prev = recession * q_prev + (1.0 - recession) * runoff_q
        q[i] = q_prev
        prev_date = date
    return q


def fit_lumped_runoff_parameters(
    rainfall: pd.Series,
    observed_flow: pd.Series,
    area_km2: float,
    runoff_grid: Iterable[float] = tuple(np.linspace(0.05, 0.95, 19)),
    recession_grid: Iterable[float] = tuple(np.linspace(0.10, 0.90, 17)),
) -> dict[str, float]:
    """Fit a transparent two-parameter runoff conversion on common dates.

    ``rainfall`` and ``observed_flow`` must be daily series indexed by date.
    Only the supplied common dates are used; no missing observed dates are
    interpolated. This is intended for training-period calibration with CDR²
    reference records, not for claiming a calibrated operational model.
    """
    if area_km2 <= 0:
        raise ValueError("area_km2 must be positive")
    joined = pd.concat([rainfall.rename("p"), observed_flow.rename("q")], axis=1)
    observed_mask = joined["q"].notna() & joined["p"].notna()
    if int(observed_mask.sum()) < 10:
        raise ValueError("at least 10 common rainfall-flow dates are required")
    best = None
    p = joined["p"].fillna(0.0).to_numpy(dtype=float)
    target = joined.loc[observed_mask, "q"].to_numpy(dtype=float)
    for coefficient in runoff_grid:
        for recession in recession_grid:
            simulated_full = simulate_lumped_runoff(
                p, coefficient, recession, area_km2
            )
            simulated = simulated_full[observed_mask.to_numpy()]
            # Log scale prevents high-flow days from completely dominating fit.
            error = np.mean((np.log1p(simulated) - np.log1p(np.maximum(target, 0.0))) ** 2)
            candidate = (float(error), float(coefficient), float(recession))
            if best is None or candidate[0] < best[0]:
                best = candidate
    assert best is not None
    return {"runoff_coeff": best[1], "recession": best[2],
            "training_common_dates": int(observed_mask.sum()),
            "log_rmse": float(np.sqrt(best[0]))}
