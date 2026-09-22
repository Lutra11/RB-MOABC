# -*- coding: utf-8 -*-
"""Data utilities used by the canonical publication workflow."""
from .gpm_pipeline import (
    RainfallWindow,
    discover_gpm_files,
    extract_window_precipitation,
    extract_window_precipitation_3h,
    PrecipitationScenarioModel,
    simulate_lumped_runoff_timestep,
    identify_rainfall_events,
    rainfall_to_inflow,
    fit_lumped_runoff_parameters,
)
from .cdr2_catalog import build_station_catalog, read_dbf
from .bbox_sensitivity import make_window_sensitivity_variants
