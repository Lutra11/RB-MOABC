# -*- coding: utf-8 -*-
"""Experiment pipelines: legacy v11, risk-aware R2, artifact store, instrumented solver."""
from .legacy_v11 import (
    NAMES, AREAS, SAFE_Q, cvar, chance_envelope, allocate_budget,
    audit_calibration, event_steps, fit_variability, rainfall_matrix,
    runoff, scenario_rain, score_objectives, seed_for,
)
from .risk_r2 import (
    VERSION, STRESS_PATTERNS, project_release_plan, plan_diagnostics,
    problem_for_r2, simulate_r2, apply_stress_pattern,
    tail_peak_ratio_cvar95, select_pressure_factor, lag_aware_guidance,
)
from .artifact_store import ArtifactStore, canonical_json
from .instrumented_solver import InstrumentedRunContext, run

__all__ = [
    'NAMES', 'AREAS', 'SAFE_Q', 'cvar', 'chance_envelope', 'allocate_budget',
    'audit_calibration', 'event_steps', 'fit_variability', 'rainfall_matrix',
    'runoff', 'scenario_rain', 'score_objectives', 'seed_for',
    'VERSION', 'STRESS_PATTERNS', 'project_release_plan', 'plan_diagnostics',
    'problem_for_r2', 'simulate_r2', 'apply_stress_pattern',
    'tail_peak_ratio_cvar95', 'select_pressure_factor', 'lag_aware_guidance',
    'ArtifactStore', 'canonical_json',
    'InstrumentedRunContext', 'run',
]
