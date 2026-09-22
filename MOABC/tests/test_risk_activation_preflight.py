"""Behavior tests for the risk-activation preflight."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from preflight_risk_activation import (  # noqa: E402
    control_risk_metrics,
    summarize_activation,
)


def test_control_risk_metrics_detects_scenario_exceedance_and_forced_spill():
    # time x scenario; two of four scenarios exceed 100 at least once.
    control_q = np.array([
        [90.0, 101.0, 80.0, 100.0],
        [95.0, 99.0, 120.0, 100.0],
    ])
    forced_spill = np.full((2, 2, 4), 1.0)

    metrics = control_risk_metrics(
        control_q, forced_spill, safe_q=100.0, dt_hours=24.0,
    )

    assert metrics["max_control_q_m3s"] == 120.0
    assert metrics["max_q_safe_ratio"] == 1.2
    assert metrics["minimum_safety_margin_m3s"] == -20.0
    assert metrics["exceedance_probability_any"] == 0.5
    assert metrics["risk_activated"] is True
    # 2 reservoirs x 2 times x 1 m3/s x 86400 s / 1e8, averaged over scenarios.
    assert abs(metrics["expected_forced_spill_1e8m3"] - 0.003456) < 1e-12


def test_summarize_activation_reports_first_factor_per_event_and_global():
    rows = pd.DataFrame([
        {"event_id": "P01", "stress_factor": 1.0, "risk_activated": False},
        {"event_id": "P01", "stress_factor": 2.0, "risk_activated": True},
        {"event_id": "P01", "stress_factor": 3.0, "risk_activated": True},
        {"event_id": "P02", "stress_factor": 1.0, "risk_activated": False},
        {"event_id": "P02", "stress_factor": 2.0, "risk_activated": False},
        {"event_id": "P02", "stress_factor": 3.0, "risk_activated": True},
        {"event_id": "P03", "stress_factor": 1.0, "risk_activated": False},
        {"event_id": "P03", "stress_factor": 2.0, "risk_activated": False},
        {"event_id": "P03", "stress_factor": 3.0, "risk_activated": False},
    ])

    summary = summarize_activation(rows)

    assert summary["first_activation_factor_by_event"] == {
        "P01": 2.0,
        "P02": 3.0,
        "P03": None,
    }
    assert summary["lowest_activation_factor_any_event"] == 2.0
    assert summary["events_activated"] == 2
    assert summary["events_scanned"] == 3


if __name__ == "__main__":
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
