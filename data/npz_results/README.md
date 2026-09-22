# NPZ Experiment Results

This directory contains the key NumPy compressed (`.npz`) experiment output files
that support the results reported in the manuscript.

## Directory Structure

### `ablation/` — Ablation Study Results (60 files, 41.3 MB)

Each file stores the complete simulation state for one ablation experiment run:
**4 variants × 3 events (P01–P03) × 5 seeds (s00–s04) = 60 files**

| Variant | Description |
|---|---|
| `full_method_full` | Full method (complete framework) |
| `full_method_no_adaptive_operator` | Without adaptive operator learning |
| `full_method_no_lag_guidance` | Without routing-aware risk guidance |
| `full_method_no_warm_start` | Without rolling warm start |

### `algorithm_comparison_P01/` — Algorithm Comparison for Event P01 (60 files, 37.4 MB)

Each file stores the complete optimization and simulation output for one algorithm run on
event P01 (August 8–23, 2020):
**6 algorithms × 2 conditions (normal + matched stress) × 5 seeds (s00–s04) = 60 files**

| Algorithm | Label |
|---|---|
| MOABC | Multi-objective Artificial Bee Colony (proposed) |
| MOPSO | Multi-objective Particle Swarm Optimization |
| NSGA-II | Non-dominated Sorting Genetic Algorithm II |
| NSGA-III | Non-dominated Sorting Genetic Algorithm III |
| MOEA/D | Multi-objective Evolutionary Algorithm based on Decomposition |
| SPEA2 | Strength Pareto Evolutionary Algorithm 2 |

## NPZ File Contents

Each `.npz` file contains the following NumPy arrays:

| Array | Shape | Description |
|---|---|---|
| `schedule` | (5, T) | Planned release schedule for 5 reservoirs over T decision periods |
| `release` | (5, T, S) | Release decisions for 5 reservoirs × T periods × S scenarios |
| `controlled_release` | (5, T, S) | Controlled (non-spill) release |
| `forced_spill` | (5, T, S) | Mandatory spill above nominal capacity |
| `storage` | (5, T, S) | Reservoir storage states |
| `inflow` | (5, T, S) | Scenario inflow to each reservoir |
| `outlet_flow` | (T, S) | Total outlet flow at downstream control section |
| `risk_budget_executed` | (T,) | Executed risk budget per period |
| `plan_violation_by_time` | (T,) | Plan-bound violation per period |
| `operational_deviation_by_time` | (T,) | Operational ramp deviation per period |
| `hard_physical_violation_by_time` | (T,) | Hard physical constraint violation |
| `feasible_front` | (N, 3) | Pareto front objectives (f1, f2, f3) |
| `last_lag_guidance` | (5, 1) | Routing-aware guidance flag per reservoir |

Where:
- **5** = number of reservoirs (Wudongde, Baihetan, Xiluodu, Xiangjiaba, Three Gorges)
- **T** = number of decision periods (typically 17 for a 16-day horizon)
- **S** = number of scenarios (typically 513)
- **N** = number of Pareto front solutions

## Filename Convention

```
<P event>_<_condition>_<strategy_variant>_<algorithm>_s<seed>.npz
```

Example: `P01_matched_full_method_full_MOABC_s00.npz`
- **P01**: Flood event P01 (August 8–23, 2020)
- **matched**: Matched synthetic stress condition (vs. `normal` for historical forcing)
- **full_method_full**: Full method strategy
- **MOABC**: Algorithm used
- **s00**: Random seed 0

## Reproducing Results

To load and inspect a result file:

```python
import numpy as np

data = np.load("ablation/P01_matched_full_method_full_MOABC_s00.npz")
for key in data.files:
    print(f"{key}: {data[key].shape} {data[key].dtype}")
```

The aggregated CSV tables in `data/` (e.g., `table4-5_algorithm_comparison.csv`)
are derived from these npz files through the experiment analysis pipeline.
