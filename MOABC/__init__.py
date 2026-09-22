# -*- coding: utf-8 -*-
"""
MOABC - Risk-Budgeted Multi-Reservoir Flood Operation
======================================================

Package layout:
  MOABC/
  ├── core/           - MOABC algorithm (pareto, solver, operators, problem, init, metrics, rolling)
  ├── simulation/     - Reservoir simulator (water balance, Muskingum, constraints)
  ├── baselines/      - NSGA-II/III, MOEA/D, MOPSO, and SPEA2
  ├── risk/           - Dynamic risk-budget allocation and CVaR
  ├── data/           - GPM, CDR2 catalogue, and forcing-window utilities
  ├── forecast/       - Ensemble forecast calibration (EMOS, ECC)
  ├── experiments/    - Publication pipelines (legacy v11, risk-aware R2, artifact store)
  └── cli.py          - Command-line entry point
"""
from .core.pareto import dominates, ranks, crowding, Archive, environmental_selection
from .core.solver import Config, RunContext, solve, run
from .core.problem import ReservoirSchedulingProblem
from .core.operators import move, reconstruct, crossover
from .core.initialization import initialize

__version__ = '0.2.0'
