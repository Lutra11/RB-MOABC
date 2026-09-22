# -*- coding: utf-8 -*-
"""
Baseline algorithms for algorithm comparison.

All baselines share the SAME RunContext, Archive, evaluate(), constraint
projection, and scenario set as MOABC — ensuring fair comparison.

Available baselines:
  - NSGA-II  : Deb et al. (2002), non-dominated sorting + crowding
  - NSGA-III : Deb & Jain (2014), reference-point-based niche preservation
  - MOEA/D   : Zhang & Li (2007), Tchebycheff decomposition
  - MOPSO    : Coello et al. (2004), particle swarm with external archive
  - SPEA2    : Zitzler et al. (2001), strength Pareto + k-NN density
"""
from .nsga2 import solve as solve_nsga2
from .nsga3 import solve as solve_nsga3
from .moead import solve as solve_moead
from .mopso import solve as solve_mopso
from .spea2 import solve as solve_spea2

__all__ = [
    'solve_nsga2', 'solve_nsga3', 'solve_moead',
    'solve_mopso', 'solve_spea2',
]
