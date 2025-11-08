from __future__ import annotations
import numpy as np
from SALib.sample import morris as morris_sample
from SALib.analyze import morris as morris_analyze

def morris_screening(problem, simulator_fn, summary_fn, N=512, levels=6, grid_jump=2, seed=123):
    """
    problem = {
      "num_vars": ...,
      "names": ["beta0","k",...],
      "bounds": [(min,max), ...]
    }
    simulator_fn(theta)-> dict with time series (incidence etc.)
    summary_fn(sim_out)-> np.array of summary outputs (e.g., [mean14d, var, dom_freq])
    """
    rng = np.random.default_rng(seed)
    X = morris_sample.sample(problem, N, num_levels=levels, optimal_trajectories=None, seed=seed, grid_jump=grid_jump)
    Y = []
    for row in X:
        pars = {n: v for n, v in zip(problem["names"], row)}
        sim_out = simulator_fn(pars)
        Y.append(summary_fn(sim_out))
    Y = np.asarray(Y)
    # analyze each output dimension separately if multi-output
    results = []
    for j in range(Y.shape[1]):
        res = morris_analyze.analyze(problem, X, Y[:, j], print_to_console=False)
        results.append(res)  # contains mu, mu_star, sigma
    return results
