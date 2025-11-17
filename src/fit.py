"""
Core model fitting and optimization routines.

This module provides the functions necessary to fit the SIRC-PF model
to observed data. It defines the main objective function `negloglik_nb`,
which integrates the simulation, observation model, and likelihood
calculation.

It also contains the primary fitting orchestration function
(`fit_pso_then_local`), which uses a global-then-local optimization
strategy (Particle Swarm Optimization followed by L-BFGS-B).
Finally, it includes tools for model analysis, such as
`profile_likelihood`.

"""
from __future__ import annotations
import numpy as np
from scipy.optimize import minimize
from tqdm import tqdm

from .observation import make_mu_from_model, nb_loglik, discrete_gamma_kernel_cached

# -----------------------------------
# HELPERS
# -----------------------------------

# for 7 days harmonic smooth seasonality
def ridge_penalty(*coeffs, lam=1e-2):
    """Computes a simple L2 ridge penalty.
    """
    return lam * float(np.sum(np.square(coeffs)))

def sanitize_seed(x, bounds, eps_frac=1e-6):
    """
    Prepares a parameter seed vector for optimization.
    
    Parameters
    ----------
    x : 1D array-like
        The initial seed vector.
    bounds : list of tuple
        A list of (min, max) tuples, one for each parameter in `x`.
    eps_frac : float, optional
        The tiny fraction of the bound width used to nudge the seed away
        from the exact boundary.

    Returns
    -------
    np.ndarray
        A sanitized 1D float array, ready for optimization.
    """
    x = np.asarray(x, dtype=float)
    lo = np.array([b[0] for b in bounds], float)
    hi = np.array([b[1] for b in bounds], float)

    # Replace non-finite entries with bound midpoints
    mid = (lo + hi) / 2.0
    mask_bad = ~np.isfinite(x)
    if np.any(mask_bad):
        x[mask_bad] = mid[mask_bad]

    # Clip into [lo, hi]
    x = np.clip(x, lo, hi)

    # Tiny nudge away from exact bounds
    width = np.maximum(hi - lo, 1e-12)
    eps = eps_frac * width
    x = np.minimum(np.maximum(x, lo + eps), hi - eps)

    return x

# -----------------------------------
# END OF HELPERS
# -----------------------------------
    
def negloglik_nb(params_vec, param_names, bounds, fixed, 
                 t_eval, y_obs, delay_w, y0_template, build_pars_fn,
                 scoring_mask=None, prev_inc_tail=None):
    """
    The primary objective function: negative log-likelihood (NLL) for Negative B
    inomial (NB) model. It takes a vector of free parameters, combines them with 
    fixed parameters, and evaluates the total cost (NLL + penalties).

    The process is as follows:
    1.  Construct the full parameter dictionary `theta` from `params_vec` and
        `fixed`.
    2.  Build the ODE parameter structure `pars` and initial conditions `y0`.
    3.  Handle dynamic delay kernels if `delay_mean` is a fitted parameter.
    4.  Simulate the model using `make_mu_from_model` to get the expected
        observations `mu`.
    5.  Apply seasonal harmonics and calculate the corresponding `ridge_penalty`.
    6.  Calculate the negative log-likelihood using `nb_loglik`.
    7.  Add a "soft hand-off" penalty to `y0` components if anchors are
        provided in `fixed`.
    8.  Return the total cost: `NLL + seasonality_penalty + y0_penalty`.

    Parameters
    ----------
    params_vec : np.ndarray
        1D array of the current values for the *free* parameters being
        optimized.
    param_names : list of str
        List of names corresponding to the values in `params_vec`.
    bounds : list of tuple
        The bounds for `params_vec`.
    fixed : dict
        A dictionary of fixed parameter values and configuration settings.
    t_eval : np.ndarray
        Array of time points for simulation and evaluation.
    y_obs : np.ndarray
        Array of observed data (e.g., daily cases).
    delay_w : np.ndarray
        The *default* reporting delay kernel. May be overridden if
        `delay_mean` is a fitted parameter.
    y0_template : dict
        A template dictionary for default initial conditions.
    build_pars_fn : callable
        A function `(theta) -> pars` that converts the flat parameter
        dictionary `theta` into the nested `pars` dict required by the ODE.
    scoring_mask : np.ndarray, optional
        Boolean mask to select which time points of `y_obs` and `mu`
        to use for likelihood calculation.
    prev_inc_tail : np.ndarray, optional
        The last L-1 values of incidence from a *previous* time slice,
        used for "warming up" the delay convolution.

    Returns
    -------
    float
        The total cost (NLL + penalties). Returns `1e50` on any simulation
        or numerical failure.

    """
    
    ###### 1. Construct full theta dict ######
    if not hasattr(negloglik_nb, 'has_printed_error'):
        negloglik_nb.has_printed_error = False
        negloglik_nb.has_printed_mu_error = False

    theta = fixed.copy()
    theta.update({k: v for k, v in zip(param_names, params_vec)})

    for k, v in theta.items():
        # Only test numeric types for finiteness
        if isinstance(v, (int, float, np.number)):
            if not np.isfinite(v):
                return 1e50

    ###### 2. Build ODE and y0 paramter structures ######
    pars = build_pars_fn(theta)
    N = float(pars["N"])

    R0 = float(theta.get("R0", y0_template.get("R0", 0.0)))
    I0 = float(theta.get("I0", y0_template.get("I0", 0.0)))
    C0 = float(theta.get("C0", y0_template.get("C0", 0.0)))
    P0 = float(theta.get("P0", y0_template.get("P0", 0.0)))
    F0 = float(theta.get("F0", y0_template.get("F0", 0.0)))
    S0 = N - I0 - C0 - R0
    
    if not np.isfinite(S0) or S0 < 0: 
        return 1e50
    if not (0.0 <= P0 <= 1.0 and 0.0 <= F0 <= 1.0):
        return 1e50

    y0 = np.array([S0, I0, C0, R0, P0, F0], float)
    if not np.all(np.isfinite(y0)):
        return 1e50

    ###### 3. Handle delay kernel and 4. run model ######
    try:
        rho_obs = float(theta["rho_obs"]) 
        delay_cfg = fixed.get("obs", {})
        delay_mean = theta.get("delay_mean", None)
        if delay_mean is not None:
            dm = round(float(delay_mean), 2)
            sd = float(delay_cfg.get("delay_sd", 2.0))
            L  = int(delay_cfg.get("delay_maxlag", 14))
            delay_w = discrete_gamma_kernel_cached(dm, sd, L)          
        
        mu, _, incidence, next_tail = make_mu_from_model(
            pars, y0, t_eval, delay_w, rho_obs,
            prev_inc_tail=prev_inc_tail
        )
    
    except Exception as e:
        if not negloglik_nb.has_printed_error:
            print(f"\n[DEBUG] SIMULATION FAILED: {e}")
            print(f"  Failing y0: {y0}")
            print(f"  Failing theta: { {k:v for k,v in theta.items() if k not in ('obs','fixed')} }")
            negloglik_nb.has_printed_error = True
        return 1e50

    if (not np.all(np.isfinite(mu))) or np.all(mu <= 1e-12):
        if not negloglik_nb.has_printed_mu_error:
            print(f"\n[DEBUG] SIMULATION 'DEAD': mu vector is non-finite or all-zero.")
            negloglik_nb.has_printed_mu_error = True
        return 1e50

    log_theta = float(theta["log_theta"])
    if not np.isfinite(log_theta):
        return 1e50

    if scoring_mask is not None:
        y_use  = y_obs[scoring_mask]
        mu_use = mu[scoring_mask]
    else:
        y_use, mu_use = y_obs, mu
    
    ###### 5. Apply seasonality hamonics ######
    pen = 0.0
    if all(k in theta for k in ("a1","b1","a2","b2")):
        from .observation import apply_weekly_harmonics
        a1 = float(theta["a1"]); b1 = float(theta["b1"])
        a2 = float(theta["a2"]); b2 = float(theta["b2"])
        mu_use = apply_weekly_harmonics(mu_use, a1, b1, a2, b2)
        pen = ridge_penalty(a1, b1, a2, b2, lam=1e-2)
        
    ###### 6. Calculate the negative log-likelihood ######
    ll = nb_loglik(y_use, mu_use, log_theta=log_theta)
    
    ###### 7. soft hands-off with penalty ######
    y0_penalty = 0.0
    anchor_weight = float(fixed.get("y0_anchor_weight", 0.0))
    
    if anchor_weight > 0:
        # Check for each anchor and add penalty if it exists
        if "I0_anchor" in fixed:
            y0_penalty += (theta["I0"] - fixed["I0_anchor"])**2
        if "C0_anchor" in fixed:
            y0_penalty += (theta["C0"] - fixed["C0_anchor"])**2
        if "R0_anchor" in fixed:
            y0_penalty += (theta["R0"] - fixed["R0_anchor"])**2
        if "P0_anchor" in fixed:
            y0_penalty += (theta["P0"] - fixed["P0_anchor"])**2
        if "F0_anchor" in fixed:
            y0_penalty += (theta["F0"] - fixed["F0_anchor"])**2
            
        y0_penalty *= anchor_weight

    ###### 8. final cost ######
    final_cost = (-ll + pen + y0_penalty)
    
    return final_cost if np.isfinite(final_cost) else 1e50


def fit_pso_then_local(
    x0_guess, # center guess
    param_names, bounds, fixed, t_eval, y_obs, delay_w, y0_template, build_pars_fn,
    use_pso=True, pso_particles=60, pso_iters=300, local_seeds=8, seed=123, scoring_mask=None, prev_inc_tail=None
):
    """
    Fits model parameters using global Particles Swarm Optimization (PSO) 
    then local Box constraints Limited-memory Broyden-Fletcher-Goldfarb-Shanno 
    (L-BFGS-B) search, using defined parameters and settings in params.yaml.

    Orchestrates the fitting process:
    1.  Runs a Particle Swarm Optimization (PSO) global search
        to find a promising region of the parameter space.
    2.  Generates a set of seeds for local refinement. these seeds are 
        centered around the PSO best-fit.
    3.  Runs a `scipy.optimize.minimize` using L-BFGS-B local search starting
        from each seed.
    4.  Tracks and returns the result from the *best* (lowest NLL) local
        run.

    Parameters
    ----------
    x0_guess : np.ndarray
        Initial guess (not used if PSO is active, but required for API).
    param_names : list of str
        Names of the parameters being fitted.
    bounds : list of tuple
        Bounds for the parameters.
    fixed : dict
        Dictionary of fixed parameters and configuration.
    t_eval : np.ndarray
        Time points for evaluation.
    y_obs : np.ndarray
        Observed data.
    delay_w : np.ndarray
        Default reporting delay kernel.
    y0_template : dict
        Template for initial conditions.
    build_pars_fn : callable
        Function to build the ODE parameter dictionary.
    use_pso : bool, optional
        Whether to run the PSO global search first. Defaults to true.
    pso_particles : int, optional
        Number of particles for PSO. Defined in params.yaml
    pso_iters : int, optional
        Number of iterations for PSO.
    local_seeds : int, optional
        Number of L-BFGS-B local searches to run.
    seed : int, optional
        Random seed for reproducibility.
    scoring_mask : np.ndarray, optional
        Boolean mask for likelihood calculation.
    prev_inc_tail : np.ndarray, optional
        Incidence tail from the previous slice for delay convolution.

    Returns
    -------
    dict
        A dictionary containing the best fit:
        - "fun": The final minimum NLL value.
        - "x": The optimal parameter vector.
        - "res": The full `scipy.optimize.OptimizeResult` object from the
        best local run.
        
    Notes
    --------
    The PSO runs on a "cold start" for the first slice, but on a
    "warm start" for following slices: we set one of the particles at
    the previous slice's best params, while allowing other particles to
    explore. 
    
    This logic is handled in the main run.py function.
    """
    rng = np.random.default_rng(seed)

    ###### 1. PSO Global Search ######
    seeds = []
    if use_pso:
        try:
            import pyswarms as ps
            lb = np.array([b[0] for b in bounds], float)
            ub = np.array([b[1] for b in bounds], float)
            n_dims = len(bounds)

            # "Selective-Random" initialization
            init_pos = None
            if x0_guess is not None:
                print("  Seeding PSO with 'Selective-Random' cloud.")
                # x0_guess was already sanitized in run.py
                
                # These are the params that MUST be kept safe
                y0_param_names = {"I0", "C0", "R0", "P0", "F0"}
                
                init_pos = np.zeros((pso_particles, n_dims))
                
                # Build the swarm
                for j, name in enumerate(param_names):
                    
                    if name in y0_param_names:
                        # Keep y0 params in a tight cloud around the anchor
                        center = x0_guess[j]
                        # Use very small noise (1% of width)
                        noise = rng.normal(0.0, 0.01, pso_particles) * (ub[j] - lb[j])
                        col = np.clip(center + noise, lb[j], ub[j])
                        # Ensure one particle is exactly the anchor
                        col[0] = center 
                        init_pos[:, j] = col
                        
                    else:
                        # Initialize mechanical params randomly across its full bounds
                        col = lb[j] + (ub[j] - lb[j]) * rng.random(pso_particles)
                        # But set the first "warm" particle to the old value
                        col[0] = x0_guess[j]
                        init_pos[:, j] = col
            
            # If x0_guess was None (Slice 1), init_pos remains None,
            # and pyswarms will use its default random initialization.

            def f_pso(X):
                return np.array([
                    negloglik_nb(row, param_names, bounds, fixed, t_eval, y_obs, delay_w, y0_template, build_pars_fn, scoring_mask, prev_inc_tail)
                for row in X
                ])

            print("Starting PSO global search...") 
            optimizer = ps.single.GlobalBestPSO(
                n_particles=pso_particles, dimensions=n_dims,
                options={"c1":1.4,"c2":1.4,"w":0.6}, 
                bounds=(lb,ub),
                init_pos=init_pos
            )
            best_cost, best_pos = optimizer.optimize(f_pso, iters=pso_iters, n_processes=None)
            print(f"PSO global search complete. Best cost: {best_cost:.4f}") 
            
            seeds.append(best_pos)
            for _ in range(local_seeds-1):
                noise = rng.normal(0, 0.05, size=n_dims)
                seeds.append(np.clip(best_pos*(1.0+noise), lb, ub))
        except Exception as e:
            print(f"[WARN] PSO skipped due to error: {e}")
           
    ###### 2. Local refinement seeds ###### 
    if not seeds and x0_guess is not None:
         print("  PSO was skipped, using x0_guess to seed local refinement.")
         seeds.append(x0_guess)
         
    if not seeds:
        seeds = []
        for _ in range(local_seeds):
            seeds.append(np.array([rng.uniform(b[0], b[1]) for b in bounds]))

    ###### 3. L-BFGS-B local fit ######
    best = {"fun": np.inf, "x": None, "res": None}
    
    for i, s in enumerate(seeds): 
        print(f"--- Starting local refinement for seed {i+1}/{len(seeds)} ---")
        s_sanitized = sanitize_seed(s, bounds)
        
        options = {"maxiter": 2000, "ftol": 1e-10, "gtol":1e-8}
        max_iter = options.get("maxiter", 2000)

        with tqdm(total=max_iter, desc=f"L-BFGS-B Seed {i+1}") as pbar:
            def callback_fn(xk):
                pbar.update(1)
                
            res = minimize(
                negloglik_nb, s_sanitized,
                args=(param_names, bounds, fixed, t_eval, y_obs, delay_w, y0_template, build_pars_fn, scoring_mask, prev_inc_tail),
                method="L-BFGS-B",
                bounds=bounds,
                options=options,
                callback=callback_fn
            )
        
        print(f"Seed {i+1} finished: {res.message}")
        print(f"  Iterations: {res.nit}")
        print(f"  Final NLL: {res.fun:.4f}")
        
        if res.fun < best["fun"]:
            best = {"fun": res.fun, "x": res.x, "res": res}
            
    return best


def profile_likelihood(param_name, grid, best_fit, param_names, bounds, fixed,
                       t_eval, y_obs, delay_w, y0_template, build_pars_fn):
    """
    Computes the profile likelihood for a single specified parameter.
    Ttraces the "valley" of the likelihood surface for one
    parameter, helping to assess its identifiability. For each value
    in `grid` for the `param_name`:
    1.  Fixes `param_name` to that value.
    2.  Re-optimizes all *other* free parameters.
    3.  Stores the resulting minimum NLL.

    Parameters
    ----------
    param_name : str
        The name of the parameter to profile.
    grid : np.ndarray
        An array of values for `param_name` to evaluate.
    best_fit : dict
        The best-fit result dictionary from `fit_pso_then_local`, used
        to provide seeds for the re-optimization.
    param_names : list of str
        List of *all* free parameter names (including `param_name`).
    bounds : list of tuple
        Bounds for all free parameters.
    fixed : dict
        Dictionary of fixed parameters.
    t_eval : np.ndarray
        Time points for evaluation.
    y_obs : np.ndarray
        Observed data.
    delay_w : np.ndarray
        Reporting delay kernel.
    y0_template : dict
        Template for initial conditions.
    build_pars_fn : callable
        Function to build the ODE parameter dictionary.

    Returns
    -------
    tuple
        (grid, prof_vals, prof_x)
        - grid : np.ndarray - The input `grid` array.
        - prof_vals : np.ndarray - The minimum NLL found at each grid value.
        - prof_x : list - A list where each element is the `res.x` vector
        (all optimized parameters) at that grid point.

    """
    ###### 1. Fix value ######
    idx = param_names.index(param_name)
    base_x = np.array(best_fit["x"], dtype=float)
    base_bounds = np.array(bounds, dtype=float)

    prof_vals, prof_x = [], []

    ###### 2. Re-optimize free parameters ######
    print(f"Profiling '{param_name}'...")
    for val in tqdm(grid, desc=f"Profiling {param_name}"):
        seed = np.clip(base_x, base_bounds[:,0], base_bounds[:,1])

        pbounds = bounds.copy()
        pbounds[idx] = (val, val)

        fixed_override = fixed.copy()
        fixed_override[param_name] = float(val)
        
        s_sanitized = sanitize_seed(seed, pbounds) 
        
        res = minimize(
            negloglik_nb, s_sanitized, 
            args=(param_names, pbounds, fixed_override, t_eval, y_obs, delay_w, y0_template, build_pars_fn),
            method="L-BFGS-B",
            bounds=pbounds,
            options={"maxiter": 1000, "ftol": 1e-9}
        )
        prof_vals.append(res.fun if np.isfinite(res.fun) else 1e50)
        prof_x.append(res.x.copy())
        
    return np.array(grid), np.array(prof_vals), prof_x