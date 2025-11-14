from __future__ import annotations
import numpy as np
from scipy.optimize import minimize
from tqdm import tqdm

# (sanitize_seed function is unchanged)
def sanitize_seed(x, bounds, eps_frac=1e-6):
    """
    x: 1D array-like seed
    bounds: list/array of (lo, hi)
    eps_frac: nudges seed a tiny fraction inside (lo, hi) if it's exactly at a bound
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

    # Tiny nudge away from exact bounds (helps some optimizers / derivatives)
    width = np.maximum(hi - lo, 1e-12)
    eps = eps_frac * width
    x = np.minimum(np.maximum(x, lo + eps), hi - eps)

    return x
    
# --- Objective on NB likelihood ---
def negloglik_nb(params_vec, param_names, bounds, fixed, 
                 t_eval, y_obs, delay_w, y0_template, build_pars_fn,
                 scoring_mask=None, prev_inc_tail=None):
    
    # --- ADDED: A global flag to prevent spamming your console ---
    # We'll use this to print errors only ONCE.
    if not hasattr(negloglik_nb, 'has_printed_error'):
        negloglik_nb.has_printed_error = False
        negloglik_nb.has_printed_mu_error = False

    # Assemble theta dict
    theta = fixed.copy()
    theta.update({k: v for k, v in zip(param_names, params_vec)})

    # ---- Guard 1: finiteness of params ----
    for k, v in theta.items():
        if not np.isfinite(v):
            return 1e50

    # ---- Build model params ----
    pars = build_pars_fn(theta)
    N = float(pars["N"])

    # ---- Initial conditions ----
    R0 = float(theta.get("R0", y0_template.get("R0", 0.0)))
    I0 = float(theta.get("I0", y0_template.get("I0", 0.0)))
    C0 = float(theta.get("C0", y0_template.get("C0", 0.0)))
    P0 = float(theta.get("P0", y0_template.get("P0", 0.0)))
    F0 = float(theta.get("F0", y0_template.get("F0", 0.0)))
    S0 = N - I0 - C0 - R0
    if S0 <= 0: 
        print("ERROR: S0 < 0, population empty")
        return 1e50
    
    if not np.isfinite(S0) or S0 < 0: 
        return 1e50
    if not (0.0 <= P0 <= 1.0 and 0.0 <= F0 <= 1.0):
        return 1e50

    y0 = np.array([S0, I0, C0, R0, P0, F0], float)
    if not np.all(np.isfinite(y0)):
        return 1e50

    # ---- Simulate; catch ODE/num errors ----
    try:
        from .observation import make_mu_from_model, nb_loglik
        rho_obs = float(theta["rho_obs"]) 
        mu, _, incidence = make_mu_from_model(pars, y0, t_eval, delay_w, rho_obs,
                                          prev_inc_tail=prev_inc_tail)
        
    except Exception as e:
        # --- DEBUG PRINT (Suspect #1) ---
        # This will print the *first* error it sees.
        is_invalid = not np.all(np.isfinite(mu))
        if is_invalid:
            print("\n" + "="*20 + " DEBUG: SIMULATION FAILED " + "="*20)
            print(f"  ERROR: {e}")
            print(f"  This error is causing negloglik_nb to return 1e50.")
            print(f"  This creates a 'flat' surface, causing 1-iteration convergence.")
            print(f"  Failing y0: {y0}")
            print(f"  Failing theta: {theta}")
            print("="*64 + "\n")
            negloglik_nb.has_printed_error = True # So we only print once
        # --- END DEBUG PRINT ---
        return 1e50

    # ---- Likelihood ----
    
    # --- DEBUG PRINT (Suspect #2) ---
    # Check if mu is "dead" (all zeros, constant, or NaN)
    if not np.all(np.isfinite(mu)) or np.all(mu < 1.0):
        is_all_zero = np.all(mu == 0)
        if not negloglik_nb.has_printed_mu_error and is_all_zero:
            print("\n" + "="*20 + " DEBUG: SIMULATION 'DEAD' " + "="*20)
            print(f"  ERROR: The 'mu' vector is all zeros, constant, or NaN.")
            print(f"  This *also* creates a 'flat' surface.")
            print(f"  Check your 'simulate' function in model.py for the 'np.diff(S)' bug.")
            print(f"  mu[0:5]: {mu[0:5]}")
            print(f"  y0: {y0}")
            print(f"  theta: {theta}")
            print("="*64 + "\n")
            negloglik_nb.has_printed_mu_error = True
    # --- END DEBUG PRINT ---

    log_theta = float(theta["log_theta"])
    if not np.isfinite(log_theta):
        return 1e50

    if scoring_mask is not None:
        y_use  = y_obs[scoring_mask]
        mu_use = mu[scoring_mask]
    else:
        y_use, mu_use = y_obs, mu
    
    ll = nb_loglik(y_use, mu_use, log_theta=log_theta)
    return -ll if np.isfinite(ll) else 1e50


# (The rest of fit.py is unchanged, including the progress bar logic)

def fit_pso_then_local(
    x0_guess, param_names, bounds, fixed, t_eval, y_obs, delay_w, y0_template, build_pars_fn,
    use_pso=True, pso_particles=60, pso_iters=300, local_seeds=8, seed=123, scoring_mask=None, prev_inc_tail=None
):
    rng = np.random.default_rng(seed)

    # --- Optional: PSO global search ---
    seeds = []
    if use_pso:
        try:
            import pyswarms as ps
            lb = np.array([b[0] for b in bounds], float)
            ub = np.array([b[1] for b in bounds], float)

            def f_pso(X):
                # X shape: (n_particles, n_dims)
                return np.array([
                    negloglik_nb(row, param_names, bounds, fixed, t_eval, y_obs, delay_w, y0_template, build_pars_fn, scoring_mask, prev_inc_tail)
                for row in X
                ])

            print("Starting PSO global search...") 
            optimizer = ps.single.GlobalBestPSO(n_particles=pso_particles, dimensions=len(bounds),
                                                options={"c1":1.4,"c2":1.4,"w":0.6}, bounds=(lb,ub))
            best_cost, best_pos = optimizer.optimize(f_pso, iters=pso_iters, n_processes=None)
            print(f"PSO global search complete. Best cost: {best_cost:.4f}") 
            
            seeds.append(best_pos)
            for _ in range(local_seeds-1):
                noise = rng.normal(0, 0.05, size=len(bounds))
                seeds.append(np.clip(best_pos*(1.0+noise), lb, ub))
        except Exception as e:
            print(f"[WARN] PSO skipped due to error: {e}")

    if not seeds:
        seeds = []
        for _ in range(local_seeds):
            seeds.append(np.array([rng.uniform(b[0], b[1]) for b in bounds]))

    # --- Local refinement L-BFGS-B ---
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
        
        # --- This is what you are seeing ---
        print(f"Seed {i+1} finished: {res.message}")
        print(f"  Iterations: {res.nit}") # <-- This is likely 1
        print(f"  Final NLL: {res.fun:.4f}")
        
        if res.fun < best["fun"]:
            best = {"fun": res.fun, "x": res.x, "res": res}
            
    return best

def profile_likelihood(param_name, grid, best_fit, param_names, bounds, fixed,
                       t_eval, y_obs, delay_w, y0_template, build_pars_fn):
    idx = param_names.index(param_name)
    base_x = np.array(best_fit["x"], dtype=float)
    base_bounds = np.array(bounds, dtype=float)

    prof_vals, prof_x = [], []

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