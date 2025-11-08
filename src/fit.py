from __future__ import annotations
import numpy as np
from scipy.optimize import minimize

# -----------------------------------
# HELPERS

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

    
# END OF HELPERS
# -----------------------------------

# --- Objective on NB likelihood ---
def negloglik_nb(params_vec, param_names, bounds, fixed, t_eval, y_obs, delay_w, y0_template, build_pars_fn):
    # Assemble theta dict
    theta = fixed.copy()
    theta.update({k: v for k, v in zip(param_names, params_vec)})

    # ---- Guard 1: finiteness of params ----
    for k, v in theta.items():
        if not np.isfinite(v):
            return 1e50  # huge penalty

    # ---- Build model params ----
    pars = build_pars_fn(theta)  # should include N, beta0, gamma, rho_proc? etc.
    N = float(pars["N"])

    # ---- Initial conditions: S0 from conservation, clamp ----
    R0 = float(theta.get("R0", y0_template.get("R0", 0.0)))
    I0 = float(theta["I0"]); C0 = float(theta["C0"])
    P0 = float(theta["P0"]); F0 = float(theta["F0"])
    S0 = N - I0 - C0 - R0
    # Early checks
    if not np.isfinite(S0) or S0 < 0: 
        # Optional debug (throttle if noisy)
        # print(f"[DBG] invalid S0={S0:.3g} (N={N}, I0={I0}, C0={C0}, R0={R0})")
        return 1e50

    if not (0.0 <= P0 <= 1.0 and 0.0 <= F0 <= 1.0):
        # print(f"[DBG] P0/F0 out of [0,1]: P0={P0}, F0={F0}")
        return 1e50

    y0 = np.array([S0, I0, C0, R0, P0, F0], float)
    if not np.all(np.isfinite(y0)):
        # print(f"[DBG] non-finite y0: {y0}")
        return 1e50

    # ---- Simulate; catch ODE/num errors ----
    try:
        from .observation import make_mu_from_model, nb_loglik
        rho_obs = float(theta["rho_obs"])  # renamed in params.yaml
        mu, _, _ = make_mu_from_model(pars, y0, t_eval, delay_w, rho_obs)
    except Exception:
        return 1e50

    # ---- Likelihood ----
    log_theta = float(theta["log_theta"])
    if not np.isfinite(log_theta):
        return 1e50

    ll = nb_loglik(y_obs, mu, log_theta=log_theta)
    # Return negative log-likelihood
    return -ll if np.isfinite(ll) else 1e50


def fit_pso_then_local(
    x0_guess, param_names, bounds, fixed, t_eval, y_obs, delay_w, y0_template, build_pars_fn,
    use_pso=True, pso_particles=60, pso_iters=300, local_seeds=8, seed=123
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
                vals = []
                for row in X:
                    v = negloglik_nb(row, param_names, bounds, fixed, t_eval, y_obs, delay_w, y0_template, build_pars_fn)
                    vals.append(v)
                return np.array(vals)

            optimizer = ps.single.GlobalBestPSO(n_particles=pso_particles, dimensions=len(bounds),
                                                options={"c1":1.4,"c2":1.4,"w":0.6}, bounds=(lb,ub))
            best_cost, best_pos = optimizer.optimize(f_pso, iters=pso_iters, n_processes=None)
            seeds.append(best_pos)
            # also sample a few neighbors around the swarmed best
            for _ in range(local_seeds-1):
                noise = rng.normal(0, 0.05, size=len(bounds))  # 5% jitter
                seeds.append(np.clip(best_pos*(1.0+noise), lb, ub))
        except Exception as e:
            print(f"[WARN] PSO skipped due to error: {e}")

    if not seeds:
        # fallback: random Latin-ish seeds within bounds
        seeds = []
        for _ in range(local_seeds):
            seeds.append(np.array([rng.uniform(b[0], b[1]) for b in bounds]))

    # --- Local refinement L-BFGS-B ---
    best = {"fun": np.inf, "x": None, "res": None}
    for s in seeds:
        
        seed = sanitize_seed(s, bounds)
        res = minimize(
            negloglik_nb, s,
            args=(param_names, bounds, fixed, t_eval, y_obs, delay_w, y0_template, build_pars_fn),
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 2000, "ftol": 1e-10, "gtol":1e-8}
        )
        if res.fun < best["fun"]:
            best = {"fun": res.fun, "x": res.x, "res": res}
    return best

def profile_likelihood(param_name, grid, best_fit, param_names, bounds, fixed,
                       t_eval, y_obs, delay_w, y0_template, build_pars_fn):
    idx = param_names.index(param_name)
    base_x = np.array(best_fit["x"], dtype=float)
    base_bounds = np.array(bounds, dtype=float)

    prof_vals, prof_x = [], []

    for val in grid:
        # 1) Clamp seed inside original bounds
        seed = np.clip(base_x, base_bounds[:,0], base_bounds[:,1])

        # 2) Fix this parameter via bounds AND via fixed override
        pbounds = bounds.copy()
        pbounds[idx] = (val, val)

        fixed_override = fixed.copy()
        fixed_override[param_name] = float(val)
        
        seed = sanitize_seed(seed, bounds)
        res = minimize(
            negloglik_nb, seed,
            args=(param_names, pbounds, fixed_override, t_eval, y_obs, delay_w, y0_template, build_pars_fn),
            method="L-BFGS-B",
            bounds=pbounds,
            options={"maxiter": 1000, "ftol": 1e-9}
        )
        prof_vals.append(res.fun if np.isfinite(res.fun) else 1e50)
        prof_x.append(res.x.copy())
    return np.array(grid), np.array(prof_vals), prof_x

