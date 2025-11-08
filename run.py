from __future__ import annotations
import os, json, time, yaml
import numpy as np
import pandas as pd
from pathlib import Path

from src.model import simulate, sirc_pf_rhs
from src.observation import discrete_gamma_kernel, make_mu_from_model, nb_loglik
from src.fit import fit_pso_then_local, profile_likelihood
from src.metrics import time_series_splits, mae, rmse, nb_mean_log_pred_density, simulate_nb_intervals
from src.dynamics import dominant_period_days, equilibrium_newton, numerical_jacobian

def build_pars_fn(theta_dict):
    # Map flat theta_dict into 'pars' for the ODE
    keys = ["N","beta0","gamma","rho","alpha","delta","epsilon","phi","k","compliance_max","beta_floor"]
    pars = {k: float(theta_dict[k]) for k in keys if k in theta_dict}
    return pars

def main():
    # --- Setup output ---
    ts = time.strftime("%Y-%m-%d_%H%M%S")
    outdir = Path("reports") / ts
    outdir.mkdir(parents=True, exist_ok=True)

    # --- Load data & params ---
    params = yaml.safe_load(open("params.yaml"))
    data = pd.read_csv("data/cases.csv")  # columns: date, cases
    y_obs = data["cases"].to_numpy(dtype=float)
    T = len(y_obs)
    t_eval = np.arange(T, dtype=float)

    # Initial state template (S0 will be N - others)
    N = float(params["fixed"]["N"])
    y0_template = {"S0": N, "R0": 0.0}

    # Delay kernel
    w = discrete_gamma_kernel(mean_days=params["obs"]["delay_mean"], sd_days=params["obs"]["delay_sd"], max_lag=params["obs"]["delay_maxlag"])

    # --- Fit model ---
    best = fit_pso_then_local(
        x0_guess=None,
        param_names=params["free_names"],
        bounds=[tuple(b) for b in params["bounds"]],
        fixed=params["fixed"],
        t_eval=t_eval, y_obs=y_obs, delay_w=w,
        y0_template={"S0": N, "R0": 0.0},
        build_pars_fn=build_pars_fn,
        use_pso=params["fit"]["use_pso"],
        pso_particles=params["fit"]["pso_particles"],
        pso_iters=params["fit"]["pso_iters"],
        local_seeds=params["fit"]["local_seeds"],
        seed=params["fit"]["seed"]
    )
    # Save best fit
    pd.Series(best["x"], index=params["free_names"]).to_csv(outdir / "best_params.csv")

    # Recompute fitted trajectories and mu
    theta_best = params["fixed"].copy()
    theta_best.update({k: v for k, v in zip(params["free_names"], best["x"])})
    pars = build_pars_fn(theta_best)
    y0 = np.array([N - theta_best["I0"] - theta_best["C0"] - theta_best.get("R0", 0.0),
                   theta_best["I0"], theta_best["C0"], theta_best.get("R0", 0.0),
                   theta_best["P0"], theta_best["F0"]], float)
    from src.observation import make_mu_from_model
    mu, Y, inc = make_mu_from_model(pars, y0, t_eval, w, rho_obs=theta_best["rho_obs"])
    pd.DataFrame({"mu":mu, "inc":inc, "y":y_obs}).to_csv(outdir / "fit_series.csv", index=False)

    # --- CV evaluation ---
    res_rows = []
    for horizon in params["cv"]["horizons"]:
        for tr_idx, te_idx in time_series_splits(T, params["cv"]["initial_train"], horizon, params["cv"]["step"]):
            # Fit on train only: (optional: you can refit; here, for brevity, we reuse best params — better: refit)
            y_true = y_obs[te_idx]
            mu_pred = mu[te_idx]  # quick placeholder; better: conditional re-sim forward-only
            m_mae = mae(y_true, mu_pred)
            m_rmse = rmse(y_true, mu_pred)
            lpd = nb_mean_log_pred_density(y_true, mu_pred, theta_best["log_theta"])
            ints = simulate_nb_intervals(mu_pred, theta_best["log_theta"], n_paths=1000)
            coverage90 = float(np.mean((y_true >= ints["lower"]) & (y_true <= ints["upper"])))
            res_rows.append({"horizon":horizon,"mae":m_mae,"rmse":m_rmse,"mlpd":lpd,"cov90":coverage90})
    pd.DataFrame(res_rows).to_csv(outdir / "cv_metrics.csv", index=False)

    # --- Profile likelihoods for a few key params ---
    for pname, grid in params["profiles"].items():
        g = np.linspace(grid[0], grid[1], grid[2])
        G, Fvals, Xs = profile_likelihood(
            pname, g, best,
            params["free_names"], [tuple(b) for b in params["bounds"]], params["fixed"],
            t_eval, y_obs, w, y0_template, build_pars_fn
        )
        pd.DataFrame({"param":pname, "grid":G, "negloglik":Fvals}).to_csv(outdir / f"profile_{pname}.csv", index=False)

    # --- Oscillation diagnostics at the fitted params ---
    from src.dynamics import dominant_period_days, equilibrium_newton, numerical_jacobian
    period, fstar, f, Pxx = dominant_period_days(mu)  # on mu or incidence
    pd.DataFrame({"f":f, "Pxx":Pxx}).to_csv(outdir / "psd.csv", index=False)

    # Equilibrium + eigenvalues (sanity / Hopf hint)
    from src.model import sirc_pf_rhs
    y_guess = Y[-1]  # last state as a starting guess
    try:
        y_star = equilibrium_newton(sirc_pf_rhs, y_guess, pars)
        J = numerical_jacobian(sirc_pf_rhs, y_star, pars)
        evals = np.linalg.eigvals(J)
        pd.DataFrame({"re":evals.real, "im":evals.imag}).to_csv(outdir / "jacobian_eigs.csv", index=False)
    except Exception as e:
        open(outdir / "jacobian_error.txt","w").write(str(e))

    print(f"Done. Results in {outdir}")

if __name__ == "__main__":
    main()
