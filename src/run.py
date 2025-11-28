"""
Main script for sequential model fitting.

This script serves as the primary entry point for running the analysis.
It loads a master parameter configuration from ``params.yaml`` and
a set of pre-sliced data files from ``data/sliced_data/``.

It iterates through each data slice sequentially, calling the 
``analyze_slice`` function. This process implements a "soft hand-off" 
between slices:
1.  The final simulated state (y-vector) from slice `N` is used as the
    anchored initial condition (`y0_anchor`) for slice `N+1`.
2.  The final `L-1` incidence values from slice `N` are used to "warm up"
    the delay convolution for slice `N+1`.
3.  The best-fit parameter vector from slice `N` is used as a "warm
    start" seed for the optimizer in slice `N+1`.

The script generates a new, timestamped directory in ``reports/`` for
each run, which contains detailed outputs for each slice, including
fitted parameters, simulation trajectories, diagnostics (AIC/BIC),
profile likelihoods, and stability analysis.

"""
from __future__ import annotations
import os, json, time, yaml
import numpy as np
import pandas as pd
from pathlib import Path

from src.observation import (
    discrete_gamma_kernel, 
    make_mu_from_model, 
    nb_loglik)
from src.fit import (
    fit_pso_then_local, 
    profile_likelihood, 
    sanitize_seed)
from src.metrics import (
    mae, 
    rmse, 
    nb_mean_log_pred_density, 
    simulate_nb_intervals)
from src.model import sirc_pf_rhs
from src.diagnostics import info_criteria_nb
from src.dynamics import (
    dominant_period_days, 
    equilibrium_newton_reduced,
    numerical_jacobian_reduced,
    sirc_pf_rhs_reduced 
)

# -----------------------------------
# HELPERS
# -----------------------------------

def build_pars_fn(theta_dict):
    """Maps flat parameter dictionary theta_dict to the nested ODE `pars` dictionary.
    """
    keys = ["N","beta0","gamma","rho","alpha","delta","epsilon",
            "phi","k","compliance_max","beta_floor"]
    pars = {k: float(theta_dict[k]) for k in keys if k in theta_dict}
    return pars

def load_sliced_data(data_dir: Path, num_slices: int = 6):
    """Loads pre-sliced CSV data files from a specified directory.

    Expects files to be named sequentially (ex: 'slice_1.csv', 'slice_2.csv', ...).

    Parameters
    ----------
    data_dir : Path
        The directory (ex: 'data/sliced_data') containing the CSV files.
    num_slices : int, optional
        The total number of slices to load.

    Returns
    -------
    list of pd.DataFrame
        A list containing the DataFrame for each slice, in order.

    Raises
    ------
    FileNotFoundError
        If a sequential slice file (e.g., 'slice_1.csv') is missing.

    """
    print(f"Loading sliced data from {data_dir}...")
    slice_dataframes = []
    
    # Find files by explicit name to ensure order
    for i in range(num_slices):
        fname = data_dir / f"slice_{i+1}.csv"
        if not fname.exists():
            raise FileNotFoundError(
                f"Missing required slice file: {fname}\n"
                f"Please ensure {num_slices} files named 'slice_1.csv' through 'slice_{num_slices}.csv' exist."
            )
        print(f"  Loading {fname.name}...")
        slice_dataframes.append(pd.read_csv(fname))
        
    print(f"Successfully loaded {len(slice_dataframes)} data slices.")
    return slice_dataframes

# -----------------------------------
# END OF HELPERS
# -----------------------------------

def analyze_slice(
    slice_id: int, 
    slice_data_df: pd.DataFrame,
    params: dict,
    base_outdir: Path,
    delay_w: np.ndarray,
    fit_params: dict = {}
):
    """Runs the entire analysis pipeline for a single data slice.

    It performs the following steps:
    1.  Sets up the output directory for this slice.
    2.  Prepares data (y_obs, t_eval) and scoring masks.
    3.  Implements the "soft hands-off" logic by setting `y0_anchor` values
        in the `current_fixed` dictionary if `y0_guess_from_prev` is provided.
    4.  Prepares a "warm start" seed (`x0_guess_for_fit`) for the optimizer
        using the best-fit parameters from the previous slice.
    5.  Calls `fit_pso_then_local` to find the best-fit parameters.
    6.  Re-computes the best-fit trajectories and simulation diagnostics
        (AIC, BIC) using the optimized parameters.
    7.  Runs `metrics` for specified parameters.
    8.  Runs `profile_likelihood` for specified parameters.
    9.  Runs stability analysis (equilibrium, Jacobian)
        on the final fitted state.

    Parameters
    ----------
    slice_id : int
        The index (e.g., 0, 1, 2...) of the current slice.
    slice_data_df : pd.DataFrame
        The DataFrame containing the data for this slice.
    params : dict
        The master parameter configuration loaded from `params.yaml`.
    base_outdir : Path
        The root output directory for this *run* (e.g., 'reports/timestamp').
    delay_w : np.ndarray
        The reporting delay kernel.
    y0_guess_from_prev : np.ndarray or None
        The final 6-state vector (S, I, C, R, P, F) from the *previous*
        slice's simulation.
    fit_params : dict
        A sub-dictionary from `params.yaml` containing optimizer settings.
    prev_inc_tail : np.ndarray or None
        The last `L-1` incidence values from the previous slice, used to
        buffer the delay convolution.
    prev_best_fit_params_vec : np.ndarray or None
        The best-fit `x` vector from the *previous* slice's optimizer.
    prev_best_fit_param_names : list[str] or None
        The list of parameter names corresponding to
        `prev_best_fit_params_vec`.

    Returns
    -------
    tuple
        A tuple `(y_final_state, next_tail, best_fit_vec, param_names)`.
        - y_final_state : np.ndarray, The final 6-state vector of the fit.
        - next_tail : np.ndarray, The last L-1 incidences for the next slice.
        - best_fit_vec : np.ndarray, The optimized parameter vector `x`.
        - param_names : list[str], The names for `best_fit_vec`.
        Returns `(None, ...)` if the analysis fails.

    """
    
    ###### 1. Setup Slice Directories ######
    slice_data = slice_data_df
    slice_outdir = base_outdir / f"slice_{slice_id+1}"
    slice_outdir.mkdir(parents=True, exist_ok=True)

    print(f"--------------------------------------------")
    print(f"--- Analyzing Slice {slice_id+1} (Length: {len(slice_data)} days) ---")
    print(f"--------------------------------------------")
    print(f" Results will be in: {slice_outdir}")

    ###### 2. Setup data and scoring masks ######
    y_obs = slice_data["cases"].to_numpy(dtype=float)
    T = len(y_obs)
    t_eval = np.arange(T, dtype=float)
    N = float(params["fixed"]["N"])
    
    burn = int(params["obs"]["delay_maxlag"])
    scoring_mask = np.ones(T, dtype=bool)
    scoring_mask[:burn] = False

    # Parameters for soft hands-off logic
    current_free_names = list(params["free_names"])
    current_bounds = [tuple(b) for b in params["bounds"]]
    
    current_fixed = params["fixed"].copy()
    current_fixed["obs"] = params["obs"]
    
    y0_template = {"S0": N, "R0": 0.0}
    print(f"  Fitting all {len(current_free_names)} free params independently: {current_free_names}")

    ###### 4. Warm start for PSO using last slice parameters ######
    x0_guess_for_fit = None
    use_pso_for_fit = fit_params.get("use_pso", True) 

    ###### 5. Model fit ######
    print("  Fitting model...")
    best = fit_pso_then_local(
        x0_guess=x0_guess_for_fit,
        param_names=current_free_names,
        bounds=current_bounds,
        fixed=current_fixed,
        t_eval=t_eval, y_obs=y_obs, delay_w=delay_w,
        y0_template=y0_template,
        build_pars_fn=build_pars_fn,
        use_pso=use_pso_for_fit, # <-- This stays TRUE
        pso_particles=fit_params.get("pso_particles", 60),
        pso_iters=fit_params.get("pso_iters", 300),
        local_seeds=fit_params.get("local_seeds", 8),
        seed=fit_params.get("seed", 123),
        scoring_mask = scoring_mask
    )
    pd.Series(best["x"], index=current_free_names).to_csv(slice_outdir / "best_params.csv")
    
    ####### 6. Recompute trajectories (AIC/BIC) for this slice ######
    print("  Recomputing best-fit trajectories...") 
    
    theta_best = current_fixed.copy() 
    theta_best.update({k: v for k, v in zip(current_free_names, best["x"])})
    pd.Series(theta_best).to_csv(slice_outdir / "all_effective_params.csv")
    
    pars = build_pars_fn(theta_best)

    R0_val = theta_best.get("R0", y0_template.get("R0", 0.0))
    y0_fit = np.array([
        N - theta_best["I0"] - theta_best["C0"] - R0_val,
        theta_best["I0"], 
        theta_best["C0"], 
        R0_val,
        theta_best["P0"], 
        theta_best["F0"]
    ], float)
    
    if not np.all(np.isfinite(y0_fit)):
        print(f"  ERROR: Non-finite y0 created. Skipping simulation.")
        return None
    
    # Rebuild kernel if it was a free parameter
    local_delay_w = delay_w 
    
    # But if 'delay_mean' was a free param, rebuild the kernel using the best-fit value
    if "delay_mean" in theta_best:
        print("  Rebuilding best-fit delay kernel for output...")
        delay_cfg = params.get("obs", {})
        # Use discrete_gamma_kernel (not _cached) since params are exact
        local_delay_w = discrete_gamma_kernel(
            mean_days=float(theta_best["delay_mean"]),
            sd_days=float(delay_cfg.get("delay_sd", 2.0)),
            max_lag=int(delay_cfg.get("delay_maxlag", 14))
        )
    
    try:
        mu, Y_fit, inc = make_mu_from_model(
            pars, y0_fit, t_eval, 
            delay_w=local_delay_w,
            rho_obs=theta_best["rho_obs"]
        )
        pd.DataFrame({"mu":mu, "inc":inc, "y":y_obs}).to_csv(slice_outdir / "fit_series.csv", index=False)
        y_final_state = Y_fit[-1] 
    
    except Exception as e:
        print(f"  ERROR: Simulation failed with best-fit params: {e}")
        (slice_outdir / "simulation_error.txt").write_text(str(e))
        return None
    
    ###### 7. Run diagnostics ######
    burn = int(params["obs"]["delay_maxlag"])
    mask = np.ones_like(y_obs, dtype=bool)
    mask[:burn] = False

    p_params = len(current_free_names)
    log_theta_best = float(theta_best["log_theta"])

    diag = info_criteria_nb(
        y=y_obs,
        mu=mu,
        log_theta=log_theta_best,
        p_params=p_params,
        mask=mask,
    )
    pd.Series(diag).to_csv(slice_outdir / "diagnostics.csv")
    print(
        "Diagnostics (masked={}): n={n:.0f}, NLL={nll:.2f}, NLL/n={nll_per_obs:.4f}, "
        "AIC={AIC:.2f}, BIC={BIC:.2f}, k={k:.0f}".format(
            "yes" if mask is not None else "no", **diag
        )
    )

    ###### 8. Run Profile likelyhoods ######
    print("  Running profile likelihoods...")
    profile_params = params.get("profiles", {})
    if not profile_params:
        print("  No profiles defined in params.yaml, skipping.")
        
    for pname, grid in profile_params.items():
        if pname not in current_free_names:
            print(f"  Skipping profile for '{pname}' (it is fixed for this slice).")
            continue 
            
        print(f"  Profiling '{pname}'...")
        g = np.linspace(grid[0], grid[1], grid[2])
        G, Fvals, Xs = profile_likelihood(
            pname, g, best,
            current_free_names,
            current_bounds,
            current_fixed,
            t_eval, y_obs, local_delay_w, y0_template, build_pars_fn
        )
        pd.DataFrame({"param":pname, "grid":G, "negloglik":Fvals}).to_csv(slice_outdir / f"profile_{pname}.csv", index=False)

    ###### 9. Runs stability analysis ######
    print("  Running oscillation diagnostics...")
    period, fstar, f, Pxx = dominant_period_days(mu)
    pd.DataFrame({"f":f, "Pxx":Pxx}).to_csv(slice_outdir / "psd.csv", index=False)
    try:
        y_guess_full = Y_fit[-1]
        y_guess_reduced = y_guess_full[1:]
        y_star_reduced = equilibrium_newton_reduced(sirc_pf_rhs_reduced, y_guess_reduced, pars, N, sirc_pf_rhs)
        J_reduced = numerical_jacobian_reduced(sirc_pf_rhs_reduced, y_star_reduced, pars, N, sirc_pf_rhs)
        evals = np.linalg.eigvals(J_reduced)
        pd.DataFrame({"re":evals.real, "im":evals.imag}).to_csv(slice_outdir / "jacobian_eigs.csv", index=False)
        I,C,R,P,F = y_star_reduced
        S = N - I - C - R
        y_star_full = np.array([S,I,C,R,P,F])
        pd.Series(y_star_full, index=list("SICRPF")).to_csv(slice_outdir / "equilibrium_state.csv")
    except Exception as e:
        print(f"  Jacobian/Equilibrium analysis failed: {e}")
        (slice_outdir / "jacobian_error.txt").write_text(str(e))

    ###### 10. External Validation ######
    print("  (Skipping external validation)")
    # We ended up not doing it but it could have been cool
    # Using mobility data as a proxy for compliance, for example
    
    print(f"  Slice {slice_id+1} complete.")

    return True



def main():
    """
    Main entry point for the sequential fitting analysis.

    Orchestrates the entire run. It loads the `params.yaml`
    config and the sliced data. It computes the (potentially fixed)
    reporting delay kernel `w`.

    It then loops through each data slice, calling `analyze_slice` and
    passing the state, incidence tail, and best-fit parameters from the
    completed slice to the next one in the loop, enabling the "soft
    hand-off" and "warm-start" sequential fitting.

    """
    # Setup output
    ts = time.strftime("%Y-%m-%d_%H%M%S")
    base_outdir = Path("reports") / ts
    base_outdir.mkdir(parents=True, exist_ok=True)
    print(f"Starting analysis. Results will be in {base_outdir}")

    # Load params once
    try:
        params = yaml.safe_load(open("params.yaml"))
    except FileNotFoundError:
        print("Error: params.yaml not found. Please create one.")
        return

    # Delay kernel (computed once if fixed, else placeholder)
    w = None 
    if "delay_mean" in params.get("free_names", []):
        print("INFO: 'delay_mean' is a free parameter. Kernel will be built inside the optimizer.")
        # Get max_lag from params to create a placeholder kernel.
        # This is needed for buffer/tail logic, but the values won't be
        # used for convolution (it will be rebuilt by the optimizer).
        max_lag = int(params["obs"]["delay_maxlag"])
        w = np.zeros(max_lag + 1)
        w[0] = 1.0
    else:
        print("INFO: Using fixed 'delay_mean' from params.yaml.")
        try:
            w = discrete_gamma_kernel(
                mean_days=params["obs"]["delay_mean"], 
                sd_days=params["obs"]["delay_sd"], 
                max_lag=params["obs"]["delay_maxlag"]
            )
        except KeyError as e:
            print(f"ERROR: '{e.args[0]}' not found in params['obs'].")
            print("If 'delay_mean' is not in 'free_names', it must be defined in 'obs'.")
            return

    # Load sliced data
    try:
        sliced_data_dir = Path("data") / "sliced_data"
        slice_dataframes = load_sliced_data(sliced_data_dir, num_slices=6)
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    
    ###### MAIN LOOP ######
    for i, slice_df in enumerate(slice_dataframes):
        
        # Pass the fit sub-dictionary for clarity
        fit_params = params.get("fit", {})
        
        analysis_success = analyze_slice(
            slice_id=i,
            slice_data_df=slice_df,
            params=params,
            base_outdir=base_outdir,
            delay_w=w,
            fit_params=fit_params
        )

        if analysis_success is None:
            print(f"CRITICAL ERROR: Slice {i+1} failed to fit. Stopping analysis.")
            break
        
    print(f"\nDone. All results in {base_outdir}")

if __name__ == "__main__":
    main()