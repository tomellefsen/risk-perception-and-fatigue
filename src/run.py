from __future__ import annotations
import os, json, time, yaml
import numpy as np
import pandas as pd
from pathlib import Path
import glob

from src.observation import discrete_gamma_kernel, make_mu_from_model, nb_loglik
from src.fit import fit_pso_then_local, profile_likelihood
from src.metrics import mae, rmse, nb_mean_log_pred_density, simulate_nb_intervals
from src.model import sirc_pf_rhs
from src.diagnostics import info_criteria_nb

from src.dynamics import (
    dominant_period_days, 
    equilibrium_newton_REDUCED,
    numerical_jacobian_REDUCED,
    sirc_pf_rhs_REDUCED     
)

def build_pars_fn(theta_dict):
    # Map flat theta_dict into 'pars' for the ODE
    keys = ["N","beta0","gamma","rho","alpha","delta","epsilon","phi","k","compliance_max","beta_floor"]
    pars = {k: float(theta_dict[k]) for k in keys if k in theta_dict}
    return pars

def load_sliced_data(data_dir: Path, num_slices: int = 6):
    """
    Loads pre-sliced CSV files from a directory.
    
    Expects files to be named sequentially, e.g.,
    'slice_0.csv', 'slice_1.csv', ..., 'slice_5.csv'
    """
    print(f"Loading sliced data from {data_dir}...")
    slice_dataframes = []
    
    # Find files by explicit name to ensure order
    for i in range(num_slices):
        fname = data_dir / f"slice_{i}.csv"
        if not fname.exists():
            raise FileNotFoundError(
                f"Missing required slice file: {fname}\n"
                f"Please ensure {num_slices} files named 'slice_0.csv' through 'slice_{num_slices-1}.csv' exist."
            )
        print(f"  Loading {fname.name}...")
        slice_dataframes.append(pd.read_csv(fname))
        
    print(f"Successfully loaded {len(slice_dataframes)} data slices.")
    return slice_dataframes

# run.py (Replace your existing analyze_slice function with this)

def analyze_slice(
    slice_id: int, 
    slice_data_df: pd.DataFrame, # <-- NEW
    params: dict,
    base_outdir: Path,
    delay_w: np.ndarray,
    y0_guess_from_prev: np.ndarray | None = None,
    fit_params: dict = {},
    prev_inc_tail: np.ndarray | None = None,
):
    """
    Runs the entire analysis pipeline for a single data slice.
    
    If y0_guess_from_prev is provided (i.e., for slices > 0),
    it will FIX the initial conditions (I0, C0, R0, P0, F0)
    to the end-state of the previous slice.
    """
    
    # --- 1. Setup Slice Directories and Data ---
    slice_data = slice_data_df
    slice_outdir = base_outdir / f"slice_{slice_id}"
    slice_outdir.mkdir(parents=True, exist_ok=True)

    print(f"\n--- Analyzing Slice {slice_id} (Length: {len(slice_data)} days) ---")
    print(f"  Results will be in: {slice_outdir}")

    y_obs = slice_data["cases"].to_numpy(dtype=float)
    T = len(y_obs)
    t_eval = np.arange(T, dtype=float)
    N = float(params["fixed"]["N"])
    
    burn = int(params["obs"]["delay_maxlag"])  # 14
    scoring_mask = np.ones(T, dtype=bool)
    scoring_mask[:burn] = False
    
    # --- Carry-in buffer safeguard (first slice) ---
    if prev_inc_tail is None:
        prev_inc_tail = np.zeros(len(delay_w) - 1, dtype=float)

    # --- 2. Setup Fit Parameters (NEW LOGIC) ---
    
    # These are the *master* lists from params.yaml
    master_free_names = list(params["free_names"])
    master_bounds = [tuple(b) for b in params["bounds"]]
    
    # These will be modified for slices 1+
    current_free_names = master_free_names.copy()
    current_bounds = master_bounds.copy()
    current_fixed = params["fixed"].copy()
    
    # This template is used by the objective function
    y0_template = {"S0": N, "R0": 0.0} 
    
    # These are the y0 state parameters we want to fix
    y0_param_names_to_fix = ["I0", "C0", "R0", "P0", "F0"]
    
    if y0_guess_from_prev is None:
        # --- SLICE 0 (Fit y0) ---
        print("  Slice 0: Fitting initial conditions (I0, C0, etc.).")
        # Use all parameters as defined in params.yaml
        
    else:
        # --- SLICE 1+ (Fix y0) ---
        print("  Slice 1+: Fixing initial conditions from previous slice's end-state.")
        
        # y0_guess_from_prev state is [S, I, C, R, P, F]
        y0_end_state_map = {
            "S0": y0_guess_from_prev[0],
            "I0": y0_guess_from_prev[1],
            "C0": y0_guess_from_prev[2],
            "R0": y0_guess_from_prev[3],
            "P0": y0_guess_from_prev[4],
            "F0": y0_guess_from_prev[5]
        }
        
        new_free_names = []
        new_bounds = []
        
        # Filter free_names and bounds, moving y0 params to 'fixed'
        for name, bound in zip(master_free_names, master_bounds):
            if name in y0_param_names_to_fix:
                # This parameter is now FIXED. Add it to the fixed dict.
                val = y0_end_state_map.get(name)
                if val is not None:
                    current_fixed[name] = float(val)
            else:
                # This parameter is still FREE.
                new_free_names.append(name)
                new_bounds.append(bound)
        
        # Update the lists we'll pass to the optimizer
        current_free_names = new_free_names
        current_bounds = new_bounds
        
        # Update the y0_template for the objective function
        if "R0" in current_fixed:
            y0_template["R0"] = current_fixed["R0"]
            
        print(f"  Fixed y0 params: "
              f"I0={current_fixed.get('I0', 'N/A'):.2f}, C0={current_fixed.get('C0', 'N/A'):.2f}, "
              f"R0={current_fixed.get('R0', 'N/A'):.2f}, P0={current_fixed.get('P0', 'N/A'):.4f}, "
              f"F0={current_fixed.get('F0', 'N/A'):.4f}")
        print(f"  Remaining {len(current_free_names)} free params: {current_free_names}")


    # --- 3. Fit Model to this Slice ---
    print("  Fitting model...")
    best = fit_pso_then_local(
        x0_guess=None, 
        param_names=current_free_names,  # <-- Use modified list
        bounds=current_bounds,           # <-- Use modified list
        fixed=current_fixed,             # <-- Use modified dict
        t_eval=t_eval, y_obs=y_obs, delay_w=delay_w,
        y0_template=y0_template,         # <-- Use modified template
        build_pars_fn=build_pars_fn,
        use_pso=fit_params.get("use_pso", True),
        pso_particles=fit_params.get("pso_particles", 60),
        pso_iters=fit_params.get("pso_iters", 300),
        local_seeds=fit_params.get("local_seeds", 8),
        seed=fit_params.get("seed", 123),
        scoring_mask = scoring_mask,
        prev_inc_tail=prev_inc_tail
    )
    # Save best *free* params
    pd.Series(best["x"], index=current_free_names).to_csv(slice_outdir / "best_params.csv")
    
    # --- 4. Recompute Trajectories for this Slice ---
    print("  Recomputing best-fit trajectories...") 
    
    # Rebuild the *full* theta dict, combining fixed and fitted params
    theta_best = current_fixed.copy() 
    theta_best.update({k: v for k, v in zip(current_free_names, best["x"])})
    
    # Save *all* effective parameters (fixed + fitted) for this slice
    pd.Series(theta_best).to_csv(slice_outdir / "all_effective_params.csv")
    
    pars = build_pars_fn(theta_best)

    # Build the y0 state vector *from theta_best*
    # This now works for both Slice 0 (fitted) and Slice 1+ (fixed)
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
        print(f"  y0_fit: {y0_fit}")
        # Return None so the next slice defaults to fitting y0
        return None 
    
    try:
        rho_obs = float(theta_best["rho_obs"])
        mu, Y_fit, inc = make_mu_from_model(pars, y0_fit, t_eval, delay_w, rho_obs)
        pd.DataFrame({"mu":mu, "inc":inc, "y":y_obs}).to_csv(slice_outdir / "fit_series.csv", index=False)
        
        # This is the y0 for the *next* slice
        y_final_state = Y_fit[-1] 
    
    except Exception as e:
        print(f"  ERROR: Simulation failed with best-fit params: {e}")
        (slice_outdir / "simulation_error.txt").write_text(str(e))
        return None # Don't pass a bad state to the next slice
    
    # Save tail for next slice
    L = len(delay_w)
    tail = inc[-(L-1):].copy() if len(inc) >= L-1 else inc.copy()
    np.savetxt(slice_outdir / "inc_tail.txt", tail)
    
    # --- 4b. Diagnostics: NLL per obs, AIC, BIC ---

    # Optional: burn-in mask to ignore delay spillover at the slice head
    burn = int(params["obs"]["delay_maxlag"])
    mask = np.ones_like(y_obs, dtype=bool)
    mask[:burn] = False

    p_params = len(current_free_names)   # number of *fitted* params this slice
    log_theta_best = float(theta_best["log_theta"])

    diag = info_criteria_nb(
        y=y_obs,
        mu=mu,
        log_theta=log_theta_best,
        p_params=p_params,
        mask=mask,
    )

    # Save and print
    pd.Series(diag).to_csv(slice_outdir / "diagnostics.csv")

    print(
        "Diagnostics (masked={}): n={n:.0f}, NLL={nll:.2f}, NLL/n={nll_per_obs:.4f}, "
        "AIC={AIC:.2f}, BIC={BIC:.2f}, k={k:.0f}".format(
            "yes" if mask is not None else "no", **diag
        )
    )

    # --- 5. Profile Likelihoods ---
    print("  Running profile likelihoods...")
    profile_params = params.get("profiles", {})
    if not profile_params:
        print("  No profiles defined in params.yaml, skipping.")
        
    for pname, grid in profile_params.items():
        # Check if this param was fixed for this slice
        if pname not in current_free_names:
            print(f"  Skipping profile for '{pname}' (it is fixed for this slice).")
            continue 
            
        print(f"  Profiling '{pname}'...")
        g = np.linspace(grid[0], grid[1], grid[2])
        G, Fvals, Xs = profile_likelihood(
            pname, g, best,
            current_free_names, # <-- Use modified
            current_bounds,     # <-- Use modified
            current_fixed,      # <-- Use modified
            t_eval, y_obs, delay_w, y0_template, build_pars_fn
        )
        pd.DataFrame({"param":pname, "grid":G, "negloglik":Fvals}).to_csv(slice_outdir / f"profile_{pname}.csv", index=False)

    # --- 6. Oscillation Diagnostics (FIXED) ---
    print("  Running oscillation diagnostics...")
    period, fstar, f, Pxx = dominant_period_days(mu)
    pd.DataFrame({"f":f, "Pxx":Pxx}).to_csv(slice_outdir / "psd.csv", index=False)

    try:
        y_guess_full = Y_fit[-1]  # Use last state as guess
        y_guess_reduced = y_guess_full[1:] # Get [I, C, R, P, F]
        
        y_star_reduced = equilibrium_newton_REDUCED(
            sirc_pf_rhs_REDUCED, y_guess_reduced, pars, N, sirc_pf_rhs
        )
        
        J_reduced = numerical_jacobian_REDUCED(
            sirc_pf_rhs_REDUCED, y_star_reduced, pars, N, sirc_pf_rhs
        )
        evals = np.linalg.eigvals(J_reduced)
        
        pd.DataFrame({"re":evals.real, "im":evals.imag}).to_csv(slice_outdir / "jacobian_eigs.csv", index=False)
        
        I,C,R,P,F = y_star_reduced
        S = N - I - C - R
        y_star_full = np.array([S,I,C,R,P,F])
        pd.Series(y_star_full, index=list("SICRPF")).to_csv(slice_outdir / "equilibrium_state.csv")

    except Exception as e:
        print(f"  Jacobian/Equilibrium analysis failed: {e}")
        (slice_outdir / "jacobian_error.txt").write_text(str(e))

    # --- 7. External Validation ---
    print("  (Skipping external validation)")
    
    print(f"  Slice {slice_id} complete.")
    return y_final_state, tail # Pass this to the next slice

def main():
    # --- Setup output ---
    ts = time.strftime("%Y-%m-%d_%H%M%S")
    base_outdir = Path("reports") / ts
    base_outdir.mkdir(parents=True, exist_ok=True)
    print(f"Starting analysis. Results will be in {base_outdir}")

    # --- Load params ONCE ---
    try:
        params = yaml.safe_load(open("params.yaml"))
    except FileNotFoundError:
        print("Error: params.yaml not found. Please create one.")
        return

    # Delay kernel (computed once)
    w = discrete_gamma_kernel(
        mean_days=params["obs"]["delay_mean"], 
        sd_days=params["obs"]["delay_sd"], 
        max_lag=params["obs"]["delay_maxlag"]
    )

    # --- Load Sliced Data ---
    try:
        sliced_data_dir = Path("data") / "sliced_data"
        slice_dataframes = load_sliced_data(sliced_data_dir, num_slices=6)
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # --- Loop and Analyze Each Slice ---
    y0_for_next_slice = None
    prev_inc_tail = None
    for i, slice_df in enumerate(slice_dataframes):
        
        # Pass the 'fit' sub-dictionary for clarity
        fit_params = params.get("fit", {})
        
        y0_for_next_slice, prev_inc_tail = analyze_slice(
            slice_id=i,
            slice_data_df=slice_df, # <-- NEW: Pass the DataFrame
            params=params,
            base_outdir=base_outdir,
            delay_w=w,
            y0_guess_from_prev=y0_for_next_slice,
            fit_params=fit_params,
            prev_inc_tail=prev_inc_tail
        )

    print(f"\nDone. All results in {base_outdir}")

if __name__ == "__main":
    main()