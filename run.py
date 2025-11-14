from __future__ import annotations
import os, json, time, yaml, glob
import numpy as np
import pandas as pd
from pathlib import Path

# Import your project's modules
try:
    from src.observation import discrete_gamma_kernel, make_mu_from_model, nb_loglik
    from src.fit import fit_pso_then_local, profile_likelihood
    from src.metrics import mae, rmse, nb_mean_log_pred_density, simulate_nb_intervals
    from src.model import sirc_pf_rhs
    #from src.external_validation import bootstrap_ci_for_lagcorr
    
    # Import the CORRECT dynamics functions (with the reduced-state fix)
    from src.dynamics import (
        dominant_period_days, 
        equilibrium_newton_REDUCED,
        numerical_jacobian_REDUCED,
        sirc_pf_rhs_REDUCED
    )
except ImportError as e:
    print(f"ImportError: {e}")
    print("Could not import from 'src'. Make sure 'run.py' is in the root directory.")
    exit()


def build_pars_fn(theta_dict):
    """Map flat theta_dict into 'pars' for the ODE"""
    keys = ["N","beta0","gamma","rho","alpha","delta","epsilon","phi","k","compliance_max","beta_floor"]
    pars = {k: float(theta_dict[k]) for k in keys if k in theta_dict}
    return pars

def load_sliced_data(data_dir: Path, num_slices: int = 6):
    """
    Loads pre-sliced CSV files from a directory.
    Expects files to be named 'slice_0.csv', 'slice_1.csv', etc.
    """
    print(f"Loading sliced data from {data_dir}...")
    slice_dataframes = []
    
    for i in range(num_slices):
        fname = data_dir / f"slice_{i+1}.csv"
        if not fname.exists():
            raise FileNotFoundError(
                f"Missing required slice file: {fname}\n"
                f"Please ensure {num_slices} files named 'slice_0.csv' through 'slice_{num_slices-1}.csv' exist."
            )
        print(f"  Loading {fname.name}...")
        slice_dataframes.append(pd.read_csv(fname))
        
    print(f"Successfully loaded {len(slice_dataframes)} data slices.")
    return slice_dataframes

def analyze_slice(
    slice_id: int, 
    slice_data_df: pd.DataFrame,
    params: dict,
    base_outdir: Path,
    delay_w: np.ndarray,
    y0_guess_from_prev: np.ndarray | None = None
):
    """
    Runs the ENTIRE analysis pipeline for a single data slice.
    This includes fitting, profiling, and dynamics checks.
    """
    
    # --- 1. Setup Slice Directories and Data ---
    slice_data = slice_data_df
    slice_outdir = base_outdir / f"slice_{slice_id}"
    slice_outdir.mkdir(parents=True, exist_ok=True)

    print(f"\n" + "="*60)
    print(f"--- Analyzing Slice {slice_id} (Length: {len(slice_data)} days) ---")
    print(f"  Results will be in: {slice_outdir}")
    print("="*60)

    y_obs_full = slice_data["cases"].to_numpy(dtype=float)
    
    # --- DATA PREPARATION FIX ---
    non_zero_indices = np.where(y_obs_full > 0)[0]
    
    if len(non_zero_indices) == 0:
        print("  WARNING: This data slice contains NO cases. Skipping.")
        return y0_guess_from_prev
    
    start_index = max(0, non_zero_indices[0] - 3) 
    y_obs = y_obs_full[start_index:]
    
    T = len(y_obs)
    t_eval = np.arange(T, dtype=float)
    N = float(params["fixed"]["N"])
    
    print(f"  Trimming data: Original length {len(y_obs_full)}, new length {T} (starting 3 days before first case).")
    # --- END DATA PREPARATION FIX ---

    # --- 2. Setup Fit Parameters (NEW LOGIC) ---
    master_free_names = list(params["free_names"])
    master_bounds = [tuple(b) for b in params["bounds"]]
    
    current_free_names = master_free_names.copy()
    current_bounds = master_bounds.copy()
    current_fixed = params["fixed"].copy()
    
    y0_template = {"S0": N, "R0": 0.0} 
    y0_param_names_to_fix = ["I0", "C0", "R0", "P0", "F0"]
    
    if y0_guess_from_prev is None or slice_id == 0:
        print("  Slice 0: Fitting initial conditions (I0, C0, etc.).")
        # We must load o_params.yaml to ensure y0 params are free
        try:
            with open('o_params.yaml', 'r') as f:
                o_params = yaml.safe_load(f)
            print("  Loaded 'o_params.yaml' to get full parameter list for slice 0.")
            current_free_names = list(o_params["free_names"])
            current_bounds = [tuple(b) for b in o_params["bounds"]]
            current_fixed = o_params.get("fixed", {}).copy() # Use only o_params fixed
        except FileNotFoundError:
            print("  WARNING: 'o_params.yaml' not found. Using 'params.yaml' for slice 0.")
            # Fallback to params.yaml, but this may fail if y0 not free
            pass # Keep params as loaded

    else:
        print("  Slice > 0: Fixing initial conditions from previous slice's end-state.")
        y0_end_state_map = {
            "S0": y0_guess_from_prev[0], "I0": y0_guess_from_prev[1],
            "C0": y0_guess_from_prev[2], "R0": y0_guess_from_prev[3],
            "P0": y0_guess_from_prev[4], "F0": y0_guess_from_prev[5]
        }
        
        new_free_names = []
        new_bounds = []
        
        for name, bound in zip(master_free_names, master_bounds):
            if name in y0_param_names_to_fix:
                val = y0_end_state_map.get(name)
                if val is not None:
                    current_fixed[name] = float(val)
            else:
                new_free_names.append(name)
                new_bounds.append(bound)
        
        current_free_names = new_free_names
        current_bounds = new_bounds
        
        if "R0" in current_fixed:
            y0_template["R0"] = current_fixed["R0"]
            
        print(f"  Fixed y0 params: I0={current_fixed.get('I0', 'N/A'):.2f}, C0={current_fixed.get('C0', 'N/A'):.2f}, ...")
        print(f"  Remaining {len(current_free_names)} free params: {current_free_names}")

    # --- 3. Fit Model to this Slice ---
    print("  Fitting model to this slice...")
    fit_params = params.get("fit", {})
    best = fit_pso_then_local(
        x0_guess=None, 
        param_names=current_free_names,
        bounds=current_bounds,
        fixed=current_fixed,
        t_eval=t_eval, y_obs=y_obs, delay_w=delay_w,
        y0_template=y0_template,
        build_pars_fn=build_pars_fn,
        use_pso=fit_params.get("use_pso", True),
        pso_particles=fit_params.get("pso_particles", 60),
        pso_iters=fit_params.get("pso_iters", 300),
        local_seeds=fit_params.get("local_seeds", 8),
        seed=fit_params.get("seed", 123)
    )
    pd.Series(best["x"], index=current_free_names).to_csv(slice_outdir / "best_params.csv")
    
    # --- 4. Recompute Trajectories for this Slice ---
    print("  Recomputing best-fit trajectories...")
    theta_best = current_fixed.copy() 
    theta_best.update({k: v for k, v in zip(current_free_names, best["x"])})
    
    pd.Series(theta_best).to_csv(slice_outdir / "all_effective_params.csv")
    pars = build_pars_fn(theta_best)

    try:
        R0_val = theta_best.get("R0", y0_template.get("R0", 0.0))
        y0_fit = np.array([
            N - theta_best["I0"] - theta_best["C0"] - R0_val,
            theta_best["I0"], theta_best["C0"], R0_val,
            theta_best["P0"], theta_best["F0"]
        ], float)
    except KeyError as e:
        print(f"  FATAL ERROR: Missing y0 parameter ({e}) even after slice 0 logic.")
        print("  This is likely because 'o_params.yaml' is missing I0, C0, P0, or F0.")
        return None

    if not np.all(np.isfinite(y0_fit)):
        print(f"  ERROR: Non-finite y0 created. Skipping simulation.")
        return None 
    
    try:
        rho_report = float(theta_best["rho_obs"])
        mu, Y_fit, inc = make_mu_from_model(pars, y0_fit, t_eval, delay_w, rho=rho_report)
        pd.DataFrame({"mu":mu, "inc":inc, "y":y_obs}).to_csv(slice_outdir / "fit_series.csv", index=False)
        y_final_state = Y_fit[-1] 
    except Exception as e:
        print(f"  ERROR: Simulation failed with best-fit params: {e}")
        (slice_outdir / "simulation_error.txt").write_text(str(e))
        return None

    # --- 5. Profile Likelihoods (MOVED INSIDE LOOP) ---
    print("  Running profile likelihoods for this slice...")
    profile_params = params.get("profiles", {})
    if not profile_params:
        print("  No profiles defined in params.yaml, skipping.")
        
    for pname, grid_config in profile_params.items():
        if pname not in current_free_names:
            print(f"  Skipping profile for '{pname}' (it is fixed for this slice).")
            continue 
            
        print(f"  Profiling '{pname}'...")
        g = np.linspace(grid_config[0], grid_config[1], int(grid_config[2]))
        G, Fvals, Xs = profile_likelihood(
            pname, g, best,
            current_free_names, current_bounds, current_fixed,
            t_eval, y_obs, delay_w, y0_template, build_pars_fn
        )
        pd.DataFrame({"param":pname, "grid":G, "negloglik":Fvals}).to_csv(slice_outdir / f"profile_{pname}.csv", index=False)

    # --- 6. Oscillation Diagnostics (MOVED INSIDE LOOP) ---
    print("  Running oscillation diagnostics for this slice...")
    period, fstar, f, Pxx = dominant_period_days(mu)
    pd.DataFrame({"f":f, "Pxx":Pxx}).to_csv(slice_outdir / "psd.csv", index=False)

    try:
        y_guess_full = Y_fit[-1]
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
    
    print(f"  Slice {slice_id} complete.")
    return y_final_state

def main():
    """
    Main controller function.
    Loads data and params ONCE, then loops through slices,
    calling 'analyze_slice' for each one.
    """
    # --- Setup output ---
    ts = time.strftime("%Y-%m-%d_%H%M%S")
    base_outdir = Path("reports") / ts
    base_outdir.mkdir(parents=True, exist_ok=True)
    print(f"Starting analysis. Results will be in {base_outdir}")

    # --- Load params ONCE ---
    try:
        # Load the REDUCED params file
        params = yaml.safe_load(open("params.yaml")) 
    except FileNotFoundError:
        print("Error: params.yaml not found.")
        print("Please run 'pre_run_sensitivity.py' first to generate it.")
        return

    # --- Load delay kernel ONCE ---
    obs_params = params.get("obs", {})
    w = discrete_gamma_kernel(
        mean_days=obs_params.get("delay_mean", 3.0), 
        sd_days=obs_params.get("delay_sd", 2.0), 
        max_lag=obs_params.get("delay_maxlag", 14)
    )

    # --- Load Sliced Data ONCE ---
    try:
        sliced_data_dir = Path("data") / "sliced_data"
        slice_dataframes = load_sliced_data(sliced_data_dir, num_slices=6)
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # --- Loop and Analyze Each Slice ---
    y0_for_next_slice = None
    for i, slice_df in enumerate(slice_dataframes):
        
        y0_for_next_slice = analyze_slice(
            slice_id=i,
            slice_data_df=slice_df,
            params=params,
            base_outdir=base_outdir,
            delay_w=w,
            y0_guess_from_prev=y0_for_next_slice
        )

    print(f"\nDone. All slice results are in {base_outdir}")

if __name__ == "__main__":
    # This block is crucial for Python scripts
    main()