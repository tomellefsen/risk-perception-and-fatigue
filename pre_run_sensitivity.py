# This module was useful for screening parameters, but the params file has
# since been modified with arbitrairy parameter choices, see said file for
# details

from __future__ import annotations
import numpy as np
import pandas as pd
import yaml
import time
from pathlib import Path
import matplotlib.pyplot as plt

# From /src
try:
    from src.model import sirc_pf_rhs, simulate
    from src.observation import make_mu_from_model, discrete_gamma_kernel
    from src.dynamics import dominant_period_days
    from run import build_pars_fn
    from SALib.sample import morris as morris_sample
    from SALib.analyze import morris as morris_analyze
    from tqdm import tqdm

except ImportError as e:
    print(f"ImportError: {e}")
    print("Mismatch in directory structure: ensure this file is in root and /src is present.")
    exit(1)

# -----------------------------------
# CONFIGS

O_PARAMS_FILE = "o_params.yaml" 
NEW_PARAMS_FILE = "params.yaml" 
DATA_SLICE_FILE = Path("data") / "sliced_data" / "slice_1.csv" 
TOP_N_PARAMS_TO_SELECT = 7 

MORRIS_N = 512
MORRIS_LEVELS = 6

# -----------------------------------

def load_all_params_from_yaml(filepath):
    """
    Loads the yaml file and prepares the 'problem' definition
    for SALib from *all* free_names.
    """
    try:
        with open(filepath, 'r') as f:
            params = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Error: {filepath} not found.")
        print(f"Please rename your original params file to {filepath}")
        exit(1)
        
    ###### SALib problem definition ######
    # We test ALL parameters listed in free_names
    problem = {
        'num_vars': len(params['free_names']),
        'names': list(params['free_names']),
        'bounds': [tuple(b) for b in params['bounds']]
    }
    
    # We also need the fixed params and the obs delay params
    fixed_params = params['fixed'].copy()
    obs_params = params['obs'].copy()
    
    return problem, fixed_params, obs_params, params

def get_simulator_fn(problem, fixed_params, obs_params, y0_template, t_eval):
    """
    Returns the simulator_fn required by SALib.
    This function will be called repeatedly by the Morris sampler.
    """
    # We pre-compute the delay kernel once
    delay_w = discrete_gamma_kernel(
        mean_days=obs_params["delay_mean"], 
        sd_days=obs_params["delay_sd"], 
        max_lag=obs_params["delay_maxlag"]
    )
    N = float(fixed_params['N'])

    def simulator_function(param_vector):
        """
        The function that SALib will call.
        
        Args:
            param_vector (np.ndarray): A vector of free parameter values.
            
        Returns:
            dict: A dictionary of simulation outputs (e.g., {"mu": ...})
        """
        # Assemble full parameter set
        theta = fixed_params.copy()
        theta.update({name: val for name, val in zip(problem['names'], param_vector)})
        
        # Build ODE params
        pars = build_pars_fn(theta)
        
        # Build y0
        # We assume y0 params (I0, C0...) are part of the test vector
        R0 = theta.get("R0", y0_template.get("R0", 0.0))
        I0 = theta.get("I0", 1.0) # Default to 1 if not in vector
        C0 = theta.get("C0", 1.0)
        P0 = theta.get("P0", 0.0)
        F0 = theta.get("F0", 0.0)
        S0 = N - I0 - C0 - R0
        
        y0 = np.array([S0, I0, C0, R0, P0, F0], float)
        
        # Run simulation
        try:
            rho_obs = float(theta["rho_obs"])
            mu, Y_fit, inc = make_mu_from_model(
                pars, y0, t_eval, delay_w, rho_obs
            )
            return {"mu": mu, "inc": inc, "Y": Y_fit}
        
        except Exception as e:
            # Simulation failed (bad params), return NaNs
            print(f"SIMULATION FAILED. Error: {e}. Params: {param_vector}")
            
            nan_array = np.full(len(t_eval), np.nan)
            return {"mu": nan_array, "inc": nan_array, "Y": None}

    return simulator_function

def get_summary_fn():
    """
    Returns the summary_fn required by SALib.
    This function computes summary statistics from the simulator output.
    """
    def summary_function(sim_out):
        """
        Args:
            sim_out (dict): The output from simulator_function
        
        Returns:
            np.ndarray: A vector of summary statistics.
        """
        mu = sim_out['mu']
        
        # Handle failed runs
        if mu is None or np.isnan(mu).all():
            return np.array([np.nan, np.nan, np.nan])
            
        ###### Define Summary Statistics ######
        # Mean of observed cases (Total magnitude)
        stat_1_mean_mu = np.nanmean(mu)
        
        # Std. Dev. of observed cases (Variability)
        stat_2_std_mu = np.nanstd(mu)
        
        # Dominant period of cases (Oscillation)
        try:
            # Filter NaNs for welch
            mu_finite = mu[np.isfinite(mu)]
            if len(mu_finite) < 20: # Not enough data
                raise ValueError("Not enough finite data for PSD")
            period, _, _, _ = dominant_period_days(mu_finite)
            stat_3_period = period if np.isfinite(period) else 0.0
        except Exception:
            stat_3_period = 0.0 # Failed Welch
            
        return np.array([stat_1_mean_mu, stat_2_std_mu, stat_3_period])
        
    summary_stat_names = ["Mean Cases", "Std. Dev. Cases", "Dominant Period"]
    
    return summary_function, summary_stat_names

def plot_tornado(res, problem, stat_name, out_file):
    """
    Generates and saves a tornado plot for the Morris results.
    Plots mu_star (mean absolute elementary effect)
    """
    mu_star = res['mu_star']
    names = problem['names']
    
    # Sort by influence
    sorted_indices = np.argsort(mu_star)
    sorted_names = [names[i] for i in sorted_indices]
    sorted_mu_star = mu_star[sorted_indices]
    
    plt.figure(figsize=(10, len(names) * 0.5))
    plt.barh(sorted_names, sorted_mu_star, align='center', color='skyblue')
    plt.xlabel("μ* (Mean Absolute Elementary Effect)")
    plt.title(f"Morris Sensitivity for: {stat_name}")
    plt.ylabel("Parameter")
    plt.tight_layout()
    plt.savefig(out_file)
    print(f"  Saved plot to {out_file}")
    plt.close()


def morris_screening_with_progress(
    problem, simulator_fn, summary_fn, 
    N=512, levels=6, grid_jump=2, seed=123
):
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
    
    X = morris_sample.sample(
        problem, N, num_levels=levels, 
        optimal_trajectories=None, seed=seed
    )
    
    Y = []
    
    # Progress bar
    print(f"  Running {len(X)} simulations for Morris screening...")
    for row in tqdm(X, total=len(X), desc="Screening Progress"):
        sim_out = simulator_fn(row)
        Y.append(summary_fn(sim_out))
        
    Y = np.asarray(Y)
    
    # Analyze each output dimension separately if multi-output
    results = []
    for j in range(Y.shape[1]):
        # Filter out NaN rows for this output, which can happen
        # if a simulation failed and summary_fn returned NaN
        valid_indices = ~np.isnan(Y[:, j])
        if np.sum(valid_indices) < 1:
            print(f"Warning: All simulations failed for output {j}. Skipping analysis.")
            # Add a dummy result
            results.append({'mu_star': np.zeros(problem['num_vars'])})
            continue

        X_filtered = X[valid_indices]
        Y_filtered = Y[valid_indices, j]

        res = morris_analyze.analyze(
            problem, X_filtered, Y_filtered, 
            print_to_console=False, 
            num_levels=levels
        )
        results.append(res)  # contains mu, mu_star, sigma
        
    return results

def propose_new_param_config(
    problem, all_params_yaml, morris_results_list, 
    stat_names, top_n, output_file
):
    """
    Analyzes combined results and SAVES a new params.yaml structure.
    """
    print("\n--- Parameter Influence Analysis ---")
    
    # Parameters ranked by average rank across all N metrics
    num_params = problem['num_vars']
    avg_rank = np.zeros(num_params)
    
    for res, name in zip(morris_results_list, stat_names):
        # Get rank for this statistic (highest mu_star = rank 0)
        ranks = np.argsort(-res['mu_star'])
        
        # Store ranks
        param_ranks_this_stat = np.empty(num_params)
        param_ranks_this_stat[ranks] = np.arange(num_params)
        avg_rank += param_ranks_this_stat

    avg_rank /= len(stat_names)
    
    # Get the indices of the top N parameters (lowest avg rank)
    top_n_indices = np.argsort(avg_rank)[:top_n]
    
    new_free_names = []
    new_bounds = []
    new_fixed_from_free = {}
    
    all_names = problem['names']
    all_bounds = problem['bounds']
    
    print(f"Top {top_n} most influential parameters (by avg. rank):")
    for i, idx in enumerate(top_n_indices):
        print(f"  {i+1}. {all_names[idx]} (Avg. Rank: {avg_rank[idx]:.2f})")
        new_free_names.append(all_names[idx])
        new_bounds.append(all_bounds[idx])
        
    print("\nParameters to be moved to 'fixed':")
    for i in range(num_params):
        if i not in top_n_indices:
            name = all_names[i]
            bounds = all_bounds[i]
            # Set to mean of range
            
            # These are parameters we are fixing. We should fix those 
            # from litterature, however most parameters depend 
            # heavily from the context, population, location, and the
            # exact strand of virus. Since fixing precise parameters
            # Is not really possible, the ranges present in o_params.yaml
            # are cannonical estimations of the valid "litterature" params.
            
            # We thus set our parameters to the mean of this range.
            # Since they are mostly inconsequential, as this screening
            # reveals, deviations from it's actual real value shouldn't
            # have much effect on the overall analysis.
            
            mean_val = (bounds[0] + bounds[1]) / 2.0
            new_fixed_from_free[name] = float(mean_val)
            print(f"  - {name} (will be fixed to {mean_val:.3g})")
            
    # Create the new YAML structure
    new_yaml_data = all_params_yaml.copy()
    new_yaml_data['free_names'] = new_free_names
    new_yaml_data['bounds'] = new_bounds
    # Add the newly fixed params to the existing ones
    new_yaml_data['fixed'].update(new_fixed_from_free)
    
    # save to file
    try:
        with open(output_file, 'w') as f:
            yaml.dump(new_yaml_data, f, sort_keys=False, default_flow_style=False, width=80)
        print(f"\nSucess: Saved new configuration to {output_file}")
    except Exception as e:
        print(f"\nError: Could not save new config file: {e}")
        print("PROPOSED YAML")
        print(yaml.dump(new_yaml_data, sort_keys=False, default_flow_style=False, width=80))
def propose_new_param_config(
    problem: dict, 
    all_params_yaml: dict, 
    morris_results_list: list, 
    summary_stat_names: list,
    top_n: int,
    out_filepath: Path
):
    """
    Analyzes Morris results, identifies the top_n most sensitive parameters,
    and saves a new params.yaml file.
    
    MODIFIED: This function now implements a 3-step selection:
    1. Force-fixes 'compliance_max' and 'beta_floor'.
    2. Selects the 'top_n' *other* most sensitive parameters.
    3. Force-keeps all y0 parameters ('I0', 'C0', etc.) in the free list.
    """
    
    # 1. Get average sensitivity (mu_star) across all outputs
    all_mu_stars = np.array([res['mu_star'] for res in morris_results_list])
    avg_mu_star = np.mean(all_mu_stars, axis=0)
    
    # 2. Get indices of params, sorted by sensitivity
    sorted_indices = np.argsort(avg_mu_star)[::-1]
    
    # --- NEW SELECTION LOGIC ---
    
    # Parameters we MUST include for y0 creation
    y0_params = {'I0', 'C0', 'P0', 'F0', 'R0'}
    
    # Parameters we want to force-fix (move to 'fixed' list)
    params_to_exclude = {'compliance_max', 'beta_floor'}
    
    print(f"Forcing exclusion of: {params_to_exclude}")

    # This set will hold the *top_n* most sensitive params
    sensitive_param_set = set()
    
    print(f"Selecting top {top_n} sensitive parameters...")
    for idx in sorted_indices:
        param_name = problem['names'][idx]
        
        # 1. Skip if it's in the exclude list
        if param_name in params_to_exclude:
            continue
            
        # 2. Skip if it's a y0 param (we'll add them all later)
        if param_name in y0_params:
            continue
            
        # 3. Add to the set
        sensitive_param_set.add(param_name)
        
        # 4. Stop when we have enough
        if len(sensitive_param_set) >= top_n:
            break
            
    print(f"  Selected sensitive params: {sensitive_param_set}")

    # 3. Create the final list of free names
    # Start with the sensitive set...
    final_free_names = sensitive_param_set.copy()
    
    # ...and add all y0 params that exist in the problem.
    problem_y0_params = {p for p in y0_params if p in problem['names']}
    print(f"Forcing inclusion of y0 params: {problem_y0_params}")
    final_free_names.update(problem_y0_params)
        
    print(f"Total set of 'free_names' to keep: {final_free_names}")
    
    # --- END NEW LOGIC ---

    # 4. Create the new parameter configuration
    new_params_config = {
        'fixed': all_params_yaml.get('fixed', {}),
        'free_names': [],
        'bounds': [],
        'obs': all_params_yaml.get('obs', {}),
        'fit': all_params_yaml.get('fit', {}),
        'cv': all_params_yaml.get('cv', {}),
        'profiles': {} # Profiles will be re-selected
    }

    # 5. Partition parameters: move non-sensitive ones to 'fixed'
    for name, bound in zip(problem['names'], problem['bounds']):
        if name in final_free_names:
            # This is a top (or required) param, keep it free
            new_params_config['free_names'].append(name)
            new_params_config['bounds'].append(bound)
        else:
            # This param is not in the final set. Move it to 'fixed'.
            # (This now includes non-influential params AND the 'params_to_exclude')
            new_params_config['fixed'][name] = float(np.mean(bound))
            
    # 6. Add back profiles *only* for params that are still free
    for name, p_grid in all_params_yaml.get('profiles', {}).items():
        if name in new_params_config['free_names']:
            new_params_config['profiles'][name] = p_grid
            
    # 6. Save the new yaml file
    print(f"\n--- New '{out_filepath}' Content ---")
    print(yaml.dump(new_params_config, sort_keys=False))
    
    with open(out_filepath, 'w') as f:
        yaml.dump(new_params_config, f, sort_keys=False)
        
    print(f"\nSuccess! New '{out_filepath}' has been created.")

def main():
    print("Starting Morris Sensitivity Analysis (pre-run)...")
    start_time = time.time()
    
    # 1. Load problem definition from o_params.yaml
    problem, fixed_params, obs_params, all_params_yaml = load_all_params_from_yaml(O_PARAMS_FILE)
    print(f"Loaded {problem['num_vars']} parameters to test from {O_PARAMS_FILE}.")

    # 2. Load representative data to get T
    try:
        data = pd.read_csv(DATA_SLICE_FILE)
    except FileNotFoundError:
        print(f"Error: Data slice file not found at {DATA_SLICE_FILE}")
        exit(1)
        
    T = len(data)
    t_eval = np.arange(T, dtype=float)
    y0_template = {"S0": float(fixed_params['N']), "R0": 0.0}
    print(f"Using {DATA_SLICE_FILE} (T={T} days) as representative data.")

    # 3. Get the specialized simulator/summary functions
    simulator_fn = get_simulator_fn(problem, fixed_params, obs_params, y0_template, t_eval)
    summary_fn, summary_stat_names = get_summary_fn()
    
    # 4. Run the screening
    print(f"Running Morris screening with N={MORRIS_N} trajectories. This may take a while...")
    
    # main SALib call, using morris_screening function from sensitivity.py
    morris_results_list = morris_screening_with_progress(
        problem=problem,
        simulator_fn=simulator_fn,
        summary_fn=summary_fn,
        N=MORRIS_N,
        levels=MORRIS_LEVELS,
        seed=int(time.time()) # Use a different seed each time
    )
    
    print(f"Morris screening complete. ({(time.time() - start_time):.2f}s)")
    
    # 5. Plot results
    out_dir = Path("reports") / "sensitivity"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving plots to {out_dir}...")
    
    for res, name in zip(morris_results_list, summary_stat_names):
        out_file = out_dir / f"tornado_{name.lower().replace(' ', '_')}.png"
        plot_tornado(res, problem, name, out_file)
        
    # 6. Propose and SAVE new parameter configuration
    propose_new_param_config(
        problem, 
        all_params_yaml, 
        morris_results_list, 
        summary_stat_names, 
        TOP_N_PARAMS_TO_SELECT,
        NEW_PARAMS_FILE # <-- Pass the new output filename
    )
    
    print("\nSensitivity pre-run finished.")
    print(f"Please review the plots in {out_dir} and the new {NEW_PARAMS_FILE}.")

if __name__ == "__main__":
    main()