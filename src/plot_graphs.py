import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path
import glob
import os

def find_latest_report_dir(reports_root="reports"):
    """Finds the most recent subdirectory in 'reports/'."""
    all_dirs = glob.glob(os.path.join(reports_root, "*/"))
    if not all_dirs:
        return None
    return max(all_dirs, key=os.path.getmtime)

def plot_fig1_fit(report_dir, out_dir):
    """
    Figure 1: Faceted Model Fit
    Plots 'mu' (model) vs 'y' (data) for each slice in a grid.
    """
    print("Generating Figure 1: Model Fit...")
    slice_dirs = sorted(glob.glob(os.path.join(report_dir, "slice_*")))
    if not slice_dirs:
        print("  No slice directories found.")
        return

    n_slices = len(slice_dirs)
    cols = 3
    rows = (n_slices + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(15, 4 * rows), constrained_layout=True)
    axes = axes.flatten()

    for i, s_dir in enumerate(slice_dirs):
        ax = axes[i]
        csv_path = os.path.join(s_dir, "fit_series.csv")
        
        if not os.path.exists(csv_path):
            ax.text(0.5, 0.5, "No Data", ha='center', va='center')
            continue

        df = pd.read_csv(csv_path)
        t = np.arange(len(df))
        
        # Plot Data
        ax.scatter(t, df["y"], color="grey", alpha=0.6, s=15, label="Observed Cases")
        # Plot Model
        ax.plot(t, df["mu"], color="tab:blue", linewidth=2, label="Model (Fit)")
        
        ax.set_title(f"Slice {i}")
        ax.set_xlabel("Days")
        ax.set_ylabel("Daily Cases")
        if i == 0:
            ax.legend()

    # Hide empty subplots
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    plt.suptitle("Figure 1: Model Fit Across Sequential Time Slices", fontsize=16)
    plt.savefig(out_dir / "fig1_model_fit.png", dpi=300)
    plt.close()
    print(f"  Saved to {out_dir / 'fig1_model_fit.png'}")

def plot_fig2_profiles(report_dir, out_dir):
    """
    Figure 2: Profile Likelihoods
    Plots profiles from a representative slice (e.g., slice 2 or 3).
    """
    print("Generating Figure 2: Profile Likelihoods...")
    # Try to find a middle slice (likely to have profiles)
    slice_dirs = sorted(glob.glob(os.path.join(report_dir, "slice_*")))
    if not slice_dirs:
        return
    
    # Pick a slice that has profile CSVs
    target_slice = None
    for s_dir in slice_dirs:
        if glob.glob(os.path.join(s_dir, "profile_*.csv")):
            target_slice = s_dir
            break
            
    if not target_slice:
        print("  No profile likelihood data found in any slice.")
        return

    profile_files = glob.glob(os.path.join(target_slice, "profile_*.csv"))
    n_profs = len(profile_files)
    if n_profs == 0:
        return

    cols = min(n_profs, 3)
    rows = (n_profs + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows), constrained_layout=True)
    if n_profs == 1: axes = [axes]
    else: axes = axes.flatten()

    for i, p_file in enumerate(profile_files):
        df = pd.read_csv(p_file)
        param_name = df["param"].iloc[0]
        
        ax = axes[i]
        ax.plot(df["grid"], df["negloglik"], 'o-', color="tab:orange")
        
        # Add confidence interval line (min + 1.92)
        min_nll = df["negloglik"].min()
        threshold = min_nll + 1.92
        ax.axhline(threshold, color='k', linestyle='--', alpha=0.5, label="95% CI Threshold")
        
        ax.set_title(f"Profile: {param_name}")
        ax.set_xlabel(param_name)
        ax.set_ylabel("Negative Log-Likelihood")
        
    # Hide empty subplots
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')
        
    plt.suptitle(f"Figure 2: Parameter Identifiability (from {os.path.basename(target_slice)})", fontsize=16)
    plt.savefig(out_dir / "fig2_profiles.png", dpi=300)
    plt.close()
    print(f"  Saved to {out_dir / 'fig2_profiles.png'}")

def plot_fig3_params(report_dir, out_dir):
    """
    Figure 3: Parameter Evolution
    Plots beta0, alpha, k, etc. across slices.
    """
    print("Generating Figure 3: Parameter Evolution...")
    slice_dirs = sorted(glob.glob(os.path.join(report_dir, "slice_*")))
    
    data_rows = []
    for i, s_dir in enumerate(slice_dirs):
        p_file = os.path.join(s_dir, "all_effective_params.csv")
        if os.path.exists(p_file):
            # Load series (no header in file usually, index is col 0)
            # Adjust based on how run.py saves it. run.py uses pd.Series.to_csv
            # which usually keeps the index.
            try:
                s = pd.read_csv(p_file, index_col=0, header=None).squeeze("columns")
                s["slice"] = i
                data_rows.append(s)
            except:
                pass
                
    if not data_rows:
        print("  No parameter data found.")
        return

    df_params = pd.DataFrame(data_rows).set_index("slice")
    
    # Key parameters to plot
    keys_to_plot = ["beta0", "alpha", "k", "rho_obs", "gamma"]
    keys_present = [k for k in keys_to_plot if k in df_params.columns]
    
    if not keys_present:
        return

    fig, axes = plt.subplots(len(keys_present), 1, figsize=(8, 2.5 * len(keys_present)), sharex=True)
    if len(keys_present) == 1: axes = [axes]
    
    for i, key in enumerate(keys_present):
        ax = axes[i]
        ax.plot(df_params.index, df_params[key], 'o-', linewidth=2, color="tab:green")
        ax.set_ylabel(key)
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.set_title("Figure 3: Evolution of Fitted Parameters")
    
    axes[-1].set_xlabel("Slice Index")
    axes[-1].set_xticks(df_params.index)
    
    plt.tight_layout()
    plt.savefig(out_dir / "fig3_param_evolution.png", dpi=300)
    plt.close()
    print(f"  Saved to {out_dir / 'fig3_param_evolution.png'}")

def plot_fig4_eigenvalues(report_dir, out_dir):
    """
    Figure 4: Dynamic Stability (Hopf Bifurcation Check)
    """
    print("Generating Figure 4: Eigenvalues...")
    slice_dirs = sorted(glob.glob(os.path.join(report_dir, "slice_*")))
    
    fig, ax = plt.subplots(figsize=(8, 8))
    
    has_data = False
    colors = plt.cm.viridis(np.linspace(0, 1, len(slice_dirs)))
    
    for i, s_dir in enumerate(slice_dirs):
        eig_file = os.path.join(s_dir, "jacobian_eigs.csv")
        if os.path.exists(eig_file):
            df = pd.read_csv(eig_file)
            ax.scatter(df["re"], df["im"], color=colors[i], label=f"Slice {i}", s=50, alpha=0.8)
            has_data = True

    if not has_data:
        print("  No eigenvalue data found.")
        return

    # Draw axes
    ax.axvline(0, color='k', linewidth=1.5, linestyle='-')
    ax.axhline(0, color='k', linewidth=0.5, linestyle='--')
    
    ax.set_xlabel("Real Part (Growth/Decay)")
    ax.set_ylabel("Imaginary Part (Oscillation)")
    ax.set_title("Figure 4: Eigenvalues of the Fitted Equilibrium")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.savefig(out_dir / "fig4_eigenvalues.png", dpi=300)
    plt.close()
    print(f"  Saved to {out_dir / 'fig4_eigenvalues.png'}")

def main():
    latest_dir = find_latest_report_dir()
    if not latest_dir:
        print("No reports found in 'reports/' directory.")
        return
    
    print(f"Processing report: {latest_dir}")
    report_path = Path(latest_dir)
    out_dir = report_path / "plots"
    out_dir.mkdir(exist_ok=True)
    
    # Generate Plots
    try: plot_fig1_fit(report_path, out_dir)
    except Exception as e: print(f"Error in Fig 1: {e}")
        
    try: plot_fig2_profiles(report_path, out_dir)
    except Exception as e: print(f"Error in Fig 2: {e}")
        
    try: plot_fig3_params(report_path, out_dir)
    except Exception as e: print(f"Error in Fig 3: {e}")
        
    try: plot_fig4_eigenvalues(report_path, out_dir)
    except Exception as e: print(f"Error in Fig 4: {e}")

    print("\nDone. Check the 'plots/' subfolder inside your report directory.")

if __name__ == "__main__":
    main()