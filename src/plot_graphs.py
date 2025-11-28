"""
Generates a standard set of figures from `run.py` output directory.

This script is designed to be run after `run.py`. It finds the
report directory (either by CLI, user input, or auto-detecting the
latest) and generates a `plots/` subdirectory within it.

It scans the `slice_*` subfolders for CSV files and generates a
standard set of 4 figures:
1.  **Fig 1: Model Fit:** A grid plot of `mu` (fit) vs. `y` (data)
    for each slice.
2.  **Fig 2: Profile Likelihoods:** A grid plot of NLL profiles
    from the first available slice.
3.  **Fig 3: Parameter Evolution:** Line plots showing how key
    parameter values change across slices.
4.  **Fig 4: Eigenvalues:** A scatter plot of the equilibrium
    Jacobian eigenvalues from each slice on the complex plane.

"""
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import config
import pandas as pd
import numpy as np
import sys

# ---------------------------
# Helper functions
# ---------------------------

def find_latest_report_dir(reports_root: Path = Path("reports")) -> Path | None:
    """Return the most recently modified subdirectory under 'reports_root', or None if none exist."""
    if not reports_root.exists():
        return None
    dirs = [p for p in reports_root.iterdir() if p.is_dir()]
    if not dirs:
        return None
    return max(dirs, key=lambda p: p.stat().st_mtime)

def has_slice_dirs(report_dir: Path) -> bool:
    """True if report_dir contains at least one 'slice_*' directory."""
    return any(p.is_dir() and p.name.startswith("slice_") for p in report_dir.iterdir())

def safe_mkdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

# ---------------------------
# Plotting functions

def plot_fig1_fit(report_dir: Path, out_dir: Path) -> None:
    """
    Figure 1: Faceted Model Fit
    Plots 'mu' (model) vs 'y' (data) for each slice in a grid.
    """
    print("Generating Figure 1: Model Fit...")
    slice_dirs = sorted([p for p in report_dir.iterdir() if p.is_dir() and p.name.startswith("slice_")])
    if not slice_dirs:
        print("  No slice directories found.")
        return

    n_slices = len(slice_dirs)
    cols = 3
    rows = (n_slices + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(15, 4 * rows), constrained_layout=True)
    axes = np.array(axes).reshape(-1)  # works for rows*cols >= 1
    
    slice_colors = ["#91CDE2", "#71E671", "#F0F060", "#F27F52", "#DF72DF", "#B6B6B6"]
    
    plt.rcParams.update({'font.size': 12})

    for i, s_dir in enumerate(slice_dirs):
        ax = axes[i]
        csv_path = s_dir / "fit_series.csv"
        if not csv_path.exists():
            ax.text(0.5, 0.5, "No Data", ha='center', va='center')
            ax.set_axis_off()
            continue

        df = pd.read_csv(csv_path)
        if not {"y", "mu"}.issubset(df.columns):
            ax.text(0.5, 0.5, "Missing columns", ha='center', va='center')
            ax.set_axis_off()
            continue

        t = np.arange(len(df))
        ax.scatter(t, df["y"], alpha=0.6, s=15, label="Cas Observés", c="red")
        ax.plot(t, df["mu"], linewidth=2, label="Modèle (Fit)", c="blue")
        for s_dir in config.SLICES:
            ax.set_title(f"{config.SLICES[i]['name']}")
            color = slice_colors[i % len(slice_colors)]
            ax.set_facecolor((*mcolors.to_rgb(color), 0.2))
        ax.set_xlabel("Jours")
        ax.set_ylabel("Cas par jour")
        if i == 0:
            ax.legend()

    # Hide empty subplots
    for j in range(i + 1, len(axes)):
        axes[j].set_axis_off()

    fig.suptitle("", fontsize=16)
    safe_mkdir(out_dir)
    fig.savefig(out_dir / "fig1_model_fit.png", dpi=300)
    plt.close(fig)
    print(f"  Saved to {out_dir / 'fig1_model_fit.png'}")

def plot_fig2_profiles(report_dir: Path, out_dir: Path) -> None:
    """
    Figure 2: Profile Likelihoods (from the first slice that has profile_*.csv).
    """
    print("Generating Figure 2: Profile Likelihoods...")
    slice_dirs = sorted([p for p in report_dir.iterdir() if p.is_dir() and p.name.startswith("slice_")])
    if not slice_dirs:
        print("  No slice directories found.")
        return

    target_slice = None
    for s_dir in slice_dirs:
        if any(s_dir.glob("profile_*.csv")):
            target_slice = s_dir
            break

    if not target_slice:
        print("  No profile likelihood data found in any slice.")
        return

    profile_files = list(target_slice.glob("profile_*.csv"))
    n_profs = len(profile_files)
    if n_profs == 0:
        print("  No profile files in target slice.")
        return

    cols = min(n_profs, 3)
    rows = (n_profs + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows), constrained_layout=True)
    axes = np.array(axes).reshape(-1)

    for i, p_file in enumerate(profile_files):
        df = pd.read_csv(p_file)
        if not {"grid", "negloglik"}.issubset(df.columns):
            continue
        param_name = df["param"].iloc[0] if "param" in df.columns and len(df) else p_file.stem.replace("profile_", "")

        ax = axes[i]
        ax.plot(df["grid"], df["negloglik"], "o-")
        min_nll = df["negloglik"].min()
        threshold = min_nll + 1.92  # ~95% for 1 dof
        ax.axhline(threshold, linestyle="--", alpha=0.5, label="95% CI Threshold")
        ax.set_title(f"Profile: {param_name}")
        ax.set_xlabel(param_name)
        ax.set_ylabel("Negative Log-Likelihood")
        ax.legend(loc="best")

    for j in range(i + 1, len(axes)):
        axes[j].set_axis_off()

    fig.suptitle(f"Figure 2: Parameter Identifiability ({target_slice.name})", fontsize=16)
    safe_mkdir(out_dir)
    fig.savefig(out_dir / "fig2_profiles.png", dpi=300)
    plt.close(fig)
    print(f"  Saved to {out_dir / 'fig2_profiles.png'}")

def _load_param_series(csv_path: Path) -> pd.Series | None:
    """
    Try to load a pd.Series produced by pd.Series.to_csv.
    Fall back to headered CSV.
    """
    try:
        # Common pattern when Series.to_csv was used (index in col 0, values in col 1)
        s = pd.read_csv(csv_path, index_col=0, header=None).squeeze("columns")
        if isinstance(s, pd.Series):
            return s
    except Exception:
        pass
    try:
        df = pd.read_csv(csv_path)
        if df.shape[1] >= 2:
            s = pd.Series(df.iloc[:, 1].values, index=df.iloc[:, 0].values)
            return s
    except Exception:
        pass
    return None

def plot_fig3_params(report_dir: Path, out_dir: Path) -> None:
    """
    Figure 3: Parameter Evolution across slices (if all_effective_params.csv exists per slice).
    """
    print("Generating Figure 3: Parameter Evolution...")
    slice_dirs = sorted([p for p in report_dir.iterdir() if p.is_dir() and p.name.startswith("slice_")])
    if not slice_dirs:
        print("  No slice directories found.")
        return

    rows = []
    for i, s_dir in enumerate(slice_dirs):
        p_file = s_dir / "all_effective_params.csv"
        if p_file.exists():
            s = _load_param_series(p_file)
            if s is not None:
                s = s.copy()
                s["slice"] = i
                rows.append(s)

    if not rows:
        print("  No parameter data found.")
        return

    df_params = pd.DataFrame(rows).set_index("slice")
    keys_to_plot = ["beta0", "alpha", "k", "rho_obs", "gamma"]
    keys_present = [k for k in keys_to_plot if k in df_params.columns]
    if not keys_present:
        print("  None of the expected parameter keys present.")
        return

    fig, axes = plt.subplots(len(keys_present), 1, figsize=(8, 2.6 * len(keys_present)), sharex=True)
    if len(keys_present) == 1:
        axes = [axes]

    for ax, key in zip(axes, keys_present):
        ax.plot(df_params.index, df_params[key], "o-", linewidth=2)
        ax.set_ylabel(key)
        ax.grid(True, alpha=0.3)
    axes[0].set_title("Figure 3: Evolution of Fitted Parameters")
    axes[-1].set_xlabel("Slice Index")

    safe_mkdir(out_dir)
    fig.tight_layout()
    fig.savefig(out_dir / "fig3_param_evolution.png", dpi=300)
    plt.close(fig)
    print(f"  Saved to {out_dir / 'fig3_param_evolution.png'}")

def plot_fig4_eigenvalues(report_dir: Path, out_dir: Path) -> None:
    """
    Figure 4: Dynamic Stability — scatter of Jacobian eigenvalues per slice.
    """
    print("Generating Figure 4: Eigenvalues...")
    slice_dirs = sorted([p for p in report_dir.iterdir() if p.is_dir() and p.name.startswith("slice_")])
    if not slice_dirs:
        print("  No slice directories found.")
        return

    fig, ax = plt.subplots(figsize=(8, 8))
    has_data = False
    colors = plt.cm.viridis(np.linspace(0, 1, max(len(slice_dirs), 2)))

    for i, s_dir in enumerate(slice_dirs):
        eig_file = s_dir / "jacobian_eigs.csv"
        if eig_file.exists():
            df = pd.read_csv(eig_file)
            if {"re", "im"}.issubset(df.columns):
                ax.scatter(df["re"], df["im"], color=colors[i], label=f"{s_dir.name}", s=50, alpha=0.8)
                has_data = True

    if not has_data:
        print("  No eigenvalue data found.")
        plt.close(fig)
        return

    ax.axvline(0, linewidth=1.5)
    ax.axhline(0, linewidth=0.8, linestyle="--")
    ax.set_xlabel("Real Part (Growth/Decay)")
    ax.set_ylabel("Imaginary Part (Oscillation)")
    ax.set_title("Figure 4: Eigenvalues of the Fitted Equilibrium")
    ax.legend()
    ax.grid(True, alpha=0.3)

    safe_mkdir(out_dir)
    fig.savefig(out_dir / "fig4_eigenvalues.png", dpi=300)
    plt.close(fig)
    print(f"  Saved to {out_dir / 'fig4_eigenvalues.png'}")

# ---------------------------
# Orchestration

def process_one_report(report_dir: Path) -> None:
    """Generate all figures for a single report root (must contain slice_* folders)."""
    if not report_dir.exists() or not report_dir.is_dir():
        print(f"[WARN] Skipping non-directory: {report_dir}")
        return
    if not has_slice_dirs(report_dir):
        print(f"[WARN] No 'slice_*' directories under: {report_dir}")
        return

    out_dir = report_dir / "plots"
    safe_mkdir(out_dir)

    try:
        plot_fig1_fit(report_dir, out_dir)
    except Exception as e:
        print(f"Error in Fig 1: {e}")

    try:
        plot_fig2_profiles(report_dir, out_dir)
    except Exception as e:
        print(f"Error in Fig 2: {e}")

    try:
        plot_fig3_params(report_dir, out_dir)
    except Exception as e:
        print(f"Error in Fig 3: {e}")

    try:
        plot_fig4_eigenvalues(report_dir, out_dir)
    except Exception as e:
        print(f"Error in Fig 4: {e}")

def main():
    """
    Entry point:
      - If a folder path is provided as CLI argument: use it.
      - Otherwise: ask user to input one or automatically select latest in 'reports/'.
    """
    if len(sys.argv) > 1:
        report_path = Path(sys.argv[1]).expanduser().resolve()
    else:
        user_input = input("Enter path to report folder (leave blank for latest in 'reports/'): ").strip()
        if user_input:
            report_path = Path(user_input).expanduser().resolve()
        else:
            report_path = find_latest_report_dir()

    if not report_path:
        print("No reports found or invalid path.")
        return

    report_path = Path(report_path)
    if not report_path.exists():
        print(f"[ERROR] Path not found: {report_path}")
        return

    print(f"[INFO] Using report folder: {report_path}")

    if has_slice_dirs(report_path):
        # Case A: directly a report root
        process_one_report(report_path)
    else:
        # Case B: contains sub-reports
        subdirs = [p for p in report_path.iterdir() if p.is_dir()]
        processed_any = False
        for sub in subdirs:
            if has_slice_dirs(sub):
                print(f"Processing sub-report: {sub}")
                process_one_report(sub)
                processed_any = True
        if not processed_any:
            print(f"[WARN] No slice_* directories found under {report_path}.")

    print("\nDone. Check 'plots/' inside the report folder(s).")

if __name__ == "__main__":
    main()
