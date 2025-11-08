# make_plots.py
from __future__ import annotations
import argparse
import glob
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Optional (for residual ACF). If not installed, that plot is skipped.
try:
    from statsmodels.tsa.stattools import acf as sm_acf
    HAS_SM = True
except Exception:
    HAS_SM = False

# ---------- helpers ----------

def _latest_report_dir(base="reports"):
    """Return the most recent *directory* inside `reports/`"""
    paths = []
    for p in Path(base).glob("*"):
        if p.is_dir():
            paths.append(p)
    if not paths:
        return None
    return sorted(paths, key=lambda p: p.stat().st_mtime)[-1]


def _ensure_figs_dir(report_dir: Path) -> Path:
    figs = report_dir / "figs"
    figs.mkdir(parents=True, exist_ok=True)
    return figs

def _load_best_params(report_dir: Path) -> dict:
    pfile = report_dir / "best_params.csv"
    if not pfile.exists():
        return {}
    s = pd.read_csv(pfile, index_col=0, header=None).squeeze("columns")
    # If names are in the index (free_names) this will work; otherwise map 0..N
    d = {str(k): float(v) for k, v in s.items()}
    return d

def _compute_nb_residuals(y, mu, log_theta):
    theta = float(np.exp(log_theta))
    var = mu + (mu**2) / theta
    return (y - mu) / np.sqrt(var + 1e-12)

# ---------- plotting primitives ----------

def plot_fit_with_band(report_dir: Path):
    figs = _ensure_figs_dir(report_dir)
    fit_path = report_dir / "fit_series.csv"
    if not fit_path.exists():
        print("[skip] fit_series.csv not found")
        return
    df = pd.read_csv(fit_path)
    y = df["y"].to_numpy(float)
    mu = df["mu"].to_numpy(float)

    # Predictive intervals (loaded if available, else quick NB resample)
    # Prefer to re-create intervals to avoid storing large arrays.
    best = _load_best_params(report_dir)
    log_theta = float(best.get("log_theta", 0.0))

    # quick parametric bootstrap for 90% interval
    theta = np.exp(log_theta)
    p = theta / (theta + mu)
    r = theta
    rng = np.random.default_rng(123)
    lam = rng.gamma(shape=r, scale=(1 - p) / p, size=(2000, len(mu)))
    ysim = rng.poisson(lam=lam)
    lower, median, upper = np.quantile(ysim, [0.05, 0.5, 0.95], axis=0)

    t = np.arange(len(y))
    plt.figure(figsize=(9, 4))
    plt.plot(t, y, lw=1, label="Observed")
    plt.plot(t, mu, lw=2, label="Model mean (μ)")
    plt.fill_between(t, lower, upper, alpha=0.2, label="90% pred. band")
    plt.xlabel("Day"); plt.ylabel("Daily cases"); plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(figs / "fit_with_band.png", dpi=200)
    plt.close()

def plot_residuals(report_dir: Path):
    figs = _ensure_figs_dir(report_dir)
    fit_path = report_dir / "fit_series.csv"
    if not fit_path.exists():
        print("[skip] fit_series.csv not found")
        return
    df = pd.read_csv(fit_path)
    y = df["y"].to_numpy(float)
    mu = df["mu"].to_numpy(float)
    log_theta = float(_load_best_params(report_dir).get("log_theta", 0.0))
    resid = _compute_nb_residuals(y, mu, log_theta)

    # Residuals over time
    plt.figure(figsize=(9, 3))
    plt.plot(resid, lw=1)
    plt.axhline(0, ls="--", lw=1)
    plt.xlabel("Day"); plt.ylabel("Pearson residual")
    plt.tight_layout()
    plt.savefig(figs / "residuals_time.png", dpi=200)
    plt.close()

    # Residual ACF (optional)
    if HAS_SM:
        vals = sm_acf(resid, nlags=min(30, len(resid)//2), fft=True)
        plt.figure(figsize=(6, 3))
        plt.stem(range(len(vals)), vals)
        plt.xlabel("Lag (days)"); plt.ylabel("ACF")
        plt.tight_layout()
        plt.savefig(figs / "residuals_acf.png", dpi=200)
        plt.close()
    else:
        print("[note] statsmodels not installed; skipping residual ACF plot")

def plot_cv_metrics(report_dir: Path):
    figs = _ensure_figs_dir(report_dir)
    cvp = report_dir / "cv_metrics.csv"
    if not cvp.exists():
        print("[skip] cv_metrics.csv not found")
        return
    df = pd.read_csv(cvp)

    def _metric_errbar(metric):
        g = df.groupby("horizon")[metric].agg(["mean", "std"]).reset_index()
        plt.figure(figsize=(5, 3))
        plt.errorbar(g["horizon"], g["mean"], yerr=g["std"], fmt="o-", capsize=3)
        plt.xlabel("Forecast horizon (days)"); plt.ylabel(metric.upper())
        plt.tight_layout()
        plt.savefig(figs / f"cv_{metric}.png", dpi=200)
        plt.close()

    for m in ["mae", "rmse", "mlpd", "cov90"]:
        if m in df.columns:
            _metric_errbar(m)

def plot_morris(report_dir: Path):
    figs = _ensure_figs_dir(report_dir)
    # Expect a file you saved from sensitivity.py, e.g., name, mu_star, sigma
    candidates = list(report_dir.glob("morris_output*.csv")) + list((report_dir.parent).glob("morris_output*.csv"))
    if not candidates:
        print("[skip] no morris_output*.csv found")
        return
    sens = pd.read_csv(candidates[0])
    if not set(["name", "mu_star"]).issubset(sens.columns):
        print("[skip] morris_output.csv missing required columns")
        return
    sens = sens.sort_values("mu_star", ascending=True)
    plt.figure(figsize=(6, max(3, 0.35 * len(sens))))
    plt.barh(sens["name"], sens["mu_star"])
    plt.xlabel("Morris μ* (elementary effect)")
    plt.tight_layout()
    plt.savefig(figs / "morris_mu_star.png", dpi=200)
    plt.close()

def plot_profiles(report_dir: Path):
    figs = _ensure_figs_dir(report_dir)
    prof_files = sorted(report_dir.glob("profile_*.csv"))
    if not prof_files:
        print("[skip] no profile_*.csv found")
        return
    for f in prof_files:
        df = pd.read_csv(f)
        if not set(["grid", "negloglik"]).issubset(df.columns):
            continue
        df["delta"] = df["negloglik"] - df["negloglik"].min()
        pname = f.stem.replace("profile_", "")
        plt.figure(figsize=(5, 3))
        plt.plot(df["grid"], df["delta"])
        plt.axhline(1.92, ls="--", lw=1)  # ~95% cutoff for 1D profile
        plt.xlabel(pname); plt.ylabel("Δ(−logL)")
        plt.tight_layout()
        plt.savefig(figs / f"profile_{pname}.png", dpi=200)
        plt.close()

def plot_lagcorr(report_dir: Path):
    figs = _ensure_figs_dir(report_dir)
    p = report_dir / "lagcorr.csv"
    if not p.exists():
        print("[skip] lagcorr.csv not found")
        return
    lc = pd.read_csv(p)
    cols = set(lc.columns)
    if not {"lag", "r"}.issubset(cols):
        print("[skip] lagcorr.csv missing columns")
        return
    plt.figure(figsize=(6, 3))
    plt.plot(lc["lag"], lc["r"], label="r(lag)")
    if {"lo", "hi"}.issubset(cols):
        plt.fill_between(lc["lag"], lc["lo"], lc["hi"], alpha=0.2, label="bootstrap CI")
    plt.axvline(0, ls="--", lw=1)
    plt.xlabel("Lag (days) [x leads > 0]"); plt.ylabel("Correlation")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(figs / "lagcorr.png", dpi=200)
    plt.close()

def plot_psd(report_dir: Path):
    figs = _ensure_figs_dir(report_dir)
    p = report_dir / "psd.csv"
    if not p.exists():
        print("[skip] psd.csv not found")
        return
    psd = pd.read_csv(p)
    if not set(["f", "Pxx"]).issubset(psd.columns):
        print("[skip] psd.csv missing columns")
        return
    f, Pxx = psd["f"].to_numpy(float), psd["Pxx"].to_numpy(float)
    if len(f) < 2:
        print("[skip] psd.csv too short")
        return
    idx = np.argmax(Pxx[1:]) + 1
    f_star = f[idx]; period = np.inf if f_star <= 0 else 1.0 / f_star

    plt.figure(figsize=(6, 3))
    plt.plot(f, Pxx, lw=1.5)
    plt.axvline(f_star, ls="--", lw=1)
    plt.text(f_star, Pxx[idx], f" f*={f_star:.3f}\n (period≈{period:.1f} d)",
             va="bottom", ha="left")
    plt.xlabel("Frequency (1/day)"); plt.ylabel("Power")
    plt.tight_layout()
    plt.savefig(figs / "psd_welch.png", dpi=200)
    plt.close()

def plot_eigs(report_dir: Path):
    figs = _ensure_figs_dir(report_dir)
    p = report_dir / "jacobian_eigs.csv"
    if not p.exists():
        print("[skip] jacobian_eigs.csv not found")
        return
    df = pd.read_csv(p)
    if not set(["re", "im"]).issubset(df.columns):
        print("[skip] jacobian_eigs.csv missing columns")
        return
    plt.figure(figsize=(5, 4))
    plt.scatter(df["re"], df["im"], s=25)
    plt.axvline(0, ls="--", lw=1)
    plt.xlabel("Re(λ)"); plt.ylabel("Im(λ)")
    plt.tight_layout()
    plt.savefig(figs / "jacobian_eigs.png", dpi=200)
    plt.close()

# ---------- main ----------

def main():
    ap = argparse.ArgumentParser(description="Generate paper-ready plots from a report run.")
    ap.add_argument("--report", type=str, default=None, help="Path to reports/<timestamp> (default = latest).")
    args = ap.parse_args()

    report_dir = Path(args.report) if args.report else _latest_report_dir()
    if report_dir is None or not report_dir.exists():
        raise SystemExit("No report directory found. Run `python run.py` first, or pass --report <path>.")

    print(f"[info] using report: {report_dir}")
    # Core figures
    plot_fit_with_band(report_dir)
    plot_residuals(report_dir)
    plot_cv_metrics(report_dir)
    plot_morris(report_dir)
    plot_profiles(report_dir)
    plot_lagcorr(report_dir)
    plot_psd(report_dir)
    plot_eigs(report_dir)
    print(f"[done] figures saved in {report_dir / 'figs'}")

if __name__ == "__main__":
    main()
