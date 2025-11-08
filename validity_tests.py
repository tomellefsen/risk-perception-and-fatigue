"""
validity_tests.py — lean, runnable validity checks using reports/<timestamp> outputs.

Usage:
  python validity_tests.py
  python validity_tests.py --report-dir reports/2025-11-06_013012
"""

from __future__ import annotations
import argparse, sys, os, math, json, glob
from pathlib import Path

import numpy as np
import pandas as pd

# Optional deps
try:
    from scipy.stats import nbinom, kstest, uniform
    from scipy.signal import welch
    SCIPY_OK = True
except Exception:
    SCIPY_OK = False

try:
    from statsmodels.stats.diagnostic import acorr_ljungbox
    STATSMODELS_OK = True
except Exception:
    STATSMODELS_OK = False


# ---------------------------
# Helpers
# ---------------------------

def find_latest_report_dir(root="reports"):
    paths = [Path(p) for p in glob.glob(os.path.join(root, "*")) if os.path.isdir(p)]
    if not paths:
        return None
    return max(paths, key=lambda p: p.stat().st_mtime)

def load_required_files(report_dir: Path):
    required = {
        "fit_series": report_dir / "fit_series.csv",
        "best_params": report_dir / "best_params.csv",
        "cv_metrics": report_dir / "cv_metrics.csv",
    }
    optional = {
        "psd": report_dir / "psd.csv",
        "jacobian_eigs": report_dir / "jacobian_eigs.csv",
        "manifest": report_dir / "manifest.json",
    }
    missing = [k for k, p in required.items() if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing required files in {report_dir}: {missing}")
    return required, {k: p for k, p in optional.items() if p.exists()}

def nb_pit(y, mu, theta, rng=np.random.default_rng(0)):
    """
    Randomized PIT for Negative Binomial with (mean=mu, dispersion=theta).
    """
    p = theta / (theta + mu)
    cdf_y = nbinom.cdf(y, theta, p)
    cdf_ym1 = nbinom.cdf(np.maximum(y - 1, 0), theta, p)
    u = rng.uniform(size=len(y))
    return cdf_ym1 + u * (cdf_y - cdf_ym1)

def pearson_residuals(y, mu, theta):
    var = mu + mu**2 / theta
    return (y - mu) / np.sqrt(var + 1e-12)

def test_result(name, passed, detail):
    return {"name": name, "passed": bool(passed), "detail": detail}

def fmt_bool(b): return "PASS" if b else "FAIL"

def safe_float(x, default=np.nan):
    try:
        return float(x)
    except Exception:
        return default


# ---------------------------
# Main checks
# ---------------------------

def run_checks(report_dir: Path, strict=True):
    results = []

    # Load files
    required, optional = load_required_files(report_dir)
    fit = pd.read_csv(required["fit_series"])       # columns: y, mu, inc (as produced by scaffold)
    best = pd.read_csv(required["best_params"], index_col=0, header=None).squeeze("columns")
    cv = pd.read_csv(required["cv_metrics"])

    # Extract parameters we need
    if "log_theta" not in best.index:
        raise ValueError("best_params.csv must contain 'log_theta' (NB dispersion in log-space).")
    theta = math.exp(float(best.loc["log_theta"]))

    # Series
    if not {"y", "mu"}.issubset(set(fit.columns)):
        raise ValueError("fit_series.csv must contain columns: 'y', 'mu'.")
    y = fit["y"].to_numpy(dtype=float)
    mu = np.clip(fit["mu"].to_numpy(dtype=float), 1e-9, None)

    # 1) NB-PIT calibration (Uniform[0,1] via KS test)
    if SCIPY_OK:
        pit = nb_pit(y, mu, theta)
        ks_stat, ks_p = kstest(pit, "uniform")
        passed = ks_p > 0.05
        results.append(test_result("PIT uniformity (KS against U[0,1])", passed,
                                   f"KS stat={ks_stat:.3f}, p={ks_p:.3f}"))
    else:
        results.append(test_result("PIT uniformity", False if strict else True,
                                   "SciPy not available; install scipy to run."))

    # 2) Residual autocorrelation (Ljung–Box up to lag=14)
    if STATSMODELS_OK:
        resid = pearson_residuals(y, mu, theta)
        lb = acorr_ljungbox(resid, lags=[7, 14], return_df=True)
        p7 = safe_float(lb["lb_pvalue"].iloc[0])
        p14 = safe_float(lb["lb_pvalue"].iloc[1]) if len(lb) > 1 else np.nan
        # Heuristic: we want no *strong* evidence of leftover autocorr: p-values not extremely small
        passed = (np.isnan(p7) or p7 > 0.01) and (np.isnan(p14) or p14 > 0.01)
        results.append(test_result("Residual autocorrelation (Ljung–Box lags 7/14)", passed,
                                   f"p7={p7:.3f}, p14={p14:.3f}"))
    else:
        results.append(test_result("Residual autocorrelation (Ljung–Box)", False if strict else True,
                                   "statsmodels not available; install statsmodels to run."))

    # 3) 7-day periodicity check (proxy weekday effect) on residuals via simple sine/cos regression
    #    H0: no weekly component -> low R^2
    try:
        t = np.arange(len(mu))
        resid = pearson_residuals(y, mu, theta)
        X = np.column_stack([np.ones_like(t),
                             np.sin(2*np.pi*t/7.0),
                             np.cos(2*np.pi*t/7.0)])
        # OLS closed form
        beta = np.linalg.pinv(X) @ resid
        fit_week = X @ beta
        ss_res = np.sum((resid - fit_week)**2)
        ss_tot = np.sum((resid - resid.mean())**2) + 1e-12
        r2 = 1 - ss_res/ss_tot
        # Heuristic: if weekly R^2 > 0.05, strong weekly leftover → consider weekday multipliers
        passed = r2 <= 0.05
        results.append(test_result("Weekly effect in residuals (R^2)", passed, f"R^2_week={r2:.3f}"))
    except Exception as e:
        results.append(test_result("Weekly effect in residuals", False if strict else True, f"Error: {e}"))

    # 4) Coverage sanity from CV metrics (expect ~0.90 for 90% PI)
    if "cov90" in cv.columns:
        cov = float(cv["cov90"].mean())
        passed = 0.85 <= cov <= 0.95
        results.append(test_result("Predictive interval coverage (90%)", passed, f"mean cov90={cov:.3f}"))
    else:
        results.append(test_result("Predictive interval coverage (90%)", False if strict else True,
                                   "cv_metrics.csv missing 'cov90'."))

    # 5) PSD quick check: non-zero dominant frequency (optional, informational)
    if SCIPY_OK:
        try:
            f_psd = None
            if (report_dir / "psd.csv").exists():
                psd = pd.read_csv(report_dir / "psd.csv")
                if {"f", "Pxx"}.issubset(psd.columns):
                    f = psd["f"].to_numpy()
                    Pxx = psd["Pxx"].to_numpy()
                    # exclude zero frequency
                    idx = np.argmax(Pxx[1:]) + 1 if len(Pxx) > 1 else 0
                    f_psd = float(f[idx]) if idx < len(f) else None
            else:
                # recompute from mu if psd file absent
                f, Pxx = welch(mu, fs=1.0, nperseg=min(256, len(mu)))
                idx = np.argmax(Pxx[1:]) + 1
                f_psd = float(f[idx])

            detail = f"dominant freq={f_psd:.4f}  (period≈{(1.0/f_psd):.1f} d)" if (f_psd and f_psd > 0) else "no nonzero peak"
            # Not a strict pass/fail unless you want to assert it's not exactly weekly:
            passed = True
            results.append(test_result("PSD dominant frequency (informational)", passed, detail))
        except Exception as e:
            results.append(test_result("PSD dominant frequency", False if strict else True, f"Error: {e}"))
    else:
        results.append(test_result("PSD dominant frequency", True, "SciPy not available; skipping (informational)."))

    # 6) Jacobian eigenvalues glance (optional sanity)
    if (report_dir / "jacobian_eigs.csv").exists():
        try:
            E = pd.read_csv(report_dir / "jacobian_eigs.csv")
            if {"re", "im"}.issubset(E.columns):
                remax = float(np.max(E["re"].to_numpy()))
                # Heuristic: we just report the max real part; pass always, but flag if > 0.2 (arbitrary)
                flag = remax > 0.2
                detail = f"max Re(eig)={remax:.3e}; {'NOTE: sizeable positive real part' if flag else 'ok'}"
                results.append(test_result("Jacobian eigenvalues (informational)", True, detail))
            else:
                results.append(test_result("Jacobian eigenvalues", True, "File present but missing columns; skipping."))
        except Exception as e:
            results.append(test_result("Jacobian eigenvalues", True, f"Error reading eigs: {e}"))
    else:
        results.append(test_result("Jacobian eigenvalues", True, "File not present; skipping (informational)."))

    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--report-dir", type=str, default=None, help="Path to a specific reports/<timestamp> folder.")
    ap.add_argument("--non-strict", action="store_true", help="Don’t fail when optional deps are missing.")
    args = ap.parse_args()

    report_dir = Path(args.report_dir) if args.report_dir else find_latest_report_dir("reports")
    if report_dir is None:
        print("ERROR: No reports/ folders found.", file=sys.stderr)
        sys.exit(2)

    print(f"Using report dir: {report_dir}")

    results = run_checks(report_dir, strict=not args.non_strict)

    print("\n=== VALIDITY TESTS SUMMARY ===")
    failures = 0
    for r in results:
        print(f"[{fmt_bool(r['passed'])}] {r['name']}: {r['detail']}")
        if not r["passed"]:
            failures += 1

    if failures:
        print(f"\nRESULT: FAIL — {failures} test(s) failed.")
        sys.exit(1)
    else:
        print("\nRESULT: PASS — all required tests passed.")
        sys.exit(0)


if __name__ == "__main__":
    main()