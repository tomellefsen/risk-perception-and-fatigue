from __future__ import annotations
import numpy as np
from scipy.integrate import solve_ivp

def sirc_pf_rhs(t, y, pars):
    """
    v2 dynamics with the original signature:
      - Inputs: (t, y, pars) where y = [S, I, C, R, P, F]
      - Returns: a numpy array [dSdt, dIdt, dCdt, dRdt, dPdt, dFdt] of dtype float

    Expected keys in `pars` (existing names kept; new ones have safe defaults):
      Required:  N, beta0, gamma, rho, alpha, epsilon, phi
      Common:    delta (or delta0), compliance_max, k, beta_floor
      Optional:  gamma_F (default 0.0), omega (waning; default 0.0),
                 P0 (sigmoid midpoint; default 0.03),
                 C50 (Hill half-sat; default 7e-4), h (Hill exponent; default 1.5)
    """
    import numpy as np

    S, I, C, R, P, F = y

    # --- STABILITY CLIPPING ---
    # We clip the parameters *inside* the ODE function to prevent
    # the optimizer from ever passing in values that cause an overflow.
    # This is the final and most robust line of defense.
    
    N         = float(pars["N"])
    beta_0    = np.clip(float(pars.get("beta0", pars.get("beta_0"))), 0.0, 2.0)
    gamma     = np.clip(float(pars["gamma"]), 0.0, 2.0)
    rho       = np.clip(float(pars["rho"]), 0.0, 2.0)

    alpha     = np.clip(float(pars["alpha"]), 0.0, 5.0)
    delta0    = np.clip(float(pars.get("delta0", pars.get("delta", 0.0))), 0.0, 5.0)
    epsilon   = np.clip(float(pars["epsilon"]), 0.0, 5.0)
    phi       = np.clip(float(pars["phi"]), 0.0, 5.0)

    comp_max  = np.clip(float(pars.get("compliance_max", pars.get("comp_max", 1.0))), 0.0, 1.0)
    k_sig     = np.clip(float(pars.get("k", 25.0)), 0.1, 100.0)
    P0        = np.clip(float(pars.get("P0", 0.03)), 0.0, 1.0)

    beta_floor= np.clip(float(pars.get("beta_floor", 0.10)), 0.0, 1.0)
    gamma_F   = np.clip(float(pars.get("gamma_F", pars.get("gammaF", 0.0))), 0.0, 5.0)
    omega     = np.clip(float(pars.get("omega", 0.0)), 0.0, 0.1)

    C50       = np.clip(float(pars.get("C50", 7.0e-4)), 1e-6, 0.1)
    h_exp     = np.clip(float(pars.get("h", 1.5)), 0.1, 10.0)

    compliance = comp_max * (1.0 / (1.0 + np.exp(-k_sig * (P - P0))))

    # effective transmission with floor
    beta_min = beta_0 * beta_floor
    beta_eff = beta_min + (beta_0 - beta_min) * (1.0 - compliance)

    total_infected = I + C
    incidence = beta_eff * S * total_infected / N

    # perception driver from reported C via Hill saturation
    C_over_N = np.clip(C / N, 0.0, np.inf)
    D = (C_over_N**h_exp) / (C50**h_exp + C_over_N**h_exp)

    # epidemic dynamics
    dSdt = -incidence + omega * R
    dIdt = incidence - (gamma + rho) * I
    dCdt = rho * I - gamma * C
    dRdt = gamma * (I + C) - omega * R

    # behavioral dynamics (P fast, F slow)
    dPdt = alpha * D - (delta0 + gamma_F * F) * P
    dFdt = epsilon * P - phi * F

    return np.array([dSdt, dIdt, dCdt, dRdt, dPdt, dFdt], dtype=float)


def simulate(pars, y0, t_eval, rtol=1e-7, atol=1e-9):
    """Deterministic simulation with LSODA."""
    
    if not np.all(np.isfinite(y0)):
        raise ValueError(f"simulate(): non-finite y0 {y0}")

    sol = solve_ivp(
        fun=lambda t, y: sirc_pf_rhs(t, y, pars),
        t_span=(t_eval[0], t_eval[-1]),
        y0=y0,
        t_eval=t_eval,
        method="LSODA",
        rtol=rtol,
        atol=atol,
        max_step=1.0  # prevent scalar vector overflow error
    )
    if not sol.success:
        raise RuntimeError(f"ODE failed: {sol.message}")
    Y = sol.y.T  # shape (T, 6)
    # Approximate discrete daily incidence from S(t)
    S = Y[:, 0]
    # incidence_day[t] ≈ max(S[t-1]-S[t], 0)
    inc = np.maximum(-np.diff(S, prepend=S[0]), 0.0)
    return Y, inc