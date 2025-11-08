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

    N         = float(pars["N"])
    beta_0    = float(pars.get("beta0", pars.get("beta_0")))
    gamma     = float(pars["gamma"])
    rho       = float(pars["rho"])

    alpha     = float(pars["alpha"])
    delta0    = float(pars.get("delta0", pars.get("delta", 0.0)))   # decay of P
    epsilon   = float(pars["epsilon"])                               # inflow to F from P
    phi       = float(pars["phi"])                                   # decay of F

    comp_max  = float(pars.get("compliance_max", pars.get("comp_max", 1.0)))
    k_sig     = float(pars.get("k", 25.0))                           # sigmoid slope
    P0        = float(pars.get("P0", 0.03))                          # sigmoid midpoint for P

    beta_floor= float(pars.get("beta_floor", 0.10))                  # min fraction of beta_0
    gamma_F   = float(pars.get("gamma_F", pars.get("gammaF", 0.0)))  # inhibition of P by F
    omega     = float(pars.get("omega", 0.0))                        # waning immunity S<->R

    # Hill function params for perception driver D(C)
    C50       = float(pars.get("C50", 7.0e-4))                       # ~70 per 100k
    h_exp     = float(pars.get("h", 1.5))

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
        atol=atol
    )
    if not sol.success:
        raise RuntimeError(f"ODE failed: {sol.message}")
    Y = sol.y.T  # shape (T, 6)
    # Approximate discrete daily incidence from S(t)
    S = Y[:, 0]
    # incidence_day[t] ≈ max(S[t-1]-S[t], 0)
    inc = np.maximum(-np.diff(S, prepend=S[0]), 0.0)
    return Y, inc