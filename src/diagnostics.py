from __future__ import annotations
import numpy as np
from typing import Dict, Optional
from .observation import nb_loglik

def _as_mask(n: int, mask: Optional[np.ndarray]) -> np.ndarray:
    if mask is None:
        return np.ones(n, dtype=bool)
    mask = np.asarray(mask, dtype=bool)
    if mask.shape[0] != n:
        raise ValueError(f"mask length {mask.shape[0]} != n={n}")
    return mask

def info_criteria_nb(
    y: np.ndarray,
    mu: np.ndarray,
    log_theta: float,
    p_params: int,
    mask: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """
    Compute log-likelihood-based diagnostics for NB(obs).
    - y, mu: arrays (same shape)
    - log_theta: scalar (your fitted dispersion parameter)
    - p_params: count of *fitted* parameters in this slice
                (i.e., len(current_free_names) in your code)
    - mask: optional boolean mask (e.g., to drop burn-in days)

    Returns dict with: n, ll, nll, nll_per_obs, AIC, BIC
    """
    y = np.asarray(y, float)
    mu = np.asarray(mu, float)
    if y.shape != mu.shape:
        raise ValueError("y and mu must have the same shape")

    m = _as_mask(len(y), mask)
    y_m = y[m]; mu_m = mu[m]
    n = y_m.size

    ll = nb_loglik(y_m, mu_m, log_theta)  # includes constants in your implementation
    nll = -ll
    aic = -2.0 * ll + 2.0 * p_params
    bic = -2.0 * ll + p_params * np.log(max(1, n))

    return dict(
        n=float(n),
        ll=float(ll),
        nll=float(nll),
        nll_per_obs=float(nll / max(1, n)),
        AIC=float(aic),
        BIC=float(bic),
        k=float(p_params),
    )
