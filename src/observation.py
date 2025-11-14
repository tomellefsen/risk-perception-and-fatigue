from __future__ import annotations
import numpy as np
from scipy.special import gammaln

def discrete_gamma_kernel(mean_days=3.0, sd_days=2.0, max_lag=14):
    """
    Returns nonnegative weights w[0..max_lag] summing to 1 (reporting delay).
    For Gamma(k,theta): mean=k*theta, var=k*theta^2
    Solve k=(mean/sd)^2, theta=sd^2/mean
    """
    k = (mean_days / sd_days) ** 2
    theta = (sd_days ** 2) / mean_days
    # Discretize via pmf of gamma mass in [i, i+1)
    # Approx: use CDF differences of continuous gamma
    from scipy.stats import gamma as gamma_dist
    edges = np.arange(0, max_lag + 2)
    cdf = gamma_dist.cdf(edges, a=k, scale=theta)
    w = np.maximum(np.diff(cdf), 0.0)
    w = w / w.sum()
    return w

def convolve_incidence_with_delay(incidence: np.ndarray, w: np.ndarray):
    """ causal convolution: mu_t = sum_{tau>=0} w_tau * incidence_{t-tau} """
    T = len(incidence)
    L = len(w)
    mu = np.zeros(T)
    for t in range(T):
        taumin = max(0, t - (L - 1))
        # reverse index: inc[t - tau] * w[tau]
        mu[t] = np.dot(incidence[taumin:t+1][::-1], w[:t - taumin + 1])
    return mu

def convolve_with_delay_with_buffer(incidence: np.ndarray, w: np.ndarray, prev_tail: np.ndarray | None = None):
    """
    If prev_tail is provided, it must be the last L-1 values of the *previous slice* incidence.
    We prepend prev_tail to the current incidence, convolve causally, then drop the first (L-1) outputs.
    """
    L = len(w)
    if prev_tail is None or len(prev_tail) != L-1:
        prev_tail = np.zeros(L-1, dtype=float)
    inc_ext = np.concatenate([prev_tail, incidence])
    # reuse your existing causal function on the extended series
    mu_ext = convolve_incidence_with_delay(inc_ext, w)
    # keep only outputs aligned with the current slice
    return mu_ext[(L-1):]

def nb_loglik(y: np.ndarray, mu: np.ndarray, log_theta: float):
    """
    NB parameterization:
      mean = mu > 0
      var  = mu + mu^2/theta  (theta>0)
    Stable log-likelihood; returns scalar log p(y|mu,theta)
    
    *** MODIFIED FOR NUMERICAL STABILITY ***
    """
    
    # --- STABILITY FIX ---
    # 1. Clip log_theta to prevent np.exp() from overflowing
    #    or theta becoming zero.
    log_theta = np.clip(log_theta, -50.0, 50.0)
    theta = np.exp(log_theta)
    
    # 2. Clip mu to avoid log(0)
    mu = np.clip(mu, 1e-9, None)
    
    y = np.asarray(y, dtype=float)

    # 3. Use the more stable log-likelihood formulation
    #    This avoids the (p = theta / (theta+mu)) division
    ll = (
        gammaln(y + theta) - gammaln(theta) - gammaln(y + 1.0)
        + theta * log_theta + y * np.log(mu)
        - (theta + y) * np.log(theta + mu)
    )
    
    return float(np.sum(ll))

def make_mu_from_model(pars, y0, t_eval, delay_w, rho_obs, 
                       Y=None, incidence=None, prev_inc_tail=None):
    from .model import simulate
    if (Y is None) or (incidence is None):
        Y, incidence = simulate(pars, y0, t_eval)
    # buffered convolution
    mu_delay = convolve_with_delay_with_buffer(incidence, delay_w, prev_inc_tail)
    mu = rho_obs * mu_delay
    return mu, Y, incidence
