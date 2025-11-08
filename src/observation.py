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

def nb_loglik(y: np.ndarray, mu: np.ndarray, log_theta: float):
    """
    NB parameterization:
      mean = mu > 0
      var  = mu + mu^2/theta  (theta>0)
    Stable log-likelihood; returns scalar log p(y|mu,theta)
    """
    mu = np.clip(mu, 1e-9, None)
    theta = np.exp(log_theta)
    # p = theta/(theta+mu); r = theta
    r = theta
    p = theta / (theta + mu)
    # logPMF: lgamma(y+r) - lgamma(r) - lgamma(y+1) + r*log(p) + y*log(1-p)
    y = np.asarray(y, dtype=float)
    ll = (gammaln(y + r) - gammaln(r) - gammaln(y + 1.0)
          + r * np.log(p) + y * np.log1p(-p))
    return float(np.sum(ll))

def make_mu_from_model(pars, y0, t_eval, delay_w, rho_obs, Y=None, incidence=None):
    """
    Convenience: simulate if not provided, convolve with delay, scale by reporting fraction rho.
    """
    from .model import simulate
    if (Y is None) or (incidence is None):
        Y, incidence = simulate(pars, y0, t_eval)
    mu_delay = convolve_incidence_with_delay(incidence, delay_w)
    mu = rho_obs * mu_delay
    
    try:
        Y, incidence = simulate(pars, y0, t_eval)
    except Exception as e:
        # Return signal upward so objective penalizes
        raise

    return mu, Y, incidence
