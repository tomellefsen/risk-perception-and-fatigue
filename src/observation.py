"""
Observation model: linking latent incidence to observed data.

This module contains all functions related to the observation process,
which transforms the model's latent `incidence` into the expected
mean `mu` of the observed data.

This process includes:
1.  Modeling the reporting delay (`discrete_gamma_kernel`).
2.  Applying the delay via convolution (`convolve_with_delay_with_buffer`).
3.  Applying weekly seasonality (`apply_weekly_harmonics`).
4.  The Negative Binomial log-likelihood function (`nb_loglik`) itself.
5.  `make_mu_from_model` that combines simulation and the full observation 
    pipeline, which is called to run the analysis.

"""
from __future__ import annotations
import numpy as np
from scipy.special import gammaln
from functools import lru_cache

from .model import simulate

# -----------------------------------
# HELPERS
# -----------------------------------

@lru_cache(maxsize=None)
def discrete_gamma_kernel_cached(mean_days=3.0, sd_days=2.0, max_lag=14):
    """
    Cached wrapper for `discrete_gamma_kernel`.

    Uses `@lru_cache` to memoize the delay kernel, avoiding redundant
    computations if the same delay parameters are requested multiple
    times (e.g., during optimization with fixed delays).
    """
    return discrete_gamma_kernel(mean_days, sd_days, max_lag)


def apply_weekly_harmonics(mu, a1, b1, a2, b2):
    """
    Multiplicative 7-day pattern via log-link harmonics, constrained 
    to mean 1.

    Modulates a mean trajectory `mu` using 1st and 2nd order Fourier terms 
    (harmonics) in a log-link model. The seasonal effect `s_t` is calculated 
    and then centered ( `log_s -= mean(log_s)` ) so that `exp(log_s)` has a 
    geometric mean of 1, preserving the overall mass of `mu` over time.

    Returns
    -------
    np.ndarray
        The seasonally-adjusted mean trajectory.
    """
    t = np.arange(len(mu), dtype=float)
    w = 2.0 * np.pi / 7.0
    log_s = (
        a1 * np.sin(w * t) + b1 * np.cos(w * t)
      + a2 * np.sin(2 * w * t) + b2 * np.cos(2 * w * t)
    )
    # Center so the multiplicative factor has average 1
    log_s -= np.mean(log_s)
    mu = np.maximum(mu, 1e-12)   # keep numerical safety
    
    return mu * np.exp(log_s)


def convolve_incidence_with_delay(incidence: np.ndarray, w: np.ndarray):
    """
    Applies a causal convolution of incidence with a delay kernel.

    Computes the expected reported cases `mu` at time `t` based on
    past incidence and the delay kernel `w`.
    `mu_t = sum_{tau>=0} w_tau * incidence_{t-tau}`

    Parameters
    ----------
    incidence : np.ndarray
        A 1D array of latent daily incidence.
    w : np.ndarray
        The 1D delay kernel (e.g., from `discrete_gamma_kernel`).

    Returns
    -------
    np.ndarray
        The 1D array of convolved-and-delayed mean observations `mu`.

    """
    T = len(incidence)
    L = len(w)
    mu = np.zeros(T)
    for t in range(T):
        taumin = max(0, t - (L - 1))
        # reverse index: inc[t - tau] * w[tau]
        mu[t] = np.dot(incidence[taumin:t+1][::-1], w[:t - taumin + 1])
        
    return mu

# -----------------------------------
# END OF HELPERS
# -----------------------------------

def discrete_gamma_kernel(mean_days=3.0, sd_days=2.0, max_lag=None):
    """
    Computes a discrete reporting delay kernel from a Gamma distribution.

    Models the reporting delay as a continuous Gamma(k, theta)
    distribution, where `k` and `theta` are solved for using the
    method of moments from `mean_days` and `sd_days`.

    The continuous distribution is then discretized by taking the
    probability mass in each 1-day bin `[i, i+1)`. The resulting
    weights `w` are normalized to sum to 1.

    Parameters
    ----------
    mean_days : float, optional
        The desired mean of the delay (in days).
    sd_days : float, optional
        The desired standard deviation of the delay (in days).
    max_lag : int, optional
        The day at which to truncate the kernel. If None, it is
        auto-calculated based on mean and SD.

    Returns
    -------
    np.ndarray
        A 1D array `w` of length `max_lag + 1` containing the
        delay weights, where `w[i]` is the probability of a
        delay of `i` days.
    """
    if max_lag is None:
        max_lag = int(np.ceil(mean_days + 4*sd_days)) # whatever
        
    k = (mean_days / sd_days) ** 2
    theta = (sd_days ** 2) / mean_days
    # Discretize via pmf of gamma mass in [i, i+1)
    # Approx: use cumulative distribution function (CDF) 
    # differences of continuous gamma
    from scipy.stats import gamma as gamma_dist
    edges = np.arange(0, max_lag + 2)
    cdf = gamma_dist.cdf(edges, a=k, scale=theta)
    w = np.maximum(np.diff(cdf), 0.0)
    w = w / w.sum()
    
    return w


def nb_loglik(y: np.ndarray, mu: np.ndarray, log_theta: float):
    """
    Calculates the log-likelihood for the NB2 (Quadratic Negative Binomial) distribution.

    Provides a numerically stable calculation of the
    sum of log-likelihoods for a set of observations `y` given their
    predicted means `mu` and a single dispersion parameter `log_theta`.

    It handles `NaN` values in the observations `y` by masking them out.

    Parameters
    ----------
    y : np.ndarray
        1D array of observed counts. Can contain NaNs.
    mu : np.ndarray
        1D array of predicted means. Must be same shape as `y`.
    log_theta : float
        The scalar log-dispersion parameter.

    Returns
    -------
    float
        The total log-likelihood `sum(log(p(y_i | mu_i, theta)))` over
        all valid (non-NaN) data points.

    """
    # Numerical safety 
    y = np.asarray(y, dtype=float)
    mu    = np.asarray(mu,    dtype=float)
    mu    = np.maximum(mu, 1e-12)      # ensure strictly positive means

    theta = np.exp(log_theta)          # NB2 parameterization: Var = mu + mu^2/theta
    theta = np.maximum(theta, 1e-12)
    
    # Clip to avoid log(0)
    log_theta = np.clip(log_theta, -50.0, 50.0)
    mu = np.clip(mu, 1e-9, None)
    
    # ignore NaNs in y_obs:
    mask = np.isfinite(y) & np.isfinite(mu)
    y = y[mask]; m = mu[mask]
    
    y = np.asarray(y, dtype=float)

    # NB2 formulation
    ll = (
        gammaln(y + theta) - gammaln(theta) - gammaln(y + 1.0)
        + theta * log_theta + y * np.log(mu)
        - (theta + y) * np.log(theta + mu)
    )
    
    return float(np.sum(ll))


def make_mu_from_model(pars, y0, t_eval, delay_w, rho_obs, 
                       Y=None, incidence=None):
    """
    Runs the full observation pipeline: simulate, convolve, and scale.
    1.  Calls `model.simulate` to get `Y` (states) and `incidence`
        (if they are not passed in).
    2.  Applies the buffered delay convolution
        (`convolve_with_delay_with_buffer`) to the `incidence`.
    3.  Scales the convolved mean by the reporting fraction `rho_obs`.
    4.  Ensures the final `mu` is numerically safe (>= 1e-12).
    5.  Extracts the `next_tail` (last L-1 incidences) to be passed
        to the *next* fitting slice.

    Parameters
    ----------
    pars : dict
        The parameter dictionary for the ODE.
    y0 : np.ndarray
        The 6-state initial conditions.
    t_eval : np.ndarray
        The time points for evaluation.
    delay_w : np.ndarray
        The delay kernel.
    rho_obs : float
        The reporting fraction (scalar).
    Y : np.ndarray, optional
        Pre-computed model states. If provided, simulation is skipped.
    incidence : np.ndarray, optional
        Pre-computed incidence. If provided, simulation is skipped.
    prev_inc_tail : np.ndarray, optional
        The incidence tail from the previous slice for buffering.

    Returns
    -------
    tuple
        (mu, Y, incidence, next_tail)
        - mu : np.ndarray - The final predicted mean observations.
        - Y : np.ndarray - The (T, 6) state trajectories.
        - incidence : np.ndarray - The (T,) latent incidence.
        - next_tail : np.ndarray - The L-1 tail of `incidence` for the
          next slice.

    """
    if (Y is None) or (incidence is None):
        Y, incidence = simulate(pars, y0, t_eval)

    mu_delay = convolve_incidence_with_delay(incidence, delay_w)
    mu = rho_obs * mu_delay
    mu = np.maximum(mu, 1e-12)  # numeric safety
    
    return mu, Y, incidence