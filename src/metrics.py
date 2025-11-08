from __future__ import annotations
import numpy as np

def time_series_splits(T, initial_train, test_horizon, step):
    """
    Rolling-origin splits:
      train: [0 .. t_end], test: (t_end+1 .. t_end+test_horizon)
      advance by `step`
    Yields (train_idx, test_idx)
    """
    t_end = initial_train - 1
    while t_end + test_horizon < T - 1:
        train_idx = np.arange(0, t_end + 1)
        test_idx  = np.arange(t_end + 1, t_end + 1 + test_horizon)
        yield train_idx, test_idx
        t_end += step

def mae(y, yhat): return float(np.mean(np.abs(y - yhat)))
def rmse(y, yhat): return float(np.sqrt(np.mean((y - yhat)**2)))

def nb_mean_log_pred_density(y, mu, log_theta):
    """
    Average log p(y_t | mu_t, theta) over test set.
    """
    from .observation import nb_loglik
    return nb_loglik(y, mu, log_theta) / len(y)

def simulate_nb_intervals(mu, log_theta, n_paths=2000, alpha=(0.1, 0.5)):
    """
    Parametric bootstrap intervals under NB:
    returns dict: {"p10":..., "p50":..., "p90":...} or using alpha→ quantiles
    """
    theta = np.exp(log_theta)
    # sample y ~ NB(mean=mu, theta)
    # Convert to (r, p) with r=theta, p=theta/(theta+mu)
    p = theta / (theta + mu)
    r = theta
    rng = np.random.default_rng(123)
    # NB as Gamma-Poisson mixture: Poisson(Gamma(r, (1-p)/p))
    lam = rng.gamma(shape=r, scale=(1-p)/p, size=(n_paths, len(mu)))  # mean=r*scale = r*(1-p)/p = mu
    y = rng.poisson(lam=lam)
    qs = np.quantile(y, q=[alpha[0]/2, 0.5, 1-alpha[0]/2], axis=0)
    return {"lower": qs[0], "median": qs[1], "upper": qs[2]}
