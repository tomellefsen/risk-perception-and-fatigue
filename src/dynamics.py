from __future__ import annotations
import numpy as np
from scipy.signal import welch, csd, hilbert

def dominant_period_days(series, fs=1.0):
    f, Pxx = welch(series, fs=fs, nperseg=min(256, len(series)))
    idx = np.argmax(Pxx[1:]) + 1  # skip zero freq
    f_star = f[idx]
    period = np.inf if f_star <= 0 else 1.0 / f_star
    return period, f_star, f, Pxx

def cross_spectrum(x, y, fs=1.0):
    f, Pxy = csd(x, y, fs=fs, nperseg=min(256, len(x)))
    return f, Pxy

def amplitude_envelope(series):
    return np.abs(hilbert(series))

def numerical_jacobian(fun, y_star, pars, eps=1e-6):
    """
    fun(t, y, pars)->dy/dt; evaluate Jacobian wrt y at equilibrium y_star
    """
    y_star = np.asarray(y_star, float)
    n = len(y_star)
    J = np.zeros((n, n))
    f0 = fun(0.0, y_star, pars)
    for j in range(n):
        e = np.zeros(n); e[j] = eps
        f1 = fun(0.0, y_star + e, pars)
        J[:, j] = (f1 - f0) / eps
    return J

def equilibrium_newton(fun, y_guess, pars, max_iter=50, tol=1e-10):
    """
    Solve f(y)=0 by Newton (finite-diff Jacobian).
    """
    y = y_guess.copy()
    for _ in range(max_iter):
        f = fun(0.0, y, pars)
        if np.linalg.norm(f, ord=np.inf) < tol:
            return y
        J = numerical_jacobian(fun, y, pars)
        try:
            step = np.linalg.solve(J, -f)
        except np.linalg.LinAlgError:
            step = -np.linalg.pinv(J) @ f
        y = y + step
        if np.linalg.norm(step, ord=np.inf) < tol:
            return y
    raise RuntimeError("Equilibrium not found")

# --- Add this code to src/dynamics.py ---

def sirc_pf_rhs_REDUCED(t, y_reduced, pars, N, original_rhs_fn):
    """
    RHS wrapper for the 5-state reduced system [I, C, R, P, F].
    It reconstructs the full 6-state vector [S, I, C, R, P, F]
    and calls the original RHS function.
    
    Args:
        t: time
        y_reduced: np.array [I, C, R, P, F]
        pars: parameters dict
        N: total population (float)
        original_rhs_fn: the original 6-state function (e.g., model.sirc_pf_rhs)
    
    Returns:
        np.array [dIdt, dCdt, dRdt, dPdt, dFdt]
    """
    I, C, R, P, F = y_reduced
    S = N - I - C - R
    
    # Check for biologically plausible state
    if S < 0:
        # Return a large derivative to push the solver back
        return np.full(5, 1e10) 
        
    y_full = np.array([S, I, C, R, P, F])
    
    # Call the original 6-state ODE function
    full_ders = original_rhs_fn(t, y_full, pars)
    
    # Return only the 5 derivatives for the reduced states
    # [dSdt, dIdt, dCdt, dRdt, dPdt, dFdt]
    return full_ders[1:] # [dIdt, dCdt, dRdt, dPdt, dFdt]


def numerical_jacobian_REDUCED(fun_reduced, y_star_reduced, pars, N, original_rhs_fn, eps=1e-6):
    """
    Calculates the 5x5 numerical Jacobian for the reduced system.
    """
    y_star_reduced = np.asarray(y_star_reduced, float)
    n = len(y_star_reduced) # n=5
    J = np.zeros((n, n))
    
    # Pass all required args to the reduced function
    f0 = fun_reduced(0.0, y_star_reduced, pars, N, original_rhs_fn)
    
    for j in range(n):
        e = np.zeros(n); e[j] = eps
        f1 = fun_reduced(0.0, y_star_reduced + e, pars, N, original_rhs_fn)
        J[:, j] = (f1 - f0) / eps
    return J


def equilibrium_newton_REDUCED(
    fun_reduced, y_guess_reduced, pars, N, original_rhs_fn, 
    max_iter=50, tol=1e-10
):
    """
    Solves f(y_reduced)=0 by Newton, using the reduced 5x5 system.
    
    Args:
        fun_reduced: The wrapper function (sirc_pf_rhs_REDUCED)
        y_guess_reduced: 5-state [I, C, R, P, F] guess
        pars: parameters dict
        N: total population (float)
        original_rhs_fn: the original 6-state function (model.sirc_pf_rhs)
        
    Returns:
        5-state equilibrium vector y_star_reduced
    """
    y = y_guess_reduced.copy()
    for _ in range(max_iter):
        f = fun_reduced(0.0, y, pars, N, original_rhs_fn)
        
        if np.linalg.norm(f, ord=np.inf) < tol:
            return y
            
        J = numerical_jacobian_REDUCED(fun_reduced, y, pars, N, original_rhs_fn)
        
        try:
            # This 5x5 solve should be stable now
            step = np.linalg.solve(J, -f)
        except np.linalg.LinAlgError:
            # Fallback just in case
            step = -np.linalg.pinv(J) @ f
            
        y = y + step
        if np.linalg.norm(step, ord=np.inf) < tol:
            return y
            
    raise RuntimeError("Equilibrium not found (reduced 5x5 system)")