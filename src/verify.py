# verify.py
# This script tests the *fixes* for the bugs identified.
# Assumes it's in the root folder, and the (uncorrected) code
# is in src/model.py, src/fit.py, etc.

import numpy as np
import importlib
from scipy.integrate import solve_ivp

# --- Helpers to load your code ---
# We will "monkey-patch" your functions with corrected versions
try:
    from src import model as model_mod
    from src import observation as obs_mod
    from src import fit as fit_mod
    from src import dynamics as dyn_mod
except ImportError:
    print("Error: Could not import from src. \n"
          "Please ensure this script is in the root directory \n"
          "and your files (model.py, fit.py, etc.) are in a 'src' subdirectory.")
    exit()

print("--- Testing Bug Fixes ---")

# --- 1. Fix for model.py: simulate() ---
print("\n[Test 1: model.py simulate()]")

def simulate_FIXED(pars, y0, t_eval, rtol=1e-7, atol=1e-9):
    """Corrected incidence calculation."""
    sol = solve_ivp(
        fun=lambda t, y: model_mod.sirc_pf_rhs(t, y, pars),
        t_span=(t_eval[0], t_eval[-1]),
        y0=y0,
        t_eval=t_eval,
        method="LSODA",
        rtol=rtol,
        atol=atol
    )
    if not sol.success:
        raise RuntimeError(f"ODE failed: {sol.message}")
    Y = sol.y.T
    S = Y[:, 0]
    # ORIGINAL BUG: inc = np.maximum(np.diff(S, prepend=S[0]), 0.0)
    # FIX:
    inc = np.maximum(-np.diff(S, prepend=S[0]), 0.0)
    return Y, inc

# --- 2. Fix for observation.py: make_mu_from_model() ---
print("\n[Test 2: observation.py make_mu_from_model()]")

def make_mu_from_model_FIXED(pars, y0, t_eval, delay_w, rho, Y=None, incidence=None):
    """
    Corrected signature (uses keyword 'rho') and 
    removes redundant internal simulation.
    """
    try:
        if (Y is None) or (incidence is None):
            # Use our *fixed* simulate function
            Y, incidence = simulate_FIXED(pars, y0, t_eval)
    except Exception as e:
        print(f"Simulation failed: {e}")
        raise
        
    mu_delay = obs_mod.convolve_incidence_with_delay(incidence, delay_w)
    # FIX: Use 'rho' to match the new signature
    mu = rho * mu_delay
    return mu, Y, incidence

# Patch the observation module to use our fixed simulator
obs_mod.simulate = simulate_FIXED

# --- 3. Fix for fit.py: negloglik_nb() ---
print("\n[Test 3: fit.py negloglik_nb()]")

# This needs to be available in the function's scope
_fixed_make_mu = make_mu_from_model_FIXED

def negloglik_nb_FIXED(params_vec, param_names, bounds, fixed, 
                       t_eval, y_obs, delay_w, y0_template, build_pars_fn):
    """
    Corrected version:
    - Removes the first broken try/except block.
    - Uses the fixed make_mu_from_model with the correct keyword 'rho'.
    - Imports at the top (good practice).
    """
    # Imports should be at top of file, but we mock it here
    from src.observation import nb_loglik

    theta = fixed.copy()
    theta.update({k: v for k, v in zip(param_names, params_vec)})

    for k, v in theta.items():
        if not np.isfinite(v):
            return 1e50

    pars = build_pars_fn(theta)
    N = float(pars["N"])
    
    R0 = float(theta.get("R0", y0_template.get("R0", 0.0)))
    I0 = float(theta["I0"]); C0 = float(theta["C0"])
    P0 = float(theta["P0"]); F0 = float(theta["F0"])
    S0 = N - I0 - C0 - R0

    if not np.isfinite(S0) or S0 < 0: 
        return 1e50
    if not (0.0 <= P0 <= 1.0 and 0.0 <= F0 <= 1.0):
        return 1e50
    y0 = np.array([S0, I0, C0, R0, P0, F0], float)
    if not np.all(np.isfinite(y0)):
        return 1e50
    
    # ---- Simulate; catch ODE/num errors ----
    # ORIGINAL BUG: Deleted the first broken try/except block.
    try:
        # FIX: Get the correct reporting parameter
        rho_report = float(theta["rho_obs"]) 
        # FIX: Call our fixed function with the 'rho' keyword
        mu, _, _ = _fixed_make_mu(pars, y0, t_eval, delay_w, rho=rho_report)
    except Exception:
        return 1e50

    # ---- Likelihood ----
    log_theta = float(theta["log_theta"])
    if not np.isfinite(log_theta):
        return 1e50

    ll = nb_loglik(y_obs, mu, log_theta=log_theta)
    return -ll if np.isfinite(ll) else 1e50

# --- 4. Fix for dynamics.py: equilibrium_newton() ---
print("\n[Test 4: dynamics.py equilibrium_newton()]")

def sirc_pf_rhs_REDUCED(t, y_reduced, pars, N):
    """
    A wrapper for the RHS function that only solves for [I, C, R, P, F].
    S is calculated from the conservation law.
    """
    I, C, R, P, F = y_reduced
    S = N - I - C - R  # Calculate S
    
    # Call the original RHS with the full state vector
    full_y = np.array([S, I, C, R, P, F])
    dSdt, dIdt, dCdt, dRdt, dPdt, dFdt = model_mod.sirc_pf_rhs(t, full_y, pars)
    
    # Return only the derivatives for the 5 states we are solving for
    # We can skip dSdt because dS/dt = -(dI/dt + dC/dt + dR/dt)
    return np.array([dIdt, dCdt, dRdt, dPdt, dFdt])

def numerical_jacobian_REDUCED(fun_reduced, y_star_reduced, pars, N, eps=1e-6):
    """
    A wrapper for the jacobian function that works on the reduced system.
    """
    y_star_reduced = np.asarray(y_star_reduced, float)
    n = len(y_star_reduced) # n=5
    J = np.zeros((n, n))
    f0 = fun_reduced(0.0, y_star_reduced, pars, N)
    for j in range(n):
        e = np.zeros(n); e[j] = eps
        f1 = fun_reduced(0.0, y_star_reduced + e, pars, N)
        J[:, j] = (f1 - f0) / eps
    return J

def equilibrium_newton_FIXED(fun_reduced, y_guess_reduced, pars, N, max_iter=50, tol=1e-10):
    """
    Modified Newton solver that uses the reduced 5x5 system.
    """
    y = y_guess_reduced.copy()
    for _ in range(max_iter):
        f = fun_reduced(0.0, y, pars, N)
        if np.linalg.norm(f, ord=np.inf) < tol:
            return y # Return the 5-state vector
        
        # Use the reduced jacobian
        J = numerical_jacobian_REDUCED(fun_reduced, y, pars, N)
        
        try:
            step = np.linalg.solve(J, -f)
        except np.linalg.LinAlgError:
            # This should be much rarer now
            print("Warning: Reduced Jacobian is singular, using pseudo-inverse.")
            step = -np.linalg.pinv(J) @ f
            
        y = y + step
        if np.linalg.norm(step, ord=np.inf) < tol:
            return y
    raise RuntimeError("Equilibrium not found (reduced system)")


# --- Verification Runs ---
print("\n--- Running Verification ---")

# --- Setup Dummy Data ---
T = 100
t_eval = np.arange(T, dtype=float)
y_obs = np.abs(np.sin(t_eval / 20) * 50 + 20 + np.random.poisson(5, T))
delay_w = obs_mod.discrete_gamma_kernel(3.0, 2.0, 14)
N = 1e6

# Dummy parameters for testing
# These *must* match the names expected by your 'params.yaml'
param_names = ["beta0", "I0", "C0", "P0", "F0", "log_theta", "rho_obs"]
params_vec = [0.3, 10.0, 5.0, 0.1, 0.1, np.log(25.0), 0.5] # A guess
bounds = [(0.1, 1.0), (1, 1000), (1, 1000), (0,1), (0,1), (np.log(1), np.log(100)), (0.1, 1.0)]
fixed = {
    "N": N, "gamma": 1/7, "rho": 1/5, "alpha": 0.1, "delta": 0.05,
    "epsilon": 0.02, "phi": 0.01, "k": 25.0, "compliance_max": 0.8,
    "beta_floor": 0.1, "R0": 0.0
}
# This needs to match the function in run.py
def build_pars_fn(theta_dict):
    keys = ["N","beta0","gamma","rho","alpha","delta","epsilon","phi","k","compliance_max","beta_floor"]
    pars = {k: float(theta_dict[k]) for k in keys if k in theta_dict}
    return pars

y0_template = {"S0": N, "R0": 0.0}

# --- Test 1 & 2 & 3: Run the fixed objective function ---
print("\n[VERIFY] Testing fixed negloglik_nb...")
nll = negloglik_nb_FIXED(params_vec, param_names, bounds, fixed, 
                         t_eval, y_obs, delay_w, y0_template, build_pars_fn)

if np.isfinite(nll) and nll < 1e49:
    print(f"  SUCCESS: negloglik_nb returned a finite value: {nll:.2f}")
else:
    print(f"  FAILURE: negloglik_nb returned a penalty: {nll}")

# --- Test 4: Run the fixed equilibrium finder ---
print("\n[VERIFY] Testing fixed equilibrium_newton...")
try:
    # Get a "guess" from the end of a simulation
    theta = fixed.copy()
    theta.update({k: v for k, v in zip(param_names, params_vec)})
    pars = build_pars_fn(theta)
    y0_full = np.array([
        N - theta["I0"] - theta["C0"] - theta["R0"],
        theta["I0"], theta["C0"], theta["R0"], theta["P0"], theta["F0"]
    ], float)
    
    Y_sim, _ = simulate_FIXED(pars, y0_full, np.arange(365))
    y_guess_full = Y_sim[-1]
    
    # Create the reduced 5-state guess vector
    y_guess_reduced = y_guess_full[1:] # [I, C, R, P, F]
    
    y_star_reduced = equilibrium_newton_FIXED(
        sirc_pf_rhs_REDUCED, y_guess_reduced, pars, N
    )
    
    # Reconstruct full state
    I,C,R,P,F = y_star_reduced
    S = N - I - C - R
    y_star_full = np.array([S,I,C,R,P,F])
    
    print(f"  SUCCESS: Found equilibrium (reduced system).")
    print(f"  Full y_star: {[f'{x:.2f}' for x in y_star_full]}")
    
    # Final check: are derivatives near zero?
    ders = model_mod.sirc_pf_rhs(0.0, y_star_full, pars)
    print(f"  Derivatives at y_star: {[f'{d:.2e}' for d in ders]}")
    if np.allclose(ders, 0.0, atol=1e-5):
        print("  Derivatives are all near zero. Fix is working.")
    else:
        print("  WARNING: Derivatives are not zero. Equilibrium find failed.")
        
except Exception as e:
    print(f"  FAILURE: equilibrium_newton_FIXED raised an error: {e}")

print("\n--- Verification Complete ---")