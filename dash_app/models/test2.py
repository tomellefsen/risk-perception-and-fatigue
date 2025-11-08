import numpy as np
from scipy.integrate import solve_ivp

def sicr_pf_delayed(
    t, y, N,
    beta_0, gamma, rho,
    alpha, delta0, compliance_max,
    gamma_F, epsilon, phi,
    omega, beta_floor=0.2,
    n_stages=4, tau=5.0
):
    """
    SICR-PF model with distributed delay (chain of n sub-compartments for C).
    """

    # Unpack state vector
    # y = [S, I, R, P, F, C1, C2, ..., Cn]
    S, I, R, P, F, *C_chain = y
    C_chain = np.array(C_chain)
    C1 = C_chain[0]
    Cn = C_chain[-1]  # "effective reported" cases

    # --- COMPLIANCE ---
    compliance = compliance_max * (1 / (1 + np.exp(-8 * (P - 0.5))))

    # --- EFFECTIVE BETA ---
    beta_min = beta_0 * beta_floor
    beta_eff = beta_min + (beta_0 - beta_min) * (1.0 - compliance)

    # --- INFECTION FORCE ---
    total_infected = I + Cn  # total infectious population
    incidence = beta_eff * S * I / N

    # --- SICR CORE ---
    dSdt = -incidence + omega * R
    dIdt = incidence - (gamma + rho) * I
    dRdt = gamma * (I + Cn) - omega * R

    # --- DELAY CHAIN FOR C ---
    k = n_stages / tau
    dC_chain = np.zeros_like(C_chain)
    dC_chain[0] = rho * I - k * C_chain[0]               # inflow from I
    for j in range(1, n_stages):
        dC_chain[j] = k * (C_chain[j - 1] - C_chain[j])  # pass-through between stages

    # --- PERCEIVED RISK & FATIGUE ---
    dPdt = alpha * (Cn / N) - (delta0 + gamma_F * F) * P
    dFdt = epsilon * P - phi * F

    return np.concatenate(([dSdt, dIdt, dRdt, dPdt, dFdt], dC_chain))


# Example simulation
if __name__ == "__main__":
    # Parameters
    N = 1_000_000
    beta_0 = 0.4
    gamma = 1/10
    rho = 1/7
    alpha = 2.0
    delta0 = 0.3
    compliance_max = 0.7
    gamma_F = 0.1
    epsilon = 0.01
    phi = 0.001
    omega = 1/180  # waning immunity
    beta_floor = 0.2
    n_stages = 4
    tau = 5.0

    # Initial conditions
    I0 = 10
    S0 = N - I0
    R0 = 0
    P0 = 0.0
    F0 = 0.0
    C_chain0 = np.zeros(n_stages)
    y0 = np.concatenate(([S0, I0, R0, P0, F0], C_chain0))

    # Time span
    t_span = (0, 800)
    t_eval = np.linspace(*t_span, 1000)

    # Solve ODE
    sol = solve_ivp(
        sicr_pf_delayed, t_span, y0, t_eval=t_eval,
        args=(N, beta_0, gamma, rho, alpha, delta0,
              compliance_max, gamma_F, epsilon, phi, omega,
              beta_floor, n_stages, tau)
    )

    # Extract results
    S, I, R, P, F = sol.y[:5]
    C_chain = sol.y[5:]
    C_eff = C_chain[-1]

    import matplotlib.pyplot as plt

    plt.figure(figsize=(10,6))
    plt.plot(sol.t, I, label="Infected (I)")
    plt.plot(sol.t, C_eff, label="Reported (C)")
    plt.plot(sol.t, P, label="Perceived risk (P)")
    plt.plot(sol.t, F, label="Fatigue (F)")
    plt.xlabel("Time (days)")
    plt.ylabel("Proportion / Count")
    plt.legend()
    plt.grid(True)
    plt.title("SICR-PF Model with Distributed Delay (τ = 5 days, n = 4)")
    plt.show()
