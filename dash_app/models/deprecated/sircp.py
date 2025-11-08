#!/usr/bin/env python3
#
# Implementation of SICR+P model

# S = Susceptible
# I = infected
# C = Reported Case
# R = Recovered/Removed
# P = Perceived Risk
#
# beta = transmission rate per person
# gamma = recovery rate (1/gamma = infectuosity period)
# alpha = perception growth factor
# delta = perception decay rate
# rho = reporting rate (average time from infection to report = 1/rho)
# k = risk perception sensibility

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

###### MODEL PARAMETERS ######
params = {
    "N": 100_000,            # Total population
    "beta_0": 0.5,          # Base transmission rate
    "gamma": 1/1,         # Recovery rate
    "alpha": 1e-5,          # Perception growth factor
    "delta": 0.05,          # Perception decay rate
    "compliance_max": 0.75, # Maximum compliance
    "k": 1,                 # Compliance sensitivity
    "rho": 0.011            # Reporting rate
}

###### INITIAL CONDITIONS ######
I0 = 10
initial_conditions = [
    params["N"] - I0,       # S0
    I0,                     # I0
    0,                      # C0
    0,                      # R0
    0                       # P0
]

###### MODEL DEFINITION ######
def sirp_model_with_compliance(t, y, beta_0, gamma, alpha, delta, compliance_max, k, rho, N):
    S, I, C, R, P = y
    
    #Calculate compliance and effective transmission rate (beta_eff)
    compliance = compliance_max * (1 - np.exp(k * P))
    beta_eff = beta_0 * (1 - compliance)

    #Calculate incidence (new infections per day)
    total_infected = I + C
    incidence = beta_eff * S * total_infected / N
   
    #ODE system
    dSdt = -incidence
    dIdt = incidence - (gamma + rho) * I
    dCdt = rho * I - gamma * C
    dRdt = gamma * (I + C)
    dPdt = alpha * C - delta * P    #perception driven by number of reported cases
    
    return [dSdt, dIdt, dCdt, dRdt, dPdt]
    
###### EQUATION INTEGRATION ######
t_span = [0, 160]
t_eval = np.linspace(t_span[0], t_span[1], t_span[1]*2)

solution = solve_ivp(
    lambda t, y: sirp_model_with_compliance(t, y, **params), #Pass parameters by name
    t_span,
    initial_conditions,
    t_eval=np.linspace(t_span[0], t_span[1], int((t_span[1]-t_span[0])*2)+1),
    method='RK45'
)

####### PLOT RESULTS ######
S, I, C, R, P = solution.y
compliance = params['compliance_max'] * (1 - np.exp(-params['k'] * P))
beta_eff = params['beta_0'] * (1 - compliance)
incidence = beta_eff * S * I / params['N']

plt.figure(figsize=(15, 10))

# Plot 1: All People Compartments
plt.subplot(2, 2, 1)
plt.plot(solution.t, S, label='Susceptible')
plt.plot(solution.t, I, label='Infected')
plt.plot(solution.t, C, label='Reported Cases', color='red') # Highlight C
plt.plot(solution.t, R, label='Recovered')
#plt.xlabel('Time (days)')
plt.ylabel('Population')
plt.title('Disease Compartments: S - I - C - R')
plt.legend()
plt.grid(True)

# Plot 2: Perception and Behavior
plt.subplot(2, 2, 2)
plt.plot(solution.t, P, 'm-', label='Risk Perception (P)')
plt.plot(solution.t, compliance, 'c--', label='Compliance')
#plt.xlabel('Time (days)')
plt.ylabel('Level')
plt.title('Risk Perception and Behavioral Response')
plt.legend()
plt.grid(True)

# Plot 3: Transmission Rate vs. Infected vs Reported
plt.subplot(2, 2, 3)
# Create a twin axis for rates and incidence
ax1 = plt.gca()
ax2 = ax1.twinx()
ax1.plot(solution.t, beta_eff, 'r-', label='β_eff')
ax1.axhline(y=params['beta_0'], color='r', linestyle='--', label='β_0 (base)')
ax1.set_ylabel('Transmission Rate', color='red')
ax1.tick_params(axis='y', labelcolor='red')
ax1.legend(loc='upper right')
ax2.plot(solution.t, I, label='Infections (I)', color='blue')
ax2.plot(solution.t, C, label='Reported Cases (C)', color='orange')
ax2.set_ylabel('Number of Cases', color='blue')
ax2.tick_params(axis='y', labelcolor='blue')
ax2.legend(loc='upper right', bbox_to_anchor=(1,0.88))
#plt.xlabel('Time (days)')
plt.title('Transmission Rate vs. Infected/Reported Cases')
plt.grid(True)

# Plot 4: Lag Demonstration
plt.subplot(2, 2, 4)
plt.plot(solution.t, I, label='Infections (I)', linewidth=3)
plt.plot(solution.t, C, label='Reported Cases (C)', linewidth=2)
# Find the peak of I and C to show the lag
peak_i_time = solution.t[np.argmax(I)]
peak_c_time = solution.t[np.argmax(C)]
plt.axvline(x=peak_i_time, color='blue', linestyle='--', alpha=0.7, label=f'Peak I (Day {peak_i_time:.1f})')
plt.axvline(x=peak_c_time, color='orange', linestyle='--', alpha=0.7, label=f'Peak C (Day {peak_c_time:.1f})')
#plt.xlabel('Time (days)')
plt.ylabel('Cases')
plt.title(f'Reporting DElay: {peak_c_time - peak_i_time:.1f} days')
plt.legend()
plt.grid(True)
plt.xlim(0, 100)

plt.tight_layout(h_pad=1)
plt.show()