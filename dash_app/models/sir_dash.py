##!/usr/bin/env python3
# A basic implementation of the SIR model

# S = Susceptible
# I = infected
# R = Recovered/Removed
# beta = transmission rate  per person
# gamma = recovery ruate (1/gamma = infectuosity period)

import numpy as np
from scipy.integrate import odeint

def run_sir(I0=10, N=10000, R0=0, beta=0.2, gamma=1/10, days=160):
    """Run SIR simulation and return (t, S, I, R), meta"""
    S0 = N - I0 - R0
    y0 = (S0, I0, R0)
    t = np.linspace(0, days, days*2)

    def deriv(y, t, N, beta, gamma):
        S, I, R = y
        dSdt = -beta * S * I / N
        dIdt = beta * S * I / N - gamma * I
        dRdt = gamma * I
        return dSdt, dIdt, dRdt

    ret = odeint(deriv, y0, t, args=(N, beta, gamma))
    S, I, R = ret.T

    # Calculate basic reproduction number
    R0_basic = beta / gamma
    if R0_basic > 1:
        state = "Epidemic will occur"
    elif R0_basic < 1:
        state = "Disease will die out"
    else:
        state = "Disease will become endemic"

    meta = {
        "R0": R0_basic,
        "state": state
    }

    return t, (S, I, R), meta

# ###### Plot Data (Matplotlib) ######
# import matplotlib.pyplot as plt

# plt.plot(t, S, "b", alpha=0.7, linewidth=2, label="Susceptible")
# plt.plot(t, I, "r", alpha=0.7, linewidth=2, label="Infected")
# plt.plot(t, R, "g", alpha=0.7, linewidth=2, label="Recovered")
# plt.xlabel("Time (days)")
# plt.ylabel("Number of individuals")
# plt.title("SIR Model Simulation")
# plt.legend()
# plt.grid(True)
# plt.figtext(0.99, 0.95, state, wrap = True, horizontalalignment='right', fontsize=10, fontweight="bold")
# plt.figtext(0.99, 0.91, r0_text, wrap = True, horizontalalignment='right', fontsize=10)
# plt.show()
