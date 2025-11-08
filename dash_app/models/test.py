import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

def stronger_feedback_model():
    # Even stronger parameters for oscillations
    params = {
        'beta_0': 0.5,        # High transmission
        'gamma': 0.05,         # Recovery
        'gamma_C': 0.02,      # Slow reporting clearance (50-day memory)
        'rho': 0.4,           # High reporting
        'alpha': 0.5,         # Very strong risk response
        'delta': 0.02,        # Slow risk decay
        'compliance_max': 0.9,# High max compliance
        'k': 1.0,             # Very sensitive compliance
        'N': 1000000
    }
    
    R0 = params['beta_0'] / params['gamma']
    print(f"R0 = {R0:.2f}")
    
    def model(y, t):
        S, I, C, R, P = y
        beta_0, gamma, gamma_C, rho, alpha, delta, compliance_max, k, N = params.values()
        
        # Stronger nonlinearity - Hill function instead of exponential
        P50 = 10  # Half-saturation constant
        compliance = compliance_max * (P**2 / (P**2 + P50**2))
        
        beta_eff = beta_0 * (1 - compliance)
        incidence = beta_eff * S * (I + C) / N

        dSdt = -incidence
        dIdt = incidence - gamma * I - rho * I
        dCdt = rho * I - gamma_C * C
        dRdt = gamma * I + gamma_C * C
        dPdt = alpha * C - delta * P
        
        return [dSdt, dIdt, dCdt, dRdt, dPdt]
    
    # Initial conditions
    I0, C0 = 5000, 2000  # Substantial initial outbreak
    S0 = params['N'] - I0 - C0
    y0 = [S0, I0, C0, 0, 0]
    
    t = np.linspace(0, 2000, 2000)  # Longer simulation
    solution = odeint(model, y0, t)
    
    S, I, C, R, P = solution.T
    total_infected = I + C
    
    # Plot last 500 days only
    mask = t > 1500
    plt.figure(figsize=(10, 4))
    plt.plot(t[mask], total_infected[mask], 'b-', linewidth=2)
    plt.xlabel('Time (days)')
    plt.ylabel('Total Infections')
    plt.title('Endemic Dynamics with Strong Feedback')
    plt.grid(True, alpha=0.3)
    plt.show()
    
    return solution, params

# Try the stronger feedback version
strong_solution, strong_params = stronger_feedback_model()