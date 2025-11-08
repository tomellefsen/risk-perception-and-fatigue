#!/usr/bin/env python3
"""
SICR-PF model
SIR variant model using P-F (Perceived Risk - Response Fatigue) dynamics.
See associated publication for details

  S = Susceptible
  I = infected
  C = Reported Case
  R = Recovered/Removed

  P = Perceived Risk
  F = Behavioral Fatigue

  beta_0: baseline infection rate
  gamma: recovery rate (1/gamma = infectuosity period)
  rho: reporting rate (average time from infection to report = 1/rho)
  alpha: perception growth factor (fast)
  delta0: baseline perception decay
  gamma_F: Perception's sensibility to fatigue
  epsilon: accumulation speed of F from P
  phi: fatigue decay rate (slow)
  beta_floor: minimum infectious (never 0)
"""

import numpy as np
from scipy.integrate import solve_ivp
import plotly.graph_objects as go
from plotly.subplots import make_subplots


###### MODEL DEFINITION ######
def sicr_pf_v2_model(
    t, y, N, beta_0, gamma, rho,
    alpha, delta0, comp_max,
    gamma_F, epsilon, phi, omega,       # <- fatigue params
    beta_floor=0.1                    # 20% floor on beta
):
    """
    SICR + PF (perceived risk and fatigue) with bounded, asymmetric dynamics:
        P: fast-reacting, inhibited by F, quick decay
        F: slow accumulator of P, very slow decay
    All states P,F remain in [0,1] by inflow_outflow structure.
    """

    S, I, C, R, P, F = y

    # Compliance -> beta
    P0 = 0.03
    k  = 25.0
    compliance = comp_max * (1.0 / (1.0 + np.exp(-k*(P - P0))))

    beta_floor = 0.10
    beta_min   = beta_0 * beta_floor
    beta_eff   = beta_min + (beta_0 - beta_min) * (1.0 - compliance)

    # Incidence 
    total_infected = I + C
    incidence = beta_eff * S * total_infected / N

    # Perception driver from delayed C
    C50 = 7.0e-4 # ~70 per 100k half-saturation
    h   = 1.5
    C_over_N = C / N
    D = (C_over_N**h) / (C50**h + C_over_N**h)
    
    # Epidemic dynamics
    dSdt = -incidence + omega * R
    dIdt = incidence - (gamma + rho) * I
    dCdt = rho * I - gamma * C
    dRdt = gamma * (I + C) - omega * R

    # Behavioural dynamics
    dPdt = alpha * D - (delta0 + gamma_F * F) * P
    dFdt = epsilon * P - phi * F
    
    return [dSdt, dIdt, dCdt, dRdt, dPdt, dFdt]



###### WRAPPER FOR DASH ######
def run_sicr_pf_v2(I0=10, N=100_000, beta_0=0.5, gamma=0.1, rho=0.01, alpha=0.01, delta=0.01,
                             compliance_max=0.9, gamma_F=0.6, epsilon=0.012, phi=0.0007, omega=1/180):
    initial_conditions = [N - I0, I0, 0, 0, 0, 0]   # S0, I0, C0, R0, P0, F0
    t_span = (0, 360)
    t_eval = np.linspace(t_span[0], t_span[1], int((t_span[1] - t_span[0]) * 2) + 1)

    sol = solve_ivp(
        sicr_pf_v2_model,
        t_span,
        initial_conditions,
        args=(N, beta_0, gamma, rho, alpha, delta, compliance_max, gamma_F, epsilon, phi, omega),
        t_eval=t_eval,
        method="LSODA"
    )

    # Extract compartments
    S, I, C, R, P, F = sol.y

    # Metadata
    R0_basic = beta_0 / gamma
    if R0_basic > 1:
        state = "Epidemic will occur"
    elif R0_basic < 1:
        state = "Disease will die out"
    else:
        state = "Disease will become endemic"

    compliance = np.clip(compliance_max * P, 0, 1)  # elementwise clamp in [0, 1]

    beta_min = beta_0 * 0.2
    beta_eff = beta_min + (beta_0 - beta_min) * (1.0 - compliance)
    beta_eff = np.clip(beta_eff, beta_min, beta_0)  # elementwise clamp in [beta_min, beta_0]

    Y = sol.y.T
    S = Y[:, 0] 
    # incidence_day[t] ≈ max(S[t-1]-S[t], 0)
    inc = np.maximum(np.diff(S, prepend=S[0]), 0.0)

    meta = {
        "compliance": compliance,
        "beta_eff": beta_eff,
        "beta_0": beta_0,
        "R0": R0_basic,
        "state": state,
        "rho": rho,
        "inc": inc
    }

    return sol.t, [S, I, C, R, P, F], meta



###### PLOTLY DASHBOARD ######
def plot_sicr_pf_dashboard(t, compartments, meta):
    # Unpack data
    try: 
        S, I, C, R, P, F = compartments
    except ValueError:
        raise ValueError("Expected compartments not match the provided data.")
    
    try:
        compliance = meta["compliance"]
        beta_eff = meta["beta_eff"]
        beta_0 = meta["beta_0"]
        R0 = meta["R0"]
        state = meta["state"]
        rho = meta["rho"]
        inc = meta["inc"]
    except ValueError:
        raise ValueError("Expected meta not match the provided data.")

    # -----------------------------------
    # HELPERS
    
    def subplot_index(row, col):
        ncols = 2
        return (row - 1) * ncols + col

    def add_subplot_box(fig, row, col, line_width=1):
        """Draws a thin border box around the subplot (domain coords 0..1)."""
        idx = subplot_index(row, col)  # Ensure this function returns the correct subplot index
        xref = f"x{idx} domain" if idx > 1 else "x domain"
        yref = f"y{idx} domain" if idx > 1 else "y domain"

        fig.add_shape(
            type="rect",
            xref=xref, yref=yref,
            x0=0, y0=0, x1=1, y1=1,
            line=dict(color="black", width=line_width),
            fillcolor="rgba(0,0,0,0)",
            layer="above",
        )

    # Because Plotly sucks
    def add_local_legend(fig, row, col, labels_colors, x_right=0.98, y_top=0.98, font_size=14):
        """
        Adds a simple fake legend box with text-based line samples.
        labels_colors: list of (label, color, dash)
        """
        subplot_id = '' if (row == 1 and col == 1) else str((row-1)*2 + col)
        xref = f"x{subplot_id} domain"
        yref = f"y{subplot_id} domain"

        # tighter background box
        box_height = 0.065 * len(labels_colors) + 0.015
        fig.add_shape(
            type="rect",
            xref=xref, yref=yref,
            x0=x_right - 0.35, y0=y_top - box_height,
            x1=x_right, y1=y_top,
            fillcolor="white", opacity=0.9,
            line=dict(color="black", width=1),
            layer="above"
        )

        # legend entries
        for i, (label, color, dash) in enumerate(labels_colors):
            if dash == "solid":
                symbol = f"<span style='color:{color};'>━━</span>"
            else:
                symbol = f"<span style='color:{color};'>╌╌</span>"

            fig.add_annotation(
                xref=xref, yref=yref,
                x=x_right - 0.33, y=y_top - i * 0.065 - 0.02,
                showarrow=False,
                text=f"{symbol} {label}",
                font=dict(size=font_size, family="Times New Roman"),
                align="left", xanchor="left", yanchor="top"
            )

        # legend entries
        for i, (label, color, dash) in enumerate(labels_colors):
            if dash == "solid":
                symbol = f"<span style='color:{color};'>━━</span>"
            else:
                symbol = f"<span style='color:{color};'>╌╌</span>"

            fig.add_annotation(
                xref=xref, yref=yref,
                x=x_right-0.33, y=y_top - i*0.065 - 0.02,
                showarrow=False,
                text=f"{symbol} {label}",
                font=dict(size=font_size, family="Times New Roman"),
                align="left", xanchor="left", yanchor="top"
            )

    # END OF HELPERS
    # -----------------------------------

    # compute tick spacing (8 intervals -> 9 ticks)
    t_max = float(np.max(t))
    ticks_count = 8  # 8 intervals (so 9 tick positions)
    raw_ticks = np.linspace(0.0, t_max, ticks_count + 1)
    # If t is integer-based days, make ticks integers; otherwise 1-decimal
    if np.allclose(np.round(t), t):
        tickvals = [int(round(v)) for v in raw_ticks]
    else:
        tickvals = [round(float(v), 1) for v in raw_ticks]

    # build subplots 
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "Disease Compartments: S — I — C — R",
            "Risk Perception and Behavioral Response",
            "Transmission Rate vs. Infected / Reported",
            f"Reporting Delay: {1/rho:.1f} days"
        ),
        specs=[[{}, {}],
            [{"secondary_y": True}, {}]],
        horizontal_spacing=0.04,  # tighten the gaps
        vertical_spacing=0.07
    )

    # dd traces (turn off global legend; fake local legends will be used)
    # Plot 1: Compartments
    fig.add_trace(go.Scatter(x=t, y=S, mode="lines", name="Susceptible",
                            line=dict(color="royalblue", width=3), showlegend=False), row=1, col=1)
    fig.add_trace(go.Scatter(x=t, y=I, mode="lines", name="Infected",
                            line=dict(color="orange", width=3), showlegend=False), row=1, col=1)
    fig.add_trace(go.Scatter(x=t, y=C, mode="lines", name="Reported Cases",
                            line=dict(color="red", width=3), showlegend=False), row=1, col=1)
    fig.add_trace(go.Scatter(x=t, y=R, mode="lines", name="Recovered",
                            line=dict(color="green", width=3), showlegend=False), row=1, col=1)
    fig.update_yaxes(title_text="Population", row=1, col=1)

    # Plot 2: Risk perception & compliance
    fig.add_trace(go.Scatter(x=t, y=P, mode="lines", name="Risk Perception (P)",
                            line=dict(color="magenta", width=3), showlegend=False), row=1, col=2)
    fig.add_trace(go.Scatter(x=t, y=F, mode="lines", name="Fatigue (F)",
                            line=dict(color="gray", width=3), showlegend=False), row=1, col=2)
    fig.add_trace(go.Scatter(x=t, y=compliance, mode="lines", name="Compliance",
                            line=dict(color="cyan", dash="dash", width=2), showlegend=False), row=1, col=2)
    fig.update_yaxes(title_text="Level", row=1, col=2)

    # Plot 3: Transmission (left) vs Cases (right)
    fig.add_trace(go.Scatter(x=t, y=beta_eff, mode="lines", name="β_eff",
                            line=dict(color="red", width=3), showlegend=False), row=2, col=1, secondary_y=False)
    fig.add_trace(go.Scatter(x=t, y=[beta_0]*len(t), mode="lines", name="β₀ (base)",
                            line=dict(color="red", dash="dash", width=2), showlegend=False), row=2, col=1, secondary_y=False)
    fig.add_trace(go.Scatter(x=t, y=I, mode="lines", name="Infections (I)",
                            line=dict(color="blue", width=3), showlegend=False), row=2, col=1, secondary_y=True)
    fig.add_trace(go.Scatter(x=t, y=C, mode="lines", name="Reported (C)",
                            line=dict(color="orange", width=3), showlegend=False), row=2, col=1, secondary_y=True)

    fig.update_yaxes(title_text="Transmission rate", row=2, col=1, secondary_y=False)
    fig.update_yaxes(title_text="Number of cases", row=2, col=1, secondary_y=True)

    # Plot 4: Reporting delay + shaded region
    fig.add_trace(go.Scatter(x=t, y=I, mode="lines", name="Infections (I)",
                            line=dict(color="blue", width=3), showlegend=False), row=2, col=2)
    fig.add_trace(go.Scatter(x=t, y=C, mode="lines", name="Reported (C)",
                            line=dict(color="orange", width=3), showlegend=False), row=2, col=2)

    peak_i_time = float(t[np.argmax(I)])
    peak_c_time = float(t[np.argmax(C)])
    if peak_c_time < peak_i_time:  # swap if reporting peak earlier (safety)
        peak_i_time, peak_c_time = peak_c_time, peak_i_time

    # shaded rect (reporting delay interval)
    fig.add_vrect(x0=peak_i_time, x1=peak_c_time, fillcolor="gray", opacity=0.20,
                layer="below", line_width=0, row=2, col=2)

    # vertical lines & annotations (kept inside the subplot)
    fig.add_vline(x=peak_i_time, line=dict(color="blue", dash="dash"),
                annotation_text=f"Peak I (Day {peak_i_time:.1f})", annotation_position="top left",
                row=2, col=2)
    fig.add_vline(x=peak_c_time, line=dict(color="orange", dash="dash"),
                annotation_text=f"Peak C (Day {peak_c_time:.1f})", annotation_position="top right",
                row=2, col=2)

    fig.update_yaxes(title_text="Cases", row=2, col=2)
    fig.update_xaxes(title_text="Time (days)", row=2, col=2)

    # Aframed boxes
    for r in (1,2):
        for c in (1,2):
            add_subplot_box(fig, r, c)

    add_local_legend(fig, 1, 1, [
        ("Susceptible", "royalblue", "solid"),
        ("Infected", "orange", "solid"),
        ("Reported", "red", "solid"),
        ("Recovered", "green", "solid"),
    ], x_right=0.97, y_top=0.95, font_size=13)

    add_local_legend(fig, 1, 2, [
        ("Risk Perception (P)", "magenta", "solid"),
        ("Fatigue (F)", "gray", "solid"),
        ("Compliance", "cyan", "dash"),
    ], x_right=0.97, y_top=0.95, font_size=13)

    add_local_legend(fig, 2, 1, [
        ("β_eff", "red", "solid"),
        ("β₀ (base)", "red", "dash"),
        ("Infections (I)", "blue", "solid"),
        ("Reported (C)", "orange", "solid"),
    ], x_right=0.97, y_top=0.95, font_size=13)

    add_local_legend(fig, 2, 2, [
        ("Infections (I)", "blue", "solid"),
        ("Reported (C)", "orange", "solid"),
        ("Reporting delay", "gray", "solid"),
    ], x_right=0.97, y_top=0.95, font_size=13)

    # Grid fix
    for r in (1,2):
        for c in (1,2):
            fig.update_xaxes(row=r, col=c,
                            tickmode="array",
                            tickvals=tickvals,
                            ticktext=[str(v) for v in tickvals],
                            showgrid=True, gridcolor="lightgray",
                            zeroline=False,
                            tickfont=dict(size=11))

    fig.update_yaxes(row=2, col=1, secondary_y=True, showgrid=True, gridcolor="lightgray", zeroline=False)
    fig.update_yaxes(row=2, col=1, secondary_y=True, showgrid=False, zeroline=False)

    # cosmetics
    fig.update_layout(
        autosize=True,
        height=920,
        margin=dict(l=50, r=20, t=80, b=50),
        plot_bgcolor="white",
        title=dict(text="SIRCP Model Dashboard", x=0.5, xanchor="center",
                font=dict(size=20, family="Times New Roman")),
        font=dict(family="Times New Roman", size=13)
    )

    # disable Plotly's global legend (we use fake in-plot legends)
    fig.update_yaxes(showgrid=True, gridcolor="lightgray", zeroline=False)
    fig.update_layout(showlegend=False)

    # matplotlib wins
    return fig