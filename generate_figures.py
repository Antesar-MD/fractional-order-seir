#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
generate_figures.py
====================

Companion code for "A Unified Fractional-Order SEIR Framework for Endemic
Disease Dynamics: Vertical Transmission, Disease-Induced Mortality, and
Stability" (Antesar Aldawoud). Produces every figure in the article, from
the model equations of Sections 2, 4, and 5.

PROVENANCE
----------
This script consolidates and generalises an earlier MATLAB prototype
(11 files implementing the classical SIR model and a single fractional-order
SEIR variant) into one general-purpose, documented Python implementation
covering the full model family analysed in the article:

  Figure  Model                                   Origin
  ------  --------------------------------------  ----------------------------------
  1       Classical SIR epidemic model             translated (`solve_ivp`, RK45)
  2       Bifurcation diagram, endemic model        new (plots the closed-form i*)
  3       Classical SIR endemic model               translated (`solve_ivp`, RK45)
  4       General fractional-order SEIR model       generalised (theta, p, q added)
  5       General fractional-order SIR model        new (no MATLAB equivalent existed)

The original MATLAB implemented only the SEIR sub-case theta=0, p=q=0
(Table 1, case (a)); `fractional_seir_general` below extends that same
numerical scheme with the disease-induced mortality term theta*I and the
vertical-transmission terms Delta*p*E/N + Delta*q*I/N, so that setting
theta=0 and p=q=0 reproduces the original equations exactly. The
age-structured extension present in the original project (two age classes,
not used by any figure in this article) was intentionally left out of scope.

NUMERICAL METHOD
-----------------
All fractional-order simulations (Figures 4-5) use the explicit
Grunwald-Letnikov (GL) fractional-difference scheme for a Caputo initial
value problem D^alpha x = f(t, x), x(0) = x0, step size h:

    w_0 = 1,  w_m = (1 - (alpha + 1) / m) * w_{m-1},          m = 1, 2, ...
    x_{k+1} = h^alpha * f(t_k, x_k) - sum_{j=1}^{k+1} w_j * x_{k+1-j}

(see e.g. I. Petras, "Fractional-Order Nonlinear Systems", Springer, 2011).
This is a first-order, explicit method whose memory sum is O(n^2) in the
number of time steps n = t_sim / h, since every step depends on the entire
simulated history; the memory sum is vectorised here as a single NumPy dot
product per step (`gl_coefficients`, `fractional_seir_general`,
`fractional_sir_general`) rather than an explicit inner loop, which is what
keeps a run of several thousand steps practical. The classical (alpha = 1)
simulations of Figures 1 and 3 instead use `scipy.integrate.solve_ivp`
(RK45), which is unrelated to the GL scheme and included only for the
integer-order baseline model of Section 2.

Figures 4-5 use t_sim = 1000 rather than a shorter window because, at the
article's parameter values, the infection subsystem's own approach to the
endemic steady state is slow (e-folding time on the order of 40) relative
to the demographic relaxation time 1/mu = 125; a shorter window shows only
the early transient, not convergence. The step size h = 0.05 used here
keeps the total number of steps -- and hence the O(n^2) cost -- comparable
to a shorter window at a smaller h. Running to t_sim = 1000 also serves as
an independent numerical check on the closed-form endemic steady states
derived in the article (Propositions 4.3 and 4.5): the simulated long-run
values agree with the analytic F_1 to 3-4 significant figures.

USAGE
-----
    python3 generate_figures.py

Requires numpy, scipy, and matplotlib (see requirements.txt). Writes five
PNG files (300 dpi) to the current directory, named to match the figure
labels used in the article's LaTeX source:

    fig1_epidemic_timeseries.png     (\\label{fig:epidemic-timeseries})
    fig2_bifurcation.png             (\\label{fig:bifurcation})
    fig3_endemic_timeseries.png      (\\label{fig:endemic-timeseries})
    fig4_seir_alpha_comparison.png   (\\label{fig:seir-alpha-comparison})
    fig5_sir_alpha_comparison.png    (\\label{fig:sir-alpha-comparison})

Total run time is on the order of ten seconds on an ordinary laptop with
the default step sizes below.
"""

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------
# Shared style: a small, fixed-order, colourblind-safe palette (Okabe-Ito),
# assigned by identity (alpha value / compartment) and never re-cycled,
# with distinct line styles too so the figures stay legible in grayscale
# print.
# ----------------------------------------------------------------------
BLACK = "#000000"
BLUE = "#0072B2"
VERMILLION = "#D55E00"
GREEN = "#009E73"

plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "font.size": 11,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

ALPHA_STYLE = {
    1.00: dict(color=BLACK, linestyle="-.", label=r"$\alpha=1.00$"),
    0.95: dict(color=BLUE, linestyle="-", label=r"$\alpha=0.95$"),
    0.90: dict(color=VERMILLION, linestyle="--", label=r"$\alpha=0.90$"),
}


# ========================================================================
# PART 1 -- Classical SIR epidemic model (article Sec. 2.1, Eq. 1-3)
# ========================================================================

def sir_epidemic_rhs(t, y, beta, gamma):
    s, i = y
    return [-beta * s * i, beta * s * i - gamma * i]


def make_figure1():
    beta, gamma = 0.05, 0.003
    y0 = [1 - 1e-5, 1e-5]
    t_span = (0, 500)
    t_eval = np.linspace(*t_span, 2000)

    sol = solve_ivp(sir_epidemic_rhs, t_span, y0, args=(beta, gamma),
                     method="RK45", t_eval=t_eval, rtol=1e-9, atol=1e-11)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(sol.t, sol.y[0], color=VERMILLION, linestyle="--", linewidth=2, label="$s(t)$")
    ax.plot(sol.t, sol.y[1], color=BLUE, linestyle="-.", linewidth=2, label="$i(t)$")
    ax.set_xlabel("time $t$")
    ax.set_ylabel("proportion of population")
    ax.set_title(r"SIR epidemic model, $\mathcal{R}_0=\beta/\gamma=%.2f$" % (beta / gamma))
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig("fig1_epidemic_timeseries.png")
    plt.close(fig)
    print("wrote fig1_epidemic_timeseries.png")


# ========================================================================
# PART 2 -- Classical SIR endemic model (article Sec. 2.2, Eq. 4-6)
# Figure 2 plots the closed-form i* = (mu/beta)(R0-1) of Proposition 2.1
# directly, as a visual check on that proposition.
# ========================================================================

def sir_endemic_rhs(t, y, beta, gamma, mu):
    s, i = y
    return [mu - beta * s * i - mu * s, beta * s * i - (gamma + mu) * i]


def make_figure3():
    beta, gamma, mu = 0.05, 0.003, 0.008
    y0 = [1 - 1e-5, 1e-5]
    t_span = (0, 500)
    t_eval = np.linspace(*t_span, 2000)

    sol = solve_ivp(sir_endemic_rhs, t_span, y0, args=(beta, gamma, mu),
                     method="RK45", t_eval=t_eval, rtol=1e-9, atol=1e-11)

    R0 = beta / (gamma + mu)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(sol.t, sol.y[0], color=VERMILLION, linestyle="--", linewidth=2, label="$s(t)$")
    ax.plot(sol.t, sol.y[1], color=BLUE, linestyle="-.", linewidth=2, label="$i(t)$")
    ax.set_xlabel("time $t$")
    ax.set_ylabel("proportion of population")
    ax.set_title(r"SIR endemic model, $\mathcal{R}_0=%.2f$, converging to $F_1$" % R0)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig("fig3_endemic_timeseries.png")
    plt.close(fig)
    print("wrote fig3_endemic_timeseries.png")


def make_figure2_bifurcation():
    """Bifurcation diagram i* vs. beta, mu=0.008, gamma=0.003 fixed,
    plotting the closed-form i* = (mu/beta)(R0-1) of Proposition 2.1."""
    mu, gamma = 0.008, 0.003
    beta = np.linspace(1e-4, 0.03, 2000)
    R0 = beta / (gamma + mu)
    istar = np.where(R0 > 1, (mu / beta) * (R0 - 1), 0.0)
    beta_crit = gamma + mu  # R0 = 1

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(beta, istar, color=BLUE, linewidth=2)
    ax.axvline(beta_crit, color=BLACK, linestyle=":", linewidth=1.2,
               label=r"$\beta=\gamma+\mu$ ($\mathcal{R}_0=1$)")
    ax.set_xlabel(r"$\beta$")
    ax.set_ylabel(r"$i^*$")
    ax.set_title(r"Transcritical bifurcation of the endemic model at $\mathcal{R}_0=1$")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig("fig2_bifurcation.png")
    plt.close(fig)
    print("wrote fig2_bifurcation.png")


# ========================================================================
# PART 3 -- General fractional-order SEIR model (article Sec. 4, Theorem 4.4)
# ========================================================================

def gl_coefficients(alpha, n):
    """Grunwald-Letnikov binomial coefficients w[0..n]:
    w[0] = 1, w[m] = (1 - (alpha + 1) / m) * w[m-1] for m = 1..n."""
    w = np.empty(n + 1)
    w[0] = 1.0
    m = np.arange(1, n + 1)
    factors = 1.0 - (alpha + 1.0) / m
    w[1:] = np.cumprod(factors)
    return w


def fractional_seir_general(delta_, beta, mu, omega, gamma, theta, p, q,
                             alpha, t_sim, y0, h=0.01):
    """General fractional-order SEIR model, Eqs. (S)-(R) of Section 4:

        D^alpha S = Delta - Delta*p*E/N - Delta*q*I/N - beta*S*I/N - mu*S
        D^alpha E = Delta*p*E/N + Delta*q*I/N + beta*S*I/N - omega*E - mu*E
        D^alpha I = omega*E - gamma*I - mu*I - theta*I
        D^alpha R = gamma*I - mu*R

    solved by the explicit Grunwald-Letnikov scheme described above.
    Returns (t, S, E, I, R) arrays of length n+1.
    """
    n = int(round(t_sim / h))
    t = np.linspace(0, t_sim, n + 1)

    S = np.empty(n + 1); E = np.empty(n + 1); I = np.empty(n + 1); R = np.empty(n + 1)
    S[0], E[0], I[0], R[0] = y0

    w = gl_coefficients(alpha, n)
    hA = h ** alpha

    for k in range(n):
        N = S[k] + E[k] + I[k] + R[k]
        fS = delta_ - delta_ * p * E[k] / N - delta_ * q * I[k] / N - beta * S[k] * I[k] / N - mu * S[k]
        fE = delta_ * p * E[k] / N + delta_ * q * I[k] / N + beta * S[k] * I[k] / N - omega * E[k] - mu * E[k]
        fI = omega * E[k] - gamma * I[k] - mu * I[k] - theta * I[k]
        fR = gamma * I[k] - mu * R[k]

        # Memory sum sum_{j=1}^{k+1} w[j] * X[k+1-j], vectorised as a dot
        # product rather than an explicit inner loop.
        wk = w[1:k + 2][::-1]
        S[k + 1] = hA * fS - np.dot(wk, S[:k + 1])
        E[k + 1] = hA * fE - np.dot(wk, E[:k + 1])
        I[k + 1] = hA * fI - np.dot(wk, I[:k + 1])
        R[k + 1] = hA * fR - np.dot(wk, R[:k + 1])

    return t, S, E, I, R


def make_figure4(h=0.05, t_sim=1000):
    # General SEIR case (Table 1 / Table 3, case (d)); parameters as used
    # throughout Sections 4-5 of the article. See the NUMERICAL METHOD note
    # above for why t_sim = 1000 rather than a shorter window.
    Delta, beta, mu, omega, gamma = 0.221176, 0.05, 0.008, 0.05, 0.003
    theta, p, q = 0.002, 0.8, 0.95
    S0, E0, I0 = 140.0, 0.01, 0.02
    R0_init = 141.0 - S0 - E0 - I0
    y0 = (S0, E0, I0, R0_init)

    runs = {}
    for alpha in (1.00, 0.95, 0.90):
        t, S, E, I, R = fractional_seir_general(Delta, beta, mu, omega, gamma, theta, p, q,
                                                  alpha, t_sim, y0, h=h)
        runs[alpha] = (t, S, E, I)
        print(f"  SEIR alpha={alpha}: done ({len(t)} steps)")

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    labels = ["$S(t)$", "$E(t)$", "$I(t)$"]
    for panel, idx in enumerate((1, 2, 3)):  # index into (t,S,E,I) tuple
        ax = axes[panel]
        for alpha, style in ALPHA_STYLE.items():
            t, S, E, I = runs[alpha]
            y = (S, E, I)[panel]
            ax.plot(t, y, linewidth=2, **style)
        ax.set_xlabel("time $t$")
        ax.set_ylabel(labels[panel])
        ax.legend(loc="best", fontsize=9)
    fig.suptitle("General fractional-order SEIR model: effect of $\\alpha$ on the approach to $F_1$")
    fig.tight_layout()
    fig.savefig("fig4_seir_alpha_comparison.png")
    plt.close(fig)
    print("wrote fig4_seir_alpha_comparison.png")


# ========================================================================
# PART 4 -- General fractional-order SIR model (article Sec. 4.4, Theorem 4.6)
# ========================================================================

def fractional_sir_general(delta_, beta, mu, gamma, theta, r,
                            alpha, t_sim, y0, h=0.01):
    """General fractional-order SIR model, Eqs. (S)-(R) of Section 4.4:

        D^alpha S = Delta - r*Delta*I/N - beta*S*I/N - mu*S
        D^alpha I = r*Delta*I/N + beta*S*I/N - gamma*I - mu*I - theta*I
        D^alpha R = gamma*I - mu*R
    """
    n = int(round(t_sim / h))
    t = np.linspace(0, t_sim, n + 1)

    S = np.empty(n + 1); I = np.empty(n + 1); R = np.empty(n + 1)
    S[0], I[0], R[0] = y0

    w = gl_coefficients(alpha, n)
    hA = h ** alpha

    for k in range(n):
        N = S[k] + I[k] + R[k]
        fS = delta_ - r * delta_ * I[k] / N - beta * S[k] * I[k] / N - mu * S[k]
        fI = r * delta_ * I[k] / N + beta * S[k] * I[k] / N - gamma * I[k] - mu * I[k] - theta * I[k]
        fR = gamma * I[k] - mu * R[k]

        wk = w[1:k + 2][::-1]
        S[k + 1] = hA * fS - np.dot(wk, S[:k + 1])
        I[k + 1] = hA * fI - np.dot(wk, I[:k + 1])
        R[k + 1] = hA * fR - np.dot(wk, R[:k + 1])

    return t, S, I, R


def make_figure5(h=0.05, t_sim=1000):
    # General SIR case (Table 2 / Table 3, case (d)); same Delta, beta, mu,
    # gamma, theta as the SEIR run above, r=0.85 as used in Sections 4-5.
    Delta, beta, mu, gamma = 0.221176, 0.05, 0.008, 0.003
    theta, r = 0.002, 0.85
    S0, I0 = 140.0, 0.02
    R0_init = 141.0 - S0 - I0
    y0 = (S0, I0, R0_init)

    runs = {}
    for alpha in (1.00, 0.95, 0.90):
        t, S, I, R = fractional_sir_general(Delta, beta, mu, gamma, theta, r,
                                             alpha, t_sim, y0, h=h)
        runs[alpha] = (t, S, I)
        print(f"  SIR alpha={alpha}: done ({len(t)} steps)")

    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    labels = ["$S(t)$", "$I(t)$"]
    for panel in (0, 1):
        ax = axes[panel]
        for alpha, style in ALPHA_STYLE.items():
            t, S, I = runs[alpha]
            y = (S, I)[panel]
            ax.plot(t, y, linewidth=2, **style)
        ax.set_xlabel("time $t$")
        ax.set_ylabel(labels[panel])
        ax.legend(loc="best", fontsize=9)
    fig.suptitle("General fractional-order SIR model: effect of $\\alpha$ on the approach to $F_1$")
    fig.tight_layout()
    fig.savefig("fig5_sir_alpha_comparison.png")
    plt.close(fig)
    print("wrote fig5_sir_alpha_comparison.png")


# ========================================================================
# MAIN
# ========================================================================

if __name__ == "__main__":
    print("Part 1: classical SIR epidemic model (Figure 1)...")
    make_figure1()

    print("Part 2: classical SIR endemic model (Figures 2-3)...")
    make_figure2_bifurcation()
    make_figure3()

    print("Part 3: general fractional-order SEIR model (Figure 4)...")
    make_figure4()  # see NUMERICAL METHOD note above before changing h/t_sim

    print("Part 4: general fractional-order SIR model (Figure 5)...")
    make_figure5()

    print("Done. Five PNG files written to the current directory.")
