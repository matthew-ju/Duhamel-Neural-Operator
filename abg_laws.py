#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""abg_laws.py -- the closed-form unit maps for a (alpha, beta, gamma) material.

This is the generalisation of alpha_laws.py from one coefficient to three.  It
answers exactly one question: given a material, what are its own units of
field, time and length, and what is left over once you have used them?

    P_s     = positive root of  |a_sat| = beta P^2 + gamma P^4   field unit
    lam     = 1 / (BigGamma |a_rate|)                            time unit  (s)
    mu      = sqrt(g11 / |a_sat|)                                length unit (m)

TWO EFFECTIVE ALPHAS, NOT ONE
-----------------------------
The depolarisation field renormalises alpha, but it does NOT renormalise it by
the same amount for the two things alpha controls, and conflating them was a
real error in this campaign's first draft:

    |a_rate| = K_r (|alpha| - 1.367e9)   sets the LINEAR GROWTH RATE, hence lam
    |a_sat|  = 1.377 (|alpha| - 1.235e9) sets the SATURATION AMPLITUDE, hence P_s

The two critical points differ by 10% and the difference is physical rather
than fitting noise: the growth rate is a property of the small-amplitude
uniform mode, where the depolarisation field is strongest, while the saturation
amplitude is a property of the settled DOMAIN state, where most of the
depolarisation is screened by the domain pattern itself.  Three independent
determinations agree on each:

    rate:  1.367e9 from the alpha43 42-point rate fit (R^2 = 0.99995)
    sat:   1.25e9  from the beta/gamma campaign's two saturation-law fits,
           1.235e9 from this campaign's 14-corner pilot (below)

Forcing a single a_eff through both -- which the first version of this file did,
pinning the saturation law to the RATE critical point -- costs 11% in a_eff and
shows up as a clean -0.92 correlation between the residual and alpha.  Two
laws, each fitted to what it governs, costs 4.5%.

and the dimensionless leftovers, the things a change of units CANNOT remove:

    theta = dt_frame / lam        how big one observed frame is, in time units
    Pi_h  = dx / mu               how well the grid resolves a domain wall
    r     = gamma P_s^4/|a_eff|   how much of the restoring force is 6th order

WHAT IS AND IS NOT FREE
-----------------------
P_s and lam are GROUP ELEMENTS: rescaling the field by P_s and time by lam is
an exact symmetry of the continuum equation, so moving along them costs
nothing and can be done in closed form.  theta, Pi_h and r are INVARIANTS: no
rescaling removes them, so they are what a conditioned model must actually
learn.  Splitting the three coefficients this way is the whole point --

    alpha  moves lam, P_s AND Pi_h        (partly free, partly not)
    beta   moves P_s AND r                (the P_s part free, the r part not)
    gamma  moves r ONLY                   (nothing free: no group action)

so gamma is the coefficient with no closed-form retarget at all, and r is the
coordinate along which a surrogate has to genuinely generalise.

r RATHER THAN PI6
-----------------
The beta/gamma campaign's invariant was PI6 = gamma |a_eff| / beta^2.  It is
the same information as r on beta > 0 (PI6 = r/(1-r)^2), but it diverges as
beta -> 0 and is meaningless for beta < 0, where the well is first-order and
stabilised only by the 6th-order term.  r stays finite, ordered and bounded
across the whole family, so it is what the conditioner is fed.

EVERY CONDITIONER IS EXACTLY ZERO AT THE REFERENCE MATERIAL.  That is not
cosmetic: the alpha campaign established that an adapter with nothing pinning
the reference walks off the specialist optimum in its first epoch and never
recovers.  Because the conditioning vector is a vector of log-ratios and
differences against the reference, `cond_vec(REF) == 0` bit-exactly, so any
bias-free head built on it contributes exactly zero there for any weights.
"""
import math
import os

import numpy as np
import torch

# ---- solver constants, must match the FerroX inputs template ----------------
BIG_GAMMA = 100.0
G11 = 1.0e-9
DX = 0.5e-9
P_NORM = 0.25          # the P_S the rest of the repo normalises by
DT_FRAME_LEGACY = 2.0e-12   # plot_int=10 * dt=2e-13, the alpha43/betagamma stacks

# ---- measured laws ----------------------------------------------------------
# RATE branch: alpha_c from the alpha43 42-point rate fit; K_r is a pure scale
# and cancels out of every ratio the conditioner uses, so it is set to make
# a_rate(alpha_ref) match the value the propagator bank was fitted at.
ALPHA_C_RATE = 1.367e9
A_RATE_K = 1.514
# SATURATION branch: fitted on the 14-corner pilot of this campaign by
# inverting  |a_sat| = beta P_meas^2 + gamma P_meas^4  (exact, no model), over
# |alpha| 1.60-4.00e9, P_s 0.10-0.39 and r 0.002-1.40 including beta < 0.
# 4.5% mean absolute error in a_sat = ~2.3% in P_s.  abg_fit_aeff.py refits
# both numbers from the full sweep; the values here are the pilot's.
# Refitted on 208 sweep configs (was 14 pilot corners), as
#     |a_sat| = K (|alpha| - alpha_c (1 + e r))
#
# The r term is not a nuisance parameter. The residual of the r-INDEPENDENT law
# correlates -0.67 with r over the full design; one extra constant removes that
# drift (-0.06) and cuts P_s error from 1.40% to 0.91% mean and 8.3% to 3.6%
# max. Physically it says the depolarisation renormalisation of alpha depends
# on the SHAPE OF THE WELL, not only on alpha -- which every campaign in this
# project had assumed away.
#
# r enters through the CRITICAL POINT, not the scale. Four forms were fitted;
# scaling the coefficient (K(1+cr)) gives 1.03%/6.3%, shifting the critical
# point gives 0.91%/3.6%, and adding a power, an r^2 term, an offset, or a
# P_s-dependence on top of the winner all fit their new coefficient to ~0.
#
# Residual: a +0.44 correlation with alpha survives in EVERY form tried, worth
# 0.97% between the shallow and deep halves of the design (-0.51% / +0.46%).
# That is smaller than the ambiguity in how P_s is measured (a 99.5% quantile
# of |P| over the settled tail), so it is left unmodelled and recorded here
# rather than absorbed by a parameter that would not mean anything.
ALPHA_C_SAT = 1.3111e9
A_SAT_K = 1.3890
E_SAT_R = -0.077492
# TEXTBOOK-UNITS SWITCH (2026-09-29 ablation): ABG_UNITS=textbook replaces both effective coefficients by the
# raw |alpha| -- what dimensional analysis of the Landau coefficients alone would give (no depolarisation
# shift, no critical point, no r term). P_s then comes from the bulk balance |alpha| = beta P^2 + gamma P^4
# and lam = 1/(Gamma |alpha|); every derived unit and dimensionless number (mu, r, theta, Pi_h) follows.
UNITS_MODE = os.environ.get("ABG_UNITS", "effective")
if UNITS_MODE == "textbook":
    ALPHA_C_RATE, A_RATE_K = 0.0, 1.0
    ALPHA_C_SAT, A_SAT_K, E_SAT_R = 0.0, 1.0, 0.0
elif UNITS_MODE != "effective":
    raise ValueError("ABG_UNITS must be 'effective' or 'textbook', got %r" % UNITS_MODE)
# kept for backward compatibility with abg_grid.py's design-time labels
ALPHA_C, A_EFF_K = ALPHA_C_SAT, A_SAT_K

# ---- reference material (the config every campaign in this project uses) ----
ALPHA_REF, BETA_REF, GAMMA_REF = -2.0e9, 3.0e10, 3.5e11
BETA_SCALE = 1.0e10     # asinh scale for the signed log-modulus of beta
GAMMA_EPS = 1.0e9       # softens log(gamma) at the pure-quartic edge, gamma = 0


def _as_t(x, like=None):
    if torch.is_tensor(x):
        return x
    t = torch.as_tensor(x, dtype=torch.float64)
    return t if like is None else t.to(like.device)


def a_sat(alpha, beta=None, gamma=None, iters=6):
    """Effective quadratic coefficient governing the SATURATION amplitude.

        |a_sat| = K (|alpha| - alpha_c (1 + e r))

    The r term makes this implicit -- r depends on P_s and P_s depends on
    a_sat -- so it is solved by fixed-point iteration from the r-independent
    value.  e = -0.077 is small, so it converges in a couple of passes; six are
    taken because they cost nothing and this runs once per material.

    beta/gamma omitted falls back to the r-independent branch, which is what
    the pure-alpha callers (and any legacy caller) want.
    """
    a_abs = torch.abs(_as_t(alpha))
    ae = A_SAT_K * torch.clamp(a_abs - ALPHA_C_SAT, min=1e6)
    if beta is None or gamma is None:
        return ae
    b, g = _as_t(beta), _as_t(gamma)
    for _ in range(iters):
        # P^2 on the stable branch, then r = gamma P^4 / a_sat.  Note r is a
        # function of (beta, gamma, P) ALONE -- the definition of the
        # conditioning coordinate is unchanged by this refit; only the
        # PREDICTION of P_s moves.
        p2 = 2.0 * ae / torch.clamp(b + torch.sqrt(b * b + 4.0 * g * ae),
                                    min=1e-30)
        r = g * p2 * p2 / torch.clamp(ae, min=1e-30)
        ae = A_SAT_K * torch.clamp(
            a_abs - ALPHA_C_SAT * (1.0 + E_SAT_R * r), min=1e6)
    return ae


def a_rate(alpha):
    """Effective quadratic coefficient governing the LINEAR GROWTH RATE."""
    return A_RATE_K * torch.clamp(torch.abs(_as_t(alpha)) - ALPHA_C_RATE, min=1e6)


def a_eff(alpha):
    """Deprecated alias for a_sat, kept so abg_grid.py's design labels do not
    silently change meaning.  New code should say which one it means."""
    return a_sat(alpha)


def p_sat(alpha, beta, gamma):
    """Local saturation polarisation, C/m^2.

    Positive root of |a_eff| = beta P^2 + gamma P^4.  Written with the
    numerically stable branch so it survives gamma -> 0 (pure quartic well) and
    beta < 0 (first-order well) without a special case:

        P^2 = 2|a_eff| / ( beta + sqrt(beta^2 + 4 gamma |a_eff|) )

    which is the same root as the usual quadratic formula but never subtracts
    two nearly equal numbers.  At gamma = 0 it collapses to |a_eff|/beta
    exactly; at beta < 0 the denominator stays positive because
    sqrt(beta^2 + 4 gamma |a_eff|) > |beta| whenever gamma > 0.
    """
    ae, b, g = a_sat(alpha, beta, gamma), _as_t(beta), _as_t(gamma)
    disc = torch.sqrt(b * b + 4.0 * g * ae)
    return torch.sqrt(2.0 * ae / torch.clamp(b + disc, min=1e-30))


def lam(alpha):
    """Time unit, seconds: the e-folding time of the linear instability.
    Uses the RATE branch -- this is the one quantity a_sat must not be used
    for, and the two differ by 10% in their critical point."""
    return 1.0 / (BIG_GAMMA * a_rate(alpha))


def mu(alpha, beta=None, gamma=None):
    """Length unit, metres: the domain-wall width.  Set by the BULK curvature
    of the settled state, hence the saturation branch.  Takes beta/gamma when
    the caller has them, so the wall width sees the same well the saturation
    does; Pi_z = L_FE/mu = 10 * Pi_h remains exact either way, since that is a
    geometric identity about the grid."""
    return torch.sqrt(_as_t(G11) / a_sat(alpha, beta, gamma))


def r_share(alpha, beta, gamma):
    """Fraction of the restoring force at saturation carried by the 6th-order
    term.  0 = pure quartic, 1 = beta vanishes, >1 = beta < 0 (first order)."""
    ps = p_sat(alpha, beta, gamma)
    return _as_t(gamma) * ps ** 4 / a_sat(alpha, beta, gamma)


def pi6(alpha, beta, gamma):
    """The beta/gamma campaign's invariant. NaN where beta <= 0, on purpose."""
    b = _as_t(beta)
    out = _as_t(gamma) * a_sat(alpha, beta, gamma) / (b * b)
    return torch.where(b > 0, out, torch.full_like(out, float("nan")))


def theta(alpha, dt_frame=DT_FRAME_LEGACY):
    """Dimensionless size of one observed frame, dt_frame / lam."""
    return _as_t(dt_frame) * BIG_GAMMA * a_rate(alpha)


def pi_h(alpha):
    """Grid resolution of a domain wall, dx / mu.  Large = under-resolved."""
    return _as_t(DX) / mu(alpha)


# ---------------------------------------------------------------------------
# The conditioning vectors
# ---------------------------------------------------------------------------
_REF = dict(alpha=ALPHA_REF, beta=BETA_REF, gamma=GAMMA_REF)
COND_NAMES_PI = ("log_theta", "log_Ps", "r", "log_Pi_h")
COND_NAMES_RAW = ("log_alpha", "beta_hat", "log_gamma")


def cond_vec(alpha, beta, gamma, dt_frame=DT_FRAME_LEGACY, mode="pi"):
    """(N, d) float32 conditioning vector, exactly 0 at the reference material.

    mode="pi"   the dimensionless invariants -- our contribution.
    mode="raw"  log of the coefficients themselves -- the literature baseline
                (FiLM/CoDA-style conditioning on the PDE coefficients).  beta
                changes sign across this design, so a plain log is impossible;
                `beta_hat` is a signed log-modulus, asinh(beta/BETA_SCALE)
                recentred on the reference: smooth THROUGH zero and monotone in
                beta.  Choosing that over log|beta| matters -- log|beta| has a
                non-removable singularity exactly in the middle of this design,
                at beta = 0, where the well is purely 6th order.  gamma gets a
                softening offset for the same reason at its own edge, gamma = 0.
                Both offsets are chosen so the reference maps to exactly 0.
    """
    alpha, beta, gamma = (_as_t(alpha).reshape(-1), _as_t(beta).reshape(-1),
                          _as_t(gamma).reshape(-1))
    dt = _as_t(dt_frame).reshape(-1).expand_as(alpha)
    if mode == "raw":
        cols = [torch.log(torch.abs(alpha) / abs(ALPHA_REF)),
                torch.asinh(beta / BETA_SCALE) - math.asinh(BETA_REF / BETA_SCALE),
                torch.log((gamma + GAMMA_EPS) / (GAMMA_REF + GAMMA_EPS))]
    elif mode == "pi":
        th_ref = theta(ALPHA_REF, DT_FRAME_LEGACY)
        ps_ref = p_sat(**_REF)
        r_ref = r_share(**_REF)
        pih_ref = pi_h(ALPHA_REF)
        cols = [torch.log(theta(alpha, dt) / th_ref),
                torch.log(p_sat(alpha, beta, gamma) / ps_ref),
                r_share(alpha, beta, gamma) - r_ref,
                torch.log(pi_h(alpha) / pih_ref)]
    elif mode == "none":
        cols = [torch.zeros_like(alpha)]
    else:
        raise ValueError(mode)
    return torch.stack(cols, dim=-1).to(torch.float32)


def cond_dim(mode="pi"):
    return {"pi": 4, "raw": 3, "none": 1}[mode]


# ---------------------------------------------------------------------------
# State-level retargets (zero-parameter, exact for the group directions)
# ---------------------------------------------------------------------------
def amp_ratio(alpha, beta, gamma):
    """P_s(config) / P_s(reference).

    A field in NORMALISED units (P / 0.25) is put into REDUCED units -- the
    units in which the nonlinearity is coefficient-free -- by dividing by this.
    Exactly 1.0 at the reference, so the reduction is the identity there.
    """
    return (p_sat(alpha, beta, gamma) / p_sat(**_REF)).to(torch.float32)


def tau_ratio(alpha, dt_frame=DT_FRAME_LEGACY):
    """theta(config) / theta(reference): how many reference frames one frame
    of this config is worth."""
    return (theta(alpha, dt_frame) / theta(ALPHA_REF, DT_FRAME_LEGACY)
            ).to(torch.float32)


def summary(alpha, beta, gamma, dt_frame=DT_FRAME_LEGACY):
    """Human-readable dict for one material (floats, not tensors)."""
    f = lambda t: float(np.asarray(t))
    return dict(alpha=float(alpha), beta=float(beta), gamma=float(gamma),
                a_sat=f(a_sat(alpha, beta, gamma)), a_rate=f(a_rate(alpha)),
                P_s=f(p_sat(alpha, beta, gamma)),
                lam=f(lam(alpha)), mu=f(mu(alpha)),
                r=f(r_share(alpha, beta, gamma)),
                pi6=f(pi6(alpha, beta, gamma)),
                theta=f(theta(alpha, dt_frame)), pi_h=f(pi_h(alpha)),
                amp_ratio=f(amp_ratio(alpha, beta, gamma)),
                tau_ratio=f(tau_ratio(alpha, dt_frame)))


if __name__ == "__main__":
    import json
    import sys
    for name, cfg in [("reference", (ALPHA_REF, BETA_REF, GAMMA_REF)),
                      ("shallow", (-1.6e9, 5.6e8, 1.5e11)),
                      ("deep", (-4.0e9, 4.1e9, 1.8e11)),
                      ("first-order", (-2.3e9, -4.5e9, 1.3e11)),
                      ("pure quartic", (-3.9e9, 2.7e10, 3.1e8))]:
        print("%-14s %s" % (name, "  ".join(
            "%s=%.4g" % (k, v) for k, v in summary(*cfg).items())))
    print("\ncond_vec(reference, pi ) =", cond_vec(*_REF.values(), mode="pi"))
    print("cond_vec(reference, raw) =", cond_vec(*_REF.values(), mode="raw"))
    assert float(cond_vec(*_REF.values(), mode="pi").abs().max()) == 0.0
    assert float(cond_vec(*_REF.values(), mode="raw").abs().max()) < 1e-6
    print("reference is pinned to exactly zero in both modes", file=sys.stderr)
