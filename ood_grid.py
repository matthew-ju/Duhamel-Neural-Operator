#!/usr/bin/env python3
"""ood_grid.py -- decoupled (alpha, beta, gamma) design for an OOD-transfer study.

THE BRIEF
---------
"Random sample alpha/beta/gamma at certain range so the energy landscape will be
different (doesn't need to be realistic)."  Stop solving for beta and gamma from
a target saturation; sample them INDEPENDENTLY of alpha, build subsets of
mathematically distinct double wells, train on one and transfer zero-shot to the
others.

This file does that.  Two things in the brief cannot be taken literally, and
both are recorded here because they change what the experiment can claim.

CORRECTION 1 -- alpha in [-1e7, -1e9] IS ENTIRELY SUBCRITICAL
-------------------------------------------------------------
The brief proposes widening alpha to -1e7 .. -1e9.  Every one of those
materials is DEAD: it never nucleates, and the dataset would be flat noise.

The film is a ferroelectric between dielectrics, so the depolarisation field
renormalises the quadratic coefficient.  The uniform mode is stabilised, and
a double well in the BARE energy is not a double well in the EFFECTIVE one.
The linear growth rate vanishes at

    |alpha_c| = 1.367e9    (alpha43: 42-point rate fit, R^2 = 0.99995,
                            plus three independent determinations agreeing
                            to +/- 0.004e9 -- see abg_laws.ALPHA_C_RATE)

and this project has the negative control on disk: at alpha = -1.40e9, only
0.03e9 PAST the critical point, incubation is already ~1000 frames, and at the
usual window it never nucleates at all (mean|Pz| = 1e-4 at the last frame, all
20 seeds -- structurally valid, physically empty).

beta and gamma CANNOT rescue a subcritical alpha.  Around P = 0 the beta P^3
and gamma P^5 terms are higher order in an amplitude that starts at the noise
floor, so they do not enter the linearised operator at all; alpha and the
depolarisation field set the growth rate by themselves.  This is exact, not
empirical.  |alpha| > |alpha_c| is therefore a hard floor for ANY (beta, gamma).

So the alpha axis here runs 1.50e9 .. 4.00e9, and the diversity the brief wants
is bought with beta and gamma, which are genuinely free.  (The one way to make
shallow alpha live is to weaken the depolarisation -- thicker film, different
dielectric stack -- which moves alpha_c itself.  That is a different and also
interesting axis; it is not this one.)

The ceiling is equally hard: the domain wall is mu = sqrt(g11/|a_eff|) wide, and
at |alpha| ~ 4.2e9 that falls below ONE grid cell, so the wall stops being
resolved and the "landscape" the model sees is a discretisation artefact.

CORRECTION 2 -- "DEEP AND WIDE vs SHALLOW AND NARROW" IS A CHANGE OF UNITS
--------------------------------------------------------------------------
This is the one that decides whether the paper's headline survives review.

The brief asks for subset A = shallow narrow wells, subset B = deep wide wells,
and a zero-shot transfer A -> B.  On this system that transfer is FREE, in
closed form, with no fitted constant -- so a model can ace it while having
learned nothing about energy minimisation.

At fixed alpha, the map

    (beta, gamma) -> (k beta, k^2 gamma)      gives      P_k(x,t) = P_1(x,t)/sqrt(k)

EXACTLY, for every x and t, including the nonlocal depolarisation term (it is
linear in P, so it carries the same 1/sqrt(k)).  Under it the well width scales
as P_s -> P_s/sqrt(k) and the barrier as depth -> depth/k, while the 6th-order
share r is INVARIANT.  So "shallow and narrow" and "deep and wide" at the same
r are the same landscape in different units, and this project has already
measured the transfer: rel L2 0.045-0.065 with zero trained parameters, twice,
by two unrelated constructions (ferrox-beta-gamma iso-PI6 ladder; abg
units_only baseline at 0.044).

A reviewer who knows this -- and for a Landau free energy it is one line of
algebra -- reads "we transferred from shallow wells to deep wells" as "we
changed units".  That is a worse position than the physically-realistic dataset
the brief is trying to replace.

WHAT IS ACTUALLY OUT OF DISTRIBUTION HERE
------------------------------------------
For the quartic-sextic family at fixed geometry there are exactly three axes,
and only two of them are hard:

  1. AMPLITUDE / DEPTH / WIDTH  (beta, gamma at fixed r)  -- a group action.
     Free.  Provably.  Keep it as a CONTROL, not as the headline.
  2. GEOMETRY  (alpha)  -- changes |a_eff|, hence the wall width mu, hence
     Pi_z = L_FE/mu, the number of wall widths the film is thick.  The
     depolarisation coupling across z is a DIFFERENT OPERATOR at each end and
     no rescaling reaches it.  Measured: rel L2 1.27 where the amplitude
     direction gives 0.044.  A factor of 30.
  3. SHAPE  (r = gamma P_s^4/|a_eff|, the 6th-order share) -- the single
     dimensionless invariant of the reduced dynamics.  r > 1 means beta < 0:
     a FIRST-ORDER well, stabilised only by the 6th-order term, where the
     energy is unbounded below without gamma and no quartic model can be right.

So the design below keeps the brief's structure -- train on one regime, test
zero-shot on distinct ones -- but makes each test subset differ from the
training subset along ONE named axis, plus one that differs along all three.
Each carries a PREDICTION.  A result that confirms the predictions is a claim
about which symmetries a surrogate must be told about; a result that violates
them is more interesting still.  Either beats "we made the wells different and
it worked".

    subset     alpha/1e9    P_s        r          differs from A by   predicted
    ---------  -----------  ---------  ---------  ------------------  ---------
    A_train    1.50-2.50    0.05-0.15  0.00-0.35  --                  --
    B_units    1.50-2.50    0.45-1.00  0.00-0.35  units only          EASY
    B_alpha    3.20-4.00    0.05-0.15  0.00-0.35  geometry (Pi_z)     HARD
    C_shape    1.50-2.50    0.05-0.15  1.15-2.00  shape (r, beta<0)   HARDEST
    D_all      3.20-4.00    0.45-1.00  1.15-2.00  all three           HARDEST

The gaps are real: alpha skips 2.50-3.20, P_s skips 0.15-0.45, r skips
0.35-1.15.  No test subset is adjacent to the training hull in the axis it is
testing, so none of these is an interpolation.

SAMPLING
--------
Raw (alpha, beta, gamma) are drawn INDEPENDENTLY -- log-uniform in |alpha|,
signed log-uniform in beta, log-uniform in gamma -- exactly as the brief asks.
Nothing is fitted or solved for.  The landscape descriptors are then COMPUTED
from each raw draw and the draw is assigned to whichever subset box it lands
in, or discarded.  Yield per box is reported rather than hidden, because a box
that only fills by construction would not be an honest "random sample".

Outputs
-------
ood_grid.txt   TAG ALPHA BETA GAMMA NSTEPS PLOT_INT DT SPLIT   (run_abg.sh format)
ood_grid.json  full manifest with every descriptor, per-subset summary, yields
"""
import argparse
import json
import math
import os

import numpy as np

import abg_laws
from abg_grid import MAX_NSTEPS, T_FRAMES, window_for

HERE = os.path.dirname(os.path.abspath(__file__))

DX = 5.0e-10          # grid cell, m
L_FE = 5.0e-9         # ferroelectric thickness, m
G11 = 1.0e-9

# ---- hard physical bounds (see CORRECTION 1) --------------------------------
ALPHA_FLOOR = 1.45e9  # |alpha_c| = 1.367e9; below this nothing nucleates
MU_CELLS_MIN = 1.0    # the domain wall must span at least one grid cell

# ---- raw sampling box (decoupled: none of these depends on any other) -------
ALPHA_LO, ALPHA_HI = 1.50e9, 4.00e9
BETA_ABS_LO, BETA_ABS_HI = 1.0e9, 1.0e12
GAMMA_LO, GAMMA_HI = 1.0e9, 1.0e14
P_BETA_NEG = 0.35     # fraction of draws with beta < 0

# ---- subset boxes: (alpha_lo, alpha_hi, ps_lo, ps_hi, r_lo, r_hi) -----------
SUBSETS = {
    "A_train": (1.50e9, 2.50e9, 0.05, 0.15, 0.00, 0.35),
    "B_units": (1.50e9, 2.50e9, 0.45, 1.00, 0.00, 0.35),
    "B_alpha": (3.20e9, 4.00e9, 0.05, 0.15, 0.00, 0.35),
    "C_shape": (1.50e9, 2.50e9, 0.05, 0.15, 1.15, 2.00),
    "D_all":   (3.20e9, 4.00e9, 0.45, 1.00, 1.15, 2.00),
}
ROLE = {"A_train": "train", "B_units": "test", "B_alpha": "test",
        "C_shape": "test", "D_all": "test"}


def descriptors(alpha_abs, beta, gamma):
    """Landscape descriptors of the EFFECTIVE well, as arrays.

    The effective well is the one the dynamics actually sees: the
    depolarisation field renormalises alpha to a_eff, and using the BARE alpha
    puts the well bottom far past the observed saturation (measured: at 1.31x
    P_sat the bare-alpha bulk energy is 22.9% LOWER than at P_sat, so an
    over-saturating model looks like it is going downhill).
    """
    a_eff = abg_laws.a_sat(-alpha_abs, beta, gamma).numpy()
    ps = abg_laws.p_sat(-alpha_abs, beta, gamma).numpy()
    r = gamma * ps ** 4 / a_eff
    # barrier height f(0) - f(P_s), using beta P_s^2 = (1-r)a_eff and
    # gamma P_s^4 = r a_eff:  depth = a_eff P_s^2 (3 + r)/12
    depth = a_eff * ps ** 2 * (3.0 + r) / 12.0
    mu = np.sqrt(G11 / a_eff)
    # curvature at the minimum: f''(P_s) = -a_eff + 3 beta P_s^2 + 5 gamma P_s^4
    #                                    = a_eff (2 + 2r)
    curv = a_eff * (2.0 + 2.0 * r)
    return dict(a_eff=a_eff, P_s=ps, r=r, depth=depth, mu=mu,
                mu_cells=mu / DX, pi_z=L_FE / mu, curv=curv)


def draw(n_pool, seed):
    """Independent raw draws of alpha, beta, gamma. Nothing is solved for."""
    rng = np.random.default_rng(seed)
    alpha_abs = np.exp(rng.uniform(math.log(ALPHA_LO), math.log(ALPHA_HI), n_pool))
    beta_abs = np.exp(rng.uniform(math.log(BETA_ABS_LO), math.log(BETA_ABS_HI), n_pool))
    sign = np.where(rng.random(n_pool) < P_BETA_NEG, -1.0, 1.0)
    beta = sign * beta_abs
    gamma = np.exp(rng.uniform(math.log(GAMMA_LO), math.log(GAMMA_HI), n_pool))
    return alpha_abs, beta, gamma


def build(n_per_subset, n_pool, seed):
    alpha_abs, beta, gamma = draw(n_pool, seed)
    d = descriptors(alpha_abs, beta, gamma)

    # global admissibility, applied before any box assignment
    ok = np.isfinite(d["P_s"]) & (d["P_s"] > 0)
    ok &= alpha_abs > ALPHA_FLOOR                 # must nucleate at all
    ok &= d["mu_cells"] >= MU_CELLS_MIN           # wall must be resolved
    ok &= gamma > 0                               # beta<0 needs gamma to bound f
    cuts = dict(
        subcritical=int((alpha_abs <= ALPHA_FLOOR).sum()),
        wall_unresolved=int((d["mu_cells"] < MU_CELLS_MIN).sum()),
        bad_ps=int((~(np.isfinite(d["P_s"]) & (d["P_s"] > 0))).sum()),
    )

    rng = np.random.default_rng(seed + 1)
    rows, yields = [], {}
    for name, (a_lo, a_hi, ps_lo, ps_hi, r_lo, r_hi) in SUBSETS.items():
        m = (ok & (alpha_abs >= a_lo) & (alpha_abs <= a_hi)
             & (d["P_s"] >= ps_lo) & (d["P_s"] <= ps_hi)
             & (d["r"] >= r_lo) & (d["r"] <= r_hi))
        idx = np.flatnonzero(m)
        yields[name] = dict(hits=int(idx.size), rate=float(idx.size) / n_pool)
        if idx.size > n_per_subset:
            idx = rng.choice(idx, size=n_per_subset, replace=False)
        for i in idx:
            dt, nsteps, plot_int, dt_frame, fe_frac = window_for(
                float(alpha_abs[i]), float(beta[i]), float(gamma[i]),
                float(d["P_s"][i]))
            if nsteps > MAX_NSTEPS:
                continue
            rows.append(dict(
                subset=name, split=ROLE[name],
                alpha=-float(alpha_abs[i]), beta=float(beta[i]),
                gamma=float(gamma[i]),
                a_eff=float(d["a_eff"][i]), P_s=float(d["P_s"][i]),
                r=float(d["r"][i]), depth=float(d["depth"][i]),
                mu_cells=float(d["mu_cells"][i]), pi_z=float(d["pi_z"][i]),
                curv=float(d["curv"][i]),
                dt=dt, nsteps=nsteps, plot_int=plot_int, dt_frame=dt_frame,
                fe_frac=fe_frac))
    rows.sort(key=lambda z: (list(SUBSETS).index(z["subset"]), z["alpha"]))
    for i, row in enumerate(rows):
        row["tag"] = "o%04d" % i
    return rows, yields, cuts


def report(rows, yields, cuts, n_pool):
    print("raw pool            %d independent (alpha,beta,gamma) draws" % n_pool)
    print("global cuts         %s" % cuts)
    print()
    print("%-9s %5s %7s  %-15s %-15s %-13s %-13s %s" % (
        "subset", "n", "yield", "|alpha|/1e9", "P_s", "r", "depth/1e6", "mu_cells"))
    for name in SUBSETS:
        sub = [z for z in rows if z["subset"] == name]
        if not sub:
            print("%-9s %5d  %6.3f%%  -- EMPTY --" % (name, 0, 100 * yields[name]["rate"]))
            continue
        f = lambda k, s=1.0: (min(z[k] for z in sub) / s, max(z[k] for z in sub) / s)
        al = f("alpha", -1e9)
        print("%-9s %5d %6.3f%%  %6.2f-%-8.2f %6.3f-%-8.3f %5.2f-%-7.2f %6.1f-%-6.1f %4.2f-%.2f" % (
            name, len(sub), 100 * yields[name]["rate"],
            al[1], al[0], *f("P_s"), *f("r"), *f("depth", 1e6), *f("mu_cells")))
    print()
    ns = sum(z["nsteps"] for z in rows)
    print("cost / seed         %.1f GPU-h = %.1f node-h   storage %.0f GB"
          % (ns * 0.0615 / 3600, ns * 0.0615 / 3600 / 4, len(rows) * 118.3e6 / 1e9))

    # the disjointness the OOD claim rests on
    A = [z for z in rows if z["subset"] == "A_train"]
    if A:
        print()
        print("gaps from A_train (the axis each test subset actually probes):")
        for k, lab in (("alpha", "|alpha|"), ("P_s", "P_s"), ("r", "r"),
                       ("depth", "depth"), ("pi_z", "Pi_z")):
            a_lo, a_hi = min(abs(z[k]) for z in A), max(abs(z[k]) for z in A)
            line = "  %-8s A=[%.3g, %.3g]" % (lab, a_lo, a_hi)
            for name in SUBSETS:
                if name == "A_train":
                    continue
                sub = [z for z in rows if z["subset"] == name]
                if not sub:
                    continue
                s_lo, s_hi = min(abs(z[k]) for z in sub), max(abs(z[k]) for z in sub)
                gap = max(s_lo / a_hi, a_lo / s_hi) if s_lo > a_hi or s_hi < a_lo else 1.0
                line += "  %s %.2gx" % (name.split("_")[0] + name.split("_")[1][:1], gap)
            print(line)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-per-subset", type=int, default=200)
    ap.add_argument("--pool", type=int, default=400000)
    ap.add_argument("--seed", type=int, default=20260918)
    ap.add_argument("--out", default=os.path.join(HERE, "ood_grid"))
    args = ap.parse_args()

    rows, yields, cuts = build(args.n_per_subset, args.pool, args.seed)
    report(rows, yields, cuts, args.pool)

    with open(args.out + ".txt", "w") as f:
        f.write("# TAG ALPHA BETA GAMMA NSTEPS PLOT_INT DT SPLIT P_S\n")
        for row in rows:
            f.write("%s %.6e %.6e %.6e %d %d %.6e %s %.6f\n" % (
                row["tag"], row["alpha"], row["beta"], row["gamma"],
                row["nsteps"], row["plot_int"], row["dt"], row["subset"],
                row["P_s"]))
    with open(args.out + ".json", "w") as f:
        json.dump(dict(
            design=dict(subsets={k: list(v) for k, v in SUBSETS.items()},
                        role=ROLE, alpha_floor=ALPHA_FLOOR,
                        mu_cells_min=MU_CELLS_MIN, pool=args.pool,
                        seed=args.seed, T_frames=T_FRAMES,
                        beta_abs=[BETA_ABS_LO, BETA_ABS_HI],
                        gamma=[GAMMA_LO, GAMMA_HI], p_beta_neg=P_BETA_NEG),
            yields=yields, cuts=cuts, configs=rows), f, indent=1)
    print("\nwrote %s.txt / .json  (%d configs)" % (args.out, len(rows)))


if __name__ == "__main__":
    main()
