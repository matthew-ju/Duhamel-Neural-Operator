#!/usr/bin/env python3
"""abg_grid.py -- build the continuous (alpha, beta, gamma) LHS design.

WHY NOT AN LHS IN THE RAW COEFFICIENTS
--------------------------------------
The physically admissible set in raw (alpha, beta, gamma) is a thin curved
sliver: the observable saturation

    P_s^2 = ( -beta + sqrt(beta^2 - 4 gamma a_eff) ) / (2 gamma)

must land in [0.10, 0.40] C/m^2, and the beta that achieves that moves by two
orders of magnitude across the alpha range (~1e9 at the shallow end, ~4e11 at
the deep end).  A box LHS + rejection would throw away >95% of the points AND
destroy the stratification that makes LHS worth using.

So we LHS in the coordinates the physics actually has:

    u1 -> |alpha|   log-uniform in [ALPHA_LO, ALPHA_HI]
    u2 -> P_s       uniform in [PS_LO, PS_HI]           (the observable)
    u3 -> r         uniform in [0, R_HI]                (6th-order share)

with  r = gamma P_s^4 / |a_eff|  = the fraction of the restoring force at
saturation that the 6th-order term carries.  Inverting,

    gamma = r |a_eff| / P_s^4 ,      beta = (1 - r) |a_eff| / P_s^2 .

Every sample is admissible by construction, every sample is a distinct float,
and the design is stratified in the three directions that matter.

WHY r IS THE RIGHT CONDITIONING COORDINATE
------------------------------------------
The beta/gamma campaign showed the reduced dynamics depend on beta and gamma
only through PI6 = gamma |a_eff| / beta^2.  In these coordinates

    PI6 = r / (1 - r)^2 ,

a bijection on r in [0,1).  PI6 diverges at r -> 1 (beta -> 0, pure 6th-order
well) and is NOT defined for beta < 0 in any useful way, while r stays finite,
ordered and bounded across the whole family:

    r = 0        pure 4th-order Landau (gamma = 0)
    0 < r < 1    2nd-order-like, beta > 0            (where the old grid lived)
    r = 1        beta = 0 exactly, pure 6th-order restoring
    r > 1        beta < 0: FIRST-ORDER-like double well, stabilised only by
                 the 6th-order term.  gamma is load-bearing by construction --
                 delete it and the energy is unbounded below.

R_HI > 1 is the point of the campaign: it is the region where no 4th-order
model can be right, so a surrogate that transfers there is doing something a
quartic rescaling cannot.

a_eff
-----
The DEPOLARISATION field renormalises the bare alpha, and TWO campaigns measure
that renormalisation from opposite directions:

  * alpha43 (42 alphas, beta/gamma fixed): the linear growth rate is affine in
    |alpha| and VANISHES at |alpha_c| = 1.367e9, four independent
    determinations agreeing (R^2 = 0.99995), and the saturated rms goes as
    (|alpha| - |alpha_c|)^0.5 -- i.e. P_sat^2 is proportional to |alpha| minus
    a critical value, which is what an effective quadratic coefficient means.
  * beta/gamma (17 (beta,gamma) configs at alpha = -2.0e9, 4 at -2.6e9): the
    saturation law fits ONE scalar a_eff per alpha, getting -1.0196e9 and
    -1.8353e9.

A two-point straight line through the second pair alone extrapolates to zero at
|alpha| = 1.25e9, near but not at the critical point the first campaign pins
much more tightly.  So the law used here is the ONE-PARAMETER form the alpha
campaign's exponent implies, with the critical point taken from the 42-point
fit and the scale fitted to the two saturation-law points:

    |a_eff| = K (|alpha| - |alpha_c|) ,   |alpha_c| = 1.367e9 , K = 1.514

which reproduces them to -6% and +2% and, unlike the two-point line, cannot go
negative anywhere in the design box.  This is a PREDICTION, not an input to the
solver -- the sweep writes the bare alpha into the inputs file, and the first
thing to check on the returned data is whether measured P_sat matches P_s here.
If the law is off, the design is still valid (the points are still spread,
still admissible, still continuous) -- only the P_s labels shift.

WINDOW
------
alpha moves the time unit, so a single (nsteps, plot_int) cannot serve the
grid: incubation t* runs from ~280 frames at |alpha|=1.5e9 to ~16 at 4.0e9
(alpha43_rate_law.json).  Instead every config gets the SAME NUMBER OF FRAMES
(T=401) and a per-config plot_int chosen so the window covers ~1.25x the time
to 99% saturation.  Consequences, all deliberate:

  * uniform tensor shape -- no ragged batching;
  * frame index means the same DEVELOPMENTAL STAGE at every config, so a batch
    compares like with like;
  * the physical frame interval dt_frame varies across configs, so the
    dimensionless sampling rate PI_tau = dt_frame / lam is NOT held constant by
    accident -- and strided pairs from a single trajectory are exact samples at
    m * PI_tau, so the training set covers the whole cadence range for free
    (no extra simulation).  See pi_universal.py.

Outputs
-------
abg_grid.txt   one line per config:  TAG ALPHA BETA GAMMA NSTEPS PLOT_INT SPLIT
abg_grid.json  full manifest with every derived quantity
"""
import argparse
import json
import math
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))

# ---- design box -------------------------------------------------------------
ALPHA_LO, ALPHA_HI = 1.6e9, 4.0e9      # |alpha|; 1.6 keeps t* <= ~170 frames and
                                       # a_eff clear of the critical point
PS_LO, PS_HI = 0.10, 0.40              # C/m^2, the user's physics filter
R_HI = 1.40                            # 6th-order share; >1 is beta < 0

# ---- solver constants (must match the template) ------------------------------
BIG_GAMMA = 100.0
DT_REF = 2.0e-13      # production dt; per-config dt never exceeds it
STAB_C = 1.2          # forward-Euler limit is 2.0; the reference config sits here
MAX_NSTEPS = 16000    # ~16 min/traj on one A100; beyond this the config is dropped
G11 = 1.0e-9
DX = 0.5e-9                            # 32 nm / 64 cells
T_FRAMES = 401

# ---- measured laws -----------------------------------------------------------
# |a_eff| = K (|alpha| - |alpha_c|); see the a_eff section of the docstring
ALPHA_C = 1.367e9
A_EFF_K = 1.514
# linear-instability incubation: 1/t* affine in |alpha| (R^2 = 0.9995)
_TSTAR_SLOPE, _TSTAR_INT = 0.02317494169943821, -0.03063775365841638
F99_OVER_TSTAR = 1.8   # measured 1.62 at -1.45e9, 1.76 at -1.50e9; round up


def a_eff_of(alpha_abs):
    """|a_eff| in SI from |alpha| in SI. Depolarisation-renormalised alpha."""
    return A_EFF_K * max(alpha_abs - ALPHA_C, 0.0)


def tstar_frames(alpha_abs):
    """Incubation time in frames of 2 ps (plot_int=10) -- the alpha43 law."""
    inv = _TSTAR_SLOPE * (alpha_abs / 1e9) + _TSTAR_INT
    return float("inf") if inv <= 0 else 1.0 / inv


def window_time(alpha_abs):
    """Physical duration of the trajectory: ~1.25x the time to 99% saturation
    plus a settled tail, expressed in seconds."""
    f99 = F99_OVER_TSTAR * tstar_frames(alpha_abs)      # frames of 2 ps
    return (1.25 * f99 + 150.0) * 10 * DT_REF


def rate_total(alpha_abs, beta, gamma, ps):
    """Worst local relaxation rate seen by the explicit update, 1/s / Gamma.

    Bulk part is |f''| at a 30% overshoot of saturation; gradient part is the
    7-point Laplacian's largest eigenvalue, 12/dx^2, and is the SAME for every
    config (g11 and the grid are fixed) -- at the production dt it alone is
    already ~48% of the forward-Euler limit, which is why the 6th-order corner
    of this design cannot reuse the production dt.
    """
    pmax = 1.3 * ps
    bulk = alpha_abs + 3.0 * abs(beta) * pmax ** 2 + 5.0 * gamma * pmax ** 4
    grad = G11 * 12.0 / DX ** 2
    return bulk + grad


def window_for(alpha_abs, beta, gamma, ps):
    """(dt, nsteps, plot_int, dt_frame, fe_frac) for T_FRAMES frames.

    dt is capped BOTH by the production value and by the stability estimate.
    STAB_C = 1.2 against a forward-Euler limit of 2.0 is not a guess: it is
    where the production reference config (alpha=-2e9, beta=3e10, gamma=3.5e11,
    dt=2e-13), which has run this campaign's entire history without blowing up,
    already sits.  So every config in this design is at most as close to the
    limit as a setting that is known to work.
    """
    rt = rate_total(alpha_abs, beta, gamma, ps)
    dt = min(DT_REF, STAB_C / (BIG_GAMMA * rt))
    plot_int = max(1, math.ceil(window_time(alpha_abs) / ((T_FRAMES - 1) * dt)))
    return dt, plot_int * (T_FRAMES - 1), plot_int, plot_int * dt, \
        BIG_GAMMA * dt * rt / 2.0


def build(n_target, seed, r_hi, extrap_lo):
    from scipy.stats import qmc

    # oversample: the map is admissible by construction, but the guards below
    # (stability, absurd coefficients) still cut a few points, and rejection
    # after the fact is what would ruin the stratification -- so draw a larger
    # LHS and thin by taking the survivors in order.
    n_draw = int(n_target * 1.25)
    u = qmc.LatinHypercube(d=3, scramble=True, seed=seed).random(n_draw)

    rows = []
    rejected = {"too_stiff": 0, "coeff_range": 0, "a_eff": 0}
    for k in range(n_draw):
        alpha_abs = math.exp(
            math.log(ALPHA_LO) + u[k, 0] * (math.log(ALPHA_HI) - math.log(ALPHA_LO)))
        ps = PS_LO + u[k, 1] * (PS_HI - PS_LO)
        r = u[k, 2] * r_hi

        ae = a_eff_of(alpha_abs)
        if ae <= 0.10e9:
            rejected["a_eff"] += 1
            continue
        gamma = r * ae / ps ** 4
        beta = (1.0 - r) * ae / ps ** 2
        if abs(beta) > 1e13 or gamma > 1e15:
            rejected["coeff_range"] += 1
            continue
        dt, nsteps, plot_int, dt_frame, fe_frac = window_for(
            alpha_abs, beta, gamma, ps)
        # a config whose stability-capped dt makes the window unaffordable is
        # dropped rather than run at a dt that would blow up.
        if nsteps > MAX_NSTEPS:
            rejected["too_stiff"] += 1
            continue
        # PI6 is only meaningful for beta > 0; report it, do not condition on it
        pi6 = gamma * ae / beta ** 2 if beta > 0 else float("nan")
        rows.append(dict(
            alpha=-alpha_abs, beta=beta, gamma=gamma,
            a_eff=-ae, P_s=ps, r=r, pi6=pi6,
            tstar_2ps=tstar_frames(alpha_abs),
            dt=dt, nsteps=nsteps, plot_int=plot_int, dt_frame=dt_frame,
            fe_frac=fe_frac, u=[float(x) for x in u[k]]))
        if len(rows) == n_target:
            break

    # ---- splits -------------------------------------------------------------
    # test_extrap: the strongly first-order corner, held out ENTIRELY, so the
    # headline generalisation claim is an extrapolation in the 6th-order share
    # and not a lucky interpolation.  test_interp: random, inside the hull.
    rng = np.random.default_rng(seed + 1)
    for row in rows:
        row["split"] = "extrap" if row["r"] >= extrap_lo else "train"
    idx_train = [i for i, row in enumerate(rows) if row["split"] == "train"]
    n_interp = max(1, int(0.10 * len(idx_train)))
    for i in rng.choice(idx_train, size=n_interp, replace=False):
        rows[i]["split"] = "interp"

    # ---- reference anchor, prepended so it is config 0 -----------------------
    # exactly the beta/gamma campaign's reference material: lets the new stack
    # be cross-checked against 6 seeds of existing ground truth.
    a_ref = 2.0e9
    ae_ref = a_eff_of(a_ref)
    ps_ref = math.sqrt((-3.0e10 + math.sqrt(9e20 + 4 * 3.5e11 * ae_ref))
                       / (2 * 3.5e11))
    dt_r, ns_r, pi_r, dtf_r, fe_r = window_for(a_ref, 3.0e10, 3.5e11, ps_ref)
    rows.insert(0, dict(
        alpha=-a_ref, beta=3.0e10, gamma=3.5e11, a_eff=-ae_ref, P_s=ps_ref,
        r=3.5e11 * ps_ref ** 4 / ae_ref, pi6=3.5e11 * ae_ref / 9e20,
        tstar_2ps=tstar_frames(a_ref), dt=dt_r, nsteps=ns_r, plot_int=pi_r,
        dt_frame=dtf_r, fe_frac=fe_r, u=[-1, -1, -1], split="train"))

    for i, row in enumerate(rows):
        row["tag"] = "c%04d" % i
    return rows, rejected


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=20260912)
    ap.add_argument("--r-hi", type=float, default=R_HI)
    ap.add_argument("--extrap-lo", type=float, default=1.15)
    ap.add_argument("--out", default=os.path.join(HERE, "abg_grid"))
    args = ap.parse_args()

    rows, rejected = build(args.n, args.seed, args.r_hi, args.extrap_lo)

    with open(args.out + ".txt", "w") as f:
        f.write("# TAG ALPHA BETA GAMMA NSTEPS PLOT_INT DT SPLIT\n")
        for row in rows:
            f.write("%s %.6e %.6e %.6e %d %d %.6e %s\n" % (
                row["tag"], row["alpha"], row["beta"], row["gamma"],
                row["nsteps"], row["plot_int"], row["dt"], row["split"]))
    with open(args.out + ".json", "w") as f:
        json.dump(dict(
            design=dict(alpha_lo=ALPHA_LO, alpha_hi=ALPHA_HI, ps_lo=PS_LO,
                        ps_hi=PS_HI, r_hi=args.r_hi, extrap_lo=args.extrap_lo,
                        seed=args.seed, n=len(rows), T_frames=T_FRAMES,
                        dt_ref=DT_REF, stab_c=STAB_C),
            rejected=rejected, configs=rows), f, indent=1)

    # ---- report -------------------------------------------------------------
    A = np.array([r["alpha"] for r in rows])
    B = np.array([r["beta"] for r in rows])
    G = np.array([r["gamma"] for r in rows])
    R = np.array([r["r"] for r in rows])
    PS = np.array([r["P_s"] for r in rows])
    PI = np.array([r["plot_int"] for r in rows])
    NS = np.array([r["nsteps"] for r in rows])
    print("configs            %d   (rejected %s)" % (len(rows), rejected))
    print("|alpha|/1e9        %.2f .. %.2f" % (-A.max() / 1e9, -A.min() / 1e9))
    print("beta               %.3e .. %.3e   (%d negative = first-order)"
          % (B.min(), B.max(), int((B < 0).sum())))
    print("gamma              %.3e .. %.3e" % (G.min(), G.max()))
    print("P_s (predicted)    %.3f .. %.3f" % (PS.min(), PS.max()))
    print("r  (6th share)     %.3f .. %.3f" % (R.min(), R.max()))
    fin = np.isfinite([r["pi6"] for r in rows])
    print("PI6 (beta>0 only)  %.3f .. %.3f  over %d configs"
          % (np.nanmin(np.array([r["pi6"] for r in rows])[fin]),
             np.nanmax(np.array([r["pi6"] for r in rows])[fin]), int(fin.sum())))
    print("plot_int           %d .. %d  (median %d)"
          % (PI.min(), PI.max(), int(np.median(PI))))
    DTV = np.array([r["dt"] for r in rows])
    print("dt                 %.3e .. %.3e  (%d below the production 2e-13)"
          % (DTV.min(), DTV.max(), int((DTV < DT_REF * 0.999).sum())))
    print("forward-Euler frac max %.3f  (reference config sits at %.3f)"
          % (max(r["fe_frac"] for r in rows), rows[0]["fe_frac"]))
    print("nsteps             %d .. %d  (median %d)"
          % (NS.min(), NS.max(), int(np.median(NS))))
    for s in ("train", "interp", "extrap"):
        print("split %-7s      %d" % (s, sum(1 for r in rows if r["split"] == s)))
    # FerroX cost model: 246 s at nsteps=4000 -> 0.0615 s/step on one A100
    cost_gpu_s = float(NS.sum()) * 0.0615
    print("cost / seed        %.1f GPU-h  = %.1f node-h (4 GPU/node)"
          % (cost_gpu_s / 3600, cost_gpu_s / 3600 / 4))
    print("storage / seed     %.0f GB" % (len(rows) * 118.3e6 / 1e9))


if __name__ == "__main__":
    main()
