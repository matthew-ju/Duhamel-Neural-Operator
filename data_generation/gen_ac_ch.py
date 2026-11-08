#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""gen_ac_ch.py -- 2D Allen-Cahn and Cahn-Hilliard datasets for the units study.

    AC:  du/dt = -M (a u + b u^3 - e lap u)          a < 0
    CH:  du/dt =  M lap (a u + b u^3 - e lap u)       a < 0

Units (textbook = effective here: there is no renormalising field in these equations)
    amplitude  U  = sqrt(|a|/b)
    length     mu = sqrt(e/|a|)                         (wall width)
    time       T  = 1/(M|a|)          (AC)      T = e/(M a^2) = mu^2/(M|a|)   (CH)
Remaining dimensionless numbers: Pi_h = dx/mu (grid spacing in wall widths; fixed domain 2pi,
64 cells -> also sets L/mu) and theta = dt_frame/T. The starting noise is fixed in ABSOLUTE
units (std 0.01), like FerroX, so its size relative to U varies.

Parameters are sampled in (U, T, Pi_h, theta) and converted to raw (a, b, M, e, dt_frame), so the
unit directions and the invariant directions can be moved independently:
    train / id    U in [0.5, 2],  |a| in [0.5, 5],   M in [0.5, 2], Pi_h in train range, theta in range
    ood_units     U in [4, 10],   |a| in [20, 100],  same Pi_h and theta ranges  (raw coeffs far outside)
    ood_inv       U, |a|, M, theta as train, Pi_h beyond the training range      (a new physical regime)
Every trajectory has its own material AND its own starting noise (unique seed).
Exact-linear ETD1 integration in Fourier, per-sample time step (sub steps per frame).
"""
import argparse, json, math, os, time
import numpy as np
import torch

OUT = os.path.join(os.environ["SCRATCH"], "ml_units")
N, LDOM = 64, 2 * math.pi
DX = LDOM / N
CFG = {
    "ac": dict(pih=(0.30, 0.70), pih_ood=(0.85, 1.20), theta=(0.10, 0.30), dt_over_T=0.04, frames=100),
    "ch": dict(pih=(0.35, 0.70), pih_ood=(0.85, 1.10), theta=(1.0, 3.0), dt_over_T=None, frames=100),
}
SPLITS = {"train": 1000, "id": 128, "ood_units": 128, "ood_inv": 128, "train4k": 4000}


def sample(system, split, n, rng):
    c = CFG[system]
    lu = lambda lo, hi: np.exp(rng.uniform(np.log(lo), np.log(hi), n))
    if split == "ood_units":
        U, A = lu(4, 10), lu(20, 100)
    else:
        U, A = lu(0.5, 2), lu(0.5, 5)
    M = lu(0.5, 2)
    pih = rng.uniform(*(c["pih_ood"] if split == "ood_inv" else c["pih"]), n)
    theta = rng.uniform(*c["theta"], n)
    mu = (LDOM / 64) / pih          # physics sampled in 64-grid units -> identical materials at every resolution
    pih = DX / mu                   # this grid's dx / mu
    a, b, e = -A, A / U ** 2, A * mu ** 2
    T = 1.0 / (M * A) if system == "ac" else mu ** 2 / (M * A)
    # starting noise: std 0.01 absolute (as in FerroX) -> noise/U = 0.01/U. For ood_units the noise is scaled
    # so that noise/U follows the TRAINING distribution; that split is then a pure change of units.
    ns = (U / lu(0.5, 2)) if split == "ood_units" else np.ones(n)
    return dict(a=a, b=b, M=M, e=e, U=U, T=T, mu=mu, pih=pih, theta=theta, dtf=theta * T, noise_scale=ns)


def k2grid(dev):
    k1 = 2 * math.pi * torch.fft.fftfreq(N, d=LDOM / N, device=dev).double()
    return (k1.view(-1, 1) ** 2 + k1.view(1, -1)[:, : N // 2 + 1] ** 2)


@torch.no_grad()
def simulate(system, p, seeds, dev, frames):
    B = len(seeds)
    t = lambda x: torch.as_tensor(x, dtype=torch.float64, device=dev).view(B, 1, 1)
    a, b, M, e = t(p["a"]), t(p["b"]), t(p["M"]), t(p["e"])
    k2 = k2grid(dev)[None]
    if system == "ac":
        Lk = -M * (a + e * k2)
        stiff = 3.0 * M * (-a)                                  # |d(nonlinear)/du| at saturation
    else:
        Lk = -M * (a * k2 + e * k2 ** 2)
        stiff = 3.0 * M * (-a) * k2.max()
    dt_max = 0.25 / stiff.view(B)                               # explicit-part stability margin
    dtf = torch.as_tensor(p["dtf"], dtype=torch.float64, device=dev)
    sub = int(torch.ceil((dtf / dt_max).max()).item())
    dt = (dtf / sub).view(B, 1, 1)
    z = Lk * dt
    E = torch.exp(z)
    phi1 = torch.where(z.abs() < 1e-10, 1.0 + 0.5 * z, (E - 1.0) / torch.where(z.abs() < 1e-10, torch.ones_like(z), z))
    ns = p.get("noise_scale", np.ones(B))
    u = torch.stack([torch.from_numpy(np.random.default_rng(int(s)).normal(0, 0.01 * float(k) * N / 64, (N, N)))
                     for s, k in zip(seeds, ns)]).to(dev)
    out = [u.float().cpu()]
    for _ in range(frames):
        for _ in range(sub):
            uh = torch.fft.rfft2(u)
            nl = torch.fft.rfft2(u ** 3)
            nh = (-M * b * nl) if system == "ac" else (-M * b * k2 * nl)
            u = torch.fft.irfft2(E * uh + dt * phi1 * nh, s=(N, N))
        out.append(u.float().cpu())
    return torch.stack(out, 1).numpy(), sub                  # (B, frames+1, N, N)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", choices=["ac", "ch"], required=True)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--splits", default="train,id,ood_units,ood_inv")
    ap.add_argument("--n", type=int, default=64, help="grid (same 2pi domain, same physics; noise std x n/64)")
    a = ap.parse_args()
    global N, DX
    N, DX = a.n, LDOM / a.n
    dev = torch.device("cuda")
    os.makedirs(OUT, exist_ok=True)
    seed_base = {"train": 10_000, "id": 20_000, "ood_units": 30_000, "ood_inv": 40_000, "train4k": 60_000}
    for split in a.splits.split(","):
        n = SPLITS[split]
        rng = np.random.default_rng({"ac": 1, "ch": 2}[a.system] * 100 + list(SPLITS).index(split))
        p = sample(a.system, split, n, rng)
        seeds = seed_base[split] + np.arange(n) + (0 if a.system == "ac" else 500_000)
        U = np.empty((n, CFG[a.system]["frames"] + 1, N, N), np.float32)
        t0 = time.time(); subs = []
        for lo in range(0, n, a.bs):
            sl = slice(lo, min(n, lo + a.bs))
            U[sl], s = simulate(a.system, {k: v[sl] for k, v in p.items()}, seeds[sl], dev, CFG[a.system]["frames"])
            subs.append(s)
        bad = ~np.isfinite(U).all(axis=(1, 2, 3))
        rms = np.sqrt((U[:, -1] ** 2).mean(axis=(1, 2))) / p["U"]
        print("%s %-9s n=%d  %.0fs  substeps/frame %s  non-finite %d  final rms/U %.2f-%.2f" % (
            a.system, split, n, time.time() - t0, subs, bad.sum(), np.nanmin(rms), np.nanmax(rms)), flush=True)
        name = a.system if a.n == 64 else "%s%d" % (a.system, a.n)
        tmp = os.path.join(OUT, ".tmp_%s_%s.npz" % (name, split))
        dml = np.stack([np.log(p["theta"]), p["pih"]], 1).astype(np.float32)
        raw = np.stack([np.log(-p["a"]), np.log(p["b"]), np.log(p["M"]), np.log(p["e"]), np.log(p["dtf"])], 1).astype(np.float32)
        np.savez(tmp, u=U, seeds=seeds, dml=dml, raw=raw, **p)
        os.replace(tmp, os.path.join(OUT, "%s_%s.npz" % (name, split)))       # atomic: readers never see a partial file


if __name__ == "__main__":
    main()
