#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""fit_alpha_bank.py -- the FerroX LINEAR operator for any material, from training data (for Duhamel-UNO).

The linear (incubation) dynamics depend only on alpha and the film/dielectric geometry, not on beta, gamma.
For every TRAINING trajectory: per integer |k| bin (0..45 on the 64x64 grid), least-squares fit of the one-frame
10x10 (z-layer) propagator B from frames 0..8 (real and imaginary Fourier parts as samples), then the
time-step-free generator G = logm(B) / theta (theta = the frame's dimensionless step, abg_laws.theta). Bins whose
propagator is not measurable (decayed to round-off: min |eig B| < 1e-6, or ill-fitted) are marked invalid -> they are
fully damped at use time (they are, physically, within one frame).
For a new material: G(alpha) = mean of the valid G of the K training materials nearest in log|alpha|.
Validation: linear-propagation correlation / rel. L2 at frames 1..8 on the held-out VALIDATION materials (vs truth),
compared with the conditioned learned Fourier SSM (handoff/runs/ssm_lin20.pt).
Out: $SCRATCH/bench_cache/alpha_bank_<task>.npz  (alpha, G (N,46,10,10), valid (N,46)) + validation json.
"""
import argparse, json, os, sys
import numpy as np
import torch
from scipy.linalg import logm, expm

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import tasks as TK
OUT = os.path.join(os.environ["SCRATCH"], "bench_cache"); os.makedirs(OUT, exist_ok=True)
N = 64
kx = np.fft.fftfreq(N) * N
KB = np.rint(np.sqrt(kx[:, None] ** 2 + kx[None, :] ** 2)).astype(int)          # (64,64) integer |k| bin
NB = KB.max() + 1


def fit_one(frames, theta, nfr=9):
    """frames (T>=nfr, Z, H, W) float64 -> G (NB,Z,Z), valid (NB,)"""
    F = np.fft.fft2(frames[:nfr], axes=(-2, -1))                    # (T,Z,H,W)
    Z = F.shape[1]
    G = np.zeros((NB, Z, Z)); ok = np.zeros(NB, bool)
    for b in range(NB):
        sel = KB == b
        if not sel.any():
            continue
        Fb = F[:, :, sel]                                           # (T,Z,m)
        X = np.concatenate([Fb[:-1].real, Fb[:-1].imag], -1).transpose(0, 2, 1).reshape(-1, Z)
        Y = np.concatenate([Fb[1:].real, Fb[1:].imag], -1).transpose(0, 2, 1).reshape(-1, Z)
        sc = np.linalg.norm(X, axis=1); keep = sc > 1e-12 * sc.max()
        X, Y = X[keep], Y[keep]
        if len(X) < 3 * Z:
            continue
        B = np.linalg.lstsq(X, Y, rcond=None)[0].T
        ev = np.linalg.eigvals(B)
        if np.min(np.abs(ev)) < 1e-6 or not np.all(np.isfinite(B)):
            continue
        Lg = logm(B)
        if np.abs(Lg.imag).max() > 1e-3 * max(np.abs(Lg.real).max(), 1e-12):
            continue
        G[b] = Lg.real / theta; ok[b] = True
    return G, ok


def propagate(x0, G, ok, theta, nfr=8):
    """linear rollout from x0 (Z,H,W) with per-bin generator -> list of frames 1..nfr"""
    Xf = np.fft.fft2(x0, axes=(-2, -1)); out = []
    A = np.zeros((NB,) + G.shape[1:])
    for b in range(NB):
        A[b] = expm(theta * G[b]) if ok[b] else 0.0
    for _ in range(nfr):
        Xf = np.einsum("hwij,jhw->ihw", A[KB], Xf)
        out.append(np.fft.ifft2(Xf, axes=(-2, -1)).real)
    return out


def corr(a, b):
    a = a.ravel() - a.mean(); b = b.ravel() - b.mean(); return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--task", default="tdgl", choices=["tdgl", "hyst"]); ap.add_argument("--K", type=int, default=8)
    a = ap.parse_args()
    import abg_laws as al
    if a.task == "tdgl":
        task = TK.TDGLTask()
        trs = [task.train[i] for i in task.tr_idx]; vals = [task.train[i] for i in task.val_idx]
        get = lambda t: task.gd.frames(t, 0, 9)[:, 0].astype(np.float64)
        th = lambda t: float(t.theta); alp = lambda t: float(t.mat[0])
    else:
        task = TK.HystTask()
        trs = task.train; vals = [task.load(p) for p in task.t_val]
        get = lambda L: L["P"][:9, 0].astype(np.float64) if L["P"] is not None else None
        # training loops' P was dropped to save memory: reload them
        trs = [task.load(os.path.join(os.environ["SCRATCH"], os.environ.get("HYST_DIR", "ferrox_hyst2"), "h_e%d_%s_V%.1f_s%d.npz" % (L["eps"], L["tag"], L["vmax"], L["seed"])))
               for L in trs]
        th = lambda L: float(al.theta(L["mat"][0], L["mat"][3])); alp = lambda L: float(L["mat"][0])
    A = np.array([alp(t) for t in trs]); Gs = []; OK = []
    for i, t in enumerate(trs):
        G, ok = fit_one(get(t), th(t)); Gs.append(G); OK.append(ok)
        if i % 100 == 0:
            print("fit %d/%d valid bins %d" % (i, len(trs), ok.sum()), flush=True)
    Gs = np.array(Gs, np.float32); OK = np.array(OK)
    np.savez_compressed(os.path.join(OUT, "alpha_bank_%s%s.npz" % (a.task, os.environ.get("BENCH_TAG", ""))), alpha=A, G=Gs, valid=OK)
    # ---- validation on held-out materials
    la = np.log(np.abs(A)); res = []
    for t in vals[:40]:
        d = np.abs(la - np.log(abs(alp(t)))); nn = np.argsort(d)[:a.K]
        G = np.zeros_like(Gs[0]); ok = np.zeros(NB, bool)
        for b in range(NB):
            v = [j for j in nn if OK[j, b]]
            if v:
                G[b] = Gs[v, b].mean(0); ok[b] = True
        fr = get(t); pr = propagate(fr[0], G, ok, th(t))
        res.append([corr(pr[q], fr[q + 1]) for q in range(8)] + [float(np.linalg.norm(pr[q] - fr[q + 1]) / np.linalg.norm(fr[q + 1])) for q in range(8)])
    res = np.array(res)
    summ = dict(task=a.task, n_train=len(trs), n_val=len(res), K=a.K, valid_bins_mean=float(OK.sum(1).mean()),
                corr_t1_8=res[:, :8].mean(0).round(4).tolist(), rel_t1_8=res[:, 8:].mean(0).round(4).tolist())
    json.dump(summ, open(os.path.join(HERE, "results_canon", "alpha_bank_%s%s_val.json" % (a.task, os.environ.get("BENCH_TAG", ""))), "w"), indent=1)
    print(json.dumps(summ), flush=True)


if __name__ == "__main__":
    main()
