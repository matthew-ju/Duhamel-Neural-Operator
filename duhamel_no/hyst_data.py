#!/usr/bin/env python3
"""hyst_data.py -- loading and scoring of the FerroX hysteresis loops (voltage sweeps), used by tasks.HystTask.

A loop file holds the polarization of every saved frame, the step of each frame and the top-electrode voltage in force
when that frame was computed. Two layouts are read:
  raw FerroX extract (data_generation/hysteresis/extract_sweep.py):  P (64, 64, 18, T)
  released benchmark (Hugging Face, hysteresis/data/*.npz):         P (T, 10, 64, 64), ferroelectric layers only
Only the regular frames (step % 20 == 0) are used, so the time step between frames is uniform.

Split (data_generation/hysteresis/hyst_split.json): materials listed under "heldout" are test_material; for every other
(eps_de, material, V_max) the lowest-seed loop trains and the others are test_noise (tasks.HystTask)."""
import json, os, re
import numpy as np
import torch

import abg_laws as al

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SPLIT2 = os.path.join(ROOT, "data_generation", "hysteresis", "hyst_split.json")
GRID = {c["tag"]: c for c in json.load(open(os.path.join(ROOT, "data_generation", "abg_grid.json")))["configs"]}


def parse(name):
    m = re.match(r"hc?_e(\d+)_(c\d+)_V([\d.]+)_s(\d+)", name)
    return dict(eps=float(m.group(1)), tag=m.group(2), vmax=float(m.group(3)), seed=int(m.group(4)))


def load_loop(path, plot_int=20):
    d = np.load(path)
    keep = d["step"] % plot_int == 0
    P = d["P"]
    if P.ndim == 4 and P.shape[2] == 18:                       # raw FerroX extract (64, 64, 18, T)
        P = P[:, :, 8:18][..., keep].transpose(3, 2, 0, 1)
    else:                                                      # released layout (T, 10, 64, 64)
        P = P[keep]
    P = P[:, None] / al.P_NORM                                 # (T, 1, Z, H, W)
    info = parse(os.path.basename(path)[:-4])
    c = GRID[info["tag"]]
    return dict(P=np.ascontiguousarray(P, dtype=np.float32), V=d["V"][keep].astype(np.float32),
                step=d["step"][keep], mat=(c["alpha"], c["beta"], c["gamma"], c["dt"] * plot_int), **info)


def extra_vec(V0, V1, eps):
    """voltage conditioning of one step: V(t), V(t+1), dV and log10(eps_de), scaled to O(1)."""
    return torch.stack([V0 / 2.0, V1 / 2.0, (V1 - V0) * 4.0, torch.log10(eps) - 2.5], dim=1)


def loop_metrics(pred, L):
    """pred (T, 1, Z, H, W) in units of P_NORM. loop_err = mean |<P>_pred - <P>_true| / P_s over the whole loop."""
    ps = float(al.p_sat(*L["mat"][:3])) / al.P_NORM
    mp, mt = pred.mean(axis=(1, 2, 3, 4)), L["P"].mean(axis=(1, 2, 3, 4))
    V = L["V"]
    r = dict(loop_err=float(np.mean(np.abs(mp - mt)) / ps))

    def crossings(m):
        return [float(V[i]) for i in range(1, len(m)) if np.sign(m[i]) != np.sign(m[i - 1]) and m[i - 1] != 0 and V[i] != 0]
    r["coercive_true"], r["coercive_pred"] = crossings(mt), crossings(mp)
    zt = np.nonzero(np.abs(V) < 1e-6)[0]
    r["remanence_true"] = [float(mt[i] * al.P_NORM) for i in zt[-2:]] if len(zt) else []
    r["remanence_pred"] = [float(mp[i] * al.P_NORM) for i in zt[-2:]] if len(zt) else []
    T = len(V)
    cs = []
    for i in (T // 8, T // 4, T // 2, 3 * T // 4, T - 1):          # pattern correlation at five points along the loop
        a = pred[i].ravel() - pred[i].mean(); b = L["P"][i].ravel() - L["P"][i].mean()
        cs.append(float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30)))
    r["field_corr"] = cs
    r["relL2_mean"] = float(np.mean([np.linalg.norm(pred[i] - L["P"][i]) / (np.linalg.norm(L["P"][i]) + 1e-30) for i in range(1, T)]))
    return r, mp, mt
