#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""abg_data.py -- one index over every FerroX stack this project has produced.

Three stacks, one interface:

  abg         the continuous (alpha,beta,gamma) LHS sweep (abg_grid.py).  Per
              config alpha, beta, gamma AND its own dt_frame, because the
              window is chosen per material.  T = 401 everywhere.
  alpha43     42 alphas at the reference beta/gamma, T = 801, dt_frame = 2 ps.
  betagamma   17 (beta,gamma) at alpha = -2e9, T = 401, dt_frame = 2 ps
              (+ the joint lane at alpha = -2.6e9).

The legacy two are a CROSS through the reference material -- one lane varies
alpha, the other varies beta and gamma -- and the LHS sweep is the filled-in
volume.  Keeping them in the same index is what lets the model be trained and
scored before the sweep finishes, and lets "trained on the cross, tested in the
volume" be an actual arm rather than a thought experiment.

STRIDED PAIRS ARE FREE DATA.  A pair (t, t+m) is an exact sample of the
operator at dimensionless step m*theta.  Since theta is what the conditioner is
told, striding covers the whole cadence range from trajectories that were
simulated at one cadence -- no extra simulation.  `stride_max` turns this on.

Units returned: NORMALISED, P / 0.25, the convention the rest of the repo uses.
Reduction to the material's own units happens inside the model, so an arm can
be run with it switched off.
"""
import glob
import json
import math
import os
import re

import numpy as np
import torch
from torch.utils.data import Dataset

import abg_laws as al

HERE = os.path.dirname(os.path.abspath(__file__))
SCRATCH = os.environ.get("SCRATCH", "/tmp")
Z_LO, Z_HI = 8, 18          # the 10 ferroelectric layers
P_NORM = al.P_NORM


# ===========================================================================
# Index
# ===========================================================================
class Traj:
    __slots__ = ("path", "alpha", "beta", "gamma", "dt_frame", "T", "split",
                 "seed", "tag", "stack")

    def __init__(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)

    def __repr__(self):
        return "<%s/%s s%d a=%.3g b=%.3g g=%.3g T=%d %s>" % (
            self.stack, self.tag, self.seed, self.alpha, self.beta,
            self.gamma, self.T, self.split)


def _seed_of(path):
    return int(re.search(r"seed_(\d+)\.npy$", path).group(1))


def index_abg(work=None, grid=None):
    work = work or os.path.join(SCRATCH, "ferrox_abg")
    grid = grid or os.path.join(HERE, "abg_grid.json")
    if not os.path.exists(grid):
        return []
    cfgs = {c["tag"]: c for c in json.load(open(grid))["configs"]}
    out = []
    for p in sorted(glob.glob(os.path.join(work, "extracted", "*", "seed_*.npy"))):
        tag = os.path.basename(os.path.dirname(p))
        c = cfgs.get(tag)
        if c is None:
            continue
        out.append(Traj(path=p, alpha=c["alpha"], beta=c["beta"],
                        gamma=c["gamma"], dt_frame=c["dt_frame"], T=401,
                        split=c["split"], seed=_seed_of(p), tag=tag,
                        stack="abg"))
    return out


# ---------------------------------------------------------------------------
# HELD-OUT MATERIALS IN THE LEGACY STACKS
# ---------------------------------------------------------------------------
# Without these the legacy cross has only held-out SEEDS, and an arm campaign
# run on it would measure "does the model generalise to a new noise field",
# which every fixed-alpha result in this project already answers yes to.  The
# question is whether it generalises to a new MATERIAL, so a subset of the
# legacy configs is withheld exactly as the LHS volume's are.
#
# interp = interior points, spread across the range, surrounded by training
#          points on both sides;
# extrap = the ENDS -- the two shallowest and two deepest alphas, and the two
#          largest gammas -- so the legacy campaign has an extrapolation test
#          of its own rather than only an interpolation one.
# The betagamma choices are the ones that campaign itself held out, so its
# numbers and these are comparable.
A43_INTERP = ("am1.75", "am2.30", "am2.95", "am3.40")
A43_EXTRAP = ("am1.45", "am1.50", "am3.45", "am3.50")
BG_INTERP = ("b2.50g3.500", "b3.00g2.000", "b3.00g9.000", "b2.10g1.715")
BG_EXTRAP = ("b3.00g14.000", "b6.00g14.000", "b3.00g0.000")


def _legacy_split(tag, interp, extrap):
    if tag in extrap:
        return "extrap"
    if tag in interp:
        return "interp"
    return "train"


def index_alpha43(work=None):
    work = work or os.path.join(SCRATCH, "ferrox_alpha43")
    out = []
    for p in sorted(glob.glob(os.path.join(work, "extracted", "am*", "seed_*.npy"))):
        tag = os.path.basename(os.path.dirname(p))
        out.append(Traj(path=p, alpha=-float(tag[2:]) * 1e9, beta=al.BETA_REF,
                        gamma=al.GAMMA_REF, dt_frame=al.DT_FRAME_LEGACY, T=801,
                        split=_legacy_split(tag, A43_INTERP, A43_EXTRAP),
                        seed=_seed_of(p), tag=tag, stack="alpha43"))
    return out


def index_betagamma(work=None, grid=None, alpha=-2.0e9):
    work = work or os.path.join(SCRATCH, "ferrox_betagamma")
    grid = grid or os.path.join(HERE, "bg_grid.txt")
    if not os.path.exists(grid):
        return []
    cfg = {}
    for line in open(grid):
        p = line.split()
        if len(p) >= 3 and not line.startswith("#"):
            cfg[p[0]] = (float(p[1]), float(p[2]))
    out = []
    for p in sorted(glob.glob(os.path.join(work, "extracted", "*", "seed_*.npy"))):
        tag = os.path.basename(os.path.dirname(p))
        if tag not in cfg:
            continue
        b, g = cfg[tag]
        out.append(Traj(path=p, alpha=alpha, beta=b, gamma=g,
                        dt_frame=al.DT_FRAME_LEGACY, T=401,
                        split=_legacy_split(tag, BG_INTERP, BG_EXTRAP),
                        seed=_seed_of(p), tag=tag, stack="betagamma"))
    return out


def index_useed(work=None, grid=None):
    """D1 configs re-run with ONE UNIQUE seed each (seed 2000+i, 2026-09-24): 1001 distinct
    starting noise fields, versus the 10 (500-509) the abg stack shares across all configs.
    Opt-in (not in the default stacks) so every existing script is unchanged. The config's own
    split applies: held-out configs stay held out, at their unique seed too."""
    work = work or os.path.join(SCRATCH, "ferrox_abg_useed")
    grid = grid or os.path.join(HERE, "abg_grid.json")
    cfgs = {c["tag"]: c for c in json.load(open(grid))["configs"]}
    out = []
    for p in sorted(glob.glob(os.path.join(work, "extracted", "c*_s*.npy"))):
        tag, sd = os.path.basename(p)[:-4].split("_s")
        c = cfgs.get(tag)
        if c is None:
            continue
        out.append(Traj(path=p, alpha=c["alpha"], beta=c["beta"], gamma=c["gamma"],
                        dt_frame=c["dt_frame"], T=401, split=c["split"], seed=int(sd),
                        tag=tag, stack="useed"))
    return out


def build_index(stacks=("abg", "alpha43", "betagamma"), max_seeds=None):
    idx = []
    for s in stacks:
        idx += {"abg": index_abg, "alpha43": index_alpha43,
                "betagamma": index_betagamma, "useed": index_useed}[s]()
    if max_seeds is not None:
        keep = {}
        for t in idx:
            keep.setdefault((t.stack, t.tag), []).append(t)
        idx = [t for v in keep.values()
               for t in sorted(v, key=lambda x: x.seed)[:max_seeds]]
    return idx


def split_index(idx, val_seeds=(508, 509, 512, 513, 514, 515, 526, 527, 528, 529)):
    """(train, val_seed, val_interp, val_extrap).

    TWO axes of held-out data, and they answer different questions:
      * held-out SEEDS at a training config -- does the model generalise to a
        new initial noise field?  (the easy one)
      * held-out CONFIGS -- does it generalise to a new material?  `interp` is
        inside the design hull, `extrap` is the strongly first-order corner
        (r >= 1.15) that is withheld ENTIRELY, so it is a genuine
        extrapolation in the 6th-order share rather than a lucky interpolation.
    A trajectory that is held out by config is never used, at any seed.
    """
    vs = set(val_seeds)
    tr, vseed, vint, vext = [], [], [], []
    for t in idx:
        if t.split == "extrap":
            vext.append(t)
        elif t.split == "interp":
            vint.append(t)
        elif t.seed in vs:
            vseed.append(t)
        else:
            tr.append(t)
    return tr, vseed, vint, vext


# ===========================================================================
# Frame access
# ===========================================================================
_CACHE = {}


def _mm(path):
    m = _CACHE.get(path)
    if m is None:
        if len(_CACHE) > 512:                 # bounded: one mmap per open file
            _CACHE.clear()
        m = np.load(path, mmap_mode="r")
        _CACHE[path] = m
    return m


def frame(traj, t):
    """(1,Z,H,W) float32 normalised field at frame t."""
    a = _mm(traj.path)[:, :, Z_LO:Z_HI, t]        # (H,W,Z)
    return np.ascontiguousarray(a.transpose(2, 0, 1))[None] / P_NORM


# ===========================================================================
# Datasets
# ===========================================================================
class PairSet(Dataset):
    """(x_t, x_{t+m}) pairs with the material label and the stride m.

    t_stride subsamples the START frames (the trajectory is heavily correlated
    frame to frame, so every frame is not an independent sample); stride_max
    controls the SPAN m, which is a genuine augmentation because the operator
    at m*theta is a different operator.
    """

    def __init__(self, idx, t_lo=0, t_hi=None, t_stride=4, stride_max=1,
                 seed=0):
        self.idx = idx
        self.items = []
        rng = np.random.default_rng(seed)
        for i, tr in enumerate(idx):
            hi = (tr.T if t_hi is None else min(t_hi, tr.T))
            for t in range(t_lo, hi - stride_max, t_stride):
                self.items.append((i, t))
        self.stride_max = stride_max
        self.rng = rng

    def __len__(self):
        return len(self.items)

    def __getitem__(self, k):
        i, t = self.items[k]
        tr = self.idx[i]
        m = 1 if self.stride_max <= 1 else int(
            np.random.randint(1, self.stride_max + 1))
        m = min(m, tr.T - 1 - t)
        return (torch.from_numpy(frame(tr, t)),
                torch.from_numpy(frame(tr, t + m)),
                torch.tensor([tr.alpha, tr.beta, tr.gamma, tr.dt_frame],
                             dtype=torch.float64),
                torch.tensor(float(m), dtype=torch.float32))


class WindowSet(Dataset):
    """k+1 consecutive frames plus the material label -- the training window.

    Mirrors the production recipe (t400_common.WindowDataset): the model is
    pushed forward k frames from a real start frame and scored at every one, so
    the loss sees the compounding error a single-step objective cannot.

    INCUBATION OVERSAMPLING.  Frame rms runs ~5e-4 at t=0 to ~0.4 at
    saturation, so uniformly sampled start frames put almost no weight on the
    regime the spectral half exists to cover.  `inc_repeat` copies of the
    windows starting before `inc_end` fixes the sampling half of that; the
    relative loss fixes the weighting half.

    STRIDE.  With stride m the window is frames t, t+m, ..., t+km, an exact
    sample of the operator at m*theta.  One trajectory therefore supplies the
    whole cadence range, which is what lets theta be a genuine conditioning
    input rather than a constant of the dataset.
    """

    def __init__(self, idx, k=5, t_stride=4, stride_max=1, inc_end=0,
                 inc_repeat=1, t_hi=None):
        self.idx, self.k, self.stride_max = idx, k, max(1, stride_max)
        self.items = []
        for i, tr in enumerate(idx):
            hi = tr.T if t_hi is None else min(t_hi, tr.T)
            last = hi - 1 - k * self.stride_max
            for t in range(0, max(0, last), t_stride):
                self.items.append((i, t))
            if inc_end and inc_repeat > 1:
                for t in range(0, min(inc_end, max(0, last))):
                    self.items += [(i, t)] * (inc_repeat - 1)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, j):
        i, t = self.items[j]
        tr = self.idx[i]
        m = (1 if self.stride_max == 1
             else int(np.random.randint(1, self.stride_max + 1)))
        m = max(1, min(m, (tr.T - 1 - t) // max(1, self.k)))
        fr = np.stack([frame(tr, t + q * m)[0] for q in range(self.k + 1)])
        return (torch.from_numpy(fr),
                torch.tensor([tr.alpha, tr.beta, tr.gamma, tr.dt_frame],
                             dtype=torch.float64),
                torch.tensor(float(m), dtype=torch.float32))


class RolloutSet(Dataset):
    """Whole trajectories, subsampled in frames, for rollout scoring."""

    def __init__(self, idx, t0=0, n=400, keep_every=10):
        self.idx = idx
        self.t0, self.n = t0, n
        self.keep = [t for t in range(0, n + 1, keep_every)]

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, k):
        tr = self.idx[k]
        fr = np.stack([frame(tr, min(self.t0 + t, tr.T - 1))[0]
                       for t in self.keep])          # (K,Z,H,W)
        return (torch.from_numpy(fr),
                torch.tensor([tr.alpha, tr.beta, tr.gamma, tr.dt_frame],
                             dtype=torch.float64), k)


def describe(idx):
    from collections import Counter
    c = Counter((t.stack, t.split) for t in idx)
    tags = len({(t.stack, t.tag) for t in idx})
    return "%d trajectories, %d configs, %s" % (
        len(idx), tags, dict(sorted(c.items())))


if __name__ == "__main__":
    idx = build_index()
    print(describe(idx))
    tr, vs, vi, ve = split_index(idx)
    for n, s in [("train", tr), ("val-seed", vs), ("val-interp", vi),
                 ("val-extrap", ve)]:
        print("%-11s %s" % (n, describe(s) if s else "empty"))
    if idx:
        t = idx[0]
        x = frame(t, 100)
        print(t, "frame:", x.shape, "rms %.4f" % float(np.sqrt((x ** 2).mean())))
        print("cond pi :", al.cond_vec(t.alpha, t.beta, t.gamma, t.dt_frame, "pi"))
