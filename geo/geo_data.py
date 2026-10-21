#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""geo_data.py -- one trajectory index over both device geometries.

  geometry "d1": 5 nm film, dx 0.5 nm, 18 z-cells, ferroelectric layers z = 8..17 (10 layers)
                 stacks: abg (seeds 500-509 shared), useed (seed 2000+i, one per config)
  geometry "d3": 40 nm film, dx 1 nm, 44 z-cells, ferroelectric layers z = 4..43 (40 layers)
                 stacks: ext (seeds 500-503 shared), extu (seed 5000+i, one per config)
Every trajectory carries its conditioning (c, amp, theta) from geo_model, its config split
(train / interp / extrap) and its seed, so train/test selection is by config AND by noise.
"""
import glob, json, os, sys
import numpy as np
import torch

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "geo"))
import abg_data as ad
import abg_laws as al
import geo_model as gm

SCR = os.environ.get("SCRATCH", "/tmp")
ZSL = {"d1": (8, 18), "d3": (4, 44)}


class GTraj:
    __slots__ = ("path", "geo", "stack", "tag", "seed", "split", "c", "amp", "theta", "T", "P_s", "mat", "box")

    def __init__(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)


def index_d1(stacks=("abg", "useed")):
    out = []
    for t in ad.build_index(stacks=tuple(stacks)):
        c, amp, th = gm.ctx_d1(t.alpha, t.beta, t.gamma, t.dt_frame)
        out.append(GTraj(path=t.path, geo="d1", stack=t.stack, tag=t.tag, seed=t.seed, split=t.split,
                         c=c, amp=amp, theta=th, T=t.T, P_s=float(al.p_sat(t.alpha, t.beta, t.gamma)),
                         mat=(t.alpha, t.beta, t.gamma, t.dt_frame), box=None))
    return out


def index_d2u():
    """Dataset 2 (D1 geometry, independently drawn alpha/beta/gamma in 5 boxes) x unique seed 6000+i.
    Never used for training: every trajectory is a new material AND new noise."""
    cfg = {c["tag"]: c for c in json.load(open(os.path.join(HERE, "ood_grid.json")))["configs"]}
    out = []
    for p in sorted(glob.glob(os.path.join(SCR, "ferrox_d2useed", "extracted", "o*_s*.npy"))):
        tag, sd = os.path.basename(p)[:-4].split("_s"); c = cfg[tag]
        cv, amp, th = gm.ctx_d1(c["alpha"], c["beta"], c["gamma"], c["dt_frame"])
        out.append(GTraj(path=p, geo="d1", stack="d2u", tag=tag, seed=int(sd), split="test", c=cv, amp=amp,
                         theta=th, T=401, P_s=c["P_s"], box=c["subset"],
                         mat=(c["alpha"], c["beta"], c["gamma"], c["dt_frame"])))
    return out


def index_d3(stacks=("ext", "extu")):
    cfg = {c["tag"]: c for c in json.load(open(os.path.join(HERE, "ext_grid.json")))["configs"]}
    dx = json.load(open(os.path.join(HERE, "ext_grid.json")))["design"]["dx"]
    roots = {"ext": os.path.join(SCR, "ferrox_ext", "extracted"),
             "extu": os.path.join(SCR, "ferrox_ext_useed", "extracted")}
    out = []
    for st in stacks:
        for p in sorted(glob.glob(os.path.join(roots[st], "x*", "seed_*.npy"))):
            tag = os.path.basename(os.path.dirname(p)); c = cfg.get(tag)
            if c is None:
                continue
            seed = int(os.path.basename(p)[5:-4])
            cv, amp, th = gm.ctx_geom(c["pi_tau"], c["P_s"], c["r"], dx / (c["mu_nm"] * 1e-9),
                                      c["d_fe_over_mu"], c["eps_de"])
            out.append(GTraj(path=p, geo="d3", stack=st, tag=tag, seed=seed, split=c["split"],
                             c=cv, amp=amp, theta=th, T=401, P_s=c["P_s"], mat=None, box=None))
    return out


_MM = {}


def frames(tr, t0, n, stride=1):
    """(n, 1, Z, H, W) float32 normalised, frames t0, t0+stride, ..., t0+(n-1)*stride."""
    m = _MM.get(tr.path)
    if m is None:
        if len(_MM) > 256:
            _MM.clear()
        m = _MM[tr.path] = np.load(tr.path, mmap_mode="r")
    if m.shape[1] == 10:                     # released benchmark layout (T, Z, H, W), ferroelectric layers only
        a = np.asarray(m[t0:t0 + (n - 1) * stride + 1:stride], dtype=np.float32)
        return np.ascontiguousarray(a)[:, None] / al.P_NORM
    zl, zh = ZSL[tr.geo]
    a = np.asarray(m[:, :, zl:zh, t0:t0 + (n - 1) * stride + 1:stride], dtype=np.float32)   # (H,W,Z,n)
    return np.ascontiguousarray(a.transpose(3, 2, 0, 1))[:, None] / al.P_NORM


class Windows(torch.utils.data.Dataset):
    """k+1 consecutive frames from one geometry (so a batch has one shape)."""

    def __init__(self, trajs, k=4, t_stride=8, inc_end=60, inc_repeat=3, stride=1):
        """stride m: windows are frames t, t+m, ..., t+k*m and theta is multiplied by m -- a
        coarser time step. D3's frames are 2-5x smaller in growth units than D1's; m=4 puts them
        inside the frame-size range the model already handles."""
        self.trajs, self.k, self.stride = trajs, k, stride
        self.items = []
        for i, tr in enumerate(trajs):
            last = tr.T - 1 - k * stride
            self.items += [(i, t) for t in range(0, last, t_stride)]
            self.items += [(i, t) for t in range(0, min(inc_end, last))] * (inc_repeat - 1)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, j):
        i, t = self.items[j]
        tr = self.trajs[i]
        c = tr.c
        if self.stride != 1:     # theta is conditioning component 0 (log theta/theta_ref)
            c = c.clone(); c[0] = c[0] + float(np.log(self.stride))
        return (torch.from_numpy(frames(tr, t, self.k + 1, self.stride)[:, 0]), c,
                torch.tensor(tr.amp, dtype=torch.float32),
                torch.tensor(tr.theta * self.stride, dtype=torch.float32))
