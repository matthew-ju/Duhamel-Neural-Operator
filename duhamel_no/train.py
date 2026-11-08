#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""train.py -- one FNO, four ways of telling it the material, on AC / CH (2D) and RD / Burgers (1D).

Arms (identical backbone, optimiser, data, steps; only the input/output map and the conditioning differ):
  raw        u' = u + F(u; c_raw)                       c_raw = logs of the raw coefficients (+ frame dt)
  revin      m,s = mean/std of u (per sample); u' = m + s (u_hat + F(u_hat; c_raw))  [RevIN, ICLR'22]
  units_amp  v = u/U;  v' = v + F(v; c_dimless incl. log theta);  u' = U v'           (amplitude unit only)
  units      v = u/U;  v' = v + theta * F(v; c_dimless);           u' = U v'          (amplitude + time unit)
U, theta and the dimensionless numbers come from the textbook units of each equation (see prep/gen scripts).
Training: k-step pushforward unroll, relative L2 per step. Evaluation: full autoregressive rollout from the
first frame on every test split; relative L2 (final frame and rollout mean), amplitude error, pattern corr.
"""
import argparse, json, math, os, time
import numpy as np
import torch
import torch.nn as nn

D = os.path.join(os.environ["SCRATCH"], "ml_units")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


# ------------------------------------------------------------------ data
def load_split(system, split):
    z = np.load(os.path.join(D, "%s_%s.npz" % (system, split)))
    u = z["u"]                                          # (N, T+1, *space)
    if system in ("ac", "ch"):
        raw = np.stack([np.log(-z["a"]), np.log(z["b"]), np.log(z["M"]), np.log(z["e"]), np.log(z["dtf"])], 1)
        dml = np.stack([np.log(z["theta"]), z["pih"]], 1)
        U, theta = z["U"], z["theta"]
    else:                                                # rd / burgers: prepared by prep_1d.py
        raw, dml, U, theta = z["raw"], z["dml"], z["U"], z["theta"]
    return dict(u=u.astype(np.float32), raw=raw.astype(np.float32), dml=dml.astype(np.float32),
                U=np.asarray(U, np.float32), theta=np.asarray(theta, np.float32))


# ------------------------------------------------------------------ model
class Spectral(nn.Module):
    def __init__(self, c, modes, nd):
        super().__init__()
        self.nd, self.modes = nd, modes
        shape = (c, c) + ((2 * modes, modes) if nd == 2 else (modes,))
        self.w = nn.Parameter(torch.randn(*shape, dtype=torch.cfloat) / (c * c))

    def forward(self, x):
        if self.nd == 1:
            xf = torch.fft.rfft(x); o = torch.zeros_like(xf)
            o[..., :self.modes] = torch.einsum("bix,iox->box", xf[..., :self.modes], self.w)
            return torch.fft.irfft(o, n=x.shape[-1])
        xf = torch.fft.rfft2(x); o = torch.zeros_like(xf); m = self.modes
        o[:, :, :m, :m] = torch.einsum("bixy,ioxy->boxy", xf[:, :, :m, :m], self.w[:, :, :m])
        o[:, :, -m:, :m] = torch.einsum("bixy,ioxy->boxy", xf[:, :, -m:, :m], self.w[:, :, m:])
        return torch.fft.irfft2(o, s=x.shape[-2:])


class FNO(nn.Module):
    def __init__(self, nd, dcond, width=32, modes=12, layers=4):
        super().__init__()
        self.nd = nd
        conv = nn.Conv1d if nd == 1 else nn.Conv2d
        self.lift = conv(1 + nd, width, 1)
        self.spec = nn.ModuleList([Spectral(width, modes, nd) for _ in range(layers)])
        self.pw = nn.ModuleList([conv(width, width, 1) for _ in range(layers)])
        self.film = nn.Sequential(nn.Linear(dcond, 64), nn.GELU(), nn.Linear(64, 2 * width * layers))
        nn.init.zeros_(self.film[-1].weight); nn.init.zeros_(self.film[-1].bias)
        self.proj = nn.Sequential(conv(width, 64, 1), nn.GELU(), conv(64, 1, 1))
        self.width, self.layers = width, layers

    def forward(self, x, c):                 # x (B, *space), c (B, dcond)
        B, sp = x.shape[0], x.shape[1:]
        grids = torch.meshgrid(*[torch.linspace(0, 1, s, device=x.device) for s in sp], indexing="ij")
        h = self.lift(torch.cat([x[:, None]] + [g.expand(B, *sp)[:, None] for g in grids], 1))
        f = self.film(c).view(B, self.layers, 2, self.width, *([1] * self.nd))
        for i, (s, p) in enumerate(zip(self.spec, self.pw)):
            h = (s(h) + p(h)) * (1 + f[:, i, 0]) + f[:, i, 1]
            if i < self.layers - 1:
                h = nn.functional.gelu(h)
        return self.proj(h)[:, 0]


class Stepper(nn.Module):
    def __init__(self, arm, nd, cmean, cstd, n_groups=3):
        super().__init__()
        self.arm = arm
        self.register_buffer("cm", torch.as_tensor(cmean)); self.register_buffer("cs", torch.as_tensor(cstd))
        if arm == "learned":
            # LEARNED UNITS: log U, log theta and the FiLM groups are LINEAR in the log raw coefficients
            # (monomials = what dimensional analysis produces), learned end to end from raw inputs only.
            d = len(cmean)
            self.wU = nn.Linear(d, 1); self.wT = nn.Linear(d, 1); self.wP = nn.Linear(d, n_groups)
            nn.init.zeros_(self.wU.weight); nn.init.zeros_(self.wU.bias)
            nn.init.zeros_(self.wT.weight); nn.init.constant_(self.wT.bias, math.log(0.1))
            self.net = FNO(nd, n_groups)
        else:
            self.net = FNO(nd, len(cmean))

    def forward(self, u, cond, U, theta):
        c = (cond - self.cm) / self.cs
        red = tuple(range(1, u.dim()))
        if self.arm == "raw":
            return u + self.net(u, c)
        if self.arm == "revin":
            m = u.mean(red, keepdim=True); s = u.std(red, keepdim=True) + 1e-6
            h = (u - m) / s
            return m + s * (h + self.net(h, c))
        sh = (-1,) + (1,) * (u.dim() - 1)
        if self.arm == "learned":
            lc = cond - self.cm                         # centred logs, NOT divided by the spread: keeps monomials exact
            Ul = torch.exp(self.wU(lc)).view(sh); th = torch.exp(self.wT(lc)).view(sh)
            v = u / Ul
            return Ul * (v + th * self.net(v, self.wP(lc)))
        v = u / U.view(sh)
        dv = self.net(v, c)
        if self.arm == "units":
            dv = theta.view(sh) * dv
        return U.view(sh) * (v + dv)


# ------------------------------------------------------------------ train / eval
def rel(a, b):
    red = tuple(range(1, a.dim()))
    return ((a - b) ** 2).sum(red) / ((b ** 2).sum(red) + 1e-12)


@torch.no_grad()
def evaluate(model, S, dev, arm, bs=32):
    rows = []
    for lo in range(0, len(S["u"]), bs):
        sl = slice(lo, lo + bs)
        u = torch.from_numpy(S["u"][sl]).to(dev)
        cond = torch.from_numpy(S["raw" if arm in ("raw", "revin", "learned") else "dml"][sl]).to(dev)
        U = torch.from_numpy(S["U"][sl]).to(dev); th = torch.from_numpy(S["theta"][sl]).to(dev)
        x = u[:, 0]; rl = []
        for t in range(1, u.shape[1]):
            x = torch.nan_to_num(model(x, cond, U, th), nan=0.0, posinf=1e3, neginf=-1e3).clamp(-1e3, 1e3)
            rl.append(rel(x, u[:, t]).sqrt())
        rl = torch.stack(rl, 1)
        red = tuple(range(1, x.dim()))
        g = u[:, -1]
        amp = (x.pow(2).mean(red).sqrt() / g.pow(2).mean(red).sqrt() - 1).abs()
        xc, gc = x - x.mean(red, keepdim=True), g - g.mean(red, keepdim=True)
        corr = (xc * gc).sum(red) / (xc.pow(2).sum(red).sqrt() * gc.pow(2).sum(red).sqrt() + 1e-12)
        for i in range(len(x)):
            rows.append(dict(rel_final=float(rl[i, -1]), rel_mean=float(rl[i].mean()), amp=float(amp[i]), corr=float(corr[i])))
    agg = {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}
    agg["rel_final_median"] = float(np.median([r["rel_final"] for r in rows]))
    return agg, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", required=True)
    ap.add_argument("--arm", choices=["raw", "revin", "units_amp", "units", "learned"], required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=10000)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--ntrain", type=int, default=0, help="use only the first n training trajectories")
    ap.add_argument("--tests", default="id,ood_units,ood_units_absnoise,ood_inv")
    ap.add_argument("--tag", default="")
    ap.add_argument("--train_split", default="train")
    a = ap.parse_args()
    torch.manual_seed(a.seed); rng = np.random.default_rng(a.seed)
    dev = torch.device("cuda")
    tr = load_split(a.system, a.train_split)
    if a.ntrain:
        tr = {k: v[:a.ntrain] for k, v in tr.items()}
    ckey = "raw" if a.arm in ("raw", "revin", "learned") else "dml"
    cm, cs = tr[ckey].mean(0), tr[ckey].std(0) + 1e-6
    nd = tr["u"].ndim - 2
    model = Stepper(a.arm, nd, cm, cs).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.steps, pct_start=0.05)
    U = torch.from_numpy(tr["u"]).to(dev)                       # whole training set on GPU (small)
    C = torch.from_numpy(tr[ckey]).to(dev); UU = torch.from_numpy(tr["U"]).to(dev); TH = torch.from_numpy(tr["theta"]).to(dev)
    T = U.shape[1]; t0 = time.time(); run = 0.0
    for step in range(1, a.steps + 1):
        i = torch.from_numpy(rng.integers(len(U), size=a.bs)).to(dev)
        t = int(rng.integers(0, T - a.k))
        x = U[i, t]; loss = 0.0
        for q in range(1, a.k + 1):
            x = model(x, C[i], UU[i], TH[i])
            loss = loss + rel(x, U[i, t + q]).mean()
        loss = loss / a.k
        opt.zero_grad(set_to_none=True); loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sch.step()
        run += float(loss)
        if step % 1000 == 0:
            print("step %d loss %.4f  %.1f min" % (step, run / 1000, (time.time() - t0) / 60), flush=True); run = 0.0
    model.eval()
    res = dict(vars(a), train_minutes=(time.time() - t0) / 60, n_train=len(U))
    tag = "%s_%s_s%d%s%s" % (a.system, a.arm, a.seed, ("_n%d" % a.ntrain) if a.ntrain else "", a.tag)
    os.makedirs(os.path.join(OUT, "ckpt"), exist_ok=True)
    torch.save(dict(model=model.state_dict(), args=vars(a)), os.path.join(OUT, "ckpt", tag + ".pt"))
    for sp in a.tests.split(","):
        if not os.path.exists(os.path.join(D, "%s_%s.npz" % (a.system, sp))):
            continue
        agg, rows = evaluate(model, load_split(a.system, sp), dev, a.arm)
        res[sp] = agg; res[sp + "_rows"] = rows
        print("%-10s %s" % (sp, "  ".join("%s %.4f" % kv for kv in agg.items())), flush=True)
    json.dump(res, open(os.path.join(OUT, tag + ".json"), "w"))


if __name__ == "__main__":
    main()
