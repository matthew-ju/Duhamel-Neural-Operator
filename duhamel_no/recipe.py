#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""recipe.py -- the FULL FerroX surrogate recipe (abg_model.CondFNO + output-switch SSM head), ported to 1D/2D.

What is carried over from the production FerroX model, piece by piece:
  * units: field divided by the equation's amplitude unit U; the frame interval enters as dimensionless theta
  * amplitude-aware input normalisation: the network sees v/rms(v) AND log10 rms(v) as a channel, so unlike
    RevIN it still knows how far the field is from its own saturation
  * FerroX update rule: v' = s * exp(tau g) * (v/s + tau * delta), tau = theta/theta_ref, g a bounded global gain
  * conditioning: a bias-free tanh trunk (exactly zero at the reference material) feeding zero-initialised input
    channels and tanh-bounded, bias-free FiLM on every FNO block
  * backbone: neuralop FNO, width 48, 4 layers, 16 Fourier modes per spatial axis
  * guard: |v| clipped at 1.35x saturation in reduced units (AC/CH/RD; off for Burgers, which has no saturation)
  * training: k=4 pushforward unroll, relative L2, growth-phase windows oversampled 3x, the equation's exact
    symmetries as augmentation (AC/CH: 8 dihedral maps x sign flip; RD: reflection; Burgers: (x,u)->(-x,-u))
  * output switch for the noise-seeded systems (AC/CH): a conditioned diagonal Fourier state-space model, whose
    per-mode growth rate is a learned function of the dimensionless wavenumber k*mu, produces frames 1-5; the FNO's
    own rollout from frame 0 is used for every later frame
"""
import argparse, json, math, os, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from neuralop.models import FNO

import train as T          # data loading + metrics shared with the arm study

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results_recipe")
SYM = {"ac": "d4sign", "ch": "d4sign", "rd": "flip", "burgers": "flipneg", "nsd": "d4pseudo", "nsf": "none"}
GUARD = {"ac": 1.35, "ch": 1.35, "rd": 1.35, "burgers": 0.0, "nsd": 0.0, "nsf": 0.0}
SSM_SYSTEMS = ("ac", "ch")


def base_of(system):              # "ac128" -> "ac": resolution variants share symmetries/guards
    return "".join(ch for ch in system if not ch.isdigit())


class Trunk(nn.Module):
    def __init__(self, d, emb=16):
        super().__init__()
        self.f = nn.Sequential(nn.Linear(d, 64, bias=False), nn.GELU(), nn.Linear(64, emb, bias=False))

    def forward(self, c):
        return torch.tanh(self.f(c))


class CondNet(nn.Module):
    def __init__(self, nd, emb=16, width=48, layers=4, modes=16, gmax=0.25, bmax=0.25):
        super().__init__()
        self.nd, self.emb, self.width, self.layers, self.gmax, self.bmax = nd, emb, width, layers, gmax, bmax
        self.fno = FNO(n_modes=(modes,) * nd, in_channels=1 + 1 + nd + emb, out_channels=2,
                       hidden_channels=width, n_layers=layers, positional_embedding=None)
        with torch.no_grad():
            self.fno.lifting.fcs[0].weight[:, -emb:] = 0.0
        self.film = nn.Sequential(nn.Linear(emb, 64, bias=False), nn.GELU(), nn.Linear(64, 2 * layers * width, bias=False))
        nn.init.zeros_(self.film[-1].weight)

    def forward(self, v, e):                          # v (B, *space) reduced units; e (B, emb)
        B, sp = v.shape[0], v.shape[1:]
        red = tuple(range(1, v.dim()))
        s = v.pow(2).mean(red, keepdim=True).sqrt().clamp_min(1e-12)
        grids = torch.meshgrid(*[torch.linspace(0, 1, n, device=v.device) for n in sp], indexing="ij")
        h = torch.cat([(v / s)[:, None], torch.log10(s).expand_as(v)[:, None]] + [g.expand(B, *sp)[:, None] for g in grids]
                      + [e.view(B, self.emb, *([1] * self.nd)).expand(B, self.emb, *sp)], 1)
        gb = self.film(e).view(B, self.layers, 2, self.width)
        f = self.fno
        h = f.lifting(h)
        for i in range(f.n_layers):
            h = f.fno_blocks(h, i)
            g = self.gmax * torch.tanh(gb[:, i, 0]).view(B, -1, *([1] * self.nd))
            b = self.bmax * torch.tanh(gb[:, i, 1]).view(B, -1, *([1] * self.nd))
            h = h * (1 + g) + b
        out = f.projection(h)
        gain = out[:, 1:].mean(tuple(range(1, out.dim())), keepdim=True).clamp(-1.0, 0.5)[:, 0]
        return out[:, 0], gain, s


class Recipe(nn.Module):
    def __init__(self, nd, cref, theta_ref, guard):
        super().__init__()
        self.register_buffer("cref", torch.as_tensor(cref)); self.theta_ref = float(theta_ref); self.guard = guard
        self.trunk = Trunk(len(cref)); self.net = CondNet(nd)

    def forward(self, u, c, U, theta):
        sh = (-1,) + (1,) * (u.dim() - 1)
        v = u / U.view(sh)
        delta, g, s = self.net(v, self.trunk(c - self.cref))
        tau = (theta / self.theta_ref).view(sh)
        v = s * torch.exp(tau * g) * (v / s + tau * delta)
        if self.guard > 0:
            v = v.clamp(-self.guard, self.guard)
        return U.view(sh) * v


class SSMHead(nn.Module):
    """exp(theta * w(k mu, emb)) per Fourier mode, w capped to [-4, 1.5] (growth ~1 per unit time by definition)."""
    def __init__(self, nd, n, cref, wmin=-4.0):
        super().__init__()
        self.wmin = wmin            # -4 = the original clamp (cannot represent stiff decay: gain >= exp(-4 theta)); None = fixed
        self.register_buffer("cref", torch.as_tensor(cref)); self.trunk = Trunk(len(cref))
        k1 = 2 * math.pi * torch.fft.fftfreq(n, d=2 * math.pi / n)
        kk = (k1.view(-1, 1) ** 2 + k1.view(1, -1)[:, : n // 2 + 1] ** 2).sqrt() if nd == 2 else k1[: n // 2 + 1].abs()
        self.register_buffer("k", kk); self.nd, self.n = nd, n
        self.w = nn.Sequential(nn.Linear(1 + 16, 64), nn.GELU(), nn.Linear(64, 64), nn.GELU(), nn.Linear(64, 1))
        if wmin is None:            # add a linear (k mu)^2 path so stiff decay ~ -(k mu)^2 (or ^4) is reachable
            self.q = nn.Sequential(nn.Linear(16, 32), nn.GELU(), nn.Linear(32, 2))
            nn.init.zeros_(self.q[-1].weight); nn.init.zeros_(self.q[-1].bias)

    def forward(self, u, c, pih, theta):                # pih = dx/mu -> k*mu = k * dx / pih
        B = u.shape[0]; dx = 2 * math.pi / self.n
        kmu = (self.k[None] * dx / pih.view(B, *([1] * self.nd))).unsqueeze(-1)
        e = self.trunk(c - self.cref).view(B, *([1] * self.nd), 16).expand(*kmu.shape[:-1], 16)
        raw = self.w(torch.cat([kmu, e], -1))[..., 0]
        if self.wmin is None:
            # capped above only (smooth), unbounded below; a (k mu)^2, (k mu)^4 path so stiff decay is reachable
            a = self.q(self.trunk(c - self.cref))                                  # (B,2)
            q2 = kmu[..., 0] ** 2
            raw = raw + a[:, 0].view(B, *([1] * self.nd)) * q2 + a[:, 1].view(B, *([1] * self.nd)) * q2 ** 2
            w = 1.5 - nn.functional.softplus(1.5 - raw)
        else:
            w = raw.clamp(self.wmin, 1.5)
        mult = torch.exp(theta.view(B, *([1] * self.nd)) * w)
        if self.nd == 2:
            return torch.fft.irfft2(torch.fft.rfft2(u) * mult, s=u.shape[-2:])
        return torch.fft.irfft(torch.fft.rfft(u) * mult, n=u.shape[-1])


def augment(x, kind):
    """x (B, K+1, *space); one random exact symmetry per sample."""
    x = x.clone()
    for b in range(x.shape[0]):
        if kind == "d4sign":
            k = int(torch.randint(4, (1,)))
            y = torch.rot90(x[b], k, dims=(-2, -1)) if k else x[b]
            if torch.rand(1) < 0.5:
                y = torch.flip(y, dims=(-1,))
            if torch.rand(1) < 0.5:
                y = -y
            x[b] = y
        elif kind == "d4pseudo":                   # vorticity is a pseudo-scalar: rotations keep it, reflections negate it
            k = int(torch.randint(4, (1,)))
            y = torch.rot90(x[b], k, dims=(-2, -1)) if k else x[b]
            if torch.rand(1) < 0.5:
                y = -torch.flip(y, dims=(-1,))
            x[b] = y
        elif kind == "none":
            pass
        elif torch.rand(1) < 0.5:
            x[b] = torch.flip(x[b], dims=(-1,)) * (-1 if kind == "flipneg" else 1)
    return x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--ssm_steps", type=int, default=3000)
    ap.add_argument("--tests", default="id,ood_units,ood_units_absnoise,ood_inv")
    ap.add_argument("--no_aug", action="store_true")
    ap.add_argument("--ssm_fix", action="store_true", help="SSM head without the w >= -4 clamp")
    ap.add_argument("--ssm_only_from", default="", help="load this recipe ckpt, keep its FNO, retrain only the SSM head")
    a = ap.parse_args()
    torch.manual_seed(a.seed); rng = np.random.default_rng(a.seed); dev = torch.device("cuda")
    tr = T.load_split(a.system, "train")
    nd = tr["u"].ndim - 2
    cref = np.median(tr["dml"], 0).astype(np.float32); theta_ref = float(np.median(tr["theta"]))
    model = Recipe(nd, cref, theta_ref, GUARD[base_of(a.system)]).to(dev)
    U = torch.from_numpy(tr["u"]).to(dev); C = torch.from_numpy(tr["dml"]).to(dev)
    UU = torch.from_numpy(tr["U"]).to(dev); TH = torch.from_numpy(tr["theta"]).to(dev)
    N, Tn = U.shape[0], U.shape[1]
    items = [(i, t) for i in range(N) for t in range(0, Tn - a.k, 2)]
    items += [(i, t) for i in range(N) for t in range(0, int(0.4 * (Tn - a.k)))] * 2     # growth phase x3
    items = np.array(items)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.steps, pct_start=0.05)
    t0 = time.time(); run = 0.0
    if a.ssm_only_from:
        ck = torch.load(a.ssm_only_from, map_location=dev, weights_only=False)
        model.load_state_dict({k: v for k, v in ck["model"].items() if not k.endswith("_metadata")}, strict=False)
        a.steps_done = 0
    for step in range(1, (0 if a.ssm_only_from else a.steps) + 1):
        sel = items[rng.integers(len(items), size=a.bs)]
        i = torch.from_numpy(sel[:, 0]).to(dev)
        win = torch.stack([U[int(ii), int(tt): int(tt) + a.k + 1] for ii, tt in sel])
        if not a.no_aug:
            win = augment(win, SYM[base_of(a.system)])
        x = win[:, 0]; loss = 0.0
        for q in range(1, a.k + 1):
            x = model(x, C[i], UU[i], TH[i]); loss = loss + T.rel(x, win[:, q]).mean()
        loss = loss / a.k
        opt.zero_grad(set_to_none=True); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sch.step()
        run += float(loss.detach())
        if step % 1000 == 0:
            print("step %d loss %.4f  %.1f min" % (step, run / 1000, (time.time() - t0) / 60), flush=True); run = 0.0
    ssm = None
    if base_of(a.system) in SSM_SYSTEMS:                        # the linear-phase head, trained on frames 0 -> 1..5 only
        ssm = SSMHead(nd, U.shape[-1], cref, wmin=None if a.ssm_fix else -4.0).to(dev)
        o2 = torch.optim.AdamW(ssm.parameters(), lr=1e-3); PIH = C[:, 1]
        for step in range(a.ssm_steps):
            i = torch.from_numpy(rng.integers(N, size=32)).to(dev)
            x = U[i, 0]; loss = 0.0
            for q in range(1, 6):
                x = ssm(x, C[i], PIH[i], TH[i]); loss = loss + T.rel(x, U[i, q]).mean()
            o2.zero_grad(set_to_none=True); (loss / 5).backward(); o2.step()
    model.eval()
    res = dict(vars(a), minutes=(time.time() - t0) / 60, n_train=N)
    os.makedirs(os.path.join(OUT, "ckpt"), exist_ok=True)
    tag = "%s_recipe_s%d%s%s" % (a.system, a.seed, "_noaug" if a.no_aug else "", "_ssmfix" if a.ssm_fix else "")
    torch.save(dict(model=model.state_dict(), ssm=ssm.state_dict() if ssm else None, args=vars(a),
                    ssm_wmin=None if a.ssm_fix else -4.0), os.path.join(OUT, "ckpt", tag + ".pt"))
    for sp in a.tests.split(","):
        if not os.path.exists(os.path.join(T.D, "%s_%s.npz" % (a.system, sp))):
            continue
        S = T.load_split(a.system, sp); rows = []
        with torch.no_grad():
            for lo in range(0, len(S["u"]), 32):
                sl = slice(lo, lo + 32)
                u = torch.from_numpy(S["u"][sl]).to(dev); c = torch.from_numpy(S["dml"][sl]).to(dev)
                Uu = torch.from_numpy(S["U"][sl]).to(dev); th = torch.from_numpy(S["theta"][sl]).to(dev)
                x = u[:, 0]; xs = u[:, 0]; preds = []
                for t in range(1, u.shape[1]):
                    x = torch.nan_to_num(model(x, c, Uu, th)).clamp(-1e3, 1e3)
                    if ssm is not None and t <= 5:
                        xs = ssm(xs, c, c[:, 1], th); preds.append(xs)
                    else:
                        preds.append(x)
                P = torch.stack(preds, 1); G = u[:, 1:]
                red = tuple(range(2, P.dim()))
                rl = ((P - G) ** 2).sum(red).sqrt() / (G ** 2).sum(red).sqrt().clamp_min(1e-12)
                xf, gf = P[:, -1], G[:, -1]; r2 = tuple(range(1, xf.dim()))
                amp = (xf.pow(2).mean(r2).sqrt() / gf.pow(2).mean(r2).sqrt() - 1).abs()
                xc, gc = xf - xf.mean(r2, keepdim=True), gf - gf.mean(r2, keepdim=True)
                corr = (xc * gc).sum(r2) / (xc.pow(2).sum(r2).sqrt() * gc.pow(2).sum(r2).sqrt() + 1e-12)
                for j in range(len(xf)):
                    rows.append(dict(rel_final=float(rl[j, -1]), rel_mean=float(rl[j].mean()), amp=float(amp[j]), corr=float(corr[j])))
        res[sp] = {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}
        res[sp]["rel_final_median"] = float(np.median([r["rel_final"] for r in rows]))
        print("%-18s %s" % (sp, "  ".join("%s %.4f" % kv for kv in res[sp].items())), flush=True)
    json.dump(res, open(os.path.join(OUT, tag + ".json"), "w"), indent=1)


if __name__ == "__main__":
    main()
