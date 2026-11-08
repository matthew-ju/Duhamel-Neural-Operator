#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""duhamel.py -- the Duhamel ("tangent") neural operator: exact learned linear semigroup + O(u^2) neural remainder.

One frame (exponential-Euler / ETD1 discretisation of the Duhamel formula, in the equation's own units v = u/U):

    v_{n+1} = exp(theta W) v_n  +  theta * phi_1(theta W) * N_psi(v_n)          (per Fourier mode)

  W(k; c)    learned LINEAR generator on ALL Fourier modes up to Nyquist (never truncated), a function of the
             dimensionless wavenumber k*mu and the material embedding:
                 W = cap( a0(c) + a1(c) (k mu)^2 + a2(c) (k mu)^4 + r(k mu, c) ),
             sign-free (unstable band allowed), smooth upper cap w_max, UNBOUNDED BELOW (stiff decay representable).
  phi_1(z)   (e^z - 1)/z.
  N_psi      the neural remainder, second order BY CONSTRUCTION:
                 N_psi(v) = P[ (A v) (.) H(v) ],
             A, P linear Fourier layers (bias-free), H a bias-free FNO with GELU (GELU(0)=0), FiLM GAINS only, no
             positional channels -> H(0) = 0 exactly and |H(v)| <= Lip(H) |v|, hence
                 |N_psi(v)| <= ||P|| ||A|| Lip(H) |v|^2 ,   N_psi(0) = 0 ,   D N_psi(0) = 0 .
             The bilinear form covers the physical nonlinearities: AC  -v^3 = (v) (.) (-v^2);  CH  lap(v^3) = P[v (.) v^2]
             with P = lap;  NS  u.grad(w) = sum_i (K_i w) (.) (d_i w).
Consequences (THEORY_fluctuation_seeded.md): in the linear phase the model IS its linear part up to O(|v|^2), so its
higher-Wiener-chaos output fraction is O(eps^2) instead of O(1); the hand-off time disappears.
Training/eval protocol identical to recipe.py (k=4 pushforward, relL2, growth-phase x3, exact symmetries as
augmentation, 20k steps, bs 16); plus a 2000-step warm start of W on frames 0 -> 1..5.
"""
import argparse, json, math, os, time
import numpy as np
import torch
import torch.nn as nn

import train as T
import recipe as R

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results_duhamel")
FR = [1, 2, 3, 5, 10, 20, 30, 50, 100]


class SpecLin(nn.Module):
    """bias-free Fourier layer: truncated spectral conv + pointwise linear (x -> 0 maps to 0)."""
    def __init__(self, cin, cout, modes):
        super().__init__()
        self.m = modes
        s = 1.0 / (cin * cout)
        self.w1 = nn.Parameter(s * torch.randn(cin, cout, modes, modes, dtype=torch.cfloat))
        self.w2 = nn.Parameter(s * torch.randn(cin, cout, modes, modes, dtype=torch.cfloat))
        self.pw = nn.Conv2d(cin, cout, 1, bias=False)

    def forward(self, x):
        B, C, H, W = x.shape; m = min(self.m, H // 2, W // 2 + 1)
        xf = torch.fft.rfft2(x)
        o = torch.zeros(B, self.w1.shape[1], H, W // 2 + 1, dtype=xf.dtype, device=x.device)
        o[:, :, :m, :m] = torch.einsum("bixy,ioxy->boxy", xf[:, :, :m, :m], self.w1[:, :, :m, :m])
        o[:, :, -m:, :m] = torch.einsum("bixy,ioxy->boxy", xf[:, :, -m:, :m], self.w2[:, :, :m, :m])
        return torch.fft.irfft2(o, s=(H, W)) + self.pw(x)


class Remainder(nn.Module):
    """N(v) = P[(A v) (.) H(v)], exactly O(|v|^2)."""
    def __init__(self, emb=16, width=48, layers=4, modes=16, ch=16, gmax=0.5, free=False):
        super().__init__()
        # free=True: ABLATION of the O(u^2) constraint -- biases everywhere plus an additive read-out Q(h), so N(0) != 0
        # and DN(0) != 0 (a generic FNO residual, Thm 2 class (a))
        self.lift = nn.Conv2d(1, width, 1, bias=free)
        self.blocks = nn.ModuleList([SpecLin(width, width, modes) for _ in range(layers)])
        if free:
            for blk in self.blocks:
                blk.pw = nn.Conv2d(width, width, 1, bias=True)
        self.hout = nn.Conv2d(width, ch, 1, bias=free)
        self.A = SpecLin(1, ch, modes)
        self.P = SpecLin(ch, 1, modes)
        self.Q = nn.Conv2d(width, 1, 1) if free else None
        if free:
            nn.init.zeros_(self.Q.weight); nn.init.zeros_(self.Q.bias)
        self.film = nn.Sequential(nn.Linear(emb, 64, bias=False), nn.GELU(), nn.Linear(64, layers * width, bias=False))
        nn.init.zeros_(self.film[-1].weight)
        nn.init.zeros_(self.P.pw.weight); nn.init.zeros_(self.P.w1); nn.init.zeros_(self.P.w2)   # starts as N = 0
        self.layers, self.width, self.gmax = layers, width, gmax

    def forward(self, v, e):
        x = v[:, None]
        g = self.gmax * torch.tanh(self.film(e)).view(-1, self.layers, self.width, 1, 1)
        h = self.lift(x)
        for i, blk in enumerate(self.blocks):
            h = nn.functional.gelu(blk(h) * (1 + g[:, i]))
        Hv = self.hout(h)
        out = self.P(self.A(x) * Hv)[:, 0]
        return out if self.Q is None else out + self.Q(h)[:, 0]


class Generator(nn.Module):
    """W(k mu; c): polynomial-in-(k mu)^2 dispersion relation + bounded correction, capped above only."""
    def __init__(self, emb=16, wmax=2.0):
        super().__init__()
        self.coef = nn.Sequential(nn.Linear(emb, 64), nn.GELU(), nn.Linear(64, 3))
        self.corr = nn.Sequential(nn.Linear(1 + emb, 64), nn.GELU(), nn.Linear(64, 64), nn.GELU(), nn.Linear(64, 1))
        nn.init.zeros_(self.coef[-1].weight); nn.init.zeros_(self.coef[-1].bias)
        nn.init.zeros_(self.corr[-1].weight); nn.init.zeros_(self.corr[-1].bias)
        self.wmax = wmax

    def forward(self, kmu, e):                          # kmu (B,H,Wf), e (B,emb)
        a = self.coef(e)                                # (B,4)
        q = kmu ** 2
        poly = a[:, 0, None, None] + a[:, 1, None, None] * q + a[:, 2, None, None] * q ** 2
        ee = e[:, None, None, :].expand(*kmu.shape, e.shape[-1])
        r = 2.0 * torch.tanh(self.corr(torch.cat([torch.tanh(kmu)[..., None], ee], -1))[..., 0])
        raw = poly + r
        return self.wmax - nn.functional.softplus(self.wmax - raw)        # smooth min(raw, wmax), unbounded below


def phi1(z):
    small = z.abs() < 1e-4
    zs = torch.where(small, torch.ones_like(z), z)
    return torch.where(small, 1 + z / 2 + z * z / 6, torch.expm1(zs) / zs)


class Duhamel(nn.Module):
    def __init__(self, n, cref, guard, emb=16, wmax=2.0, pih_const=None, free_rem=False):
        super().__init__()
        self.pih_const = pih_const          # NS: no wall width; use kmu = k / k0 (pih = dx * k0)
        self.substeps = 1                   # Duhamel quadrature: m exponential-Euler sub-steps of theta/m per frame
        self.register_buffer("cref", torch.as_tensor(cref))
        self.trunk = R.Trunk(len(cref), emb)
        self.gen = Generator(emb, wmax)
        self.rem = Remainder(emb, free=free_rem)
        k1 = 2 * math.pi * torch.fft.fftfreq(n, d=2 * math.pi / n)
        kk = (k1.view(-1, 1) ** 2 + k1.view(1, -1)[:, : n // 2 + 1] ** 2).sqrt()
        self.register_buffer("k", kk); self.n = n; self.guard = guard
        self.use_rem = True
        self.trunc_w = 0                    # ABLATION: >0 -> W learned only in the FNO mode box, one learned constant above

    def set_trunc(self, K, emb=16):
        self.trunc_w = K
        box = torch.zeros(self.n, self.n // 2 + 1, dtype=torch.bool)
        box[:K, :K] = True; box[-K:, :K] = True                     # the same box SpecLin keeps
        self.register_buffer("box", box.to(self.k.device))
        self.w_inf = nn.Linear(emb, 1).to(self.k.device)            # FNO-like: one multiplier for every mode above cutoff

    def symbol(self, c, pih):                           # pih = dx/mu -> k mu = k dx / pih
        e = self.trunk(c - self.cref)
        dx = 2 * math.pi / self.n
        kmu = self.k[None] * dx / pih.view(-1, 1, 1)
        W = self.gen(kmu, e)
        if self.trunc_w:
            wi = self.gen.wmax - nn.functional.softplus(self.gen.wmax - self.w_inf(e))[:, :, None]
            W = torch.where(self.box[None], W, wi)
        return W, e

    def forward(self, u, c, U, theta, pih):
        sh = (-1, 1, 1)
        if self.pih_const is not None:
            pih = torch.full_like(theta, self.pih_const)
        v = u / U.view(sh)
        v_in = v
        W, e = self.symbol(c, pih)
        h = theta.view(sh) / self.substeps
        z = h * W; E = torch.exp(z); P1 = h * phi1(z)
        for _ in range(self.substeps):
            vh = E * torch.fft.rfft2(v)
            if self.use_rem:
                nv = self.rem(v, e)
                if getattr(self, "conserve", False):
                    # <N(v), v> = 0 exactly (2D NS advection conserves enstrophy); the projection keeps N = O(|v|^2)
                    nv = nv - (nv * v).sum((1, 2), keepdim=True) / (v * v).sum((1, 2), keepdim=True).clamp_min(1e-20) * v
                vh = vh + P1 * torch.fft.rfft2(nv)
            v = torch.fft.irfft2(vh, s=v.shape[-2:])
        if getattr(self, "zclip", False):
            # decaying 2D NS: enstrophy is non-increasing in the TRUE dynamics (dZ/dt = -2 nu ||grad w||^2 <= 0), so the
            # step may not raise ||w||; exact for the truth, unlike a proxy energy
            ratio = (v_in.pow(2).sum((1, 2), keepdim=True) / v.pow(2).sum((1, 2), keepdim=True).clamp_min(1e-30)).sqrt()
            v = v * ratio.clamp(max=1.0)
        if self.guard > 0:
            v = v.clamp(-self.guard, self.guard)
        return U.view(sh) * v


def pih_of(c):
    return c[:, 1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", required=True)          # ac, ch, ac32, ac128, ...
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--warm", type=int, default=2000)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--tests", default="id,ood_units,ood_inv")
    ap.add_argument("--substeps", type=int, default=1)
    ap.add_argument("--conserve", action="store_true", help="project the neural term so <N(v), v> = 0 (enstrophy-conserving)")
    ap.add_argument("--zclip", action="store_true", help="never increase the field's L2 norm in a step (enstrophy monotonicity; decaying NS only)")
    ap.add_argument("--wmax", type=float, default=2.0, help="upper cap of the linear generator (0 = dissipative-only, for systems without a linear instability)")
    ap.add_argument("--free_rem", action="store_true", help="ABLATION: unconstrained neural term (biases + additive read-out; not O(u^2))")
    ap.add_argument("--trunc_w", type=int, default=0, help="ABLATION: linear generator only inside the K-mode box, one constant above (FNO-like)")
    a = ap.parse_args()
    base = "".join(ch for ch in a.system if not ch.isdigit())
    torch.manual_seed(a.seed); rng = np.random.default_rng(a.seed); dev = torch.device("cuda")
    tr = T.load_split(a.system, "train")
    cref = np.median(tr["dml"], 0).astype(np.float32)
    n = tr["u"].shape[-1]
    model = Duhamel(n, cref, R.GUARD.get(base, 1.35), wmax=a.wmax, pih_const=(2 * math.pi / n) * 4.0 if base == "nsd" else None,
                    free_rem=a.free_rem).to(dev)
    if a.trunc_w:
        model.set_trunc(a.trunc_w)
    model.substeps = a.substeps
    model.conserve = a.conserve
    model.zclip = a.zclip
    U = torch.from_numpy(tr["u"]).to(dev); C = torch.from_numpy(tr["dml"]).to(dev)
    UU = torch.from_numpy(tr["U"]).to(dev); TH = torch.from_numpy(tr["theta"]).to(dev)
    N, Tn = U.shape[0], U.shape[1]
    t0 = time.time()
    # ---- stage 1: the linear generator alone on frames 0 -> 1..5 (the remainder is O(u^2) ~ 0 there)
    model.use_rem = False
    opt = torch.optim.AdamW(list(model.trunk.parameters()) + list(model.gen.parameters())
                            + (list(model.w_inf.parameters()) if a.trunc_w else []), lr=3e-3)
    for step in range(a.warm):
        i = torch.from_numpy(rng.integers(N, size=32)).to(dev)
        x = U[i, 0]; loss = 0.0
        for q in range(1, 6):
            x = model(x, C[i], UU[i], TH[i], pih_of(C[i])); loss = loss + T.rel(x, U[i, q]).mean()
        opt.zero_grad(set_to_none=True); (loss / 5).backward(); opt.step()
    print("warm-start linear loss %.4f  %.1f min" % (float(loss) / 5, (time.time() - t0) / 60), flush=True)
    model.use_rem = True
    # ---- stage 2: joint, same protocol as recipe.py
    items = [(i, t) for i in range(N) for t in range(0, Tn - a.k, 2)]
    items += [(i, t) for i in range(N) for t in range(0, int(0.4 * (Tn - a.k)))] * 2
    items = np.array(items)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.steps, pct_start=0.05)
    run = 0.0
    for step in range(1, a.steps + 1):
        sel = items[rng.integers(len(items), size=a.bs)]
        i = torch.from_numpy(sel[:, 0]).to(dev)
        win = torch.stack([U[int(ii), int(tt): int(tt) + a.k + 1] for ii, tt in sel])
        win = R.augment(win, R.SYM.get(base, "none"))
        x = win[:, 0]; loss = 0.0
        for q in range(1, a.k + 1):
            x = model(x, C[i], UU[i], TH[i], pih_of(C[i])); loss = loss + T.rel(x, win[:, q]).mean()
        loss = loss / a.k
        opt.zero_grad(set_to_none=True); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sch.step()
        run += float(loss.detach())
        if step % 1000 == 0:
            print("step %d loss %.4f  %.1f min" % (step, run / 1000, (time.time() - t0) / 60), flush=True); run = 0.0
    model.eval()
    res = dict(vars(a), minutes=(time.time() - t0) / 60, n_train=N)
    os.makedirs(os.path.join(OUT, "ckpt"), exist_ok=True)
    tag = "%s_duhamel_s%d%s%s%s" % (a.system, a.seed, "_sub%d" % a.substeps if a.substeps > 1 else "", "_cons" if a.conserve else "",
                                    "_diss" if a.wmax <= 0 else "") + ("_zclip" if a.zclip else "") \
        + ("_free" if a.free_rem else "") + ("_trunc%d" % a.trunc_w if a.trunc_w else "")
    torch.save(dict(model=model.state_dict(), args=vars(a), cref=cref, pih_const=model.pih_const), os.path.join(OUT, "ckpt", tag + ".pt"))
    for sp in a.tests.split(","):
        if not os.path.exists(os.path.join(T.D, "%s_%s.npz" % (a.system, sp))):
            continue
        res[sp] = evaluate(model, T.load_split(a.system, sp), dev)
        print("%-10s %s" % (sp, json.dumps({k: round(v, 4) for k, v in res[sp].items()})), flush=True)
    json.dump(res, open(os.path.join(OUT, tag + ".json"), "w"), indent=1)


@torch.no_grad()
def evaluate(model, S, dev, step_fn=None):
    """final relL2 / corr, plus relL2 / corr at early frames FR (per-trajectory means)."""
    rows = []
    for lo in range(0, len(S["u"]), 32):
        sl = slice(lo, lo + 32)
        u = torch.from_numpy(S["u"][sl]).to(dev); c = torch.from_numpy(S["dml"][sl]).to(dev)
        Uu = torch.from_numpy(S["U"][sl]).to(dev); th = torch.from_numpy(S["theta"][sl]).to(dev)
        x = u[:, 0]; preds = []
        for t in range(1, u.shape[1]):
            x = torch.nan_to_num(model(x, c, Uu, th, c[:, 1]) if step_fn is None else step_fn(x, c, Uu, th, t)).clamp(-1e3, 1e3)
            preds.append(x)
        P = torch.stack(preds, 1); G = u[:, 1:]
        rows.append(metrics(P, G))
    return {k: float(np.nanmean(np.concatenate([r[k] for r in rows]))) for k in rows[0]}


def metrics(P, G):
    red = (2, 3)
    rl = (((P - G) ** 2).sum(red).sqrt() / (G ** 2).sum(red).sqrt().clamp_min(1e-12)).cpu().numpy()
    Pc, Gc = P - P.mean(red, keepdim=True), G - G.mean(red, keepdim=True)
    cr = ((Pc * Gc).sum(red) / ((Pc ** 2).sum(red).sqrt() * (Gc ** 2).sum(red).sqrt() + 1e-12)).cpu().numpy()
    gstd = Gc.pow(2).mean(red).sqrt().cpu().numpy()
    cr = np.where(gstd > 1e-6, cr, np.nan)
    out = {"rel_final": rl[:, -1], "corr_final": cr[:, -1], "rel_mean": rl.mean(1)}
    for t in FR:
        if t <= rl.shape[1]:
            out["rel%d" % t] = rl[:, t - 1]; out["corr%d" % t] = cr[:, t - 1]
    return out


if __name__ == "__main__":
    main()
