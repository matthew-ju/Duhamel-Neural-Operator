#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Baseline neural PDE surrogates for the 2026-10 comparative benchmark (bench/), dimension-generic (1D / 2D).

Every model maps (x, cond) -> x_next with x (B, C, *S) and cond (B, D); the parameter vector is broadcast as
constant input channels (the conditioning used by PDEArena / The Well / PDEBench for parametric PDEs), and every
model predicts a residual (x_next = x + net(...)), except PDE-Refiner which follows its paper.

  fno         Li et al. ICLR 2021 (neuralop FNO, 16 modes, width 64, 4 layers)
  tfno        Kossaifi et al. (neuralop Tucker-factorised FNO) -- a baseline of The Well (NeurIPS 2024)
  ffno        Tran et al. ICLR 2023, factorised FNO (per-axis spectral mixing + FF residual), 8 layers width 64
  unet        Gupta & Brandstetter 2022 (PDEArena modern U-Net: residual GroupNorm/GELU blocks, mults 1,2,2,4)
  cnext       The Well's CNextU-Net (ConvNeXt blocks in a U-Net) -- the overall winner of The Well's one-step table
  cno         Raonic et al. NeurIPS 2023, Convolutional Neural Operator (U-Net with alias-aware up/down-sampled
              activations, implemented with Fourier resampling)
  transolver  Wu et al. ICML 2024, physics-attention transformer (slice tokens), 8 layers, width 192
  refiner     Lippe et al. NeurIPS 2023, PDE-Refiner: the U-Net above + K = 3 denoising refinement steps
"""
import math
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F


def convnd(nd):
    return nn.Conv1d if nd == 1 else nn.Conv2d


def poolnd(nd):
    return nn.AvgPool1d if nd == 1 else nn.AvgPool2d


def bcast(cond, S):
    return cond.view(*cond.shape, *([1] * len(S))).expand(*cond.shape, *S)


def npos(nd):
    return 0 if os.environ.get("BENCH_NOPOS") else 2 * nd


def posenc(x):
    """periodic coordinate channels sin/cos(2 pi x_i), (B, 2*nd, *S) -- lets translation-equivariant conv nets see
    fixed spatial structure (e.g. the Kolmogorov / Li-et-al. forcing pattern); FNO gets coordinates by default."""
    S = x.shape[2:]
    if os.environ.get("BENCH_NOPOS"):                    # load checkpoints trained before the coordinate-channel fix
        return x.new_zeros(x.shape[0], 0, *S)
    g = torch.stack(torch.meshgrid(*[torch.arange(n, device=x.device, dtype=torch.float32) * 2 * math.pi / n for n in S], indexing="ij"))
    return torch.cat([g.sin(), g.cos()], 0).unsqueeze(0).expand(x.shape[0], -1, *S)


# --------------------------------------------------------------------------------------------- FNO / TFNO
class NeuralopFNO(nn.Module):
    def __init__(self, nd, cin, cond, modes=16, width=64, layers=4, tucker=False):
        super().__init__()
        from neuralop.models import FNO, TFNO
        kw = dict(n_modes=(modes,) * nd, hidden_channels=width, in_channels=cin + cond, out_channels=cin, n_layers=layers)
        self.net = TFNO(factorization="tucker", rank=0.42, **kw) if tucker else FNO(**kw)

    def forward(self, x, c):
        return x + self.net(torch.cat([x, bcast(c, x.shape[2:])], 1))


# --------------------------------------------------------------------------------------------- F-FNO
class FactSpectral(nn.Module):
    """per-axis Fourier mixing, summed over axes (Tran et al. 2023)."""

    def __init__(self, nd, width, modes):
        super().__init__()
        self.nd, self.m = nd, modes
        s = 1 / (width * width)
        self.w = nn.ParameterList([nn.Parameter(s * torch.randn(width, width, modes, dtype=torch.cfloat)) for _ in range(nd)])

    def forward(self, x):
        out = 0
        for ax in range(self.nd):
            dim = 2 + ax
            xf = torch.fft.rfft(x.float(), dim=dim)
            m = min(self.m, xf.shape[dim])
            xf = xf.narrow(dim, 0, m)
            perm = list(range(xf.dim())); perm[dim], perm[-1] = perm[-1], perm[dim]
            xf = xf.permute(*perm)                                         # (..., m) on the last axis
            yf = torch.einsum("bi...m,iom->bo...m", xf, self.w[ax][..., :m])
            yf = yf.permute(*perm)
            full = list(yf.shape); full[dim] = x.shape[dim] // 2 + 1
            yz = torch.zeros(full, dtype=yf.dtype, device=x.device)
            yz.narrow(dim, 0, m).copy_(yf)
            out = out + torch.fft.irfft(yz, n=x.shape[dim], dim=dim)
        return out


class FFNO(nn.Module):
    def __init__(self, nd, cin, cond, modes=16, width=64, layers=8):
        super().__init__()
        C = convnd(nd)
        self.lift = C(cin + cond + nd, width, 1)
        self.spec = nn.ModuleList([FactSpectral(nd, width, modes) for _ in range(layers)])
        self.ff = nn.ModuleList([nn.Sequential(C(width, 2 * width, 1), nn.GELU(), C(2 * width, width, 1)) for _ in range(layers)])
        self.proj = nn.Sequential(C(width, 128, 1), nn.GELU(), C(128, cin, 1))
        self.nd = nd

    def forward(self, x, c):
        S = x.shape[2:]
        grid = torch.stack(torch.meshgrid(*[torch.linspace(0, 1, n, device=x.device) for n in S], indexing="ij"))
        h = self.lift(torch.cat([x, bcast(c, S), grid.unsqueeze(0).expand(x.shape[0], *grid.shape)], 1))
        for sp, ff in zip(self.spec, self.ff):
            h = h + ff(sp(h))
        return x + self.proj(h)


# --------------------------------------------------------------------------------------------- U-Nets
class ResBlock(nn.Module):
    def __init__(self, nd, cin, cout):
        super().__init__()
        C = convnd(nd)
        self.n1 = nn.GroupNorm(8, cin); self.c1 = C(cin, cout, 3, padding=1, padding_mode="circular")
        self.n2 = nn.GroupNorm(8, cout); self.c2 = C(cout, cout, 3, padding=1, padding_mode="circular")
        self.sk = C(cin, cout, 1) if cin != cout else nn.Identity()

    def forward(self, x):
        h = self.c1(F.gelu(self.n1(x)))
        h = self.c2(F.gelu(self.n2(h)))
        return h + self.sk(x)


class ConvNeXtBlock(nn.Module):
    def __init__(self, nd, ch):
        super().__init__()
        C = convnd(nd)
        self.dw = C(ch, ch, 7, padding=3, groups=ch, padding_mode="circular")
        self.norm = nn.GroupNorm(1, ch)                                     # = channel LayerNorm
        self.pw1 = C(ch, 4 * ch, 1); self.pw2 = C(4 * ch, ch, 1)
        self.gamma = nn.Parameter(1e-6 * torch.ones(ch))

    def forward(self, x):
        h = self.pw2(F.gelu(self.pw1(self.norm(self.dw(x)))))
        return x + self.gamma.view(1, -1, *([1] * (x.dim() - 2))) * h


class UNetGeneric(nn.Module):
    """U-Net with residual (PDEArena) or ConvNeXt (The Well) blocks, 2 blocks per level, 4 levels."""

    def __init__(self, nd, cin, cond, width=64, mults=(1, 2, 2, 4), block="res", extra_in=0):
        super().__init__()
        C = convnd(nd); self.nd = nd
        mk = (lambda a, b: ResBlock(nd, a, b)) if block == "res" else None
        self.inp = C(cin + cond + extra_in + npos(nd), width, 3, padding=1, padding_mode="circular")
        chs = [width * m for m in mults]
        self.down, self.ds = nn.ModuleList(), nn.ModuleList()
        prev = width
        for i, ch in enumerate(chs):
            if block == "res":
                self.down.append(nn.Sequential(ResBlock(nd, prev, ch), ResBlock(nd, ch, ch)))
            else:
                self.down.append(nn.Sequential(C(prev, ch, 1), ConvNeXtBlock(nd, ch), ConvNeXtBlock(nd, ch)))
            self.ds.append(C(ch, ch, 2, stride=2) if i < len(chs) - 1 else nn.Identity())
            prev = ch
        mid = chs[-1]
        self.mid = nn.Sequential(ResBlock(nd, mid, mid), ResBlock(nd, mid, mid)) if block == "res" else \
            nn.Sequential(ConvNeXtBlock(nd, mid), ConvNeXtBlock(nd, mid))
        self.up, self.us = nn.ModuleList(), nn.ModuleList()
        CT = nn.ConvTranspose1d if nd == 1 else nn.ConvTranspose2d
        for i, ch in reversed(list(enumerate(chs))):
            if block == "res":
                self.up.append(nn.Sequential(ResBlock(nd, prev + ch, ch), ResBlock(nd, ch, ch)))
            else:
                self.up.append(nn.Sequential(C(prev + ch, ch, 1), ConvNeXtBlock(nd, ch), ConvNeXtBlock(nd, ch)))
            self.us.append(CT(ch, chs[i - 1], 2, stride=2) if i > 0 else nn.Identity())
            prev = chs[i - 1] if i > 0 else ch
        self.out = nn.Sequential(nn.GroupNorm(8 if block == "res" else 1, prev), nn.GELU(), C(prev, cin, 1))

    def core(self, h):
        skips = []
        for blk, ds in zip(self.down, self.ds):
            h = blk(h); skips.append(h); h = ds(h)
        h = self.mid(h)
        for blk, us in zip(self.up, self.us):
            h = blk(torch.cat([h, skips.pop()], 1)); h = us(h)
        return self.out(h)

    def forward(self, x, c):
        return x + self.core(self.inp(torch.cat([x, bcast(c, x.shape[2:]), posenc(x)], 1)))


# --------------------------------------------------------------------------------------------- CNO
def fourier_resample(x, size):
    """band-limited resampling of a periodic signal to `size` points per spatial axis."""
    nd = x.dim() - 2
    if all(s == size for s in x.shape[2:]):
        return x
    dims = tuple(range(2, 2 + nd))
    X = torch.fft.rfftn(x.float(), dim=dims)
    N_in = x.shape[2]
    out_shape = [size] * nd
    Y = torch.zeros(*x.shape[:2], *out_shape[:-1], size // 2 + 1, dtype=X.dtype, device=x.device)
    m = min(N_in, size) // 2
    if nd == 1:
        Y[..., :m] = X[..., :m]
    else:
        Y[..., :m, :m] = X[..., :m, :m]
        Y[..., -m:, :m] = X[..., -m:, :m]
    y = torch.fft.irfftn(Y, s=out_shape, dim=dims)
    return y * (size ** nd) / (N_in ** nd)


class CNOAct(nn.Module):
    """alias-aware activation: upsample x2 (band-limited), LeakyReLU, downsample back."""

    def forward(self, x):
        n = x.shape[2]
        return fourier_resample(F.leaky_relu(fourier_resample(x, 2 * n), 0.1), n)


class CNOBlock(nn.Module):
    def __init__(self, nd, cin, cout):
        super().__init__()
        C = convnd(nd)
        self.c = C(cin, cout, 3, padding=1, padding_mode="circular"); self.n = nn.GroupNorm(8 if cout % 8 == 0 else 1, cout)
        self.act = CNOAct()

    def forward(self, x):
        return self.act(self.n(self.c(x)))


class CNO(nn.Module):
    def __init__(self, nd, cin, cond, width=48, levels=3, nres=2):
        super().__init__()
        C = convnd(nd)
        self.lift = CNOBlock(nd, cin + cond + npos(nd), width)
        chs = [width * 2 ** i for i in range(levels + 1)]
        self.enc = nn.ModuleList([CNOBlock(nd, chs[i], chs[i + 1]) for i in range(levels)])
        self.res = nn.ModuleList([nn.ModuleList([CNOBlock(nd, chs[i], chs[i]) for _ in range(nres)]) for i in range(levels + 1)])
        self.dec = nn.ModuleList([CNOBlock(nd, chs[i + 1] * (1 if i == levels - 1 else 2), chs[i]) for i in range(levels)])
        self.proj = C(2 * width, cin, 1)
        self.levels = levels

    def forward(self, x, c):
        n = x.shape[2]
        h = self.lift(torch.cat([x, bcast(c, x.shape[2:]), posenc(x)], 1))
        skips = []
        for i in range(self.levels):
            r = h
            for b in self.res[i]:
                r = r + b(r)
            skips.append(r)
            h = fourier_resample(self.enc[i](h), h.shape[2] // 2)
        for b in self.res[-1]:
            h = h + b(h)
        for i in reversed(range(self.levels)):
            h = fourier_resample(self.dec[i](h), h.shape[2] * 2)
            h = torch.cat([h, skips[i]], 1)
        return x + self.proj(h)


# --------------------------------------------------------------------------------------------- Transolver
class PhysicsAttention(nn.Module):
    def __init__(self, dim, heads=8, slices=32):
        super().__init__()
        self.h, self.dh, self.m = heads, dim // heads, slices
        self.temp = nn.Parameter(0.5 * torch.ones(1, heads, 1, 1))
        self.in_x = nn.Linear(dim, dim); self.in_fx = nn.Linear(dim, dim)
        self.slice = nn.Linear(self.dh, slices)
        self.q = nn.Linear(self.dh, self.dh, bias=False); self.k = nn.Linear(self.dh, self.dh, bias=False)
        self.v = nn.Linear(self.dh, self.dh, bias=False)
        self.out = nn.Linear(dim, dim)

    def forward(self, x):                                                   # x (B, N, dim)
        B, N, _ = x.shape
        fx = self.in_fx(x).view(B, N, self.h, self.dh).transpose(1, 2)       # (B,h,N,dh)
        xm = self.in_x(x).view(B, N, self.h, self.dh).transpose(1, 2)
        w = torch.softmax(self.slice(xm) / self.temp.clamp_min(0.05), dim=-1)  # (B,h,N,m)
        norm = w.sum(2, keepdim=True).transpose(-1, -2)                      # (B,h,m,1)
        tok = (w.transpose(-1, -2) @ fx) / (norm + 1e-5)                     # (B,h,m,dh)
        att = torch.softmax(self.q(tok) @ self.k(tok).transpose(-1, -2) / math.sqrt(self.dh), -1)
        tok = att @ self.v(tok)
        y = (w @ tok).transpose(1, 2).reshape(B, N, -1)
        return self.out(y)


class TransolverBlock(nn.Module):
    def __init__(self, dim, heads, slices):
        super().__init__()
        self.n1 = nn.LayerNorm(dim); self.att = PhysicsAttention(dim, heads, slices)
        self.n2 = nn.LayerNorm(dim); self.mlp = nn.Sequential(nn.Linear(dim, 2 * dim), nn.GELU(), nn.Linear(2 * dim, dim))

    def forward(self, x):
        x = x + self.att(self.n1(x))
        return x + self.mlp(self.n2(x))


class Transolver(nn.Module):
    def __init__(self, nd, cin, cond, dim=192, layers=8, heads=8, slices=32):
        super().__init__()
        self.nd = nd
        self.pre = nn.Sequential(nn.Linear(cin + cond + npos(nd), 2 * dim), nn.GELU(), nn.Linear(2 * dim, dim))
        self.blocks = nn.ModuleList([TransolverBlock(dim, heads, slices) for _ in range(layers)])
        self.post = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, cin))

    def forward(self, x, c):
        B, C = x.shape[:2]; S = x.shape[2:]
        grid = torch.stack(torch.meshgrid(*[torch.arange(n, device=x.device, dtype=torch.float32) * 2 * math.pi / n for n in S],
                                          indexing="ij"))
        pos = torch.cat([grid.sin(), grid.cos()], 0)                         # periodic coordinates
        feat = torch.cat([x, bcast(c, S), pos.unsqueeze(0).expand(B, *pos.shape)], 1)
        h = self.pre(feat.flatten(2).transpose(1, 2))
        for b in self.blocks:
            h = b(h)
        return x + self.post(h).transpose(1, 2).reshape(B, C, *S)


# --------------------------------------------------------------------------------------------- PDE-Refiner
class Refiner(nn.Module):
    """Lippe et al. 2023. Same U-Net; input = [u_t, noisy u_{t+1} (or zeros at k=0), cond, k-embedding channel].
    k = 0 predicts the residual u_{t+1}-u_t (MSE); k >= 1 predicts the added noise. sigma_k = min_std^(k/K)."""

    def __init__(self, nd, cin, cond, K=3, min_std=4e-7, width=64):
        super().__init__()
        self.K, self.min_std = K, min_std
        self.net = UNetGeneric(nd, cin, cond + 1, width=width, block="res", extra_in=cin)
        self.cin = cin

    def _call(self, x, y, c, k):
        kc = torch.full((x.shape[0], 1), k / self.K, device=x.device)
        h = torch.cat([x, y, bcast(torch.cat([c, kc], 1), x.shape[2:]), posenc(x)], 1)
        return self.net.core(self.net.inp(h))

    def loss(self, x, target, c):
        """training objective for one window step (x -> target), random refinement step."""
        B = x.shape[0]
        k = int(torch.randint(0, self.K + 1, (1,)))
        d = target - x
        if k == 0:
            pred = self._call(x, torch.zeros_like(x), c, 0)
            return F.mse_loss(pred, d)
        sig = self.min_std ** (k / self.K)
        eps = torch.randn_like(d)
        pred = self._call(x, d + sig * eps, c, k)
        return F.mse_loss(pred, eps)

    def forward(self, x, c):
        d = self._call(x, torch.zeros_like(x), c, 0)
        for k in range(1, self.K + 1):
            sig = self.min_std ** (k / self.K)
            noisy = d + sig * torch.randn_like(d)
            eps = self._call(x, noisy, c, k)
            d = noisy - sig * eps
        return x + d


def build(name, nd, cin, cond):
    if name == "fno":
        return NeuralopFNO(nd, cin, cond)
    if name == "tfno":
        return NeuralopFNO(nd, cin, cond, tucker=True)
    if name == "ffno":
        return FFNO(nd, cin, cond)
    if name == "unet":
        return UNetGeneric(nd, cin, cond, block="res")
    if name == "cnext":
        return UNetGeneric(nd, cin, cond, width=48, mults=(1, 2, 4, 8), block="cnext")
    if name == "cno":
        return CNO(nd, cin, cond)
    if name == "transolver":
        return Transolver(nd, cin, cond)
    if name == "refiner":
        return Refiner(nd, cin, cond)
    if name == "poseidon":
        return Poseidon(nd, cin, cond)
    raise ValueError(name)


# --------------------------------------------------------------------------------------------- Poseidon (fine-tuned)
class Poseidon(nn.Module):
    """Herde et al. NeurIPS 2024 -- pretrained scOT foundation model (Poseidon-T, 21M), fine-tuned. Fields (+ the
    conditioning channels) are band-limited-upsampled to its native 128x128, the patch embedding / recovery layers
    are re-initialised for our channel counts (the paper's 'task-specific parameters'), lead time fixed at one frame."""

    def __init__(self, nd, cin, cond, name=os.path.join(os.environ["SCRATCH"], "poseidon_T"), lead=0.1):
        super().__init__()
        assert nd == 2, "Poseidon is 2D only"
        import os, sys
        sys.path.insert(0, os.path.join(os.environ["SCRATCH"], "poseidon_src"))
        from scOT.model import ScOT
        if not hasattr(ScOT, "get_head_mask") or True:            # removed in transformers 5
            ScOT.get_head_mask = lambda self, hm, n, *a, **k: [None] * n
        self.net = ScOT.from_pretrained(name, num_channels=cin + cond, num_out_channels=cin, ignore_mismatched_sizes=True)
        self.lead = lead

    def forward(self, x, c):
        n = x.shape[-1]
        h = torch.cat([x, bcast(c, x.shape[2:])], 1)
        h = fourier_resample(h, 128)
        t = torch.full((x.shape[0],), self.lead, device=x.device)
        y = self.net(pixel_values=h, time=t).output
        return x + fourier_resample(y.float(), n)


# --------------------------------------------------------------------------------------------- our training recipe, any backbone
class RecipeWrap(nn.Module):
    """The FerroX recipe around ANY backbone (to separate architecture from recipe):
      units      v = x / U   (U = the sample's amplitude unit), tau = theta / theta_ref (time unit)
      amp-aware  backbone sees v/rms(v) and log10 rms(v) (not RevIN: it still knows how far it is from saturation)
      cond       bias-free tanh trunk of the DIMENSIONLESS parameters (exactly zero at the reference), as channels
      update     v' = s exp(tau g) (v/s + tau delta), g a bounded global gain;  guard |v'| <= guard (reduced units)
    Trained with the relative-L2 rollout loss (run.py --wrap)."""

    def __init__(self, backbone, nd, cin, dcu, theta_ref, guard, emb=16):
        super().__init__()
        self.trunk = nn.Sequential(nn.Linear(dcu, 64, bias=False), nn.GELU(), nn.Linear(64, emb, bias=False))
        self.net = build2(backbone, nd, cin + 1, emb)
        self.cin, self.theta_ref, self.guard = cin, float(theta_ref), float(guard)

    def forward(self, x, cu, U, th):
        sh = (-1,) + (1,) * (x.dim() - 1)
        v = x / U.view(sh)
        red = tuple(range(1, v.dim()))
        s = v.pow(2).mean(red, keepdim=True).sqrt().clamp_min(1e-12)
        h = torch.cat([v / s, torch.log10(s).expand_as(v[:, :1])], 1)
        e = torch.tanh(self.trunk(cu))
        out = self.net(h, e) - h
        delta = out[:, :self.cin]
        g = out[:, self.cin:].mean(tuple(range(1, out.dim())), keepdim=True).clamp(-1.0, 0.5)
        tau = (th / self.theta_ref).view(sh)
        v = s * torch.exp(tau * g) * (v / s + tau * delta)
        if self.guard > 0:
            v = v.clamp(-self.guard, self.guard)
        return v * U.view(sh)


# --------------------------------------------------------------------------------------------- U-shaped neural operator
class SpecConvND(nn.Module):
    def __init__(self, nd, cin, cout, modes):
        super().__init__()
        self.nd, self.m = nd, modes
        s = 1 / (cin * cout)
        shape = (cin, cout) + (modes,) * nd
        self.w = nn.ParameterList([nn.Parameter(s * torch.randn(*shape, dtype=torch.cfloat)) for _ in range(1 if nd == 1 else 2)])

    def forward(self, x):
        nd, m = self.nd, self.m
        dims = tuple(range(2, 2 + nd))
        X = torch.fft.rfftn(x.float(), dim=dims)
        Y = torch.zeros(x.shape[0], self.w[0].shape[1], *X.shape[2:], dtype=X.dtype, device=x.device)
        mm = min(m, X.shape[-1], (X.shape[2] // 2) if nd == 2 else m)
        if nd == 1:
            Y[..., :mm] = torch.einsum("bix,iox->box", X[..., :mm], self.w[0][..., :mm])
        else:
            Y[..., :mm, :mm] = torch.einsum("bixy,ioxy->boxy", X[..., :mm, :mm], self.w[0][..., :mm, :mm])
            Y[..., -mm:, :mm] = torch.einsum("bixy,ioxy->boxy", X[..., -mm:, :mm], self.w[1][..., :mm, :mm])
        return torch.fft.irfftn(Y, s=x.shape[2:], dim=dims)


class UNOBlock(nn.Module):
    """Fourier layer that changes resolution (Rahman et al. 2022): spectral conv + pointwise, then band-limited resample."""

    def __init__(self, nd, cin, cout, modes, scale):
        super().__init__()
        C = convnd(nd)
        self.spec = SpecConvND(nd, cin, cout, modes); self.pw = C(cin, cout, 1); self.scale = scale

    def forward(self, x):
        y = F.gelu(self.spec(x) + self.pw(x))
        n = int(round(x.shape[2] * self.scale))
        return fourier_resample(y, n) if n != x.shape[2] else y


class UNO(nn.Module):
    """U-shaped neural operator (Rahman et al., TMLR 2023); E-UNO (2025, the Cahn-Hilliard SOTA) = UNO trained with
    D4 equivariance -- here supplied by the exact-symmetry augmentation every model gets."""

    def __init__(self, nd, cin, cond, width=32, modes=16):
        super().__init__()
        C = convnd(nd)
        self.lift = C(cin + cond + nd, width, 1)
        w = width
        self.e1 = UNOBlock(nd, w, 2 * w, modes, 0.5)          # N   -> N/2
        self.e2 = UNOBlock(nd, 2 * w, 4 * w, modes // 2, 0.5)  # N/2 -> N/4
        self.m = UNOBlock(nd, 4 * w, 4 * w, modes // 2, 1.0)
        self.d2 = UNOBlock(nd, 8 * w, 2 * w, modes // 2, 2.0)  # N/4 -> N/2
        self.d1 = UNOBlock(nd, 4 * w, w, modes, 2.0)           # N/2 -> N
        self.out = UNOBlock(nd, 2 * w, w, modes, 1.0)
        self.proj = nn.Sequential(C(w, 128, 1), nn.GELU(), C(128, cin, 1))

    def forward(self, x, c):
        S = x.shape[2:]
        grid = torch.stack(torch.meshgrid(*[torch.linspace(0, 1, n, device=x.device) for n in S], indexing="ij"))
        h0 = self.lift(torch.cat([x, bcast(c, S), grid.unsqueeze(0).expand(x.shape[0], *grid.shape)], 1))
        h1 = self.e1(h0); h2 = self.e2(h1); h = self.m(h2)
        h = self.d2(torch.cat([h, h2], 1)); h = self.d1(torch.cat([h, h1], 1)); h = self.out(torch.cat([h, h0], 1))
        return x + self.proj(h)


# --------------------------------------------------------------------------------------------- generative next-step models
class GenBase(nn.Module):
    """shared plumbing for the generative next-step models: target = residual d = x_{t+1} - x_t in units of
    `dscale` -- a global constant (scale='global') or rms(x_t) per sample (scale='inst'). Backbone = the U-Net used
    by PDE-Refiner / ACDM / FlowRefiner-type models; input [x_t, noisy target, cond, level]."""

    def __init__(self, nd, cin, cond, scale="global"):
        super().__init__()
        self.net = UNetGeneric(nd, cin, cond + 1, width=64, block="res", extra_in=cin)
        self.scale_mode = scale
        self.register_buffer("dscale", torch.tensor(1.0))

    def set_dscale(self, v):
        self.dscale.fill_(float(v))

    def unit(self, x):
        if self.scale_mode == "inst":
            red = tuple(range(1, x.dim()))
            return x.pow(2).mean(red, keepdim=True).sqrt().clamp_min(1e-8) * 0.1
        return self.dscale

    def call(self, x, z, c, lev):
        lv = torch.as_tensor(lev, device=x.device, dtype=torch.float32).reshape(-1)
        lv = lv.expand(x.shape[0]) if lv.numel() == 1 else lv
        h = torch.cat([x, z, bcast(torch.cat([c, lv[:, None]], 1), x.shape[2:]), posenc(x)], 1)
        return self.net.core(self.net.inp(h))


class Refiner2(GenBase):
    """PDE-Refiner (Lippe et al. 2023) with the target in data-scaled units (the original's min_std = 4e-7 assumes
    O(1) targets; ours are residuals, so they are rescaled first). K refinement steps, sigma_k = min_std^(k/K)."""

    def __init__(self, nd, cin, cond, scale="global", K=3, min_std=1e-3):
        super().__init__(nd, cin, cond, scale); self.K, self.min_std = K, min_std

    def loss(self, x, y, c):
        u = self.unit(x); d = (y - x) / u
        k = int(torch.randint(0, self.K + 1, (1,)))
        if k == 0:
            return F.mse_loss(self.call(x, torch.zeros_like(d), c, 0.0), d)
        sig = self.min_std ** (k / self.K); eps = torch.randn_like(d)
        return F.mse_loss(self.call(x, d + sig * eps, c, k / self.K), eps)

    def forward(self, x, c):
        u = self.unit(x)
        d = self.call(x, torch.zeros_like(x), c, 0.0)
        for k in range(1, self.K + 1):
            sig = self.min_std ** (k / self.K)
            noisy = d + sig * torch.randn_like(d)
            d = noisy - sig * self.call(x, noisy, c, k / self.K)
        return x + u * d


class ACDM(GenBase):
    """autoregressive conditional diffusion (Kohl et al. 2023; Shysheya et al. NeurIPS 2024): DDPM, cosine schedule,
    v-prediction, 20-step DDIM with the x0 estimate clipped to +-clip (target units) -- without the clip, the 1/sqrt(abar)
    amplification at high noise levels blows rollouts up."""

    def __init__(self, nd, cin, cond, scale="global", T=200, steps=20, clip=6.0):
        super().__init__(nd, cin, cond, scale)
        s = 0.008; t = torch.linspace(0, 1, T + 1)
        ab = torch.cos((t + s) / (1 + s) * math.pi / 2) ** 2; ab = (ab / ab[0]).clamp(1e-5, 1)
        self.register_buffer("ab", ab); self.T, self.steps, self.clip = T, steps, clip

    def loss(self, x, y, c):
        u = self.unit(x); d = (y - x) / u
        t = torch.randint(1, self.T + 1, (x.shape[0],), device=x.device)
        a = self.ab[t].view(-1, *([1] * (x.dim() - 1)))
        eps = torch.randn_like(d)
        z = a.sqrt() * d + (1 - a).sqrt() * eps
        v = a.sqrt() * eps - (1 - a).sqrt() * d
        return F.mse_loss(self.call(x, z, c, t.float() / self.T), v)

    def forward(self, x, c):
        u = self.unit(x)
        ts = torch.linspace(self.T, 0, self.steps + 1).round().long().tolist()
        z = torch.randn_like(x)
        for t, tn in zip(ts[:-1], ts[1:]):
            a, an = self.ab[t], self.ab[tn]
            v = self.call(x, z, c, t / self.T)
            d0 = (a.sqrt() * z - (1 - a).sqrt() * v).clamp(-self.clip, self.clip)
            eps = ((1 - a).sqrt() * z + a.sqrt() * v)
            z = an.sqrt() * d0 + (1 - an).sqrt() * eps
        return x + u * z


class FlowMatch(GenBase):
    """conditional flow matching / rectified flow next-step model (Lipman et al. 2023; PDE uses: FlowRefiner 2026,
    memory-conditioned FM 2026, flow marching 2025): z_s = (1-s) z0 + s d, predict v = d - z0, 8 Euler steps."""

    def __init__(self, nd, cin, cond, scale="global", steps=8):
        super().__init__(nd, cin, cond, scale); self.steps = steps

    def loss(self, x, y, c):
        u = self.unit(x); d = (y - x) / u
        z0 = torch.randn_like(d)
        s = torch.rand(x.shape[0], device=x.device).view(-1, *([1] * (x.dim() - 1)))
        zs = (1 - s) * z0 + s * d
        return F.mse_loss(self.call(x, zs, c, s.flatten()), d - z0)

    def forward(self, x, c):
        u = self.unit(x); z = torch.randn_like(x); h = 1.0 / self.steps
        for i in range(self.steps):
            z = z + h * self.call(x, z, c, i * h)
        return x + u * z


GENERATIVE = ("refiner2", "acdm", "flowmatch")


def build2(name, nd, cin, cond, scale="global"):
    if name == "uno":
        return UNO(nd, cin, cond)
    if name == "refiner2":
        return Refiner2(nd, cin, cond, scale)
    if name == "acdm":
        return ACDM(nd, cin, cond, scale)
    if name == "flowmatch":
        return FlowMatch(nd, cin, cond, scale)
    if name == "sgno":
        return SGNOBench(nd, cin, cond)
    if name == "persist":
        return Persist()
    if name == "transolver_grid":
        return TransolverGrid(nd, cin, cond)
    return build(name, nd, cin, cond)


class InstNorm(nn.Module):
    """per-sample normalisation (RevIN-style, Kim et al. ICLR 2022): the backbone sees x / rms(x) and its residual is
    rescaled by rms(x). A standard remedy for inputs whose amplitude varies by orders of magnitude."""

    def __init__(self, net):
        super().__init__(); self.net = net

    def forward(self, x, c):
        red = tuple(range(1, x.dim()))
        s = x.pow(2).mean(red, keepdim=True).sqrt().clamp_min(1e-8)
        return self.net(x / s, c) * s


class DuhamelTDGL(nn.Module):
    """our Duhamel model (q_probe/volt_sweep/duhamel_hyst.DuhamelHyst) on multi-material TDGL: V = 0, eps_de = 10.
    Exact conditioned linear semigroup on all modes + O(u^2) remainder, no hand-off. Called as model(x, aux)."""

    def __init__(self, scale):
        super().__init__()
        import sys as _s
        _s.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "q_probe", "volt_sweep"))
        from duhamel_hyst import DuhamelHyst
        import train_hyst as th
        self.m = DuhamelHyst(); self.th = th; self.scale = float(scale)

    def forward(self, x, aux):
        B = x.shape[0]; z = torch.zeros(B)
        extra = self.th.extra_vec(z, z, torch.full((B,), 10.0))
        self.m.set(list(aux["mat"]), extra, x.device)
        return self.m.step(x[:, None] * self.scale)[:, 0] / self.scale


# --------------------------------------------------------------------------------------------- Duhamel-UNO (ours, v4)
def _phi1(M):
    """exp(M) and phi1(M) = M^-1 (exp(M) - I) for batched square matrices via the augmented exponential."""
    z = M.shape[-1]
    aug = M.new_zeros(*M.shape[:-2], 2 * z, 2 * z)
    aug[..., :z, :z] = M; aug[..., :z, z:] = torch.eye(z, device=M.device, dtype=M.dtype)
    X = torch.matrix_exp(aug)
    return X[..., :z, :z], X[..., :z, z:]


class DuhamelUNO(nn.Module):
    """v' = e^{theta L} v + theta phi1(theta L) N(v)   (exponential-Euler / Duhamel step), in the material's units.
      L  FerroX ('bank'): the per-|k| 10x10 generator measured from TRAINING trajectories (fit_alpha_bank.py) and
         interpolated in log|alpha| (K nearest); bins the data cannot resolve (decayed within a frame / too few
         modes) get a learned per-bin diagonal rate. Other PDEs ('poly'): learned conditioned symbol
         W(k) = sum_j c_j(cond) (k^2)^j on ALL modes (sign-free).
      N  U-shaped neural operator (UNO) on [v/rms(v), log10 rms(v)] + conditioning channels, output zero-initialised
         (so the untrained model IS the linear propagator); multiplied back by rms(v).
      recipe: units (v = x/U), dimensionless step theta, relative-L2 rollout loss, guard."""

    def __init__(self, nd, cin, dcu, guard, linear="poly", bank=None, K=8, emb=16, npoly=4, persist_k0=False, cref=None, substeps=1,
                 trunc_w=0, remainder="uno"):
        super().__init__()
        self.nd, self.cin, self.guard, self.linear, self.K = nd, cin, float(guard), linear, K
        self.substeps, self.use_rem = substeps, True
        if linear == "kmu":
            # learned conditioned symbol in PHYSICAL units: W(k mu; c), mu = wall width from Pi_h (64-grid units) -- the same
            # parameterisation as Duhamel-O(u^2) (ml_units/duhamel.Generator), so the step is grid-agnostic
            import duhamel as DH
            self.gen = DH.Generator(emb, 2.0)
            self.trunc_w = trunc_w                      # ABLATION: symbol only inside the FNO-like K-mode box, one learned constant above
            if trunc_w:
                self.w_inf = nn.Linear(emb, 1)
            self.register_buffer("pih_ref", torch.tensor(float(np.asarray(cref)[1])))
        self.trunk = nn.Sequential(nn.Linear(dcu, 64, bias=False), nn.GELU(), nn.Linear(64, emb, bias=False))
        self.rem_kind = remainder
        if remainder == "ou2":
            self.rem = RemainderOU2(cin, emb)
        else:
            self.net = UNO(nd, cin + 1, emb)
            with torch.no_grad():
                self.net.proj[-1].weight.zero_(); self.net.proj[-1].bias.zero_()
        if linear == "bank":
            d = np.load(bank)
            self.register_buffer("b_la", torch.tensor(np.log(np.abs(d["alpha"])), dtype=torch.float32))
            self.register_buffer("b_G", torch.tensor(d["G"], dtype=torch.float32))          # (Ntr, NB, Z, Z)
            self.register_buffer("b_ok", torch.tensor(d["valid"]))                           # (Ntr, NB)
            NB = self.b_G.shape[1]
            self.w_inv = nn.Parameter(torch.full((NB,), -50.0))                              # unresolved bins: decayed within a frame -> strong damping (learnable)
            if persist_k0:                                                                   # k=0 is ONE coefficient -> never fittable (not decayed); it carries the
                with torch.no_grad():                                                        # driven net polarisation (hysteresis), so start from persistence
                    self.w_inv[0] = 0.0
            n = 64; kx = torch.fft.fftfreq(n) * n
            self.register_buffer("KB", torch.round(torch.sqrt(kx[:, None] ** 2 + kx[None, :] ** 2)).long())
        else:
            self.poly = nn.Sequential(nn.Linear(dcu, 64), nn.GELU(), nn.Linear(64, npoly))
            nn.init.zeros_(self.poly[-1].weight); nn.init.zeros_(self.poly[-1].bias)
        self._cache = {}

    # ---------------- linear operators
    def _bank_ops(self, alphas, th):
        key = (tuple(round(float(a), 3) for a in alphas), tuple(round(float(t), 6) for t in th.tolist()),
               float(self.w_inv.detach().sum()), torch.is_grad_enabled())
        if key in self._cache:
            return self._cache[key]
        Es, Ps = [], []
        Z = self.b_G.shape[-1]; I = torch.eye(Z, device=th.device)
        for a, t in zip(alphas, th):
            nn_ = torch.argsort((self.b_la - math.log(abs(float(a)))).abs())[:self.K]
            ok = self.b_ok[nn_].float()                                                      # (K, NB)
            cnt = ok.sum(0)
            G = (self.b_G[nn_] * ok[..., None, None]).sum(0) / cnt.clamp_min(1)[:, None, None]  # (NB,Z,Z)
            inv = (cnt == 0).float()[:, None, None]
            G = G * (1 - inv) + inv * (self.w_inv.view(-1, 1, 1) * I)
            E, P = _phi1(G * t)
            Es.append(E); Ps.append(P * t)
        out = (torch.stack(Es), torch.stack(Ps))                                             # (B, NB, Z, Z)
        self._cache = {key: out}
        return out

    def _kgrid(self, S, dev):
        ks = [torch.fft.fftfreq(n, device=dev) * 2 * math.pi for n in S]
        k2 = sum(g ** 2 for g in torch.meshgrid(*ks, indexing="ij"))
        return k2

    def forward(self, x, aux):
        B = x.shape[0]; sh = (-1,) + (1,) * (x.dim() - 1)
        U, th, cu = aux["U"].view(-1), aux["th"].view(-1), aux["cu"]
        v = x / U.view(sh)
        red = tuple(range(1, v.dim()))
        s = (v.pow(2).mean(red, keepdim=True) + 1e-24).sqrt()                                # eps inside: sqrt'(0) = inf -> NaN grads on an all-zero field
        e = torch.tanh(self.trunk(cu))
        h = torch.cat([v / s, torch.log10(s).expand_as(v[:, :1])], 1)
        Nv = self.rem(v, e) if self.rem_kind == "ou2" else (self.net(h, e) - h)[:, :self.cin] * s       # remainder, in v units
        dims = tuple(range(2, 2 + self.nd))
        if self.linear == "kmu":
            n = x.shape[-1]
            k1 = torch.fft.fftfreq(n, d=1.0 / n, device=x.device)
            kk = (k1.view(-1, 1) ** 2 + k1.view(1, -1) ** 2).sqrt()                          # integer wavenumbers, 2pi domain
            mu = (2 * math.pi / 64) / (cu[:, 1] + self.pih_ref)
            W = self.gen(kk[None] * mu.view(-1, 1, 1), e)                                    # (B,n,n)
            if getattr(self, "trunc_w", 0):
                K = self.trunc_w
                box = ((k1.view(-1, 1) >= -K) & (k1.view(-1, 1) < K) & (k1.view(1, -1) >= -K) & (k1.view(1, -1) < K))[None]
                wi = (2.0 - F.softplus(2.0 - self.w_inf(e))).view(-1, 1, 1)
                W = torch.where(box, W, wi)
            hstep = (th / self.substeps).view(-1, 1, 1)
            z = W * hstep; E = torch.exp(z)
            P = torch.where(z.abs() > 1e-4, (E - 1) / torch.where(z.abs() > 1e-4, z, torch.ones_like(z)), 1 + z / 2) * hstep
            for i in range(self.substeps):
                if i > 0 and self.use_rem:
                    s = (v.pow(2).mean(red, keepdim=True) + 1e-24).sqrt()
                    h = torch.cat([v / s, torch.log10(s).expand_as(v[:, :1])], 1)
                    Nv = self.rem(v, e) if self.rem_kind == "ou2" else (self.net(h, e) - h)[:, :self.cin] * s
                Vf = torch.fft.fftn(v.float(), dim=dims)
                Yf = E.unsqueeze(1) * Vf + (P.unsqueeze(1) * torch.fft.fftn(Nv.float(), dim=dims) if self.use_rem else 0)
                v = torch.fft.ifftn(Yf, dim=dims).real
            y = v
            if self.guard > 0:
                y = y.clamp(-self.guard, self.guard)
            return y * U.view(sh)
        if self.linear == "bank":
            E, P = self._bank_ops(aux["mat"] if isinstance(aux["mat"][0], float) else [m[0] for m in aux["mat"]], th)
            Em, Pm = E[:, self.KB], P[:, self.KB]                                            # (B,H,W,Z,Z)
            Vf = torch.fft.fft2(v.float(), dim=dims).permute(0, 2, 3, 1).unsqueeze(-1)       # (B,H,W,Z,1)
            Nf = torch.fft.fft2(Nv.float(), dim=dims).permute(0, 2, 3, 1).unsqueeze(-1)
            Yf = (torch.complex(Em @ Vf.real, Em @ Vf.imag) + torch.complex(Pm @ Nf.real, Pm @ Nf.imag)).squeeze(-1).permute(0, 3, 1, 2)
            y = torch.fft.ifft2(Yf, dim=dims).real
        else:
            k2 = self._kgrid(x.shape[2:], x.device)                                         # (*S)
            c = self.poly(cu)                                                                # (B, npoly)
            W = sum(c[:, j].view((-1,) + (1,) * self.nd) * k2 ** j for j in range(c.shape[1]))  # (B,*S)
            z = (W * th.view((-1,) + (1,) * self.nd)).clamp(max=5.0)
            E = torch.exp(z)
            P = torch.where(z.abs() > 1e-4, (E - 1) / torch.where(z.abs() > 1e-4, z, torch.ones_like(z)), 1 + z / 2) * th.view((-1,) + (1,) * self.nd)
            Vf = torch.fft.fftn(v.float(), dim=dims); Nf = torch.fft.fftn(Nv.float(), dim=dims)
            y = torch.fft.ifftn(E.unsqueeze(1) * Vf + P.unsqueeze(1) * Nf, dim=dims).real
        if self.guard > 0:
            y = y.clamp(-self.guard, self.guard)
        return y * U.view(sh)


def _duno_linear_params(self):
    return list(self.trunk.parameters()) + list(self.gen.parameters()) + (list(self.w_inf.parameters()) if getattr(self, "trunc_w", 0) else [])


DuhamelUNO.linear_params = _duno_linear_params


class DuhamelOU2(nn.Module):
    """ml_units/duhamel.py (Duhamel O(u^2)) inside the bench harness:
        v' = e^{theta W} v + theta phi1(theta W) N(v),   W = W(k mu; c) on ALL Fourier modes (sign-free, capped above),
        N(v) = P[(A v) (.) H(v)]  exactly O(|v|^2).
    The generator is a function of the DIMENSIONLESS wavenumber k mu (mu = wall width, from the conditioning Pi_h = dx/mu
    in 64-grid units), so the same weights run on any grid of the 2pi domain (zero-shot resolution transfer).
    CH: 4 exponential-Euler sub-steps per frame (as in the ml_units run)."""

    def __init__(self, cref, guard, system, emb=16, free_rem=False, trunc_w=0):
        super().__init__()
        import duhamel as DH
        base = "".join(ch for ch in system if not ch.isdigit())
        nsd = base == "nsd"
        # ablations: free_rem = unconstrained remainder (biases + additive read-out, NOT O(u^2));
        #            trunc_w = linear generator only inside the FNO's K-mode box, one learned constant above (FNO-like)
        self.m = DH.Duhamel(64, np.zeros_like(np.asarray(cref, np.float32)), guard, emb=emb, wmax=0.0 if nsd else 2.0,
                            pih_const=(2 * math.pi / 64) * 4.0 if nsd else None, free_rem=free_rem)
        if trunc_w:
            self.m.set_trunc(trunc_w)
        self.m.substeps = 4 if base == "ch" else 1
        self.m.conserve = nsd; self.m.zclip = nsd
        self.register_buffer("cref", torch.as_tensor(np.asarray(cref, np.float32)))

    def _grid(self, n):
        if n != self.m.n:
            k1 = 2 * math.pi * torch.fft.fftfreq(n, d=2 * math.pi / n, device=self.m.k.device)
            self.m.k = (k1.view(-1, 1) ** 2 + k1.view(1, -1)[:, : n // 2 + 1] ** 2).sqrt(); self.m.n = n

    def forward(self, x, aux):
        n = x.shape[-1]; self._grid(n)
        cu = aux["cu"]
        pih = (cu[:, 1] + self.cref[1]) * 64.0 / n                     # this grid's dx/mu
        return self.m(x[:, 0], cu, aux["U"].view(-1), aux["th"].view(-1), pih)[:, None]

    def linear_params(self):
        return list(self.m.trunk.parameters()) + list(self.m.gen.parameters()) + (list(self.m.w_inf.parameters()) if self.m.trunc_w else [])


class RemainderOU2(nn.Module):
    """multi-channel version of ml_units/duhamel.Remainder: N(v) = P[(A v) (.) H(v)], exactly O(|v|^2) (bias-free Fourier layers
    A, P; bias-free FNO H with GELU and FiLM gains only, no positional channels). v (B, C, H, W) -> (B, C, H, W)."""

    def __init__(self, cin, emb=16, width=48, layers=4, modes=16, ch=16, gmax=0.5):
        super().__init__()
        import duhamel as DH
        self.lift = nn.Conv2d(cin, width, 1, bias=False)
        self.blocks = nn.ModuleList([DH.SpecLin(width, width, modes) for _ in range(layers)])
        self.hout = nn.Conv2d(width, ch, 1, bias=False)
        self.A = DH.SpecLin(cin, ch, modes)
        self.P = DH.SpecLin(ch, cin, modes)
        self.film = nn.Sequential(nn.Linear(emb, 64, bias=False), nn.GELU(), nn.Linear(64, layers * width, bias=False))
        nn.init.zeros_(self.film[-1].weight)
        nn.init.zeros_(self.P.pw.weight); nn.init.zeros_(self.P.w1); nn.init.zeros_(self.P.w2)       # starts as N = 0
        self.layers, self.width, self.gmax = layers, width, gmax

    def forward(self, v, e):
        g = self.gmax * torch.tanh(self.film(e)).view(-1, self.layers, self.width, 1, 1)
        h = self.lift(v)
        for i, blk in enumerate(self.blocks):
            h = F.gelu(blk(h) * (1 + g[:, i]))
        return self.P(self.A(v) * self.hout(h))


# --------------------------------------------------------------------------------------------- added 2026-10-06: closest-method and reference baselines
class SGNOBench(nn.Module):
    """SGNO (Li et al. 2026, arXiv:2602.18801), the OFFICIAL implementation (github.com/cruiseresearchgroup/SGNO @ b07cb4c,
    bench/external/SGNO): residual neural operator whose layers are exponential-Euler spectral steps with a learned,
    NON-POSITIVE diagonal generator on a truncated mode set. Conditioning enters as constant input channels (as for every
    baseline); the official core appends its own coordinate grid. Width / modes / depth sized like our other operators."""

    def __init__(self, nd, cin, cond, width=48, modes=12, n_blocks=4):
        super().__init__()
        import sys as _s
        _s.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "external", "SGNO", "src"))
        from sgno import sgno as _sg
        assert nd == 2, "SGNO wrapper: 2D only here"
        if os.environ.get("SGNO_EXACT_SVD") != "1":
            # same projection, faster: ||M||_2 by power iteration (as torch.nn.utils.spectral_norm does) instead of the
            # official batched complex SVD (matrix_norm ord=2), which made a training step take seconds on an A100
            def _proj(mix, channels, _cap=_sg._MIXER_NORM_CAP, iters=int(os.environ.get("SGNO_POWER_IT", "100"))):
                mats = mix.reshape(channels, channels, -1).permute(2, 0, 1)
                with torch.no_grad():                                      # power iteration for the top right-singular vector
                    g = mats.conj().transpose(-1, -2) @ mats
                    v = torch.ones(g.shape[0], channels, 1, dtype=g.dtype, device=g.device)
                    for _ in range(iters):
                        v = g @ v; v = v / v.norm(dim=1, keepdim=True).clamp_min(1e-30)
                norms = (mats @ v).norm(dim=(1, 2))                        # differentiable, as in torch's spectral_norm
                sc = torch.clamp(_cap / torch.clamp(norms, min=1e-12), max=1.0)
                return (mats * sc[:, None, None]).permute(1, 2, 0).reshape_as(mix)
            _sg._project_mixer = _proj
        self._sg = _sg; self.cin = cin
        self.core = _sg.SGNO2d(cin + cond, modes, modes, width, 1, n_blocks)

    def forward(self, x, c):
        S = x.shape[2:]
        feat = torch.cat([x, bcast(c, S)], 1).permute(0, 2, 3, 1).contiguous()
        grid = self._sg._make_grid_2d(x.shape[0], S[0], S[1], x.device, x.dtype)
        d = self.core(feat, grid).squeeze(-2)[..., :self.cin].permute(0, 3, 1, 2)
        return x + d


class Persist(nn.Module):
    """reference: persistence u_{t+1} = u_t (no parameters)."""

    def __init__(self):
        super().__init__(); self.dummy = nn.Parameter(torch.zeros(1))

    def forward(self, x, c=None):
        return x


class PODLSTM(nn.Module):
    """POD/PCA + LSTM reduced-order surrogate, the approach of the ferroelectric phase-field surrogate literature
    (PCA-compressed polarisation fields evolved by an LSTM). Given the recipe's advantages for fairness: fields in each
    material's units (v = x/U), amplitude-aware latent input [z/|z|, log|z|], the dimensionless parameters, the frame
    interval, and a multiplicative amplitude update z' = |z| exp(g) (z/|z| + d). The POD basis (k modes) is fitted on
    training frames sampled across the whole trajectory (60% from the growth phase, as for every model)."""

    def __init__(self, D, dcu, k=256, hidden=512, layers=2):
        super().__init__()
        self.k = k
        self.register_buffer("Phi", torch.zeros(D, k))
        self.register_buffer("mu", torch.zeros(D))
        self.lstm = nn.LSTM(k + 1 + dcu + 1, hidden, layers, batch_first=True)
        self.head = nn.Linear(hidden, k + 1)
        nn.init.zeros_(self.head.weight); nn.init.zeros_(self.head.bias)

    @torch.no_grad()
    def fit_basis(self, task, rng, nframes=16000):
        N, Tn = task.Umm.shape[:2]
        i = task.tr_idx[rng.integers(len(task.tr_idx), size=nframes)]
        t = np.where(rng.random(nframes) < 0.6, rng.integers(0, 121, size=nframes), rng.integers(0, Tn, size=nframes))
        X = []
        for a_, b_ in zip(i, t):
            X.append(np.asarray(task.Umm[a_, b_], np.float32).reshape(-1) / task.scale / float(task.AUX["U"][a_]))
        X = torch.from_numpy(np.stack(X)).to(self.Phi.device)
        X = X / X.pow(2).mean(1, keepdim=True).sqrt().clamp_min(1e-12)        # shape, not amplitude (early frames are tiny)
        U, S, V = torch.svd_lowrank(X, q=self.k + 16, niter=4, M=None)
        self.Phi.copy_(V[:, :self.k])
        ev = (S[:self.k] ** 2).sum() / X.pow(2).sum()
        print("POD: %d modes capture %.4f of the energy of the normalised training frames" % (self.k, float(ev)), flush=True)
        return float(ev)

    def step(self, x, cu, U, th, state=None):
        """x (B,C,H,W) normalised field -> next field; state = LSTM (h, c)."""
        B = x.shape[0]; sh = (-1,) + (1,) * (x.dim() - 1)
        v = (x / U.view(sh)).reshape(B, -1)
        z = v @ self.Phi
        n = z.norm(dim=1, keepdim=True).clamp_min(1e-12)
        inp = torch.cat([z / n, torch.log10(n), cu, th.view(-1, 1)], 1).unsqueeze(1)
        o, state = self.lstm(inp, state)
        d = self.head(o[:, 0])
        g = d[:, -1:].clamp(-1.0, 0.5)
        z2 = n * torch.exp(g) * (z / n + d[:, :-1])
        v2 = z2 @ self.Phi.T
        return (v2.view_as(x) * U.view(sh)), state


class TransolverGrid(nn.Module):
    """Transolver (Wu et al., ICML 2024), the OFFICIAL structured-grid model (github.com/thuml/Transolver @ 75e0f67,
    PDE-Solving-StandardBenchmark/model/Transolver_Structured_Mesh_2D.py) with the authors' settings for 2D grids
    (exp_ns.sh: 8 layers, width 256, 8 heads, 32 slices, unified position encoding with ref 8). Unlike the
    irregular-mesh variant above, its slice projections are 3x3 convolutions. One input frame + conditioning channels."""

    def __init__(self, nd, cin, cond, n_hidden=256, n_layers=8, slice_num=32):
        super().__init__()
        import sys as _s
        _s.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "external", "Transolver", "PDE-Solving-StandardBenchmark"))
        try:
            import timm  # noqa: F401
        except ImportError:                                                # the official code only uses timm's trunc_normal_
            import types
            tm, tl = types.ModuleType("timm"), types.ModuleType("timm.models"); ly = types.ModuleType("timm.models.layers")
            ly.trunc_normal_ = nn.init.trunc_normal_; tl.layers = ly; tm.models = tl
            _s.modules.update({"timm": tm, "timm.models": tl, "timm.models.layers": ly})
        from model import Transolver_Structured_Mesh_2D as _T
        assert nd == 2
        H = int(os.environ.get("BENCH_HW", "64"))
        self.cin, self.H = cin, H
        self.core = _T.Model(space_dim=2, n_layers=n_layers, n_hidden=n_hidden, dropout=0.0, n_head=8, Time_Input=False, act="gelu",
                             mlp_ratio=1, fun_dim=cin + cond, out_dim=cin, slice_num=slice_num, ref=8, unified_pos=1, H=H, W=H)

    def forward(self, x, c):
        B, C, H, W = x.shape
        assert H == self.H and W == self.H, "TransolverGrid built for %d^2 (set BENCH_HW)" % self.H
        fx = torch.cat([x, bcast(c, (H, W))], 1).permute(0, 2, 3, 1).reshape(B, H * W, -1)
        pos = x.new_zeros(B, H * W, 2)                                     # replaced by the unified position encoding inside
        out = self.core(pos, fx)                                           # (B, N, cin)
        return x + out.reshape(B, H, W, C).permute(0, 3, 1, 2)
