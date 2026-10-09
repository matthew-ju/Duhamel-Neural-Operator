#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""probe_chaos.py -- empirical check of the decorrelation theorem (THEORY_fluctuation_seeded.md, Thm 1-2).

For one in-range Allen-Cahn material and fresh white noise at amplitude eps (a factor lam times the training noise
level), measure for each architecture class
  H(eps)    = E||G - G_1||^2 / E||G||^2       non-first-Wiener-chaos output fraction (G_1 = per-mode best linear
                                              response m(k) xi(k); includes the deterministic offset G(0))
  rho(eps)  = E<G,T> / sqrt(E||G||^2 E||T||^2) population correlation with the exact linear truth T = e^{dt L} eps xi
  bound     = sqrt(1 - H)                       Thm 1: rho <= bound
  corr      = mean per-sample spatial pattern correlation (what we report elsewhere)
Classes (Thm 2):
  (a) biased, unnormalised FNO            -> H -> 1, rho -> 0 as eps -> 0        (arm-study 'raw' and 'units' arms)
  (b) scale-normalised FNO                 -> H ~ const in eps                      (recipe FNO: v/rms + log rms channel)
  (c) tangent / Duhamel (neural part O(u^2)) -> H = O(eps^2) -> 0                   (duhamel.py)
Mode k=0 (the spatial mean) is excluded everywhere, so rho is a PATTERN correlation.
"""
import json, math, os
import numpy as np, torch
import train as T
import recipe as R
import duhamel as DU

dev = torch.device("cuda")
N = 64; NB = 512; LAMS = [0.01, 0.03, 0.1, 0.3, 1.0, 3.0]
D = os.path.join(os.environ["SCRATCH"], "ml_units")


def load_raw(arm, seed):
    ck = torch.load(os.path.join("results", "ckpt", "ac_%s_s%d_v2.pt" % (arm, seed)), map_location=dev, weights_only=False)
    ncond = 5 if arm in ("raw", "revin") else 2
    m = T.Stepper(arm, 2, np.zeros(ncond, np.float32), np.ones(ncond, np.float32)).to(dev)
    m.load_state_dict(ck["model"]); return m.eval()


def load_recipe(seed):
    z = np.load(os.path.join(D, "ac_train.npz"))
    dml = np.stack([np.log(z["theta"]), z["pih"]], 1)
    m = R.Recipe(2, np.median(dml, 0).astype(np.float32), float(np.median(z["theta"])), R.GUARD["ac"]).to(dev)
    ck = torch.load(os.path.join(R.OUT, "ckpt", "ac_recipe_s%d.pt" % seed), map_location=dev, weights_only=False)
    m.load_state_dict({k: v for k, v in ck["model"].items() if not k.endswith("_metadata")}, strict=False); return m.eval()


def load_duhamel(seed):
    ck = torch.load(os.path.join(DU.OUT, "ckpt", "ac_duhamel_s%d.pt" % seed), map_location=dev, weights_only=False)
    m = DU.Duhamel(N, ck["cref"], R.GUARD["ac"]).to(dev); m.load_state_dict(ck["model"]); return m.eval()


@torch.no_grad()
def main():
    z = np.load(os.path.join(D, "ac_id.npz"))
    i = int(np.argsort(z["U"])[len(z["U"]) // 2])                        # mid-range material
    a, b, M, e, dtf, U, th, pih = [float(z[k][i]) for k in ("a", "b", "M", "e", "dtf", "U", "theta", "pih")]
    raw = torch.tensor([[math.log(-a), math.log(b), math.log(M), math.log(e), math.log(dtf)]], device=dev)
    dml = torch.tensor([[math.log(th), pih]], device=dev)
    Ut = torch.tensor([U], device=dev); tht = torch.tensor([th], device=dev)
    k1 = np.fft.fftfreq(N, d=1.0 / N); kr = np.fft.rfftfreq(N, d=1.0 / N); K2 = k1[:, None] ** 2 + kr[None, :] ** 2
    g = torch.tensor(np.exp(dtf * (-M * (a + e * K2))), device=dev)
    w = torch.ones_like(g); w[:, 1:-1] = 2.0; w[0, 0] = 0.0                # rfft Parseval weights, k=0 excluded
    ex = lambda t, n: t.expand(n, *t.shape[1:])
    models = {"a_raw(biased)": lambda x: arm_raw(x), "a_units(biased)": lambda x: arm_units(x),
              "b_recipe(rms-norm)": lambda x: rec(x, ex(dml, x.shape[0]), ex(Ut, x.shape[0]), ex(tht, x.shape[0])),
              "c_duhamel(O(u^2))": lambda x: duh(x, ex(dml, x.shape[0]), ex(Ut, x.shape[0]), ex(tht, x.shape[0]), ex(dml, x.shape[0])[:, 1])}
    mraw, munits = load_raw("raw", 0), load_raw("units", 0)
    arm_raw = lambda x: mraw(x, ex(raw, x.shape[0]), ex(Ut, x.shape[0]), ex(tht, x.shape[0]))
    arm_units = lambda x: munits(x, ex(dml, x.shape[0]), ex(Ut, x.shape[0]), ex(tht, x.shape[0]))
    rec, duh = load_recipe(0), load_duhamel(0)
    rng = np.random.default_rng(0)
    xi = torch.tensor(rng.normal(0, 1, (NB, N, N)), dtype=torch.float32, device=dev)
    xih = torch.fft.rfft2(xi.double())
    out = {"material": dict(U=U, theta=th, pih=pih), "lams": LAMS, "noise_train_std": 0.01}
    for name, f in models.items():
        rows = []
        for lam in LAMS:
            eps = 0.01 * lam
            G = torch.cat([f(eps * xi[j:j + 64]) for j in range(0, NB, 64)]).double()
            Gh = torch.fft.rfft2(G)
            Th = g[None] * eps * xih
            m = (Gh * xih.conj()).mean(0) / (xih.abs() ** 2).mean(0)          # first-chaos multiplier (per mode)
            G1 = m[None] * xih
            E_G = float((w * Gh.abs() ** 2).sum() / NB); E_T = float((w * Th.abs() ** 2).sum() / NB)
            H = float((w * (Gh - G1).abs() ** 2).sum() / NB) / E_G
            rho = float((w * (Gh * Th.conj()).real).sum() / NB) / math.sqrt(E_G * E_T)
            Gc = G - G.mean((1, 2), keepdim=True); Tc = torch.fft.irfft2(Th, s=(N, N)); Tc = Tc - Tc.mean((1, 2), keepdim=True)
            corr = float(((Gc * Tc).sum((1, 2)) / (Gc.norm(dim=(1, 2)) * Tc.norm(dim=(1, 2)) + 1e-30)).mean())
            rows.append(dict(lam=lam, eps_over_U=eps / U, H=H, rho=rho, bound=math.sqrt(max(0.0, 1 - H)), corr=corr))
            print("%-20s eps/U %.1e  H %.3e  rho %.4f  sqrt(1-H) %.4f  per-sample corr %.4f" % (name, eps / U, H, rho, rows[-1]["bound"], corr), flush=True)
        out[name] = rows
    json.dump(out, open("figs_recipe/probe_chaos.json", "w"), indent=1)


if __name__ == "__main__":
    main()
