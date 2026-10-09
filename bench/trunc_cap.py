#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""bench/trunc_cap.py -- the Fourier-truncation cap of Theorem 4 for the FerroX TDGL test materials, computed from the
MEASURED linear operator (the alpha bank used by Duhamel-UNO), for the mode sets of the benchmark's spectral models.

For white seeds and a layered field, the best linear response that is exact inside the retained set K and one constant
Z x Z matrix outside has  rho_K(n)^2 = 1 - sum_{k notin K, k!=0} ||G_n(k) - C||_F^2 / sum_{k!=0} ||G_n(k)||_F^2,
C = mean_{k notin K} G_n(k),  G_n(k) = exp(n theta L(k)) (per |k| bin).  Averaged over the 256 held-out D1 materials.
Mode sets (64x64 grid, integer kx, ky): neuralop FNO/TFNO n_modes=(16,16) -> kx in [-8,8), ky in [-8,8];
the 16-mode box of UNO's spectral convolutions -> kx in [-16,16), ky in [-15,15].
Out: paper_tmlr/data/theory_cap_bank.json
"""
import json, math, os, sys
import numpy as np
from scipy.linalg import expm

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import tasks as TK

N = 64
kx = (np.fft.fftfreq(N) * N).astype(int)
KX, KY = np.meshgrid(kx, kx, indexing="ij")
KB = np.rint(np.sqrt(KX ** 2 + KY ** 2)).astype(int)
SETS = {"fno8": (KX >= -8) & (KX < 8) & (np.abs(KY) <= 8), "box16": (KX >= -16) & (KX < 16) & (np.abs(KY) <= 15)}
FR = [1, 2, 3, 5, 10]


def main():
    import geo_data as gd
    d = np.load(os.path.join(os.environ["SCRATCH"], "bench_cache", "alpha_bank_tdgl.npz"))
    la, G, ok = np.log(np.abs(d["alpha"])), d["G"].astype(np.float64), d["valid"]
    tests = [t for t in gd.index_d1(("useed",)) if t.split in ("interp", "extrap")][:256]
    out = {s: {n: [] for n in FR} for s in SETS}
    nz = KB != 0
    for t in tests:
        nn = np.argsort(np.abs(la - math.log(abs(t.mat[0]))))[:8]
        L = np.zeros(G.shape[1:])
        valid = np.zeros(G.shape[1], bool)
        for b in range(G.shape[1]):
            v = [j for j in nn if ok[j, b]]
            if v:
                L[b] = G[v, b].mean(0); valid[b] = True
        for n in FR:
            Gb = np.array([expm(n * float(t.theta) * L[b]) if valid[b] else np.zeros_like(L[b]) for b in range(L.shape[0])])
            Gk = Gb[KB]                                                   # (64,64,Z,Z)
            tot = (Gk[nz] ** 2).sum()
            for s, K in SETS.items():
                out_ = (~K) & nz
                C = Gk[out_].mean(0)
                out[s][n].append(float(np.sqrt(max(0.0, 1 - ((Gk[out_] - C) ** 2).sum() / tot))))
    res = {"frames": FR, "n_materials": len(tests),
           **{"cap_%s" % s: [float(np.mean(out[s][n])) for n in FR] for s in SETS},
           **{"cap_%s_min" % s: [float(np.min(out[s][n])) for n in FR] for s in SETS}}
    p = os.path.join(TK.HERE, "paper_tmlr", "data", "theory_cap_bank.json")
    json.dump(res, open(p, "w"), indent=1)
    print(json.dumps(res))


if __name__ == "__main__":
    main()
