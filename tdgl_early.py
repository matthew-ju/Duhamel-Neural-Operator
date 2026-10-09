#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tdgl_early.py -- FNO alone vs Fourier-SSM (linear propagator) -> FNO on the paper's TDGL setup, FRAME BY FRAME.

Same protocol as eval_paper_final.py: 20 held-out trajectories (fc3 split, seed 42, alpha = -2e9), every one of the
20 frozen FNO checkpoints, cold start from the raw t=0 noise, top z-slice. Arms:
  fno_alone   FNO rolled out from the raw noise (t_switch = 0)
  pipeline    linear propagator bank for frames 1..50, FNO afterwards (the paper's model)
  prop_only   the propagator alone (shows where linearity ends)
Metrics per frame t: pattern corr, relL2 (top slice) and relL2 over the full 10-layer FE volume; sat-RMSE as in the
paper (frames >= 100, every 10). Reported: median over the 20 checkpoints (paper convention) and min-max.
"""
import json
import numpy as np
import torch

import ferro_common as fc
import t400_common as tc
import ferro_ssm3d as fs
from eval_paper_final import snaps_of, load_snap, T0
from eval_t400_showdown import two_stage

DEV = "cuda"
FR = [1, 2, 3, 5, 10, 15, 20, 30, 40, 50, 55, 60, 70, 80, 100, 150, 200, 300, 400]


def rel_vol(p, g):
    p = p.reshape(p.shape[0], -1).double(); g = g.reshape(g.shape[0], -1).double()
    return ((p - g).norm(dim=1) / g.norm(dim=1).clamp_min(1e-30)).numpy()


def score(tr, gt, sat):
    out = {}
    for t in FR:
        p, g = tr[t][:, :, -1], gt[t][:, :, -1]
        out["corr%d" % t] = float(np.mean(tc.corr_batch(p, g)))
        out["rel%d" % t] = float(np.mean(tc.rel_l2_batch(p, g)))
        out["relvol%d" % t] = float(np.mean(rel_vol(tr[t], gt[t])))
    out["sat_rmse"] = float(np.mean([float(torch.sqrt(((tr[t][:, :, -1] - gt[t][:, :, -1]) ** 2).mean())) for t in sat]))
    return out


def main():
    _, val = tc.load_split(2)
    x0 = tc.val_tensor(val, 0)
    sat = [t for t in range(fc.NUCLEATION_END, 401, 10)]
    keep = sorted(set(FR + sat + [T0]))
    gt = {t: tc.val_tensor(val, t) for t in keep}
    prop = fs.PropagatorModel("incubation_propagator_bank.npz").to(DEV).eval()
    res = {"prop_only": score(two_stage(prop, prop, x0, 400, 400, keep), gt, sat), "fno_alone": [], "pipeline": []}
    for path in snaps_of("t400_fno_an_k8"):
        m, ep = load_snap(path)
        res["fno_alone"].append(dict(epoch=ep, **score(two_stage(prop, m, x0, 0, 400, keep), gt, sat)))
        res["pipeline"].append(dict(epoch=ep, **score(two_stage(prop, m, x0, T0, 400, keep), gt, sat)))
        print("E%d done" % ep, flush=True)
        del m; torch.cuda.empty_cache()
    summ = {}
    for arm in ("fno_alone", "pipeline"):
        summ[arm] = {k: dict(median=float(np.median([r[k] for r in res[arm]])), lo=float(np.min([r[k] for r in res[arm]])),
                             hi=float(np.max([r[k] for r in res[arm]]))) for k in res[arm][0] if k != "epoch"}
    res["summary"] = summ
    json.dump(res, open("tdgl_early.json", "w"), indent=1)
    print("%6s | %-26s | %-26s | %-26s" % ("frame", "corr  fno / pipe", "relL2 top  fno / pipe", "relL2 vol  fno / pipe"))
    for t in FR:
        f, p = summ["fno_alone"], summ["pipeline"]
        print("%6d | %.4f / %.4f            | %.4f / %.4f            | %.4f / %.4f" % (
            t, f["corr%d" % t]["median"], p["corr%d" % t]["median"], f["rel%d" % t]["median"], p["rel%d" % t]["median"],
            f["relvol%d" % t]["median"], p["relvol%d" % t]["median"]))
    print("sat_rmse fno %.4f  pipeline %.4f   prop_only corr1 %.6f rel1 %.2e" % (
        summ["fno_alone"]["sat_rmse"]["median"], summ["pipeline"]["sat_rmse"]["median"], res["prop_only"]["corr1"], res["prop_only"]["rel1"]))


if __name__ == "__main__":
    main()
