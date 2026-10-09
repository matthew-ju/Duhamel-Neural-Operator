#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tdgl_tangent_eval.py -- frame-by-frame scores of the equal-budget TDGL runs, paper protocol (20 held-out
trajectories, raw-noise cold start, top slice): tangent O(u^2) model vs the from-scratch FNO trained with the same
data/windows/loss/epochs, next to the paper's pipeline and FNO-alone (tdgl_early.json). Uses each run's best
(cold-start corr@400) checkpoint and, for an equal-epoch comparison, the snapshot at the largest epoch both runs have.
"""
import glob, json, os, argparse
import numpy as np
import torch

import ferro_common as fc
import t400_common as tc
from eval_t400_showdown import build_model
from tdgl_tangent import Tangent
from tdgl_early import FR, score

DEV = "cuda"


def load(exp, which):
    d = os.path.join(os.environ["SCRATCH"], "ferrox_checkpoints", exp)
    path = os.path.join(d, "%s_best.pt" % exp) if which == "best" else os.path.join(d, "snapshots", "e%04d.pt" % which)
    if not os.path.exists(path):
        return None, None
    sd = torch.load(path, map_location=DEV, weights_only=False)
    m = Tangent("incubation_propagator_bank.npz") if "tangent" in exp else build_model(argparse.Namespace(
        **torch.load("paper_frozen/t400_fno_an_k8/e0090.pt", map_location="cpu", weights_only=False)["args"]))
    m.load_state_dict({k: v for k, v in sd["model"].items() if k != "_metadata"})
    return m.to(DEV).eval(), sd.get("epoch")


@torch.no_grad()
def main():
    _, val = tc.load_split(2)
    x0 = tc.val_tensor(val, 0)
    sat = [t for t in range(fc.NUCLEATION_END, 401, 10)]
    keep = sorted(set(FR + sat))
    gt = {t: tc.val_tensor(val, t) for t in keep}
    snaps = {}
    EXPS = [e for e in ("tdgl_tangent_eq30", "tdgl_tangent_eq30_r2", "tdgl_fno_eq30")
            if os.path.isdir(os.path.join(os.environ["SCRATCH"], "ferrox_checkpoints", e))]
    for exp in EXPS:
        d = os.path.join(os.environ["SCRATCH"], "ferrox_checkpoints", exp, "snapshots")
        snaps[exp] = sorted(int(os.path.basename(p)[1:5]) for p in glob.glob(os.path.join(d, "e*.pt")))
    common = sorted(set(snaps["tdgl_tangent_eq30"]) & set(snaps["tdgl_fno_eq30"]))
    out = {"common_epochs": common, "snapshots": snaps}
    for exp in EXPS:
        # paper protocol: EVERY archived snapshot (no best-checkpoint selection), plus the selected best for reference
        for which in ["best"] + snaps[exp]:
            m, ep = load(exp, which)
            if m is None:
                continue
            tr = tc.rollout(m, x0.to(DEV), 400, DEV, keep=keep)
            out["%s_%s" % (exp, which)] = dict(epoch=ep, **score(tr, gt, sat))
            r = out["%s_%s" % (exp, which)]
            print("%-28s e%-4s " % (exp, ep) + "  ".join("t%d %.4f/%.3g" % (t, r["corr%d" % t], r["rel%d" % t]) for t in (1, 2, 5, 10, 50, 100, 200, 400))
                  + "  sat %.4f" % r["sat_rmse"], flush=True)
            del m; torch.cuda.empty_cache()
    summ = {}
    for exp in EXPS:
        rows = [out["%s_%s" % (exp, e)] for e in snaps[exp] if "%s_%s" % (exp, e) in out]
        if not rows:
            continue
        summ[exp] = {k: dict(median=float(np.median([r[k] for r in rows])), lo=float(np.min([r[k] for r in rows])),
                             hi=float(np.max([r[k] for r in rows])), n=len(rows)) for k in rows[0] if k != "epoch"}
        s = summ[exp]
        print("MEDIAN over %d snapshots %-20s " % (len(rows), exp) + "  ".join(
            "t%d %.4f/%.3g" % (t, s["corr%d" % t]["median"], s["rel%d" % t]["median"]) for t in (1, 2, 5, 10, 50, 100, 200, 400))
            + "  sat %.4f" % s["sat_rmse"]["median"], flush=True)
    out["summary"] = summ
    json.dump(out, open("tdgl_tangent_eval.json", "w"), indent=1)


if __name__ == "__main__":
    main()
