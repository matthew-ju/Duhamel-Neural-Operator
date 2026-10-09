#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""bench/restransfer.py -- zero-shot resolution transfer on Allen-Cahn: a model trained on the 64x64 grid is rolled out,
unchanged, on the SAME physics simulated on 32x32 and 128x128 grids of the same 2pi domain (ml_units ac32_* / ac128_*;
physics sampled in 64-grid units, white noise of the same physical strength). Fields use the 64-task normalisation; the
conditioning is resolution-free (raw physical parameters; Pi_h expressed in 64-grid units). Metrics as bench/run.py.
Out: bench/results_res/<tag>.json  {grid: {split: metrics}}
"""
import argparse, json, os, sys
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import models as M
import tasks as TK
import run as RUN

DEV = TK.DEV
OUT = os.path.join(HERE, "results_res"); os.makedirs(OUT, exist_ok=True)


def sets_for(task, name, n):
    out = {}
    for sp in ("id", "ood_units", "ood_inv"):
        p = os.path.join(task.T.D, "%s_%s.npz" % (name, sp))
        if not os.path.exists(p):
            continue
        S = task.T.load_split(name, sp)
        dml = S["dml"].copy(); dml[:, 1] *= n / 64.0                     # Pi_h = dx/mu in 64-grid units
        out[sp] = dict(u=torch.from_numpy(S["u"] / task.scale).unsqueeze(2), cond=torch.from_numpy((S["raw"] - task.cm) / task.cs).float(),
                       batched=True, aux=dict(U=torch.from_numpy(S["U"] / task.scale).float(), th=torch.from_numpy(S["theta"]).float(),
                                              cu=torch.from_numpy(dml - task.cref).float()))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="ac")
    ap.add_argument("--model", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--loss", default="mse")
    ap.add_argument("--wrap", action="store_true")
    ap.add_argument("--scale", default="global")
    ap.add_argument("--norm", default="global")
    a = ap.parse_args()
    torch.manual_seed(1234)
    task = TK.make(a.task)
    if a.model in ("duhamel_ou2",) or a.model.startswith("duhamel_uno"):
        a.wrap = True
    gen = a.model in M.GENERATIVE
    if a.wrap:
        a.loss = "rel"
    cfg = "lr%g_%s%s" % (a.lr, ("scale-" + a.scale) if gen else ("recipe" if a.wrap else a.loss), "_inst" if (a.norm == "inst" and not gen and not a.wrap) else "")
    tag = "%s_%s_%s_s%d" % (a.task, a.model, cfg, a.seed)
    if a.model == "duhamel_ou2":
        model = M.DuhamelOU2(task.cref, task.guard, a.task)
    elif a.model.startswith("duhamel_uno"):
        model = M.DuhamelUNO(task.nd, task.cin, task.dcu, task.guard, linear="kmu" if a.model == "duhamel_uno_kmu" else "poly",
                             cref=task.cref)
    elif a.wrap:
        model = M.RecipeWrap(a.model, task.nd, task.cin, task.dcu, task.theta_ref, task.guard)
    else:
        model = M.build2(a.model, task.nd, task.cin, task.dcond, a.scale)
        if a.norm == "inst" and not gen:
            model = M.InstNorm(model)
    model = model.to(DEV)
    ck = torch.load(os.path.join(RUN.CK, tag + ".pt"), map_location=DEV, weights_only=False)
    model.load_state_dict({k: v for k, v in ck["model"].items() if not k.endswith("_metadata")}, strict=False)
    model.eval()

    def step(x, c, aux=None):
        if a.model == "duhamel_ou2" or a.model.startswith("duhamel_uno"):
            return RUN.clampb(task, model(x, aux).float())
        if a.wrap:
            return RUN.clampb(task, model(x, aux["cu"], aux["U"], aux["th"]).float())
        return RUN.clampb(task, model(x, c).float())
    step.wrap = a.wrap
    res = dict(tag=tag, model=a.model, seed=a.seed, grids={})
    for name, n in ((a.task, 64), (a.task + "32", 32), (a.task + "128", 128)):
        sets = sets_for(task, name, n)
        if not sets:
            continue
        try:
            ev = RUN.evaluate(task, a.task, step, sets=sets)
            res["grids"][str(n)] = {sp: {k: v for k, v in m.items() if k not in ("rows", "subsets")} for sp, m in ev.items()}
        except Exception as e:
            res["grids"][str(n)] = {"error": repr(e)[:300]}
            print("grid", n, "failed:", repr(e)[:300], flush=True)
    json.dump(res, open(os.path.join(OUT, tag + ".json"), "w"), indent=1)
    print("DONE", tag, flush=True)


if __name__ == "__main__":
    main()
