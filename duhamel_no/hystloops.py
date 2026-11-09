#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""duhamel_no/hystloops.py -- save polarisation-voltage loops (spatial mean of P over the ferroelectric, per frame) of the
held-out-material hysteresis test loops for a few models, for the paper figure. Same checkpoints / rollout as run.py.
Out: results/phys/hyst_loops<BENCH_TAG>.npz  (V, truth and one array per model, per loop)
"""
import os, sys
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import models as M
import tasks as TK
import run as RUN

DEV = TK.DEV
CFGS = [("cno", 0.0003, True), ("duhamel_uno", 0.001, True), ("fno", 0.001, True)]


def main():
    task = TK.make("hyst")
    loops = [task.load(p) for p in task.t_mat[:: max(1, len(task.t_mat) // 6)][:6]]
    out = {}
    for i, L in enumerate(loops):
        out["V_%d" % i] = L["V"]; out["true_%d" % i] = L["P"][:, 0].mean(axis=(1, 2, 3)); out["name_%d" % i] = L["name"]
    for m, lr, wrap in CFGS:
        if m == "duhamel_uno":
            bank = os.path.join(os.environ["SCRATCH"], "bench_cache", "alpha_bank_hyst.npz")
            model = M.DuhamelUNO(task.nd, task.cin, task.dcu, task.guard, linear="bank", bank=bank, persist_k0=True)
        else:
            model = M.RecipeWrap(m, task.nd, task.cin, task.dcu, task.theta_ref, task.guard)
        model = model.to(DEV)
        tag = "hyst_%s_lr%g_recipe_s0%s" % (m, lr, os.environ.get("BENCH_TAG", ""))
        sd = {k: v for k, v in torch.load(os.path.join(RUN.CK, tag + ".pt"), map_location=DEV, weights_only=False)["model"].items()
              if not k.endswith("_metadata")}
        model.load_state_dict(sd); model.eval()

        def step(x, c, aux=None):
            if m == "duhamel_uno":
                return model(x, aux).float().clamp(-task.bound, task.bound)
            return model(x, aux["cu"], aux["U"], aux["th"]).float().clamp(-task.bound, task.bound)
        preds = []
        for L in loops:
            x0 = torch.from_numpy(L["P"][0, 0] / task.scale)[None].to(DEV).float()
            cs = task.cond_seq(L)[None].to(DEV)
            A = task.loop_aux(L)
            aux = dict(U=A["U"].view(1).to(DEV), th=A["th"].view(1).to(DEV), cu=A["cu"][None], mat=[tuple(float(v) for v in L["mat"])])
            with torch.no_grad():
                preds.append(torch.cat([x0[None], RUN.rollout(step, x0, cs, aux)], 1)[0])
        for i, p in enumerate(preds):
            out["%s_%d" % (m, i)] = (p.float().cpu().numpy() * task.scale).mean(axis=(1, 2, 3))
        print("done", m, flush=True)
    np.savez(os.path.join(os.path.dirname(HERE), "results", "phys", "hyst_loops%s.npz" % os.environ.get("BENCH_TAG", "")), **out)
    print("DONE hystloops", flush=True)


if __name__ == "__main__":
    main()
