#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""bench/speed.py -- wall-clock seconds per 400-frame FerroX TDGL rollout for every benchmark model, on ONE dedicated
GPU (run it in its own allocation: GPU contention silently inflates timings). Batch 1 and batch 16; 1 warm-up rollout
(20 frames), then median of 3 full rollouts; fp32 as in the evaluation. Same checkpoints / constructors as physmetrics.
FerroX reference: median 196 s per Dataset-1 trajectory (range 147-319 s; bench_abg.json, nsteps x 0.06125 s/step).
Out: bench/results_phys/speed.json
"""
import json, os, sys, time
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import models as M
import tasks as TK
import run as RUN
import physmetrics as PM
from types import SimpleNamespace as NS

DEV = TK.DEV
CFGS = [  # (label, model, lr, wrap, loss, scale)
    ("Duhamel-UNO (ours)", "duhamel_uno", 0.0003, True, "rel", "global"),
    ("F-FNO + recipe", "ffno", 0.001, True, "rel", "global"),
    ("U-Net + recipe", "unet", 0.0003, True, "rel", "global"),
    ("UNO + recipe", "uno", 0.001, True, "rel", "global"),
    ("CNextU-Net + recipe", "cnext", 0.0003, True, "rel", "global"),
    ("CNO + recipe", "cno", 0.001, True, "rel", "global"),
    ("TFNO + recipe", "tfno", 0.001, True, "rel", "global"),
    ("FNO + recipe", "fno", 0.001, True, "rel", "global"),
    ("Duhamel, learned linear (ours)", "duhamel", 0.001, True, "rel", "global"),
    ("Transolver + recipe", "transolver", 0.001, True, "rel", "global"),
    ("ACDM", "acdm", 0.001, False, "mse", "global"),
    ("Flow matching", "flowmatch", 0.001, False, "mse", "global"),
    ("PDE-Refiner", "refiner2", 0.001, False, "mse", "global"),
]


def main():
    task = TK.make("tdgl")
    trajs = task.eval_sets()["D1_heldout"]["trajs"][:16]
    u, c = task.load_batch(trajs)
    u = u.to(DEV).float(); c = c.to(DEV).float()
    A0 = RUN._aux_to(task.batch_aux)
    out = dict(gpu=torch.cuda.get_device_name(0), ferrox_s_median=196.0, ferrox_s_range=[147.0, 318.5], models={})
    for lab, mname, lr, wrap, loss, scale in CFGS:
        a = NS(model=mname, lr=lr, wrap=wrap, loss=loss, scale=scale, norm="global", seed=0)
        gen = mname in M.GENERATIVE
        cfg = "lr%g_%s" % (lr, ("scale-" + scale) if gen else ("recipe" if wrap else loss))
        tag = "tdgl_%s_%s_s0" % (mname, cfg)
        try:
            model, gen = PM.build(a, task)
            ck = os.path.join(RUN.CK, tag + ".pt")
            if os.path.exists(ck):
                model.load_state_dict({k: v for k, v in torch.load(ck, map_location=DEV, weights_only=False)["model"].items()
                                       if not k.endswith("_metadata")})
            model.eval()
        except Exception as e:
            print("skip", lab, e, flush=True); continue

        def step(x, cc, aux=None):
            if mname == "duhamel" or mname.startswith("duhamel_uno"):
                return model(x, aux).float().clamp(-task.bound, task.bound)
            if wrap:
                return model(x, aux["cu"], aux["U"], aux["th"]).float().clamp(-task.bound, task.bound)
            return model(x, cc).float().clamp(-task.bound, task.bound)
        res = {"params": sum(p.numel() for p in model.parameters())}
        for B in (1, 16):
            A = {k: (v[:B] if torch.is_tensor(v) or isinstance(v, list) else v) for k, v in A0.items()}
            def roll(T):
                AA = dict(A); AA["cu"] = A["cu"][:B].unsqueeze(1).expand(-1, T, -1)
                cs = c[:B].unsqueeze(1).expand(-1, T, -1)
                torch.cuda.synchronize(); t0 = time.time()
                RUN.rollout(step, u[:B, 0], cs, AA)
                torch.cuda.synchronize(); return time.time() - t0
            roll(20)
            ts = [roll(400) for _ in range(3)]
            res["s_per_traj_b%d" % B] = float(np.median(ts)) / B
            res["speedup_b%d" % B] = 196.0 / res["s_per_traj_b%d" % B]
        out["models"][lab] = res
        print(lab, json.dumps(res), flush=True)
        del model; torch.cuda.empty_cache()
    # our production FNO pipeline (per-trajectory roll, batch 1)
    try:
        f = RUN.ours_tdgl(task, "ours_fno"); f(trajs[:1]); torch.cuda.synchronize()
        ts = []
        for _ in range(3):
            t0 = time.time(); f(trajs[:1]); torch.cuda.synchronize(); ts.append(time.time() - t0)
        out["models"]["production FNO (ours, earlier)"] = dict(s_per_traj_b1=float(np.median(ts)), speedup_b1=196.0 / float(np.median(ts)))
    except Exception as e:
        print("skip production", e, flush=True)
    json.dump(out, open(os.path.join(PM.OUT, "speed.json"), "w"), indent=1)
    print("DONE speed", flush=True)


if __name__ == "__main__":
    main()
