#!/usr/bin/env python3
"""tdgl_frame_table.py -- per-frame corr/relL2 at t=1..10,50,100,200,400: tangent O(u^2) (Duhamel) vs paper SSM->FNO
pipeline vs FNO alone. Paper protocol (20 held-out trajectories, raw-noise cold start, top slice); median over every
archived snapshot (tangent: 2 runs x 15; FNO/pipeline: the 20 frozen checkpoints). Also wall-clock per 400-frame rollout.
"""
import glob, json, os, time
import numpy as np
import torch

import t400_common as tc
import ferro_ssm3d as fs
from eval_paper_final import snaps_of, load_snap, T0
from eval_t400_showdown import two_stage
from tdgl_tangent import Tangent

DEV = "cuda"
FR = list(range(1, 11)) + [20, 30, 40, 50, 55, 60, 100, 200, 400]


def score(tr, gt):
    out = {}
    for t in FR:
        p, g = tr[t][:, :, -1], gt[t][:, :, -1]
        out["corr%d" % t] = float(np.mean(tc.corr_batch(p, g)))
        out["rel%d" % t] = float(np.mean(tc.rel_l2_batch(p, g)))
    return out


def timed(fn):
    torch.cuda.synchronize(); t = time.perf_counter(); r = fn(); torch.cuda.synchronize()
    return r, time.perf_counter() - t


@torch.no_grad()
def main():
    _, val = tc.load_split(2)
    x0 = tc.val_tensor(val, 0)
    gt = {t: tc.val_tensor(val, t) for t in FR}
    prop = fs.PropagatorModel("incubation_propagator_bank.npz").to(DEV).eval()
    res = {"frames": FR, "fno_alone": [], "pipeline": [], "tangent": [], "time": {}}
    for path in snaps_of("t400_fno_an_k8"):
        m, ep = load_snap(path)
        tr, dt_f = timed(lambda: two_stage(prop, m, x0, 0, 400, FR))
        res["fno_alone"].append(score(tr, gt)); res["time"].setdefault("fno_alone", []).append(dt_f)
        tr, dt_p = timed(lambda: two_stage(prop, m, x0, T0, 400, FR))
        res["pipeline"].append(score(tr, gt)); res["time"].setdefault("pipeline", []).append(dt_p)
        print("fno e%s  %.1fs / %.1fs" % (ep, dt_f, dt_p), flush=True)
        del m; torch.cuda.empty_cache()
    for exp in ("tdgl_tangent_eq30", "tdgl_tangent_eq30_r2"):
        for p in sorted(glob.glob(os.path.join(os.environ["SCRATCH"], "ferrox_checkpoints", exp, "snapshots", "e*.pt"))):
            m = Tangent("incubation_propagator_bank.npz")
            sd = torch.load(p, map_location=DEV, weights_only=False)["model"]
            m.load_state_dict({k: v for k, v in sd.items() if k != "_metadata"}); m = m.to(DEV).eval()
            tr, dt = timed(lambda: tc.rollout(m, x0.to(DEV), 400, DEV, keep=FR))
            res["tangent"].append(score(tr, gt)); res["time"].setdefault("tangent", []).append(dt)
            print("%s %s %.1fs" % (exp, os.path.basename(p), dt), flush=True)
            del m; torch.cuda.empty_cache()
    res["summary"] = {a: {k: float(np.median([r[k] for r in res[a]])) for k in res[a][0]} for a in ("fno_alone", "pipeline", "tangent")}
    res["summary_time_s"] = {a: float(np.median(v)) for a, v in res["time"].items()}
    json.dump(res, open("tdgl_frame_table.json", "w"), indent=1)
    s = res["summary"]
    print("%5s | %-17s | %-17s | %-17s" % ("t", "FNO alone", "SSM->FNO", "Duhamel/tangent"))
    for t in FR:
        print("%5d | " % t + " | ".join("%.4f / %.2e" % (s[a]["corr%d" % t], s[a]["rel%d" % t]) for a in ("fno_alone", "pipeline", "tangent")))
    print("rollout time (20 traj x 400 frames, median):", res["summary_time_s"])


if __name__ == "__main__":
    main()
