#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""bench/curves.py -- per-frame error curves for the AI4Mat hysteresis / Allen-Cahn figure. Re-loads the validation-selected
checkpoints ($SCRATCH/bench_ckpt/<tag>.pt) and repeats the rollout of run.py, but keeps the whole error curve instead of
only the final-frame numbers. Never writes to results_final.

  hyst (export HYST_DIR=ferrox_hyst_closed2 BENCH_TAG=_closed):  python bench/curves.py --task hyst
      per loop: |<P>_pred - <P>_true| / P_s per frame (its time mean is the loop error), field relative L2 per frame;
      splits test_material (30 held-out-material loops) and closed (10 extra closed loops)
  ac:  python bench/curves.py --task ac
      relative L2 per frame; splits ood_inv on the trained 64^2 grid and id on the zero-shot 128^2 and 32^2 grids;
      a file missing a split is completed
Out: bench/results_curves/<tag>.json
"""
import argparse, json, os, re, sys
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import models as M
import tasks as TK
import run as RUN
import summarize2 as S2

DEV = TK.DEV
OUT = os.path.join(HERE, "results_curves"); os.makedirs(OUT, exist_ok=True)
MODELS = {"hyst": ["duhamel_uno", "ffno", "unet", "uno", "cnext", "cno", "tfno", "fno"],
          "ac": ["duhamel_uno_kmu", "ffno", "unet", "uno", "cnext", "cno", "tfno", "fno"]}


def build(task, tname, d):
    m, cfg = d["model"], d["cfg"]
    wrap = "recipe" in cfg
    if m == "duhamel_uno":
        bank = os.path.join(os.environ["SCRATCH"], "bench_cache", "alpha_bank_%s.npz" % tname)
        model = M.DuhamelUNO(task.nd, task.cin, task.dcu, task.guard, linear="bank", bank=bank, persist_k0=tname == "hyst",
                             cref=getattr(task, "cref", None))
    elif m == "duhamel_uno_kmu":
        model = M.DuhamelUNO(task.nd, task.cin, task.dcu, task.guard, linear="kmu", cref=task.cref)
    elif wrap:
        model = M.RecipeWrap(m, task.nd, task.cin, task.dcu, task.theta_ref, task.guard)
    else:
        model = M.build2(m, task.nd, task.cin, task.dcond, "global")
        if cfg.endswith("_inst"):
            model = M.InstNorm(model)
    model = model.to(DEV)
    ck = torch.load(os.path.join(RUN.CK, d["tag"] + ".pt"), map_location=DEV, weights_only=False)
    model.load_state_dict({k: v for k, v in ck["model"].items() if not k.endswith("_metadata")})
    model.eval()

    def step(x, c, aux=None):
        if m.startswith("duhamel"):
            return RUN.clampb(task, model(x, aux).float())
        if wrap:
            return RUN.clampb(task, model(x, aux["cu"], aux["U"], aux["th"]).float())
        return RUN.clampb(task, model(x, c).float())
    step.wrap = wrap or m.startswith("duhamel")
    return step


@torch.no_grad()
def hyst_curves(task, step, paths):
    out = []
    for Lp in paths:
        L = task.load(Lp)
        x0 = torch.from_numpy(L["P"][0, 0] / task.scale)[None].to(DEV)
        cs = task.cond_seq(L)[None].to(DEV)
        aux = None
        if step.wrap:
            A = task.loop_aux(L)
            aux = dict(U=A["U"].view(1).to(DEV), th=A["th"].view(1).to(DEV), cu=A["cu"][None], mat=[tuple(float(v) for v in L["mat"])])
        pred = torch.cat([x0[None], RUN.rollout(step, x0, cs, aux)], 1)[0]
        P = (pred * task.scale).unsqueeze(1).cpu().numpy()
        r, mp, mt = task.th.loop_metrics(P, L)
        ps = float(task.al.p_sat(*L["mat"][:3])) / task.al.P_NORM
        G = torch.from_numpy(L["P"][1:, 0] / task.scale).to(DEV)[None]
        rl, _, _ = RUN.per_traj_metrics(pred[1:][None].to(DEV), G)
        out.append(dict(name=L["name"], V=[float(v) for v in L["V"]], dev=(np.abs(mp - mt) / ps).tolist(),
                        rel=rl[0].cpu().numpy().tolist(), loop_err=r["loop_err"]))
    return out


@torch.no_grad()
def field_curves(task, step, S):
    N = S["u"].shape[0]; curves = []
    for i in range(0, N, 32):
        b = slice(i, i + 32)
        u, c = S["u"][b].to(DEV).float(), S["cond"][b].to(DEV).float()
        aux = {k: v[b] for k, v in S["aux"].items()}
        if hasattr(task, "AUX"):
            task._bs = torch.clamp(aux["U"].to(DEV).float() / task.AUX["U"].max(), min=1.0)
        cs = c.unsqueeze(1).expand(-1, u.shape[1] - 1, -1)
        A = RUN._aux_to(aux); A["cu"] = A["cu"].unsqueeze(1).expand(-1, u.shape[1] - 1, -1)
        P = RUN.rollout(step, u[:, 0], cs, A) if step.wrap else RUN.rollout(step, u[:, 0], cs)
        rl, _, _ = RUN.per_traj_metrics(P, u[:, 1:])
        task._bs = None
        curves.append(rl.cpu().numpy())
    return np.concatenate(curves).tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=["hyst", "ac"])
    ap.add_argument("--only", default="", help="comma list of models")
    a = ap.parse_args()
    task = TK.make(a.task)
    R = S2.load(closed=a.task == "hyst")
    sets = None
    if a.task == "ac":
        import restransfer as RT
        sets = {"ood_inv_64": task.eval_sets()["ood_inv"], "id_128": RT.sets_for(task, "ac128", 128)["id"],
                "id_32": RT.sets_for(task, "ac32", 32)["id"]}
    for m in (a.only.split(",") if a.only else MODELS[a.task]):
        for d in sorted(R.get((a.task, m), []), key=lambda d: d["tag"]):
            fo = os.path.join(OUT, d["tag"] + ".json")
            res = json.load(open(fo)) if os.path.exists(fo) else dict(tag=d["tag"], model=m, seed=d["seed"], task=a.task)
            need = ["test_material", "closed"] if a.task == "hyst" else list(sets)
            if all(k in res for k in need):
                print("have", d["tag"], flush=True); continue
            torch.manual_seed(1234)
            step = build(task, a.task, d)
            if a.task == "hyst":
                for sp, ps in (("test_material", task.t_mat), ("closed", task.t_closed)):
                    res[sp] = hyst_curves(task, step, ps)
                    print(d["tag"], sp, "loop_err %.4f (table %.4f)" % (np.mean([r["loop_err"] for r in res[sp]]),
                                                                       d["eval"][sp]["loop_err"]), flush=True)
            else:
                for sp, S in sets.items():
                    if sp in res:
                        continue
                    res[sp] = field_curves(task, step, S)
                    print(d["tag"], sp, "rel_final %.4f" % np.mean([c[-1] for c in res[sp]]), flush=True)
            json.dump(res, open(fo, "w"))
            del step; torch.cuda.empty_cache()
    print("DONE curves", a.task, flush=True)


if __name__ == "__main__":
    main()
