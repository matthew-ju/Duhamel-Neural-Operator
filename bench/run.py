#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""bench/run.py -- train one baseline on one task (equal budget for every model), then roll out every test set.

  python bench/run.py --task ac --model unet --seed 0            # train + eval
  python bench/run.py --task tdgl --model ours_fno               # eval our trained FerroX model in the same harness
Budget (all baselines): --steps 20000, --bs 16, k = 4 step unrolled relative-L2 loss (PDE-Refiner: its own one-step
denoising objective), AdamW + OneCycle, grad clip 1. Metrics (identical to ml_units/recipe.py): per trajectory
relative L2 (not squared) per frame, pattern correlation at the final frame; hysteresis also loop metrics.
Out: bench/results/<task>_<model>_s<seed>.json ; checkpoints $SCRATCH/bench_ckpt/
"""
import argparse, json, os, sys, time
import numpy as np
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import models as M
import tasks as TK

OUT = os.path.join(HERE, "results"); os.makedirs(OUT, exist_ok=True)
CK = os.path.join(os.environ["SCRATCH"], "bench_ckpt"); os.makedirs(CK, exist_ok=True)
DEV = TK.DEV
torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True
NO_AMP = ("fno", "tfno", "ffno", "refiner", "cno")
FRAMES = {"tdgl": [1, 2, 5, 10, 50, 100, 200, 400]}


def relsq(a, b):
    red = tuple(range(1, a.dim()))
    return ((a - b) ** 2).sum(red) / ((b ** 2).sum(red) + 1e-12)


def per_traj_metrics(P, G):
    """P, G (B, T, C, *S) predicted / true frames 1..T (normalised) -> per-trajectory dict lists."""
    red = tuple(range(2, P.dim()))
    rl = ((P - G) ** 2).sum(red).sqrt() / (G ** 2).sum(red).sqrt().clamp_min(1e-12)        # (B,T)
    xf, gf = P[:, -1], G[:, -1]; r2 = tuple(range(1, xf.dim()))
    xc, gc = xf - xf.mean(r2, keepdim=True), gf - gf.mean(r2, keepdim=True)
    corr = (xc * gc).sum(r2) / (xc.pow(2).sum(r2).sqrt() * gc.pow(2).sum(r2).sqrt() + 1e-12)
    amp = (xf.pow(2).mean(r2).sqrt() / gf.pow(2).mean(r2).sqrt().clamp_min(1e-12) - 1).abs()
    return rl, corr, amp


@torch.no_grad()
def rollout(step, x0, cond_seq, aux=None):
    """step(x, c[, aux_t]) ; x0 (B,C,*S) ; cond_seq (B,T-1,D) -> (B,T-1,C,*S). aux: dict(U (B,), th (B,), cu (B,T-1,Du))."""
    x = x0; out = []
    for t in range(cond_seq.shape[1]):
        if aux is None:
            y = step(x, cond_seq[:, t])
        else:
            y = step(x, cond_seq[:, t], dict(U=aux["U"], th=aux["th"], cu=aux["cu"][:, t], mat=aux.get("mat")))
        x = torch.nan_to_num(y).clamp(-50, 50)
        out.append(x)
    return torch.stack(out, 1)


def clampb(task, y):
    """output bound: 3x the training max, scaled per sample by its amplitude unit when that exceeds the training range
    (the 'units' test split has amplitudes up to 5x the training max; a fixed bound would truncate the TRUE field --
    harness bug found 2026-10-04, affected every model's ood_units score)."""
    bs = getattr(task, "_bs", None)
    if bs is None or bs.shape[0] != y.shape[0]:
        return y.clamp(-task.bound, task.bound)
    b = (task.bound * bs).view(-1, *([1] * (y.dim() - 1)))
    return torch.maximum(torch.minimum(y, b), -b)


def _aux_to(aux, n=None):
    return {k: (v.to(DEV).float() if torch.is_tensor(v) else v) for k, v in aux.items()}


def evaluate(task, tname, step, ours=None, sets=None):
    res = {}
    NE = int(os.environ.get("BENCH_EVAL_N", "0")) or None          # >0: tiny eval for testing the code path
    for sp, S in (sets if sets is not None else task.eval_sets()).items():
        rows = []
        if NE:
            for key in ("loops", "trajs"):
                if key in S:
                    S[key] = S[key][:NE]
            if "u" in S:
                S["u"], S["cond"] = S["u"][:NE], S["cond"][:NE]
                if "aux" in S:
                    S["aux"] = {k: v[:NE] for k, v in S["aux"].items()}
        if tname == "hyst":
            for Lp in S["loops"]:
                L = task.load(Lp)
                if ours is not None:
                    pred = ours(L)                                           # (T, C, H, W) normalised, incl. frame 0
                else:
                    x0 = torch.from_numpy(L["P"][0, 0] / task.scale)[None].to(DEV)
                    cs = task.cond_seq(L)[None].to(DEV)
                    aux = None
                    if getattr(step, "wrap", False):
                        A = task.loop_aux(L); aux = dict(U=A["U"].view(1).to(DEV), th=A["th"].view(1).to(DEV), cu=A["cu"][None],
                                                         mat=[tuple(float(v) for v in L["mat"])])
                    pred = torch.cat([x0[None], rollout(step, x0, cs, aux)], 1)[0]
                m = task.loop_metrics(pred, L)
                G = torch.from_numpy(L["P"][1:, 0] / task.scale).to(DEV)[None]
                rl, corr, amp = per_traj_metrics(pred[1:][None].to(DEV), G)
                m.update(rel_final=float(rl[0, -1]), corr_final=float(corr[0]), rel_t1=float(rl[0, 0]), name=L.get("name", ""),
                         tag=L.get("tag", ""), vmax="V%.1f" % L.get("vmax", 0), seed=int(L.get("seed", -1)))
                rows.append(m)
        else:
            if S.get("lazy"):
                batches = [S["trajs"][i:i + 16] for i in range(0, len(S["trajs"]), 16)]
            else:
                N = S["u"].shape[0]; batches = [slice(i, i + 32) for i in range(0, N, 32)]
            for b in batches:
                if S.get("lazy"):
                    u, c = task.load_batch(b)
                    if ours is not None:
                        P = ours(b)
                        u = u.to(DEV)
                        rl, corr, amp = per_traj_metrics(P, u[:, 1:]); _store(rows, rl, corr, amp, tname); _meta(rows, b); continue
                    aux = getattr(task, "batch_aux", None)
                else:
                    u, c = S["u"][b], S["cond"][b]
                    aux = {k: v[b] for k, v in S["aux"].items()} if "aux" in S else None
                    if aux is not None and hasattr(task, "AUX"):
                        task._bs = torch.clamp(aux["U"].to(DEV).float() / task.AUX["U"].max(), min=1.0)
                u, c = u.to(DEV).float(), c.to(DEV).float()
                cs = c.unsqueeze(1).expand(-1, u.shape[1] - 1, -1)
                if getattr(step, "wrap", False):
                    A = _aux_to(aux); A["cu"] = A["cu"].unsqueeze(1).expand(-1, u.shape[1] - 1, -1)
                    P = rollout(step, u[:, 0], cs, A)
                else:
                    P = rollout(step, u[:, 0], cs)
                rl, corr, amp = per_traj_metrics(P, u[:, 1:])
                task._bs = None
                _store(rows, rl, corr, amp, tname)
                if S.get("lazy"):
                    _meta(rows, b)
        keys = [k for k in rows[0] if isinstance(rows[0][k], float)]
        res[sp] = {k: float(np.mean([r[k] for r in rows])) for k in keys}
        res[sp].update({k + "_median": float(np.median([r[k] for r in rows])) for k in keys})
        res[sp]["n"] = len(rows)
        res[sp]["subsets"] = _subsets(rows, keys)
        res[sp]["rows"] = rows
        print("%-14s %s" % (sp, "  ".join("%s %.4f" % (k, res[sp][k]) for k in keys)), flush=True)
    return res


def _meta(rows, trajs):
    for r, t in zip(rows[-len(trajs):], trajs):
        r.update(tag=t.tag, seed=int(t.seed), subset=(t.box if getattr(t, "box", None) else t.split))


def _subsets(rows, keys):
    """mean / median of every float metric per subset label (TDGL: interp / extrap, D2 box; hysteresis: V amplitude)."""
    out = {}
    for lab in ("subset", "vmax"):
        groups = {}
        for r in rows:
            if lab in r:
                groups.setdefault(r[lab], []).append(r)
        for g, rs in sorted(groups.items()):
            out[g] = dict({k: float(np.mean([r[k] for r in rs])) for k in keys},
                          **{k + "_median": float(np.median([r[k] for r in rs])) for k in keys}, n=len(rs))
    return out


def _store(rows, rl, corr, amp, tname):
    for j in range(rl.shape[0]):
        r = dict(rel_final=float(rl[j, -1]), rel_mean=float(rl[j].mean()), corr_final=float(corr[j]), amp_final=float(amp[j]),
                 rel_t1=float(rl[j, 0]))
        for f in FRAMES.get(tname, []):
            if f - 1 < rl.shape[1]:
                r["rel_t%d" % f] = float(rl[j, f - 1])
        rows.append(r)


# ------------------------------------------------------------------------------------------- our models
def ours_tdgl(task, which):
    sys.path.insert(0, os.path.join(TK.HERE, "geo"))
    import plot_rollouts4 as P4
    gm = P4.gm
    m = gm.from_d1_checkpoint(os.path.join(TK.HERE, "q_probe", "runs", "ft_useedonly.pt")).to(DEV).eval()
    ssm = None
    if which == "ours_switch5":
        from train_handoff import build_ssm
        ssm = build_ssm(os.path.join(TK.HERE, "handoff", "runs", "ssm_lin20.pt")).to(DEV).eval()

    @torch.no_grad()
    def f(trajs):
        outs = []
        for t in trajs:
            pr = P4.roll(m, t, ssm=ssm)
            outs.append(torch.stack([torch.from_numpy(pr[s][0]) for s in range(1, 401)]))
        return torch.stack(outs).to(DEV) / task.scale
    return f


def ours_hyst(task, which):
    th = task.th
    if which == "ours_fno_pre":
        hm = th.HystFNO().to(DEV).eval(); ck = os.environ.get("HYST_OURS_CKPT", "hyst4_k12_ft.pt")
    else:
        from duhamel_hyst import DuhamelHyst
        hm = DuhamelHyst(ssm_init="").to(DEV).eval(); ck = "hystD_k8.pt"
    hm.load_state_dict({k: v for k, v in torch.load(os.path.join(TK.HERE, "q_probe", "volt_sweep", "runs", ck), map_location=DEV,
                                                   weights_only=False)["model"].items() if k != "_metadata"})

    def f(L):
        return torch.from_numpy(th.roll_loop(hm, L, DEV)[:, 0] / task.scale).to(DEV)
    return f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--mode", default="final", choices=["tune", "final"], help="tune: short run, scored on the validation split only")
    ap.add_argument("--steps", type=int, default=0, help="0 = 4000 (tune) / 20000 (final)")
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--loss", default="mse", choices=["mse", "rel"])
    ap.add_argument("--wrap", action="store_true", help="our training recipe around the backbone (models.RecipeWrap); forces rel")
    ap.add_argument("--scale", default="global", choices=["global", "inst"], help="generative models: residual unit")
    ap.add_argument("--norm", default="global", choices=["global", "inst"], help="deterministic models: per-sample (RevIN-style) normalisation")
    ap.add_argument("--no_aug", action="store_true")
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--fp32", action="store_true", help="(legacy, fp32 is now the default)")
    ap.add_argument("--eval_only", action="store_true", help="load $SCRATCH/bench_ckpt/<tag>.pt and re-run the final evaluation")
    a = ap.parse_args()
    torch.manual_seed(a.seed); rng = np.random.default_rng(a.seed)
    gen = a.model in M.GENERATIVE
    if a.wrap:
        a.loss = "rel"
    steps = 60 if a.smoke else (a.steps or (4000 if a.mode == "tune" else 20000))
    if (a.model == "duhamel" or a.model.startswith(("duhamel_uno", "duhamel_ou2"))):
        a.wrap = True; a.loss = "rel"
    if a.model in ("persist", "duhamel_uno_lin"):          # reference rows: no training (persistence; the measured linear propagator alone)
        steps = 0
    if a.model == "podlstm":
        a.wrap = True; a.loss = "rel"
    cfg = "lr%g_%s%s" % (a.lr, ("scale-" + a.scale) if gen else ("recipe" if a.wrap else a.loss), "_inst" if (a.norm == "inst" and not gen and not a.wrap) else "")
    tag = "%s_%s_%s_s%d%s%s" % (a.task, a.model, cfg, a.seed, os.environ.get("BENCH_TAG", ""), "_smoke" if a.smoke else "")
    outdir = os.path.join(HERE, "results_tune" if a.mode == "tune" else "results_final"); os.makedirs(outdir, exist_ok=True)
    task = TK.make(a.task)
    res = dict(vars(a), tag=tag, cfg=cfg, bound=getattr(task, "bound", None), scale_field=task.scale)
    if a.model.startswith("ours"):
        ours = ours_tdgl(task, a.model) if a.task == "tdgl" else ours_hyst(task, a.model)
        res["eval"] = evaluate(task, a.task, None, ours=ours, sets=task.val_set() if a.mode == "tune" else None)
        json.dump(res, open(os.path.join(outdir, tag + ".json"), "w"), indent=1)
        return
    if a.model == "duhamel":
        model = M.DuhamelTDGL(task.scale).to(DEV)
    elif a.model.startswith("duhamel_uno"):
        # variants (ablations): duhamel_uno_poly = learned conditioned symbol instead of the measured bank;
        # duhamel_uno_k0 = keep k=0 persistent (as for hysteresis); BENCH_BANK = alternative bank file (data-efficiency runs)
        lin = "kmu" if a.model.startswith("duhamel_uno_kmu") else ("bank" if a.task in ("tdgl", "hyst") and a.model != "duhamel_uno_poly" else "poly")
        bank = os.environ.get("BENCH_BANK") or os.path.join(os.environ["SCRATCH"], "bench_cache", "alpha_bank_%s.npz" % a.task)
        model = M.DuhamelUNO(task.nd, task.cin, task.dcu, task.guard, linear=lin, bank=bank if lin == "bank" else None,
                             persist_k0=a.task == "hyst" or a.model == "duhamel_uno_k0",
                             cref=getattr(task, "cref", None), substeps=4 if (lin == "kmu" and a.task.startswith("ch") and "_s1" not in a.model) else 1,
                             trunc_w=16 if "_trunc" in a.model else 0, remainder="ou2" if "_rou2" in a.model else "uno").to(DEV)
    elif a.model == "podlstm":
        D = int(np.prod(task.Umm.shape[2:]))
        model = M.PODLSTM(D, task.dcu, k=int(os.environ.get("POD_K", "256"))).to(DEV)
        res["pod_energy"] = model.fit_basis(task, rng)
    elif a.model.startswith("duhamel_ou2"):
        model = M.DuhamelOU2(task.cref, task.guard, a.task, free_rem=a.model.endswith("_free"),
                             trunc_w=16 if "_trunc" in a.model else 0).to(DEV)
    elif a.wrap:
        model = M.RecipeWrap(a.model, task.nd, task.cin, task.dcu, task.theta_ref, task.guard).to(DEV)
    else:
        model = M.build2(a.model, task.nd, task.cin, task.dcond, a.scale).to(DEV)
        if a.norm == "inst" and not gen:
            model = M.InstNorm(model).to(DEV)
    if gen:                                                             # global residual unit from the training data
        ds = []
        for _ in range(32):
            w, _ = task.sample(16, 1, rng); ds.append((w[:, 1] - w[:, 0]).float().pow(2).mean().item())
        model.set_dscale(float(np.sqrt(np.mean(ds))) + 1e-8)
    res["params"] = sum(p.numel() for p in model.parameters())
    print("%s params %.2fM steps %d" % (tag, res["params"] / 1e6, steps), flush=True)
    kind = "none" if a.no_aug else TK.SYM.get(a.task, "none")
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=max(steps, 1), pct_start=0.05)
    amp = a.bf16
    t0 = time.time(); run = 0.0
    if a.eval_only:
        ck = torch.load(os.path.join(CK, tag + ".pt"), map_location=DEV, weights_only=False)
        model.load_state_dict({k: v for k, v in ck["model"].items() if not k.endswith("_metadata")}); steps = 0
        print("eval_only: loaded", tag, flush=True)
    if (a.model.startswith("duhamel_ou2") or a.model.startswith("duhamel_uno_kmu")) and steps > 0:
        warm = 60 if a.smoke else 2000                                  # stage 1: linear generator alone on frames 0 -> 1..5
        (model.m if hasattr(model, "m") else model).use_rem = False
        wopt = torch.optim.AdamW(model.linear_params(), lr=3e-3)
        Nw = task.U.shape[0]
        for _ in range(warm):
            ii = torch.from_numpy(rng.integers(Nw, size=32)).to(DEV)
            ax = dict(U=task.AUX["U"][ii], th=task.AUX["th"][ii], cu=task.AUX["cu"][ii])
            x = task.U[ii, 0].unsqueeze(1).float(); wl = 0.0
            for q in range(1, 6):
                x = model(x, ax); wl = wl + relsq(x, task.U[ii, q].unsqueeze(1).float()).sqrt().mean()
            wopt.zero_grad(set_to_none=True); (wl / 5).backward(); wopt.step()
        (model.m if hasattr(model, "m") else model).use_rem = True
        print("warm-start linear loss %.4f" % (float(wl.detach()) / 5), flush=True)
    model.train()
    for s_ in range(1, steps + 1):
        win, cond = task.sample(a.bs, a.k, rng)
        ax = getattr(task, "last_aux", None) if a.wrap else None
        win, cond, ax = TK.augment(win, cond, ax, kind, task.nd)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
            if gen:
                q = int(rng.integers(a.k))
                loss = model.loss(win[:, q], win[:, q + 1], cond[:, q])
            else:
                x = win[:, 0]; loss = 0.0; st = None
                for q in range(1, a.k + 1):
                    if a.model == "podlstm":
                        y, st = model.step(x, ax["cu"][:, q - 1], ax["U"], ax["th"], st)
                    elif a.model == "duhamel":
                        y = model(x, ax)
                    elif a.model.startswith(("duhamel_uno", "duhamel_ou2")):
                        y = model(x, dict(U=ax["U"], th=ax["th"], cu=ax["cu"][:, q - 1], mat=ax.get("mat")))
                    elif a.wrap:
                        y = model(x, ax["cu"][:, q - 1], ax["U"], ax["th"])
                    else:
                        y = model(x, cond[:, q - 1])
                    x = y.float().clamp(-task.bound, task.bound)
                    loss = loss + (relsq(x, win[:, q]).mean() if a.loss == "rel" else (x - win[:, q]).pow(2).mean())
                loss = loss / a.k
        opt.zero_grad(set_to_none=True); loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sch.step()
        lv = float(loss.detach()); run += lv
        if s_ % 500 == 0 or s_ == steps:
            print("step %d loss %.5f  %.1f min" % (s_, run / min(500, s_), (time.time() - t0) / 60), flush=True); run = 0.0
        if not np.isfinite(lv):
            print("NaN loss at step", s_, flush=True); res["diverged_at"] = s_; break
    res["train_minutes"] = (time.time() - t0) / 60
    if a.eval_only:
        old = os.path.join(outdir, tag + ".json")
        if os.path.exists(old):
            res["train_minutes"] = json.load(open(old)).get("train_minutes")
    if a.mode == "final" and not a.eval_only:
        torch.save(dict(model=model.state_dict(), args=vars(a)), os.path.join(CK, tag + ".pt"))
    model.eval()

    mem = {}

    def step(x, c, aux=None):
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
            if a.model == "podlstm":                            # stateful: a new rollout starts when x is not our last output
                last = mem.get("y")
                st = mem.get("st") if (last is not None and last.shape == x.shape and torch.equal(last, x)) else None
                y, st = model.step(x, aux["cu"], aux["U"], aux["th"], st)
                y = clampb(task, y.float()); mem["st"] = st; mem["y"] = torch.nan_to_num(y).clamp(-50, 50)
                return y
            if a.model == "duhamel" or a.model.startswith(("duhamel_uno", "duhamel_ou2")):
                return clampb(task, model(x, aux).float())
            if a.wrap:
                return clampb(task, model(x, aux["cu"], aux["U"], aux["th"]).float())
            return clampb(task, model(x, c).float())
    step.wrap = a.wrap
    t1 = time.time(); torch.manual_seed(1234)
    if a.smoke and not os.environ.get("BENCH_EVAL_N"):
        res["eval"] = "skipped (smoke)"
    elif a.mode == "tune":
        res["eval"] = evaluate(task, a.task, step, sets=task.val_set())
    else:
        sets = dict(task.val_set()); sets.update(task.eval_sets())
        res["eval"] = evaluate(task, a.task, step, sets=sets)
    res["eval_minutes"] = (time.time() - t1) / 60
    json.dump(res, open(os.path.join(outdir, tag + ".json"), "w"), indent=1)
    print("DONE", tag, flush=True)


if __name__ == "__main__":
    main()
