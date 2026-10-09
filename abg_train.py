#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""abg_train.py -- ONE trainer for every arm of the (alpha,beta,gamma) campaign.

Every arm goes through this file with the same loss, schedule, data split and
eval; the only thing that varies is the model's flags.  That is deliberate and
it is the lesson of the t400 showdown: when each model family had its own
driver, every difference was confounded with the training recipe.

THE LOSS IS RELATIVE, PER FRAME.  Field rms runs from ~5e-4 at t=0 to ~0.4 at
saturation, so a plain MSE weights one saturated frame like a million
incubation frames and the spectral half of the model receives essentially no
gradient.  Dividing each frame's error by that frame's own mean square makes
every frame count equally.

WHAT IS SCORED, AND WHY NOT CORRELATION.  Two metrics, both reported at
milestones:
    relL2   ||pred - true|| / ||true||        the honest one
    ampErr  |rms(pred)/rms(true) - 1|         what beta and gamma actually move
Correlation is NOT the headline here.  On this system a frozen propagator with
zero trained parameters scores corr@400 = 0.987, and switching gamma off
entirely leaves corr at 0.9998 -- correlation reports the initial noise
pattern, which the seed fixes, not the dynamics the coefficients set.

MILESTONES ARE PER MATERIAL.  alpha moves the time unit by 11x across this
design, so there is no single frame index at which the whole grid is at the
same stage.  The abg sweep is generated with a per-config window that puts the
same DEVELOPMENTAL stage at the same frame index, so frame milestones are
comparable within it; for the legacy alpha43 stack (one fixed cadence) the
milestones are reported in units of that material's own t*.
"""
import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import abg_data as ad
import abg_laws as al
from abg_model import ABGOperator, count_params

HERE = os.path.dirname(os.path.abspath(__file__))
CKPT = os.path.join(HERE, "checkpoints")


# ===========================================================================
# Loss and metrics
# ===========================================================================
def rel_loss(pred, target, eps=1e-12):
    d = tuple(range(1, pred.dim()))
    return ((pred - target).pow(2).mean(d)
            / target.pow(2).mean(d).clamp_min(eps)).mean()


def rel_l2(pred, target):
    d = tuple(range(1, pred.dim()))
    return ((pred - target).pow(2).mean(d).sqrt()
            / target.pow(2).mean(d).sqrt().clamp_min(1e-12))


def amp_err(pred, target):
    d = tuple(range(1, pred.dim()))
    return (pred.pow(2).mean(d).sqrt()
            / target.pow(2).mean(d).sqrt().clamp_min(1e-12) - 1.0).abs()


def corr(pred, target):
    a = pred.reshape(pred.shape[0], -1).double()
    b = target.reshape(target.shape[0], -1).double()
    a, b = a - a.mean(1, keepdim=True), b - b.mean(1, keepdim=True)
    return ((a * b).sum(1)
            / (a.norm(dim=1) * b.norm(dim=1)).clamp_min(1e-30))


# ===========================================================================
# Eval
# ===========================================================================
def subsample(idx, n):
    """At most n trajectories, SPREAD ACROSS CONFIGS.

    Not idx[:n].  The index is built stack by stack and config by config, so
    taking the head silently scores one stack and never the others -- the first
    baseline run reported a held-out-material number computed entirely on
    alpha43, with all 24 betagamma trajectories present in the split and none
    of them evaluated.  Round-robin over configs instead, deterministically, so
    every arm is scored on exactly the same trajectories.
    """
    if n <= 0 or len(idx) <= n:
        return list(idx)
    by = {}
    for t in idx:
        by.setdefault((t.stack, t.tag), []).append(t)
    order = sorted(by)
    out, k = [], 0
    while len(out) < n:
        added = False
        for key in order:
            if k < len(by[key]):
                out.append(by[key][k])
                added = True
                if len(out) == n:
                    break
        if not added:
            break
        k += 1
    return out


def by_stack(idx):
    """{stack: [traj]} -- so a mixed split can be reported per stack instead of
    as one average that hides which half is failing."""
    out = {}
    for t in idx:
        out.setdefault(t.stack, []).append(t)
    return out



@torch.no_grad()
def evaluate(model, idx, device, t0=0, milestones=(50, 100, 200, 400),
             max_traj=64, chunk=8, split_stacks=True):
    """Cold-start rollout from frame t0 to the last milestone."""
    if not idx:
        return {}
    if hasattr(model, "eval"):
        model.eval()          # the zero-training baselines are not nn.Modules
    idx = subsample(idx, max_traj)
    n = max(milestones)
    acc = {m: {"relL2": [], "ampErr": [], "corr": []} for m in milestones}
    for lo in range(0, len(idx), chunk):
        sub = idx[lo:lo + chunk]
        n_ok = min(n, min(t.T - 1 - t0 for t in sub))
        ms = [m for m in milestones if m <= n_ok]
        # frame() is (1,Z,H,W), so stacking B of them gives (B,Z,H,W) -- the
        # model wants an explicit channel axis, (B,1,Z,H,W).
        x = torch.from_numpy(
            np.stack([ad.frame(t, t0)[0] for t in sub])).unsqueeze(1).to(device)
        model.set_material([t.alpha for t in sub], [t.beta for t in sub],
                           [t.gamma for t in sub], [t.dt_frame for t in sub],
                           device=device)
        want = set(ms)
        for s in range(1, n_ok + 1):
            x, _ = model.step(x)
            if not torch.isfinite(x).all():
                x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
            if s in want:
                tgt = torch.from_numpy(
                    np.stack([ad.frame(t, t0 + s)[0] for t in sub])
                ).unsqueeze(1).to(device)
                acc[s]["relL2"] += rel_l2(x, tgt).cpu().tolist()
                acc[s]["ampErr"] += amp_err(x, tgt).cpu().tolist()
                acc[s]["corr"] += corr(x, tgt).cpu().tolist()
    out = {m: {k: float(np.mean(v)) for k, v in d.items() if v}
           for m, d in acc.items() if d["relL2"]}
    if split_stacks:
        st = by_stack(idx)
        if len(st) > 1:
            out["by_stack"] = {
                name: evaluate(model, sub, device, t0=t0, milestones=milestones,
                               max_traj=max_traj, chunk=chunk,
                               split_stacks=False)
                for name, sub in sorted(st.items())}
    return out


def fmt_eval(e):
    # FILTER BEFORE SORTING.  `e` carries an extra "by_stack" key alongside the
    # integer milestones, and sorted() compares the str against the ints before
    # any downstream filter can drop it -- TypeError at the END of the first
    # epoch, i.e. after all the work and before anything was written.  Every
    # arm died there for three hours.
    ms = sorted(k for k in e if isinstance(k, int))
    return " ".join("t%d[L2 %.3f amp %.3f corr %.4f]"
                    % (m, e[m]["relL2"], e[m]["ampErr"], e[m]["corr"])
                    for m in ms)


# ===========================================================================
# Train
# ===========================================================================
def train(args):
    dev = torch.device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    idx = ad.build_index(stacks=tuple(args.stacks.split(",")),
                         max_seeds=args.max_seeds)
    tr, vseed, vint, vext = ad.split_index(idx)
    if args.max_train_traj:
        tr = tr[:args.max_train_traj]
    print("data | " + ad.describe(idx), flush=True)
    for nm, s in [("train", tr), ("val-seed", vseed), ("val-interp", vint),
                  ("val-extrap", vext)]:
        print("  %-11s %s" % (nm, ad.describe(s) if s else "empty"), flush=True)

    ds = ad.WindowSet(tr, k=args.k, t_stride=args.t_stride,
                      stride_max=args.stride_max, inc_end=args.inc_end,
                      inc_repeat=args.inc_repeat)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                    num_workers=args.num_workers, drop_last=True,
                    pin_memory=True, persistent_workers=args.num_workers > 0)

    model = ABGOperator(cond_mode=args.cond, units=not args.no_units,
                        tau_scale=not args.no_tau, use_ssm=not args.no_ssm,
                        use_fno=not args.no_fno, width=args.width,
                        n_layers=args.n_layers, modes_xy=args.modes_xy,
                        use_film=not args.no_film, guard=args.guard).to(dev)
    print("%s | %d params (trunk %d, ssm %d, fno %d) | %d windows"
          % (args.name, count_params(model), count_params(model.trunk),
             count_params(model.ssm) if model.ssm else 0,
             count_params(model.fno) if model.fno else 0, len(ds)), flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=args.wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    os.makedirs(CKPT, exist_ok=True)
    ck = os.path.join(CKPT, args.name + ".pt")
    hist, ep0, best = [], 0, float("inf")
    if args.resume and os.path.exists(ck):
        st = torch.load(ck, map_location=dev, weights_only=False)
        # STRIP "_metadata".  state_dict() carries it as an ATTRIBUTE, but it
        # round-trips through torch.save/torch.load as a KEY, and
        # load_state_dict then rejects the whole checkpoint with "Unexpected
        # key(s)".  Wave 1 sat dead for fourteen hours on this: the arms
        # relaunched every four hours, crashed instantly on resume, and
        # produced nothing, while a process-liveness watchdog saw only that
        # something had been started.
        sd = {k: v for k, v in st["model"].items() if k != "_metadata"}
        model.load_state_dict(sd); opt.load_state_dict(st["opt"])
        sched.load_state_dict(st["sched"])
        # T_max IS BAKED INTO THE CHECKPOINT.  Restoring the scheduler restores
        # the OLD T_max, so changing --epochs later is silently ignored -- and
        # once last_epoch passes T_max the cosine turns around and the learning
        # rate CLIMBS BACK UP.  Measured: T_max=60 resumed to epoch 89 put the
        # lr back at 4.7e-4 (half the initial value), training loss rose
        # 0.00016 -> 0.00035 and held-out relL2 went 0.19 -> 0.40.  Nothing
        # crashed; the model just quietly got worse for 29 epochs.
        if sched.T_max != args.epochs:
            print("resume: T_max %d -> %d (from --epochs)" % (sched.T_max, args.epochs),
                  flush=True)
            sched.T_max = args.epochs
        if sched.last_epoch > sched.T_max:
            print("resume: WARNING last_epoch %d > T_max %d; lr was on the way "
                  "back UP" % (sched.last_epoch, sched.T_max), flush=True)
        hist = st["hist"]
        ep0, best = st["epoch"], st.get("best", float("inf"))
        print("resumed %s at epoch %d" % (args.name, ep0), flush=True)

    t_start = time.time()
    for ep in range(ep0, args.epochs):
        model.train()
        tot, nb = 0.0, 0
        for nstep, (frames, mat, stride) in enumerate(dl):
            if args.steps_per_epoch and nstep >= args.steps_per_epoch:
                break
            frames = frames.to(dev, non_blocking=True).unsqueeze(2)  # (B,k+1,1,Z,H,W)
            mat = mat.to(dev)
            model.set_material(mat[:, 0], mat[:, 1], mat[:, 2], mat[:, 3],
                               device=dev)
            m = stride.to(dev)
            x = frames[:, 0]
            loss = 0.0
            for q in range(1, args.k + 1):
                x, _ = model.step(x, steps=m)
                loss = loss + rel_loss(x, frames[:, q])
                if args.detach:
                    x = x.detach()
            loss = loss / args.k
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            opt.step()
            tot += float(loss.detach()); nb += 1
            if args.max_minutes and (time.time() - t_start) / 60 > args.max_minutes:
                break
        sched.step()
        row = dict(epoch=ep + 1, loss=tot / max(nb, 1),
                   minutes=(time.time() - t_start) / 60)
        if (ep + 1) % args.eval_every == 0 or ep + 1 == args.epochs:
            for nm, s in [("seed", vseed), ("interp", vint), ("extrap", vext)]:
                if s:
                    row[nm] = evaluate(model, s, dev, milestones=args.milestones,
                                       max_traj=args.eval_traj)
        hist.append(row)
        print("ep %3d | loss %.5f | %.1f min | %s" % (
            ep + 1, row["loss"], row["minutes"],
            " ".join("%s %s" % (k, fmt_eval(v)) for k, v in row.items()
                     if isinstance(v, dict))), flush=True)
        score = row["loss"]
        if "seed" in row and row["seed"]:
            # SAME BUG AS fmt_eval, SECOND OCCURRENCE.  The eval dict carries
            # integer milestones plus the string key "by_stack"; max() compares
            # them and raises.  Fixing only fmt_eval left this one live, and it
            # killed every arm for a further ten hours -- the checkpoint-
            # selection line runs AFTER the epoch's work, so each relaunch
            # trained, crashed, and saved nothing.
            ms = [k for k in row["seed"] if isinstance(k, int)]
            if ms:
                score = row["seed"][max(ms)]["relL2"]
        torch.save(dict(model=dict(model.state_dict()), opt=opt.state_dict(),
                        sched=sched.state_dict(), hist=hist, epoch=ep + 1,
                        best=min(best, score), args=vars(args)), ck)
        if score < best:
            best = score
            torch.save(model.state_dict(), ck.replace(".pt", "_best.pt"))
        with open(os.path.join(HERE, "results_abg", args.name + ".json"), "w") as f:
            json.dump(dict(args=vars(args), hist=hist, best=best), f, indent=1)
        if args.max_minutes and (time.time() - t_start) / 60 > args.max_minutes:
            print("time budget reached", flush=True)
            break
    return model, hist


def build_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--name", required=True)
    p.add_argument("--cond", default="pi", choices=["pi", "raw", "none"])
    p.add_argument("--no_units", action="store_true")
    p.add_argument("--no_tau", action="store_true")
    p.add_argument("--no_ssm", action="store_true")
    p.add_argument("--no_fno", action="store_true")
    p.add_argument("--no_film", action="store_true")
    p.add_argument("--guard", type=float, default=1.35)
    p.add_argument("--width", type=int, default=48)
    p.add_argument("--n_layers", type=int, default=4)
    p.add_argument("--modes_xy", type=int, default=16)
    p.add_argument("--stacks", default="abg,alpha43,betagamma")
    p.add_argument("--max_seeds", type=int, default=None)
    p.add_argument("--max_train_traj", type=int, default=0)
    p.add_argument("--k", type=int, default=4)
    p.add_argument("--t_stride", type=int, default=8)
    p.add_argument("--stride_max", type=int, default=1)
    p.add_argument("--inc_end", type=int, default=60)
    p.add_argument("--inc_repeat", type=int, default=3)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--wd", type=float, default=1e-4)
    p.add_argument("--clip", type=float, default=1.0)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--eval_every", type=int, default=2)
    p.add_argument("--steps_per_epoch", type=int, default=0,
                   help="cap optimiser steps per epoch so the EVAL CADENCE is "
                        "predictable and independent of how big the dataset "
                        "happens to be. The legacy cross gives 166k windows = "
                        "~20.8k steps/epoch at batch 8, so an arm on a 200 min "
                        "budget can spend the whole budget inside epoch 1 and "
                        "report no held-out number at all. An epoch is a "
                        "reporting unit here, not a pass over the data.")
    p.add_argument("--eval_traj", type=int, default=32)
    p.add_argument("--milestones", type=int, nargs="+",
                   default=[50, 100, 200, 400])
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--max_minutes", type=float, default=0)
    p.add_argument("--detach", action="store_true")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p


if __name__ == "__main__":
    os.makedirs(os.path.join(HERE, "results_abg"), exist_ok=True)
    train(build_parser().parse_args())
