#!/usr/bin/env python3
"""make_rollout_fig.py -- Fig. rollouts of the camera-ready: Duhamel-NO vs FerroX at t = 1,2,5,10,50,100,200,400 for
three test materials with their own (unseen) initial noise. Same checkpoint, data and rollout as bench/physmetrics.py
(tdgl_duhamel_uno_lr0.0003_recipe_s0). Each material is the trajectory with the MEDIAN frame-400 error of its subset
in results_phys/<tag>.json (no cherry-picking):
    D1_heldout / interp   held-out material inside the sampled box
    D2_ood / C_shape      first-order wells (beta < 0), r beyond the training range
Writes figures/06_duhamel/rollouts_multi.pdf and rollouts_multi.json (per-panel rel. L2 + material numbers).
Run from anywhere:  python make_rollout_fig.py   (needs a GPU with ~3 GB free, or BENCH_DEV=cpu)
"""
import argparse, json, os, sys
OUTNAME = "rollouts_multi"
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
BENCH = os.path.join(os.path.dirname(HERE), "bench")
sys.path.insert(0, BENCH)
os.chdir(BENCH)
import tasks as TK          # noqa: E402
import run as RUN           # noqa: E402
import physmetrics as PM    # noqa: E402

TAG = "tdgl_duhamel_uno_lr0.0003_recipe_s0"
T = [1, 2, 5, 10, 50, 100, 200, 400]
PICK = [("D1_heldout", "interp", "held-out material\n(interpolation)"),
        ("D2_ood", "B_units", r"$P_{\mathrm{S}}$ above" + "\ntraining range"),
        ("D2_ood", "C_shape", "first-order well\n" + r"($\beta<0$, $r$ beyond range)")]


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--skip", nargs="*", default=[]); ap.add_argument("--out", default="rollouts_multi")
    args = ap.parse_args()
    global OUTNAME; OUTNAME = args.out
    task = TK.make("tdgl")
    a = argparse.Namespace(model="duhamel_uno", seed=0, lr=3e-4, loss="rel", wrap=True, scale="global", norm="global")
    model, _ = PM.build(a, task)
    ck = torch.load(os.path.join(RUN.CK, TAG + ".pt"), map_location=TK.DEV, weights_only=False)
    model.load_state_dict({k: v for k, v in ck["model"].items() if not k.endswith("_metadata")})
    model.eval()
    res = json.load(open(os.path.join(BENCH, "results_phys", TAG + ".json")))
    sets = task.eval_sets()
    out = []
    for sp, sub, label in [p for p in PICK if p[1] not in args.skip]:
        trajs = sets[sp]["trajs"]; rows = res["sets"][sp]["rows"]
        idx = [i for i, r in enumerate(rows) if r["subset"] == sub]
        med = np.median([rows[i]["rel_t400"] for i in idx])
        j = min(idx, key=lambda i: abs(rows[i]["rel_t400"] - med))
        b = [trajs[j]]
        assert b[0].tag == rows[j]["tag"]
        u, c = task.load_batch(b)
        u = u.to(TK.DEV).float()
        A = RUN._aux_to(task.batch_aux); A["cu"] = A["cu"].unsqueeze(1).expand(-1, u.shape[1] - 1, -1)
        cs = c.to(TK.DEV).float().unsqueeze(1).expand(-1, u.shape[1] - 1, -1)
        step = lambda x, cc, aux=None: model(x, aux).float().clamp(-task.bound, task.bound)
        P = torch.cat([u[:, :1], RUN.rollout(step, u[:, 0], cs, A)], 1)[0].cpu().numpy()
        G = u[0].cpu().numpy()
        rel = [float(np.linalg.norm(P[t] - G[t]) / np.linalg.norm(G[t])) for t in T]
        out.append(dict(set=sp, subset=sub, label=label, tag=b[0].tag, mat=[float(v) for v in b[0].mat], P_s=float(b[0].P_s),
                        rel=rel, rel400_json=rows[j]["rel_t400"], subset_median=float(med),
                        pred=P[T][:, -1], true=G[T][:, -1]))
        print(sp, sub, b[0].tag, "mat", b[0].mat, "P_s %.3f" % b[0].P_s, "rel400 %.4f (json %.4f, subset median %.4f)"
              % (rel[-1], rows[j]["rel_t400"], med), flush=True)
    plot(out)
    json.dump([{k: v for k, v in o.items() if k not in ("pred", "true")} for o in out],
              open(os.path.join(HERE, "figures", "06_duhamel", OUTNAME + ".json"), "w"), indent=1)


def plot(out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import gridspec
    plt.rcParams.update({"font.size": 7, "font.family": "serif"})
    nr = 2 * len(out)
    fig = plt.figure(figsize=(6.9, 0.78 * nr + 0.3))
    gs = gridspec.GridSpec(nr, len(T) + 1, width_ratios=[1] * len(T) + [0.06], wspace=0.04,
                           hspace=0.06, figure=fig)
    for m, o in enumerate(out):
        for jj, t in enumerate(T):
            ref = o["true"][jj].astype(np.float32); lim = float(np.abs(ref).max()) + 1e-12
            for k, (fld, name) in enumerate(((ref, "FerroX"), (o["pred"][jj].astype(np.float32), "Duhamel-NO"))):
                ax = fig.add_subplot(gs[2 * m + k, jj])
                im = ax.imshow(fld / lim, cmap="RdBu_r", vmin=-1, vmax=1, interpolation="nearest")
                ax.set_xticks([]); ax.set_yticks([])
                if m == 0 and k == 0:
                    ax.set_title("$t=%d$" % t, fontsize=7.5)
                if k == 1:
                    ax.text(0.03, 0.05, "%.3f" % o["rel"][jj], transform=ax.transAxes, fontsize=5.6,
                            bbox=dict(boxstyle="square,pad=0.12", fc="white", ec="none", alpha=0.85))
                if jj == 0:
                    ax.set_ylabel(name, fontsize=6.4, rotation=0, ha="right", va="center")
                if jj == 0 and k == 0:
                    ax.text(-1.25, 0.0, o["label"], transform=ax.transAxes, fontsize=6.6, ha="center", va="center",
                            rotation=90, fontweight="bold")
    cax = fig.add_subplot(gs[:, -1]); cb = fig.colorbar(im, cax=cax); cb.set_ticks([-1, 0, 1])
    cb.ax.tick_params(labelsize=5.5); cb.set_label(r"$P_z/\max|P_z^{\mathrm{FerroX}}|$ (per frame)", fontsize=6)
    fig.savefig(os.path.join(HERE, "figures", "06_duhamel", OUTNAME + ".pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(HERE, "figures", "06_duhamel", OUTNAME + ".png"), dpi=110, bbox_inches="tight")


if __name__ == "__main__":
    main()
