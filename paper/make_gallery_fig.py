#!/usr/bin/env python3
"""make_gallery_fig.py -- appendix rollout galleries figures/appendix/gallery_tdgl_d1.pdf and gallery_tdgl_d2.pdf: every model
on the same test trajectory (D1 held-out, D2 out of range) at frames 1, 5, 20, 50, 100 and 400, top ferroelectric layer.
Input: results/phys/<tag>_examples.npz, written by duhamel_no/physmetrics.py (scripts/evaluate_all.sh) for the
validation-selected run of every model. The trajectory shown is the one with the median Duhamel-NO frame-400 error among
the saved examples. Panels: prediction (or truth) / max|truth| of that frame; number = relative L2 error of that frame."""
import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import gridspec

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "tools")); os.chdir(os.path.join(HERE, "tools"))
import make_figs as MF  # noqa: E402

OUT = os.path.join(HERE, "figures", "appendix"); os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"font.size": 6.5, "font.family": "serif"})
TDGL = [("duhamel_uno", "Duhamel-NO (ours)"), ("ffno", "F-FNO"), ("unet", "U-Net"), ("uno", "UNO"), ("cnext", "CNextU-Net"),
        ("cno", "CNO"), ("tfno", "TFNO"), ("fno", "FNO"), ("poseidon", "Poseidon-T"), ("acdm", "U-Net + ACDM"),
        ("flowmatch", "U-Net + flow m."), ("refiner2", "U-Net + PDE-Ref."), ("transolver", "Transolver")]


def grid2d(truth, preds, frames, title, out, rel=None):
    """truth (F, H, W); preds list of (label, (F, H, W), rel (F,))"""
    nr, nc = 1 + len(preds), len(frames)
    s = min(0.66, 8.4 / nr)
    fig = plt.figure(figsize=(1.25 + s * nc + 0.35, s * nr + 0.35))
    gs = gridspec.GridSpec(nr, nc + 1, width_ratios=[1] * nc + [0.07], wspace=0.04, hspace=0.06,
                           left=1.2 / (1.25 + s * nc + 0.35), right=0.97, top=1 - 0.25 / (s * nr + 0.35), bottom=0.01)
    for j, f in enumerate(frames):
        lim = float(np.abs(truth[j]).max()) + 1e-12
        rows = [("truth", truth[j], None)] + [(lab, P[j], r[j]) for lab, P, r in preds]
        for i, (lab, fld, rr) in enumerate(rows):
            ax = fig.add_subplot(gs[i, j])
            im = ax.imshow(fld / lim, cmap="RdBu_r", vmin=-1, vmax=1, interpolation="nearest")
            ax.set_xticks([]); ax.set_yticks([])
            if i == 0:
                ax.set_title(f, fontsize=6.5)
            if rr is not None:
                ax.text(0.03, 0.04, "%.3f" % rr, transform=ax.transAxes, fontsize=4.6,
                        color="darkred" if rr > 0.3 else "k", bbox=dict(boxstyle="square,pad=0.08", fc="white", ec="none", alpha=0.8))
            if j == 0:
                ax.set_ylabel(lab, fontsize=6.2, rotation=0, ha="right", va="center", fontweight="bold" if i in (0, 1) else "normal")
    cax = fig.add_subplot(gs[:, -1]); cb = fig.colorbar(im, cax=cax); cb.set_ticks([-1, 0, 1]); cb.ax.tick_params(labelsize=5)
    cb.set_label("field / max|truth| (per frame)", fontsize=5.5)
    fig.savefig(out); plt.close(fig)


def tdgl(sp, name):
    files = {}
    for m, _ in TDGL:                                    # the validation-selected configuration, first training seed
        ds = sorted(MF.runs("tdgl", m), key=lambda d: d["seed"])
        f = os.path.join(MF.RES, "phys", "%s_examples.npz" % ds[0]["tag"]) if ds else ""
        if os.path.exists(f):
            files[m] = np.load(f)
        else:
            print("no example rollouts for", m, "(run scripts/evaluate_all.sh)")
    du = files["duhamel_uno"]; T = list(du["t"])
    cand = sorted(int(k.split("_")[-2]) for k in du.files if k.startswith(sp) and k.endswith("_true"))
    rel = lambda P, G: float(np.linalg.norm(P - G) / np.linalg.norm(G))
    r400 = {j: rel(du["%s_%d_pred" % (sp, j)][-1].astype(np.float32), du["%s_%d_true" % (sp, j)][-1].astype(np.float32)) for j in cand}
    j = sorted(cand, key=lambda i: r400[i])[len(cand) // 2]
    G = du["%s_%d_true" % (sp, j)].astype(np.float32)
    preds = []
    for m, lab in TDGL:
        if m in files and "%s_%d_pred" % (sp, j) in files[m].files:
            P = files[m]["%s_%d_pred" % (sp, j)].astype(np.float32)
            preds.append((lab, P[:, -1], [rel(P[f], G[f]) for f in range(len(T))]))
    grid2d(G[:, -1], preds, ["$t=%d$" % t for t in T], name, os.path.join(OUT, "gallery_%s.pdf" % name))
    print(name, "traj", j, "models", len(preds))


if __name__ == "__main__":
    tdgl("D1_heldout", "tdgl_d1")
    tdgl("D2_ood", "tdgl_d2")
