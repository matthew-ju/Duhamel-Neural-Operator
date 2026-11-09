#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""fig_design.py -- (1) figures/design_space.png: where D1 and the five D2 regions sit in
(|alpha|, P_s, r); (2) figures/supp_satlaw.png: closed-form P_s vs FerroX for all 1001 D1 materials."""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # repository root
FIG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
d1 = json.load(open(os.path.join(HERE, "data_generation", "abg_grid.json")))["configs"]
d2 = json.load(open(os.path.join(HERE, "data_generation", "ood_grid.json")))
BOX = d2["design"]["subsets"]
COL = {"A_train": "C2", "B_units": "C1", "B_alpha": "C4", "C_shape": "C3", "D_all": "C5"}
LAB = {"A_train": "A", "B_units": "B$_{\\mathrm{units}}$", "B_alpha": "B$_\\alpha$", "C_shape": "C", "D_all": "D"}

fig, axs = plt.subplots(1, 2, figsize=(10, 4.2))
sp = {"train": ("0.55", "D1 train"), "interp": ("C0", "D1 held-out (interior)"), "extrap": ("C9", "D1 held-out (first-order)")}
for s, (c, lab) in sp.items():
    q = [x for x in d1 if x["split"] == s]
    axs[0].scatter([abs(x["alpha"]) / 1e9 for x in q], [x["P_s"] for x in q], s=4, color=c, label=lab)
    axs[1].scatter([x["r"] for x in q], [x["P_s"] for x in q], s=4, color=c)
for b in BOX:
    q = [x for x in d2["configs"] if x["subset"] == b]
    axs[0].scatter([abs(x["alpha"]) / 1e9 for x in q], [x["P_s"] for x in q], s=5, marker="^", color=COL[b], label="D2 " + LAB[b])
    axs[1].scatter([x["r"] for x in q], [x["P_s"] for x in q], s=5, marker="^", color=COL[b])
axs[0].set_xlabel(r"$|\alpha|$ ($10^9$ V m C$^{-1}$)"); axs[0].set_ylabel(r"$P_s$ (C m$^{-2}$)")
axs[1].set_xlabel(r"$r = \gamma P_s^4/|a_{sat}|$  (6th-order share; $r>1$: $\beta<0$)"); axs[1].set_ylabel(r"$P_s$ (C m$^{-2}$)")
axs[0].legend(fontsize=7, loc="upper center", markerscale=3, ncol=2)
axs[0].set_title("a  D1 materials and the five D2 regions", fontsize=10, loc="left")
axs[1].set_title("b  well shape", fontsize=10, loc="left")
fig.tight_layout(); fig.savefig(os.path.join(FIG, "design_space.png"), dpi=150); plt.close(fig)

ps = json.load(open(os.path.join(HERE, "results", "analysis", "abg_check_ps.json")))["rows"]
e = np.array([100 * r["err"] for r in ps])
fig, axs = plt.subplots(1, 3, figsize=(13, 3.8))
axs[0].scatter([r["pred"] for r in ps], [r["meas"] for r in ps], s=5, c=[r["r"] for r in ps], cmap="viridis")
lo, hi = 0.08, 0.44; axs[0].plot([lo, hi], [lo, hi], "k--", lw=0.8)
axs[0].set_xlabel(r"$P_s$ closed form (C m$^{-2}$)"); axs[0].set_ylabel(r"$P_s$ FerroX (C m$^{-2}$)")
axs[0].set_title("mean |error| %.2f%%, max %.2f%% (n = %d)" % (np.mean(np.abs(e)), np.max(np.abs(e)), len(e)), fontsize=9)
axs[1].scatter([abs(r["alpha"]) / 1e9 for r in ps], e, s=5, color="C0"); axs[1].axhline(0, color="k", lw=0.6)
axs[1].set_xlabel(r"$|\alpha|$ ($10^9$ V m C$^{-1}$)"); axs[1].set_ylabel("error (%)")
axs[1].set_title("corr = %.2f" % np.corrcoef([abs(r["alpha"]) for r in ps], e)[0, 1], fontsize=9)
axs[2].scatter([r["r"] for r in ps], e, s=5, color="C0"); axs[2].axhline(0, color="k", lw=0.6)
axs[2].set_xlabel(r"$r$"); axs[2].set_ylabel("error (%)")
axs[2].set_title("corr = %.2f" % np.corrcoef([r["r"] for r in ps], e)[0, 1], fontsize=9)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "supp_satlaw.png"), dpi=150); plt.close(fig)
print("ok")
