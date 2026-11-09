#!/usr/bin/env python3
"""make_evt_ab_fig.py -- Figure 5, panels (a,b): relative error at every frame on D1 held-out and D2, stacked,
plus the model legend as its own small PDF (placed under the table of panel (c) in main.tex; the table itself is
tables/physics_fig.tex, typeset by LaTeX).
Curves: paper/tools/make_figs.PH <- results/phys; mean over test
trajectories, bands = range over training seeds.
Writes figures/06_duhamel/error_vs_time_ab.pdf and error_vs_time_legend.pdf.
Run:  python paper/make_evt_ab_fig.py"""
import os, sys
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import NullFormatter, LogLocator, FuncFormatter

HERE = os.path.dirname(os.path.abspath(__file__))                     # paper/
OUT = os.path.join(HERE, "figures", "06_duhamel")
NPJ = os.path.join(HERE, "tools")
sys.path.insert(0, NPJ); os.chdir(NPJ)
import make_figs as MF  # noqa: E402
plt = MF.plt
ORDER = ["duhamel_uno", "ffno", "unet", "uno", "cnext", "cno", "tfno", "fno"]
STY = {"duhamel_uno": ("#b2182b", "o"), "ffno": ("#2a78d6", "s"), "unet": ("#eb6834", "^"), "uno": ("#1baf7a", "v"),
       "cnext": ("#6a3d9a", "D"), "cno": ("#f781bf", "P"), "tfno": ("#bcbd22", "X"), "fno": ("#000000", "*")}
LW = lambda m: 1.6 if m == "duhamel_uno" else 0.9
MS = lambda m: 3.0 if m == "duhamel_uno" else 2.4


def tdgl(m, sp):
    return [np.asarray(d["sets"][sp]["rel_curve_mean"]) for d in MF.PH.get(m, []) if sp in d["sets"]]


PANELS = [("D1_heldout", "(a) D1 held-out materials"), ("D2_ood", "(b) D2, out of range")]
fig, axs = plt.subplots(2, 1, figsize=(3.25, 3.35), sharex=True)
for ax, (sp, ttl) in zip(axs, PANELS):
    for m in ORDER:
        cs = tdgl(m, sp)
        if not cs:
            continue
        cs = np.array(cs); t = np.arange(1, cs.shape[1] + 1); mu = cs.mean(0)
        c, mk = STY[m]
        ax.plot(t, mu, color=c, lw=LW(m), marker=mk, ms=MS(m), mew=0.4,
                markevery=list(np.unique(np.round(np.geomspace(1, len(t), 11)).astype(int)) - 1),
                zorder=5 if m == "duhamel_uno" else 2)
        if len(cs) > 1:
            ax.fill_between(t, cs.min(0), cs.max(0), color=c, alpha=0.15, lw=0)
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_title(ttl, loc="left")
    ax.set_xlim(1, t[-1]); ax.grid(alpha=0.25, lw=0.4, which="both")
    ax.yaxis.set_minor_formatter(NullFormatter()); ax.set_ylabel("relative L2 error")
    ax.axvspan(1, 10, color="0.92", zorder=0)
    ax.text(0.03, 0.06, "incubation", fontsize=6.5, color="0.35", transform=ax.transAxes)
lo = min(a.get_ylim()[0] for a in axs); hi = max(a.get_ylim()[1] for a in axs)
for a in axs:
    a.set_ylim(lo, hi)
    if hi / lo < 100:
        a.yaxis.set_major_locator(LogLocator(base=10, subs=(1, 2, 5)))
        a.yaxis.set_major_formatter(FuncFormatter(lambda v, _: ("%g" % v)))
axs[1].set_xlabel("frame $t$")
fig.tight_layout(h_pad=0.6)
fig.savefig(os.path.join(OUT, "error_vs_time_ab.pdf"), bbox_inches="tight")

# legend on its own, 4 columns x 2 rows, to sit under the table of panel (c)
handles = [Line2D([], [], color=STY[m][0], lw=LW(m), marker=STY[m][1], ms=MS(m), mew=0.4) for m in ORDER]
lf = plt.figure(figsize=(3.6, 0.42))
lf.legend(handles, [MF.LAB(m) for m in ORDER], ncol=4, loc="center", frameon=False, fontsize=6.6,
          handlelength=1.8, columnspacing=1.0, handletextpad=0.5)
lf.savefig(os.path.join(OUT, "error_vs_time_legend.pdf"), bbox_inches="tight")
print("wrote error_vs_time_ab.pdf and error_vs_time_legend.pdf")
