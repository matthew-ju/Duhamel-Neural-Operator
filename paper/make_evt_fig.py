#!/usr/bin/env python3
"""make_evt_fig.py -- Fig. 3 of the camera-ready: relative error at every frame, 2x2.
(a,b) FerroX TDGL, D1 held-out materials and D2 out of range (paper_tools/make_figs.PH <- bench/results_phys; no ablation curve).
(c,d) Allen-Cahn, new-physics split on the trained 64^2 grid and in-range materials on the zero-shot 128^2 grid
(bench/results_curves <- bench/curves.py: the validation-selected checkpoints, every seed).
Mean over test trajectories; bands: range over training seeds."""
import glob, json, os, sys
import numpy as np
from matplotlib.ticker import NullFormatter, LogLocator, FuncFormatter
HERE = os.path.dirname(os.path.abspath(__file__))
NPJ = os.path.join(os.path.dirname(HERE), "paper_tools")
sys.path.insert(0, NPJ); os.chdir(NPJ)
import make_figs as MF  # noqa: E402
plt = MF.plt
ORDER = ["duhamel_uno", "ffno", "unet", "uno", "cnext", "cno", "tfno", "fno"]
# one hue per model (checked for normal and colour-blind vision: worst pair dE 16.8 normal, 9.2 under protan/deutan/tritan);
# all lines solid, identity = colour + a small marker
STY = {"duhamel_uno": ("#b2182b", "o"), "ffno": ("#2a78d6", "s"), "unet": ("#eb6834", "^"), "uno": ("#1baf7a", "v"),
       "cnext": ("#6a3d9a", "D"), "cno": ("#f781bf", "P"), "tfno": ("#bcbd22", "X"), "fno": ("#000000", "*")}
CUR = os.path.join(MF.BENCH, "results_curves")
AC = {}
for f in glob.glob(os.path.join(CUR, "ac_*.json")):
    d = json.load(open(f))
    AC.setdefault("duhamel_uno" if d["model"].startswith("duhamel") else d["model"], []).append(d)


def tdgl(m, sp):
    return [np.asarray(d["sets"][sp]["rel_curve_mean"]) for d in MF.PH.get(m, []) if sp in d["sets"]]


def ac(m, sp):
    return [np.asarray(d[sp]).mean(0) for d in AC.get(m, []) if sp in d]


PANELS = [(tdgl, "D1_heldout", "(a) TDGL D1: held-out materials, unseen noise", True),
          (tdgl, "D2_ood", "(b) TDGL D2: out of range, zero-shot", True),
          (ac, "ood_inv_64", "(c) Allen–Cahn: new physics ($64^2$)", False),
          (ac, "id_128", "(d) Allen–Cahn: zero-shot $128^2$ grid", False)]
fig, axs = plt.subplots(2, 2, figsize=(6.9, 4.4))
for ax, (fn, sp, ttl, inc) in zip(axs.ravel(), PANELS):
    for m in ORDER:
        cs = fn(m, sp)
        if not cs:
            continue
        cs = np.array(cs); t = np.arange(1, cs.shape[1] + 1); mu = cs.mean(0)
        c, mk = STY[m]
        ax.plot(t, mu, color=c, ls="-", lw=1.6 if m == "duhamel_uno" else 0.9, label=MF.LAB(m), marker=mk,
                ms=3.0 if m == "duhamel_uno" else 2.4, markevery=list(np.unique(np.round(np.geomspace(1, len(t), 11)).astype(int)) - 1), mew=0.4,
                zorder=5 if m == "duhamel_uno" else 2)
        if len(cs) > 1:
            ax.fill_between(t, cs.min(0), cs.max(0), color=c, alpha=0.15, lw=0)
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("frame $t$"); ax.set_title(ttl)
    ax.set_xlim(1, t[-1]); ax.grid(alpha=0.25, lw=0.4, which="both")
    ax.yaxis.set_minor_formatter(NullFormatter())
    if inc:
        ax.axvspan(1, 10, color="0.92", zorder=0); ax.text(0.03, 0.04, "incubation", fontsize=6.5, color="0.35", transform=ax.transAxes)
for r in range(2):                                   # share y within a row
    lo = min(a.get_ylim()[0] for a in axs[r]); hi = max(a.get_ylim()[1] for a in axs[r])
    for a in axs[r]:
        a.set_ylim(lo, hi)
    axs[r, 1].tick_params(labelleft=False)
    if hi / lo < 100:                                # less than two decades: label 1-2-5 steps
        for a in axs[r]:
            a.yaxis.set_major_locator(LogLocator(base=10, subs=(1, 2, 5)))
            a.yaxis.set_major_formatter(FuncFormatter(lambda v, _: ("%g" % v)))
    axs[r, 0].set_ylabel("relative L2 error")
axs[0, 1].legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False)
fig.tight_layout()
fig.savefig(os.path.join(HERE, "figures", "06_duhamel", "error_vs_time.pdf"), bbox_inches="tight")
fig.savefig(os.path.join(HERE, "figures", "06_duhamel", "error_vs_time.png"), bbox_inches="tight", dpi=130)
print("wrote error_vs_time.pdf")
