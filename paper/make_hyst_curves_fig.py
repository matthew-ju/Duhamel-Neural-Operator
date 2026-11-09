#!/usr/bin/env python3
"""make_hyst_curves_fig.py -- figures/appendix/hyst_curves.pdf (appendix): 1x2 hysteresis error curves in the style of error_vs_time.pdf.
(a,b) hysteresis: |<P>_pred - <P>_true| / P_s along the sweep (its time mean is the loop error), held-out materials and the
ten extra closed loops; each loop is resampled onto sweep progress in [0, 1] before averaging.
Data: results/curves/<tag>.json <- duhamel_no/curves.py (validation-selected checkpoints, all training seeds)."""
import glob, json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
NPJ = os.path.join(HERE, "tools")
sys.path.insert(0, NPJ); os.chdir(NPJ)
import make_figs as MF  # noqa: E402
plt = MF.plt
CUR = os.path.join(MF.RES, "curves")
ORDER = ["duhamel_uno", "ffno", "unet", "uno", "cnext", "cno", "tfno", "fno"]
# one hue per model (checked for normal and colour-blind vision: worst pair dE 16.8 normal, 9.2 under protan/deutan/tritan);
# all lines solid, identity = colour + a small marker
STY = {"duhamel_uno": ("#b2182b", "o"), "ffno": ("#2a78d6", "s"), "unet": ("#eb6834", "^"), "uno": ("#1baf7a", "v"),
       "cnext": ("#6a3d9a", "D"), "cno": ("#f781bf", "P"), "tfno": ("#bcbd22", "X"), "fno": ("#000000", "*")}
D = {}
for f in glob.glob(os.path.join(CUR, "*.json")):
    d = json.load(open(f))
    m = "duhamel_uno" if d["model"].startswith("duhamel") else d["model"]
    D.setdefault((d["task"], m), []).append(d)
G = np.linspace(0, 1, 201)


def hyst_mean(d, sp):
    """mean over loops of the deviation, each loop resampled onto sweep progress (frame / last frame)."""
    ys = []
    for r in d[sp]:
        y = np.asarray(r["dev"])[1:]; x = np.linspace(0, 1, len(y))          # frame 0 is the given field (zero error)
        ys.append(np.interp(G, x, y))
    return G, np.mean(ys, 0)


def field_mean(d, sp):
    c = np.asarray(d[sp]); return np.arange(1, c.shape[1] + 1), c.mean(0)


PANELS = [("hyst", "test_material", "(a) hysteresis: held-out materials", hyst_mean),
          ("hyst", "closed", "(b) hysteresis: ten independent closed loops", hyst_mean)]
fig, axs = plt.subplots(1, 2, figsize=(6.9, 2.3), squeeze=False)
for ax, (task, sp, ttl, fn) in zip(axs.ravel(), PANELS):
    for m in ORDER:
        ds = D.get((task, m), [])
        if not ds:
            continue
        cs = [fn(d, sp) for d in ds]; t = cs[0][0]; cs = np.array([c[1] for c in cs]); mu = cs.mean(0)
        c, mk = STY[m]
        ax.plot(t, mu, color=c, ls="-", lw=1.6 if m == "duhamel_uno" else 0.9, label=MF.LAB(m), marker=mk,
                ms=3.0 if m == "duhamel_uno" else 2.4, markevery=list(np.linspace(0, len(t) - 1, 11).round().astype(int)), mew=0.4,
                zorder=5 if m == "duhamel_uno" else 2)
        if len(cs) > 1:
            ax.fill_between(t, cs.min(0), cs.max(0), color=c, alpha=0.15, lw=0)
    ax.set_yscale("log"); ax.set_title(ttl); ax.grid(alpha=0.25, lw=0.4, which="both")
    if task == "hyst":
        ax.set_xlim(0, 1); ax.set_xlabel("sweep progress")
    else:
        ax.set_xscale("log"); ax.set_xlim(1, t[-1]); ax.set_xlabel("frame $t$")
axs[0, 0].set_ylabel(r"$|\langle P\rangle-\langle P\rangle_{\mathrm{true}}|/P_{\mathrm{S}}$")
for r in range(1):                                   # share y within a row
    lo = min(a.get_ylim()[0] for a in axs[r]); hi = max(a.get_ylim()[1] for a in axs[r])
    lo = max(lo, 1e-3)
    for a in axs[r]:
        a.set_ylim(lo, hi)
    axs[r, 1].tick_params(labelleft=False)
axs[0, 1].legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False)
fig.tight_layout()
os.makedirs(os.path.join(HERE, "figures", "appendix"), exist_ok=True)
fig.savefig(os.path.join(HERE, "figures", "appendix", "hyst_curves.pdf"), bbox_inches="tight")
fig.savefig(os.path.join(HERE, "figures", "appendix", "hyst_curves.png"), bbox_inches="tight", dpi=130)
# check: time mean of the hysteresis curve per loop = loop error of the benchmark rows
for (task, m), ds in sorted(D.items()):
    if task == "hyst":
        print(m, "loop_err held-out %.3f closed %.3f" % tuple(np.mean([np.mean([r["loop_err"] for r in d[sp]]) for d in ds])
                                                             for sp in ("test_material", "closed")))
    else:
        print(m, "AC final new-physics %.3f 128^2 %.3f" % tuple(np.mean([np.mean(np.asarray(d[sp])[:, -1]) for d in ds])
                                                             for sp in ("ood_inv_64", "id_128")))
print("wrote hyst_curves.pdf")
