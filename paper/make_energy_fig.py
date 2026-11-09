#!/usr/bin/env python3
"""make_energy_fig.py -- free-energy landscapes: (a) bare Landau potential f(P) = a/2 P^2 + b/4 P^4 + g/6 P^6 of the three test
materials of the rollout figures (coefficients from figures/06_duhamel/rollouts_multi_2row.json, written by make_rollout_fig.py and the D2 large-P_S material
o0257 printed by make_rollout_fig.py) ; (b) the reduced potential that every material shares in its own units,
phi_r(v) = -v^2/2 + (1-r) v^4/4 + r v^6/6 (minima at v = +-1 for every r), for r across and beyond the training range."""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__)); WS = HERE   # paper/
plt.rcParams.update({"font.size": 7, "font.family": "serif"})
R = json.load(open(os.path.join(WS, "figures", "06_duhamel", "rollouts_multi_2row.json")))
mats = [("D1 held-out (interior)", R[0]["mat"][:3]), ("D2 large $P_S$", [-2307739265.7255735, 3013761488.5485153, 3414320883.429762]),
        ("D2 first-order ($\\beta<0$)", R[-1]["mat"][:3])]
fig, axs = plt.subplots(1, 2, figsize=(6.6, 2.75))
P = np.linspace(-0.8, 0.8, 1601)
for (lab, (a, b, g)), c in zip(mats, ["#b2182b", "#2166ac", "#1b7837"]):
    f = a / 2 * P ** 2 + b / 4 * P ** 4 + g / 6 * P ** 6
    dep = -f.min()
    axs[0].plot(P, f / dep, color=c, lw=1.2, label=lab + ": " + r"$P_{\min}$=%.2f C/m$^2$, depth %.1e J/m$^3$" % (abs(P[np.argmin(f)]), dep))
axs[0].set_xlabel(r"$P$ (C/m$^2$)"); axs[0].set_ylabel(r"$f(P)\,/\,|\min f|$"); axs[0].set_ylim(-1.15, 1.6)
axs[0].legend(fontsize=5.5, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.24))   # below the axes, clear of the curves
axs[0].set_title("(a) bare Landau potential (own depth)")
v = np.linspace(-1.35, 1.35, 801)
for r, c in zip([0.0, 0.5, 1.0, 1.4, 2.0], plt.cm.viridis(np.linspace(0, 0.9, 5))):
    axs[1].plot(v, -v ** 2 / 2 + (1 - r) * v ** 4 / 4 + r * v ** 6 / 6, color=c, lw=1.2,
                ls="-" if r <= 1.4 else "--", label=("$r=%.1f$" % r) + (" (D2 only)" if r > 1.4 else ""))
axs[1].axvline(1, color="0.6", lw=0.5, ls=":"); axs[1].axvline(-1, color="0.6", lw=0.5, ls=":")
axs[1].set_xlabel(r"$v=P/P_S$"); axs[1].set_ylabel(r"$\phi_r(v)$"); axs[1].set_ylim(-0.32, 0.25)
axs[1].legend(fontsize=5.5, frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.24)); axs[1].set_title("(b) reduced potential, material units")
os.makedirs(os.path.join(WS, "figures", "appendix"), exist_ok=True)
fig.tight_layout(pad=0.4); fig.savefig(os.path.join(WS, "figures", "appendix", "energy_landscapes.pdf"))
print("ok")
