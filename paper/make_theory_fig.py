#!/usr/bin/env python3
"""Truncation-cap figure for the TMLR paper (main text): same data as paper/tools/make_figs.py fig_theory, but with
plain model names and the palette of figures/06_duhamel/error_vs_time.pdf.
Run:  python paper/make_theory_fig.py"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))                     # paper/
sys.path.insert(0, os.path.join(HERE, "tools")); os.chdir(os.path.join(HERE, "tools"))
import make_figs as mf
import matplotlib.pyplot as plt

STY = {"fno": ("FNO", "#000000", "*"), "tfno": ("TFNO", "#bcbd22", "X"), "uno": ("UNO", "#1baf7a", "v"),
       "ffno": ("F-FNO", "#2a78d6", "s"), "duhamel_uno": ("Duhamel-NO (ours)", "#b2182b", "o")}
cb = mf.jload(os.path.join(mf.RES, "analysis", "theory_cap_bank.json"))
fr = cb["frames"]
fig, ax = plt.subplots(figsize=(4.0, 2.5))
ax.plot(fr, cb["cap_fno8"], "k--", lw=1.0, label="cap, FNO mode set")
ax.plot(fr, cb["cap_box16"], "k:", lw=1.0, label="cap, 16-mode box")
for m, (lab, col, mk) in STY.items():
    y = [mf.pmean(m, "D1_heldout", "corr_t%d" % f)[0] for f in fr]
    ax.plot(fr, y, color=col, marker=mk, ms=3.5, label=lab)
ax.set_xscale("log"); ax.set_xticks(fr); ax.set_xticklabels([str(x) for x in fr]); ax.minorticks_off()
ax.set_xlabel("frame $n$"); ax.set_ylabel("pattern correlation")
ax.legend(frameon=False, fontsize=6.2, loc="lower right"); ax.grid(alpha=0.25, lw=0.4)
fig.tight_layout()
out = os.path.join(HERE, "figures", "tmlr", "theory_cap.pdf"); os.makedirs(os.path.dirname(out), exist_ok=True)
fig.savefig(out); print("wrote", out, "cap", cb["cap_fno8"])
