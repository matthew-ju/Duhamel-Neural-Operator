#!/usr/bin/env python3
"""make_ac_table.py -- tables/ac_main.tex: Allen-Cahn (our exponential-integrator data, bench task 'ac') relative L2 error at
the final frame: in range / new units / new physics (validation-selected runs, mean over seeds, as paper/tools/tables/acch.tex)
and zero-shot grids 32^2 / 128^2 for models trained on 64^2 (results/curves <- duhamel_no/curves.py: the same
validation-selected runs, every seed)."""
import glob, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
NPJ = os.path.join(HERE, "tools")
sys.path.insert(0, NPJ); os.chdir(NPJ)
import make_figs as MF  # noqa: E402

OURS = MF.ours("ac")
MODELS = [(OURS, r"\DU{} (ours)"), ("ffno", "F-FNO"), ("cnext", "CNextU-Net"), ("unet", "U-Net"), ("cno", "CNO"),
          ("uno", "UNO"), ("tfno", "TFNO"), ("fno", "FNO")]
SPL = [("id", "in range"), ("ood_units", "new units"), ("ood_inv", "new physics")]
rr = {}
for f in glob.glob(os.path.join(MF.RES, "res", "ac_*.json")):
    d = MF.jload(f)
    if d:
        rr.setdefault(d["model"], []).append(d)


CUR = os.path.join(MF.RES, "curves")


def res(m, n):
    """zero-shot grid n: final-frame error of the in-range test materials, mean over the validation-selected runs (all
    seeds), from results/curves (duhamel_no/curves.py); falls back to results/res where a curve file is missing."""
    vals = []
    for d in MF.runs("ac", m):
        f = os.path.join(CUR, d["tag"] + ".json")
        c = MF.jload(f) if os.path.exists(f) else None
        if c and ("id_%s" % n) in c:
            vals.append(float(np.mean(np.asarray(c["id_%s" % n])[:, -1])))
        else:
            r = [x for x in rr.get(m, []) if x["tag"] == d["tag"]]
            if r:
                vals.append(((r[0]["grids"].get(n) or {}).get("id") or {}).get("rel_final"))
    return MF.mean_std(vals)[0]


V = {}
for m, _ in MODELS:
    V[m] = [MF.metric("ac", m, s, "rel_final")[0] for s, _ in SPL] + [res(m, "32"), res(m, "128")]
n = len(V[OURS])
col = lambda j: sorted(v[j] for v in V.values() if np.isfinite(v[j]))
L = [r"\begin{tabular}{lccccc}", r"\toprule",
     r" & \multicolumn{3}{c}{trained grid ($64^2$)} & \multicolumn{2}{c}{zero-shot grid (in range)} \\",
     r"\cmidrule(lr){2-4}\cmidrule(lr){5-6}",
     "model & " + " & ".join([s for _, s in SPL] + [r"$32^2$", r"$128^2$"]) + r" \\", r"\midrule"]
for i, (m, lab) in enumerate(MODELS):
    cells = []
    for j, v in enumerate(V[m]):
        if not np.isfinite(v):
            cells.append("--"); continue
        s = "%.3f" % v; c = col(j)
        cells.append(r"\textbf{%s}" % s if v == c[0] else (r"\underline{%s}" % s if len(c) > 1 and v == c[1] else s))
    L.append(lab + " & " + " & ".join(cells) + r" \\")
    if i == 0:
        L.append(r"\midrule")
L += [r"\bottomrule", r"\end{tabular}"]
open(os.path.join(HERE, "tables", "ac_main.tex"), "w").write("\n".join(L) + "\n")
print("\n".join(L))
