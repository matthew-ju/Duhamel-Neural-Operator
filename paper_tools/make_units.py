#!/usr/bin/env python3
"""tables/units_new.tex + numbers_units.tex: recipe backbones with / without the change of units (bench runs tdgl_<m>_lr0.001_recipe_s{0,1,2} vs ..._nounits)."""
import glob, json, os, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); BENCH = os.path.join(os.path.dirname(HERE), "bench", "results_final")
def get(m, tag):
    out = []
    for s in range(3):
        f = os.path.join(BENCH, "tdgl_%s_lr0.001_recipe_s%d%s.json" % (m, s, tag))
        if os.path.exists(f):
            e = json.load(open(f))["eval"]
            if isinstance(e, dict): out.append(e)
    return out
def stat(rs, sp, key, sub=None):
    v = [(r[sp]["subsets"][sub] if sub else r[sp])[key] for r in rs]
    return (float(np.mean(v)), float(np.std(v)), len(v)) if v else (np.nan, np.nan, 0)
NAMES = {"fno": "FNO + recipe", "uno": "UNO + recipe"}
N = {}; L = [r"\begin{tabular}{llcccc}", r"\toprule", r"backbone & change of units & D1 rel.\,$L_2$ & D1 amplitude err. & D2 B$_{\mathrm{units}}$ rel.\,$L_2$ & D2 B$_{\mathrm{units}}$ amplitude err. \\", r"\midrule"]
ok = True
for m, nm in NAMES.items():
    for tag, lab, k in (("", "yes", "On"), ("_nounits", "no", "Off")):
        rs = get(m, tag)
        if not rs: ok = False; continue
        cells = [stat(rs, "D1_heldout", "rel_final"), stat(rs, "D1_heldout", "amp_final")]
        try:
            cells += [stat(rs, "D2_ood", "rel_final", "B_units"), stat(rs, "D2_ood", "amp_final", "B_units")]
        except KeyError:
            ok = False; cells += [(np.nan, np.nan, 0)] * 2
        L.append("%s & %s & %s \\\\" % (nm, lab, " & ".join("%.3f {\\scriptsize$\\pm$%.3f}" % (c[0], c[1]) for c in cells)))
        key = {"fno": "Fno", "uno": "Uno"}[m]
        N["nu%s%sDone" % (key, k)] = "%.2f" % cells[0][0]; N["nu%s%sBu" % (key, k)] = "%.2f" % cells[3][0]
L += [r"\bottomrule", r"\end{tabular}"]
open(os.path.join(HERE, "tables", "units_new.tex"), "w").write("\n".join(L))
with open(os.path.join(HERE, "numbers_units.tex"), "w") as f:
    f.write("%% written by make_units.py\n")
    for k, v in N.items(): f.write("\\newcommand{\\%s}{%s}\n" % (k, v))
print(N, "complete" if ok else "INCOMPLETE")
