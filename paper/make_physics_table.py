#!/usr/bin/env python3
"""make_physics_table.py -- tables/main_combined.tex: field errors (D1, D2, frame 1; from tables/tdgl_compact.tex, written by
make_duhamel_macros.py from paper_tools) and physical-quantity errors on the D1 held-out materials (mean over training
seeds, from bench/results_phys via paper_tools/make_figs.phys_runs) for Duhamel-NO and every baseline.
Also writes tables/app_physics_full.tex (all eight physical metrics, incl. domain size and wall density) for the appendix."""
import os, re, sys
HERE = os.path.dirname(os.path.abspath(__file__))
NPJ = os.path.join(os.path.dirname(HERE), "paper_tools")
sys.path.insert(0, NPJ); os.chdir(NPJ)
import make_figs as MF  # noqa: E402

MODELS = [("duhamel_uno", r"\DU{} (ours)", "Duhamel-NO (ours)"), ("ffno", "F-FNO", "F-FNO"), ("unet", "U-Net", "U-Net"),
          ("uno", "UNO", "UNO"), ("cnext", "CNextU-Net", "CNextU-Net"), ("cno", "CNO", "CNO"), ("tfno", "TFNO", "TFNO"),
          ("fno", "FNO", "FNO"), ("poseidon", "Poseidon-T (best generative / pretrained)", "Poseidon-T (pretrained)")]
PH_ALL = [("tnuc_err", r"$t_{\mathrm{nuc}}$", "%.2f"), ("psat_err", r"$P_{\mathrm{S}}$", "%.3f"),
          ("dsize_err_t400", "domain size", "%.3f"), ("wall_err_t400", "wall density", "%.3f"),
          ("hist_w1", r"$P/P_{\mathrm{S}}$ hist.", "%.3f"), ("phi_rel_t400", r"$\varphi$", "%.3f"),
          ("ez_rel_t400", r"$E_z$", "%.3f"), ("energy_traj_err", r"$F(t)$", "%.3f")]
PH_MAIN = [c for c in PH_ALL if c[0] not in ("dsize_err_t400", "wall_err_t400")]
comp = open(os.path.join(HERE, "tables", "tdgl_compact.tex")).read()
field = {}
for line in comp.splitlines():
    for m, _, src in MODELS:
        if line.startswith(src + " &"):
            cells = [re.sub(r"\\(textbf|underline)\{([^}]*)\}", r"\2", c).strip(" \\") for c in line.split("&")[1:]]
            field[m] = [cells[1], cells[2], cells[3]]          # D1 error, D2 error, frame-1 error (drop D1 corr.)


def marks(col):
    v = sorted(set(float(x) for x in col if x != "--"))
    return v[0], (v[1] if len(v) > 1 else None)


def table(cols, with_field, bottom_row=True):
    rows = []
    for m, lab, _ in MODELS:
        if m == "poseidon" and not bottom_row:
            continue
        vv = [MF.pmean(m, "D1_heldout", k)[0] if MF.PH.get(m) else None for k, _, _ in cols]
        ph = ["--" if v is None else ("%.0f" % v if v >= 100 else "%.1f" % v if v >= 10 else fmt % v) for v, (_, _, fmt) in zip(vv, cols)]
        rows.append((m, lab, (field.get(m, ["--"] * 3) if with_field else []) + ph))
    n = len(rows[0][2])
    best = [marks([r[2][j] for r in rows if r[0] != "poseidon"]) for j in range(n)]
    hdr1 = (r" & \multicolumn{3}{c}{field error (rel.\ $L_2$)} & \multicolumn{%d}{c}{physical-quantity error (D1)} \\" % len(cols)
            + "\n" + r"\cmidrule(lr){2-4}\cmidrule(lr){5-%d}" % (4 + len(cols))) if with_field else ""
    L = [r"\begin{tabular}{l" + "c" * n + "}", r"\toprule"] + ([hdr1] if hdr1 else [])
    L.append("model & " + " & ".join((["D1", "D2", "frame 1"] if with_field else []) + [c for _, c, _ in cols]) + r" \\")
    L.append(r"\midrule")
    for i, (m, lab, vals) in enumerate(rows):
        cells = []
        for j, x in enumerate(vals):
            if x != "--" and m != "poseidon" and float(x) == best[j][0]:
                x = r"\textbf{%s}" % x
            elif x != "--" and m != "poseidon" and best[j][1] is not None and float(x) == best[j][1]:
                x = r"\underline{%s}" % x
            cells.append(x)
        if m == "poseidon":
            L.append(r"\midrule")
        L.append(lab + " & " + " & ".join(cells) + r" \\")
        if i == 0:
            L.append(r"\midrule")
    L += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(L) + "\n"


open(os.path.join(HERE, "tables", "main_combined.tex"), "w").write(table(PH_MAIN, True))
open(os.path.join(HERE, "tables", "app_physics_full.tex"), "w").write(table(PH_ALL, False, bottom_row=False))
print(open(os.path.join(HERE, "tables", "main_combined.tex")).read())
