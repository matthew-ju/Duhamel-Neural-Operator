#!/usr/bin/env python3
"""make_appendix_tables.py -- the remaining tables of the paper, all built from tables that paper/tools/make_figs.py,
make_physics_table.py and make_ac_table.py wrote from the result files. Only labels and layout change here; every number
is copied, none is typed by hand.

  paper/tables/app_{tdgl_main,tdgl_subsets,models,recipe,hyst,units_new,ablation_tdgl}.tex   appendix tables
  paper/tables/tmlr_ablation.tex      ablation table of the main text (rows (a)-(f))
  paper/tables/physics_fig.tex        physical-quantity table inside Figure 5 (same numbers as app_physics_full.tex)
  paper/tables/ac_main.tex            Allen-Cahn table, column "new physics" renamed "new wall widths"
  paper/figures/appendix/*            appendix figures drawn by paper/tools (copied)"""
import os, re, shutil

HERE = os.path.dirname(os.path.abspath(__file__))
TOOLS = os.path.join(HERE, "tools")
TAB = os.path.join(HERE, "tables")
os.makedirs(TAB, exist_ok=True)


def read(p):
    return open(p).read()


def write(name, text):
    open(os.path.join(TAB, name), "w").write(text)


def fix(t):
    """wording of the paper: no "recipe", no arrows."""
    t = t.replace(" + our recipe", " (our setup)").replace("with our recipe", "with our training setup")
    t = t.replace("our recipe", "our setup").replace("UNO + recipe (no Duhamel step)", "UNO, no Duhamel step")
    t = re.sub(r" \+ recipe\b", "", t)
    return t.replace(r" $\uparrow$", "").replace(r" $\downarrow$", "")


def keep_ferro(t):
    """model table: the FerroX TDGL and hysteresis blocks only."""
    i = t.find("\\midrule\nAllen--Cahn")
    return t if i < 0 else t[:i] + t[t.index("\\bottomrule"):]


# 1. ferroelectric part of the combined ablation table
L = read(os.path.join(TOOLS, "tables", "ablation.tex")).split("\n")
i = next(k for k, l in enumerate(L) if l.startswith(r"\midrule") and "Allen" in L[k + 1])
open(os.path.join(TOOLS, "tables", "ablation_tdgl.tex"), "w").write("\n".join(L[:i] + [r"\bottomrule", r"\end{tabular}"]))

# 2. appendix tables
for n in ["tdgl_main", "tdgl_subsets", "models", "recipe", "hyst", "units_new", "ablation_tdgl"]:
    t = fix(read(os.path.join(TOOLS, "tables", n + ".tex")))
    write("app_%s.tex" % n, keep_ferro(t) if n == "models" else t)

# 3. main-text ablation table: plain row and column labels
t = read(os.path.join(TAB, "app_ablation_tdgl.tex"))
for a, b in [("FerroX TDGL (same 9.5M-parameter UNO backbone)", "variant"), ("corr@400 (D1)", "D1 corr."),
             (r"rel.\,L2@400 (D1)", "D1 error"), (r"rel.\,L2, frame 1", "frame-1 error")]:
    t = t.replace(a, b)
for a, b in [("UNO, no Duhamel step", "(a) UNO alone, no Duhamel step"),
             (r"+ Duhamel step, measured operator (\DU{})", r"(b) \DU{} (measured Fourier SSM)"),
             (r"\DU{} with an $O(v^2)$ remainder instead of UNO", r"(c) \DU{}, small $O(v^2)$ network instead of UNO"),
             ("+ Duhamel step, learned conditioned symbol", r"(d) \DU{}, learned symbol instead of measured"),
             (r"\DU{}, uniform mode kept instead of damped", r"(e) \DU{}, uniform mode kept instead of damped"),
             (r"Duhamel, learned generator + small $O(v^2)$ network", r"(f) learned symbol and small $O(v^2)$ network")]:
    t = "\n".join(b + l[len(a):] if l.startswith(a) else l for l in t.split("\n"))
write("tmlr_ablation.tex", t)

# 4. physical-quantity table of Figure 5: app_physics_full.tex with a compact header
t = read(os.path.join(TAB, "app_physics_full.tex"))
t = t.replace(r"\begin{tabular}{lcccccccc}", r"\begin{tabular}{@{}lcccccccc@{}}", 1)
t = t.replace(r"model & $t_{\mathrm{nuc}}$ & $P_{\mathrm{S}}$ & domain size & wall density & $P/P_{\mathrm{S}}$ hist. &",
              r" & $t_{\mathrm{nuc}}$ & $P_{\mathrm{S}}$ & \begin{tabular}[b]{@{}c@{}}domain\\size\end{tabular} & "
              r"\begin{tabular}[b]{@{}c@{}}wall\\density\end{tabular} & \begin{tabular}[b]{@{}c@{}}$P/P_{\mathrm{S}}$\\hist.\end{tabular} &", 1)
t = t.replace(r"\DU{} (ours) &", r"\DU{} &", 1)
write("physics_fig.tex", t)

# 5. Allen-Cahn table: the paper calls the out-of-range split "new wall widths"
p = os.path.join(TAB, "ac_main.tex")
write("ac_main.tex", read(p).replace("& new physics &", "& new wall widths &"))

# 6. appendix figures drawn by paper/tools
dst = os.path.join(HERE, "figures", "appendix")
os.makedirs(dst, exist_ok=True)
n = 0
for f in ["design_space.png", "supp_satlaw.png", "subsets.pdf", "physics_panels.pdf", "spectrum.pdf", "pareto.pdf",
          "data_eff.pdf", "hyst_loops.pdf"]:
    src = os.path.join(TOOLS, "figures", f)
    if os.path.exists(src):
        shutil.copy2(src, os.path.join(dst, f)); n += 1
    else:
        print("missing figure", f)
print("wrote 10 tables, copied %d appendix figures" % n)
