#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""paper_tmlr/make_figs.py -- every figure, table and number of the TMLR draft, regenerated from result files.

Sources (never hand-typed): bench/results_final (benchmark finals, val-selected via bench/summarize2.load),
bench/results_phys (physics metrics, speed), bench/results_res (resolution transfer), ml_units/figs_recipe/probe_chaos.json
(Wiener-chaos probe), data/theory_cap.json (Fourier-truncation cap, computed in ml_units/THEORY_decorrelation.md),
$SCRATCH/bench_cache/alpha_bank_tdgl_val.json (linear-propagation accuracy of the measured operator).
Writes figures/*.pdf, tables/*.tex, numbers.tex. Missing inputs -> the figure/table is skipped with a warning.
"""
import glob, json, os, re, sys, warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
BENCH = os.path.join(ROOT, "bench")
sys.path.insert(0, BENCH)
import summarize2 as S2

FIG = os.path.join(HERE, "figures"); TAB = os.path.join(HERE, "tables")
os.makedirs(FIG, exist_ok=True); os.makedirs(TAB, exist_ok=True)
NUM = {}

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["DejaVu Serif", "Times New Roman", "Times"], "mathtext.fontset": "dejavuserif",
    "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8, "legend.fontsize": 6.8, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "lines.linewidth": 1.3, "savefig.bbox": "tight", "savefig.pad_inches": 0.02, "figure.dpi": 200, "pdf.fonttype": 42})

# one colour per model, used in every figure
STY = {
    "duhamel_uno": ("Duhamel-NO (ours)", "#b2182b", "o", "-"),
    "OURS": ("Duhamel-NO (ours)", "#b2182b", "o", "-"),
    "duhamel_uno_kmu": ("Duhamel-NO (ours)", "#b2182b", "o", "-"),
    "duhamel_uno_rou2": (r"Duhamel-NO, $O(u^2)$ remainder (abl.)", "#e66101", "D", "--"),
    "duhamel_uno_kmu_trunc": ("Duhamel-NO, 16-mode linear part (abl.)", "#f4a582", "o", ":"),
    "duhamel_ou2": (r"Duhamel, $O(u^2)$ remainder (abl.)", "#e66101", "D", "--"),
    "ffno": ("F-FNO", "#2166ac", "s", "-"),
    "unet": ("U-Net", "#5aae61", "^", "-"),
    "uno": ("UNO", "#7b3294", "v", "-"),
    "cnext": ("CNextU-Net", "#1b7837", "<", "--"),
    "cno": ("CNO", "#c2a5cf", ">", "--"),
    "tfno": ("TFNO", "#67a9cf", "p", "--"),
    "fno": ("FNO", "#404040", "x", "-"),
    "transolver": ("Transolver", "#a6611a", "*", ":"),
    "poseidon": ("Poseidon-T", "#dfc27d", "h", ":"),
    "refiner2": ("U-Net + PDE-Refiner", "#80cdc1", "1", ":"),
    "acdm": ("U-Net + ACDM diffusion", "#018571", "2", ":"),
    "flowmatch": ("U-Net + flow matching", "#f4a582", "3", ":"),
    "duhamel": ("Duhamel, learned linear (abl.)", "#f1948a", "o", ":"),
    "duhamel_uno_poly": ("Duhamel-NO, learned symbol (abl.)", "#d6604d", "o", "--"),
    "duhamel_uno_k0": ("Duhamel-NO, $k{=}0$ kept (abl.)", "#67001f", "o", "--"),
    "ours_fno": ("FNO, production (earlier)", "#999999", "x", ":"),
    "ours_fno_pre": ("C-FNO, D1-pretrained (ours)", "#b2182b", "o", "-"),
    "ours_duhamel": ("Duhamel-FNO (ours)", "#e66101", "D", "-"),
}
LAB = lambda m: STY.get(m, (m, "k", "o", "-"))[0]
COL = lambda m: STY.get(m, (m, "k", "o", "-"))[1]
MRK = lambda m: STY.get(m, (m, "k", "o", "-"))[2]
LS = lambda m: STY.get(m, (m, "k", "o", "-"))[3]


def warn(msg):
    print("WARNING:", msg, flush=True)


def jload(p):
    try:
        return json.load(open(p))
    except Exception:
        return None


def mean_std(v):
    v = [x for x in v if x is not None and np.isfinite(x)]
    return (float(np.mean(v)), float(np.std(v)), len(v)) if v else (np.nan, np.nan, 0)


def fmt(x, nd=3):
    return ("%%.%df" % nd) % x if np.isfinite(x) else "--"


def put(name, val):
    NUM[name] = val


# ------------------------------------------------------------------------------------------------ benchmark rows
R = S2.load()


def runs(task, model):
    if model == "OURS":
        return R.get((task, ours(task)), [])
    return R.get((task, model), [])


# Duhamel-NO: the same Duhamel step and exact linear operator on every task; the REMAINDER network is a hyperparameter
# chosen on the validation split (like every baseline's configuration): UNO or an O(u^2) network.
CANDIDATES = {"tdgl": ["duhamel_uno", "duhamel_uno_rou2"], "hyst": ["duhamel_uno"],
              "ac": ["duhamel_uno_kmu", "duhamel_uno_kmu_rou2", "duhamel_ou2"],
              "ch": ["duhamel_uno_kmu", "duhamel_uno_kmu_s1", "duhamel_uno_kmu_rou2", "duhamel_ou2"]}
_OURS = {}


def ours(task):
    if task not in _OURS:
        best, bm = -9e9, None
        for m in CANDIDATES.get(task, []):
            ds = R.get((task, m), [])
            sc = [S2.vscore(d) for d in ds]; sc = [x for x in sc if x is not None]
            if sc and np.mean(sc) > best:
                best, bm = np.mean(sc), m
        _OURS[task] = bm
        print("ours(%s) = %s (val score %.4f)" % (task, bm, best), flush=True)
    return _OURS[task]


def remainder_of(m):
    return "O(u^2)" if ("rou2" in m or "ou2" in m) else "UNO"


def metric(task, model, split, key):
    return mean_std([(d["eval"].get(split) or {}).get(key) for d in runs(task, model)])


# ================================================================================================ tables
TDGL_MODELS = ["duhamel_uno", "ffno", "unet", "uno", "cnext", "cno", "tfno", "fno", "poseidon", "acdm", "flowmatch", "refiner2", "transolver"]
FAMILY = dict(duhamel_uno="ours", duhamel_ou2="ours", ffno="spectral NO", fno="spectral NO", tfno="spectral NO", uno="spectral NO",
              cno="conv. NO", unet="U-Net", cnext="U-Net", transolver="transformer", poseidon="foundation", acdm="diffusion",
              refiner2="diffusion", flowmatch="flow matching", duhamel="ours (abl.)", ours_fno="ours (earlier)", ours_fno_pre="ours",
              ours_duhamel="ours")


def cell(m, s, n, nd=3, bold=False):
    if not np.isfinite(m):
        return "--"
    t = fmt(m, nd) + (r"{\scriptsize$\pm$%s}" % fmt(s, nd) if n > 1 else "")
    return r"\textbf{%s}" % t if bold else t


def table_tdgl():
    rows = []
    for m in TDGL_MODELS:
        if not runs("tdgl", m):
            continue
        c1, r1 = metric("tdgl", m, "D1_heldout", "corr_final"), metric("tdgl", m, "D1_heldout", "rel_final")
        c2, r2 = metric("tdgl", m, "D2_ood", "corr_final"), metric("tdgl", m, "D2_ood", "rel_final")
        t1 = metric("tdgl", m, "D1_heldout", "rel_t1")
        rows.append((m, c1, r1, c2, r2, t1))
    if not rows:
        return warn("no tdgl rows")
    best = {i: (max if i in (1, 3) else min)(range(len(rows)), key=lambda j: rows[j][i][0] if np.isfinite(rows[j][i][0]) else (-9 if i in (1, 3) else 9)) for i in range(1, 6)}
    L = [r"\begin{tabular}{llcccccc}", r"\toprule",
         r" & & & \multicolumn{2}{c}{D1 held-out (256)} & \multicolumn{2}{c}{D2 out-of-range (300)} & frame 1 \\",
         r"\cmidrule(lr){4-5}\cmidrule(lr){6-7}\cmidrule(lr){8-8}",
         r"model & family & seeds & corr@400 $\uparrow$ & rel.\,L2@400 $\downarrow$ & corr@400 $\uparrow$ & rel.\,L2@400 $\downarrow$ & rel.\,L2 $\downarrow$ \\", r"\midrule"]
    for j, (m, *cs) in enumerate(rows):
        n = cs[0][2]
        L.append("%s & %s & %d & %s \\\\" % (LAB(m), FAMILY.get(m, ""), n, " & ".join(cell(*c, bold=(best[i + 1] == j)) for i, c in enumerate(cs))))
        if m == "duhamel_uno":
            L.append(r"\midrule")
    L += [r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(TAB, "tdgl_main.tex"), "w").write("\n".join(L))
    du = dict((r[0], r) for r in rows)
    if "duhamel_uno" in du:
        r = du["duhamel_uno"]
        put("duCorrDone", fmt(r[1][0])); put("duRelDone", fmt(r[2][0])); put("duCorrDtwo", fmt(r[3][0])); put("duRelDtwo", fmt(r[4][0]))
        put("duRelTone", fmt(r[5][0]))
        base = [x for x in rows if x[0] != "duhamel_uno" and np.isfinite(x[2][0])]
        b = min(base, key=lambda x: x[2][0])
        put("bestBaseName", LAB(b[0])); put("bestBaseRelDone", fmt(b[2][0])); put("bestBaseCorrDone", fmt(b[1][0]))
        put("bestBaseRelDtwo", fmt(b[4][0])); put("bestBaseCorrDtwo", fmt(b[3][0]))
        put("relGainDone", "%d" % round(100 * (1 - r[2][0] / b[2][0]))); put("relGainDtwo", "%d" % round(100 * (1 - r[4][0] / b[4][0])))
        if "fno" in du:
            put("fnoRelDone", fmt(du["fno"][2][0])); put("fnoRatioDone", "%.1f" % (du["fno"][2][0] / r[2][0]))
            put("fnoRelTone", fmt(du["fno"][5][0])); put("ffnoRelTone", fmt(du.get("ffno", du["fno"])[5][0]))
        gen = [x for x in rows if x[0] in ("acdm", "flowmatch", "refiner2", "transolver", "poseidon")]
        if gen:
            put("genMaxCorr", fmt(max(x[1][0] for x in gen), 2))
    # per-subset table (corr@400)
    subs = ["extrap", "interp", "A_train", "B_alpha", "B_units", "C_shape", "D_all"]
    SUBN = dict(extrap="D1 extrap.", interp="D1 interp.", A_train="D2-A", B_alpha=r"D2-B$_\alpha$", B_units=r"D2-B$_{\mathrm{units}}$", C_shape="D2-C", D_all="D2-D")
    L = [r"\begin{tabular}{l" + "c" * len(subs) + "}", r"\toprule", "model & " + " & ".join(SUBN[s] for s in subs) + r" \\", r"\midrule"]
    for m in TDGL_MODELS[:8]:
        if not runs("tdgl", m):
            continue
        cs = []
        for s in subs:
            sp = "D1_heldout" if s in ("extrap", "interp") else "D2_ood"
            v = [((d["eval"].get(sp) or {}).get("subsets") or {}).get(s, {}).get("corr_final") for d in runs("tdgl", m)]
            cs.append(mean_std(v))
        L.append(LAB(m) + " & " + " & ".join(cell(*c) for c in cs) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(TAB, "tdgl_subsets.tex"), "w").write("\n".join(L))


RC = S2.load(closed=True)          # closed-loop hysteresis benchmark (retrained on closed loops)


def runs_h(model):
    return RC.get(("hyst", model), []) if RC else R.get(("hyst", model), [])      # never mix closed- and open-loop results


def metric_h(model, split, key):
    return mean_std([(d["eval"].get(split) or {}).get(key) for d in runs_h(model)])


def table_hyst():
    closed = bool(RC)
    put("hystClosed", "closed" if closed else "open")
    ms = ["cno", "duhamel_uno", "uno", "cnext", "poseidon", "unet", "tfno", "fno", "transolver", "refiner2", "ffno", "flowmatch"]
    L = [r"\begin{tabular}{lccccc}", r"\toprule", r"model & seeds & held-out material & new noise & extra closed set & field corr. (held-out mat.) \\", r"\midrule"]
    for m in ms:
        if not runs_h(m):
            continue
        cs = [metric_h(m, sp, "loop_err") for sp in ("test_material", "test_noise", "closed")]
        fc = metric_h(m, "test_material", "field_corr")
        L.append("%s & %d & %s & %s \\\\" % (LAB(m) if m != "duhamel_uno" else "Duhamel-NO (ours)", cs[0][2], " & ".join(cell(*c) for c in cs), cell(*fc)))
    L += [r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(TAB, "hyst.tex"), "w").write("\n".join(L))
    b = metric_h("cno", "test_material", "loop_err")
    c = metric_h("duhamel_uno", "test_material", "loop_err")
    put("hystGapDu", "%.2f" % (c[0] / b[0]))
    put("hystCno", fmt(b[0])); put("hystCnoStd", fmt(b[1])); put("hystDuno", fmt(c[0]))
    put("hystFno", fmt(metric_h("fno", "test_material", "loop_err")[0]))


def ours_acch():
    return "OURS"


ACCH_MODELS = ["OURS", "ffno", "cnext", "unet", "cno", "uno", "tfno", "fno", "refiner2", "flowmatch", "acdm", "poseidon", "transolver"]


def table_acch():
    L = [r"\begin{tabular}{l" + "ccc" * 2 + "}", r"\toprule",
         r" & \multicolumn{3}{c}{Allen--Cahn (rel.\,L2 at the final frame)} & \multicolumn{3}{c}{Cahn--Hilliard (rel.\,L2 at the final frame)} \\",
         r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}",
         r"model & in range & units & new physics & in range & units & new physics \\", r"\midrule"]
    vals = {}
    models = [ours_acch() if m == "OURS" else m for m in ACCH_MODELS]
    for m in models:
        row = []
        for t in ("ac", "ch"):
            for sp in ("id", "ood_units", "ood_inv"):
                row.append(metric(t, m, sp, "rel_final"))
        vals[m] = row
    present = [m for m in models if any(np.isfinite(c[0]) for c in vals[m])]
    for i in range(6):
        fin = [m for m in present if np.isfinite(vals[m][i][0])]
        if fin:
            bm = min(fin, key=lambda m: vals[m][i][0])
            vals[bm][i] = vals[bm][i] + ("best",)
    for m in present:
        L.append(("Duhamel-NO (ours)" if m == "OURS" else LAB(m)) + " & " + " & ".join(cell(c[0], c[1], c[2], bold=len(c) > 3) for c in vals[m]) + r" \\")
        if m == "OURS":
            L.append(r"\midrule")
    L += [r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(TAB, "acch.tex"), "w").write("\n".join(L))
    for t in ("ac", "ch"):
        for sp, nm in (("id", "Id"), ("ood_units", "Units"), ("ood_inv", "Inv")):
            v = metric(t, ours_acch(), sp, "rel_final")
            put("ou%s%s" % (t.upper(), nm), fmt(v[0]))
            others = [(metric(t, m, sp, "rel_final")[0], m) for m in models if m != "OURS"]
            others = [o for o in others if np.isfinite(o[0])]
            if others:
                b = min(others)
                put("base%s%s" % (t.upper(), nm), fmt(b[0])); put("base%s%sName" % (t.upper(), nm), LAB(b[1]))
            put("du%s%s" % (t.upper(), nm), fmt(metric(t, "duhamel_uno", sp, "rel_final")[0]))
            if others and np.isfinite(v[0]):
                put("ratio%s%s" % (t.upper(), nm), "%.1f" % (b[0] / v[0]))


# ================================================================================================ figures
def phys_runs():
    out = {}
    for f in glob.glob(os.path.join(BENCH, "results_phys", "tdgl_*.json")):
        d = jload(f)
        if not d or "sets" not in d:
            continue
        d.setdefault("tag", os.path.basename(f)[:-5])
        out.setdefault(d["model"], []).append(d)
    # physical quantities only from the validation-selected runs (the same checkpoints as Table 1); models without a
    # benchmark selection (ablation references) keep all their files
    for m in list(out):
        sel = {r["tag"] for r in R.get(("tdgl", m), [])}
        if sel:
            out[m] = [d for d in out[m] if d["tag"] in sel]
    return out


PH = phys_runs()


def pmean(m, sp, key):
    return mean_std([d["sets"][sp].get(key) for d in PH.get(m, []) if sp in d["sets"]])


def fig_error_vs_time():
    ms = [m for m in ["duhamel_uno", "ffno", "unet", "uno", "cnext", "cno", "tfno", "fno", "duhamel"] if m in PH]
    if not ms:
        return warn("no physmetrics for error-vs-time")
    fig, axs = plt.subplots(1, 2, figsize=(6.9, 2.35), sharey=True)
    for ax, sp, ttl in zip(axs, ("D1_heldout", "D2_ood"), ("D1: held-out materials, new noise", "D2: out of range, zero-shot")):
        for m in ms:
            cs = np.array([d["sets"][sp]["rel_curve_mean"] for d in PH[m] if sp in d["sets"]])
            if not len(cs):
                continue
            t = np.arange(1, cs.shape[1] + 1); mu = cs.mean(0)
            ax.plot(t, mu, color=COL(m), ls=LS(m), lw=1.6 if m == "duhamel_uno" else 1.0, label=LAB(m), zorder=5 if m == "duhamel_uno" else 2)
            if len(cs) > 1:
                ax.fill_between(t, cs.min(0), cs.max(0), color=COL(m), alpha=0.18, lw=0)
        ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("frame $t$"); ax.set_title(ttl)
        ax.axvspan(1, 10, color="0.92", zorder=0); ax.text(0.03, 0.04, "linear phase", fontsize=6.5, color="0.35", transform=ax.transAxes)
        ax.set_xlim(1, 400); ax.grid(alpha=0.25, lw=0.4, which="both")
    axs[0].set_ylabel("relative L2 error")
    axs[1].legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False)
    fig.savefig(os.path.join(FIG, "error_vs_time.pdf")); plt.close(fig)


def fig_rollouts():
    """truth (top row) and, for each model, the ERROR field (prediction - truth) on one shared symmetric scale per frame:
    correlation saturates near 1 after the first frames for every model, the error field does not."""
    ms = ["duhamel_uno", "ffno", "unet", "fno"]
    ex = {}
    for m in ms:
        fs = sorted(glob.glob(os.path.join(BENCH, "results_phys", "tdgl_%s_*_s0_examples.npz" % m)))
        fs = [f for f in fs if not re.search(r"_K\d+_examples", f)]
        if fs:
            ex[m] = np.load(fs[0])
    if "duhamel_uno" not in ex or "duhamel_uno" not in PH:
        return warn("no rollout examples")
    T = list(ex["duhamel_uno"]["t"])
    d0 = [d for d in PH["duhamel_uno"] if d["seed"] == 0][0]
    for sp, nm in (("D1_heldout", "rollouts_d1"), ("D2_ood", "rollouts_d2")):
        rows_ = d0["sets"][sp]["rows"]; med = np.median([r["rel_t400"] for r in rows_])
        cand = [int(k.split("_")[-2]) for k in ex["duhamel_uno"].files if k.startswith(sp) and k.endswith("_true")]
        if not cand:
            continue
        j = min(cand, key=lambda i: abs(rows_[i]["rel_t400"] - med))           # typical trajectory (median error of ours)
        key = "%s_%d" % (sp, j)
        truth = ex["duhamel_uno"][key + "_true"].astype(np.float32)               # (nT, Z, H, W)
        mods = [m for m in ms if m in ex and key + "_pred" in ex[m]]
        nr = 1 + len(mods)
        fig = plt.figure(figsize=(0.98 * len(T) + 1.15, 0.98 * nr + 0.25))
        gs = gridspec.GridSpec(nr, len(T) + 1, width_ratios=[1] * len(T) + [0.06], wspace=0.05, hspace=0.08)
        for jj, t in enumerate(T):
            ref = truth[jj][-1]
            lim = float(np.abs(ref).max()) + 1e-12
            ax = fig.add_subplot(gs[0, jj]); im0 = ax.imshow(ref / lim, cmap="RdBu_r", vmin=-1, vmax=1, interpolation="nearest")
            ax.set_title("$t=%d$" % t, fontsize=7.5); ax.set_xticks([]); ax.set_yticks([])
            if jj == 0:
                ax.set_ylabel("FerroX\n(truth)", fontsize=6.6, rotation=0, ha="right", va="center")
            for i, m in enumerate(mods):
                p = ex[m][key + "_pred"].astype(np.float32)[jj][-1]
                err = (p - ref) / lim
                ax = fig.add_subplot(gs[i + 1, jj])
                im1 = ax.imshow(err, cmap="PuOr_r", vmin=-0.5, vmax=0.5, interpolation="nearest")
                ax.set_xticks([]); ax.set_yticks([])
                rl = float(np.linalg.norm(ex[m][key + "_pred"].astype(np.float32)[jj] - truth[jj]) / np.linalg.norm(truth[jj]))
                ax.text(0.03, 0.05, "%.3f" % rl, transform=ax.transAxes, fontsize=5.6,
                        bbox=dict(boxstyle="square,pad=0.12", fc="white", ec="none", alpha=0.8))
                if jj == 0:
                    lab = LAB(m) + ("" if m == "duhamel_uno" else " + recipe")
                    ax.set_ylabel(lab.replace(" (ours)", "\n(ours)").replace(" + recipe", "\n+ recipe") + "\nerror", fontsize=6.4,
                                  rotation=0, ha="right", va="center")
        cax = fig.add_subplot(gs[0, -1]); cb = fig.colorbar(im0, cax=cax); cb.set_ticks([-1, 0, 1]); cb.ax.tick_params(labelsize=5.5)
        cb.set_label(r"$P_z/\max|P_z|$", fontsize=6)
        cax = fig.add_subplot(gs[1:, -1]); cb = fig.colorbar(im1, cax=cax); cb.ax.tick_params(labelsize=5.5)
        cb.set_label(r"$(P_z^{\mathrm{pred}}-P_z)/\max|P_z|$", fontsize=6)
        fig.savefig(os.path.join(FIG, nm + ".pdf")); plt.close(fig)
        put("ex%sRel" % ("D" if sp == "D1_heldout" else "DD"), "%.3f" % rows_[j]["rel_t400"]); put("ex%sMedRel" % ("D" if sp == "D1_heldout" else "DD"), "%.3f" % med)


PHYS_ROWS = [("rel_t1", "rel. L2, frame 1"), ("rel_t10", "rel. L2, frame 10"), ("rel_t400", "rel. L2, frame 400"),
             ("tnuc_err", "nucleation time |Δt| (frames)"), ("psat_err", "saturation ⟨|P|⟩ error"),
             ("dsize_err_t400", "domain size error"), ("wall_err_t400", "wall density error"), ("hist_w1", r"$P/P_s$ histogram $W_1$"),
             ("phi_rel_t400", "potential φ rel. L2"), ("ez_rel_t400", r"depol. field $E_z$ rel. L2"),
             ("energy_traj_err", "free energy F(t) error"), ("E_grad_err_t400", "wall (gradient) energy error")]
PHYS_MODELS = ["duhamel_uno", "ffno", "unet", "uno", "cnext", "cno", "tfno", "fno", "duhamel", "poseidon", "acdm", "flowmatch", "refiner2", "transolver"]


def fig_physics(sp="D1_heldout", nums=True):
    ms = [m for m in PHYS_MODELS if m in PH]
    if len(ms) < 2:
        return warn("physics heatmap: too few models")
    V = np.array([[pmean(m, sp, k)[0] for m in ms] for k, _ in PHYS_ROWS])
    with np.errstate(divide="ignore", invalid="ignore"):
        Rr = V / np.nanmin(V, 1, keepdims=True)
    fig, ax = plt.subplots(figsize=(0.52 * len(ms) + 2.2, 0.27 * len(PHYS_ROWS) + 0.9))
    im = ax.imshow(np.log10(np.clip(Rr, 1, 100)), cmap="YlOrRd", vmin=0, vmax=2, aspect="auto")
    for i in range(V.shape[0]):
        for j in range(V.shape[1]):
            v = V[i, j]
            if np.isfinite(v):
                s = ("%.2f" % v) if v >= 0.1 else ("%.3f" % v) if v >= 0.001 else "%.0e" % v
                if PHYS_ROWS[i][0] == "tnuc_err":
                    s = "%.1f" % v
                ax.text(j, i, s, ha="center", va="center", fontsize=5.4, color="white" if Rr[i, j] > 20 else "black",
                        fontweight="bold" if Rr[i, j] < 1.0001 else "normal")
    ax.set_xticks(range(len(ms))); ax.set_xticklabels([LAB(m).replace(" (ours)", "*").replace(" (abl.)", "†") for m in ms], rotation=40, ha="right")
    ax.set_yticks(range(len(PHYS_ROWS))); ax.set_yticklabels([r for _, r in PHYS_ROWS])
    for s in ax.spines.values():
        s.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.01); cb.set_label("error ÷ best (log scale)")
    cb.set_ticks([0, 1, 2]); cb.set_ticklabels(["1×", "10×", "≥100×"])
    fig.savefig(os.path.join(FIG, "physics_%s.pdf" % sp)); plt.close(fig)
    if not nums:
        return
    # numbers for the text
    for k, nm in (("tnuc_err", "Tnuc"), ("psat_err", "Psat"), ("dsize_err_t400", "Dsize"), ("wall_err_t400", "Wall"), ("phi_rel_t400", "Phi"),
                  ("ez_rel_t400", "Ez"), ("energy_traj_err", "Energy"), ("rel_t1", "RelTone"), ("rel_t10", "RelTten"), ("hist_w1", "Wone")):
        for m, mn in (("duhamel_uno", "du"), ("ffno", "ffno"), ("fno", "fno"), ("unet", "unet")):
            v = pmean(m, sp, k)[0]
            put("ph%s%s" % (mn, nm), ("%.1f" % v) if k == "tnuc_err" else fmt(v, 3))
    # rank count: on how many physics rows is duhamel_uno best
    if "duhamel_uno" in ms:
        j = ms.index("duhamel_uno")
        put("phBestCount", "%d" % int(np.sum(np.isclose(Rr[:, j], 1.0)))); put("phRows", "%d" % len(PHYS_ROWS))
        put("phWorstGap", "%d" % max(1, round(100 * (np.nanmax(Rr[:, j]) - 1))))


def fig_physics_panels(sp="D1_heldout"):
    """research-grade replacement of the heat map: (a) free-energy trajectories, (b,c) per-trajectory distributions (ECDFs) of
    the nucleation-time and saturation errors, (d) Cleveland dot plot of every physical metric relative to the best model."""
    det = [m for m in ["duhamel_uno", "ffno", "unet", "uno", "cnext", "cno", "tfno", "fno"] if m in PH]
    if len(det) < 3:
        return warn("physics panels: too few models")
    fig = plt.figure(figsize=(6.9, 5.3))
    gs = gridspec.GridSpec(2, 3, height_ratios=[1, 1.15], hspace=0.42, wspace=0.34)
    # (a) free energy: error of the free-energy trajectory (main), the normalised truth (inset)
    ax = fig.add_subplot(gs[0, 0])
    Et = np.array(PH["duhamel_uno"][0]["sets"][sp]["energy_true_mean"]); t = np.arange(len(Et))
    F0, F1 = Et[0], Et[-1]
    for m in ["duhamel_uno", "ffno", "unet", "uno", "fno"]:
        if m in PH:
            Ep = np.mean([d["sets"][sp]["energy_pred_mean"] for d in PH[m]], 0)
            ax.plot(t[1:], np.abs(Ep - Et)[1:] / abs(F0 - F1), color=COL(m), lw=1.5 if m == "duhamel_uno" else 0.9, ls=LS(m))
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(1, 400); ax.set_xlabel("frame $t$")
    ax.set_ylabel(r"$|F_{\mathrm{pred}}-F|\,/\,(F_0-F_{400})$"); ax.set_title("(a) free-energy trajectory", loc="left", fontsize=8)
    ax.grid(alpha=0.25, lw=0.4, which="both")
    ins = ax.inset_axes([0.6, 0.1, 0.36, 0.3])
    ins.plot(t[1:], ((Et - F1) / (F0 - F1))[1:], color="k", lw=1.0); ins.set_xscale("log"); ins.set_xlim(1, 400)
    ins.tick_params(labelsize=4.5); ins.set_title("FerroX $F(t)$", fontsize=5, pad=1); ins.set_yticks([0, 1])
    # (b, c) ECDFs of per-trajectory errors
    for k_, (key, ttl, xl, logx) in enumerate((("tnuc_err", "(b) nucleation time", r"$|\Delta t_{\mathrm{nuc}}|$ (frames)", False),
                                              ("psat_err", "(c) saturation polarisation", r"$|\langle|P|\rangle_{\mathrm{pred}}/\langle|P|\rangle-1|$", True))):
        ax = fig.add_subplot(gs[0, 1 + k_])
        if k_ == 0:
            ax_b = ax
        for m in ["duhamel_uno", "ffno", "unet", "uno", "fno"]:
            if m not in PH:
                continue
            v = np.sort(np.concatenate([[r[key] for r in d["sets"][sp]["rows"]] for d in PH[m]]))
            if logx:
                v = np.maximum(v, 1e-5)
            ax.step(v, np.arange(1, len(v) + 1) / len(v), where="post", color=COL(m), lw=1.5 if m == "duhamel_uno" else 0.9, ls=LS(m),
                    label=LAB(m).replace(" (ours)", ""))
        if logx:
            ax.set_xscale("log")
        else:
            ax.set_xlim(-0.2, 4.2)
        ax.set_ylim(0, 1.01); ax.set_xlabel(xl); ax.set_ylabel("fraction of trajectories"); ax.set_title(ttl, loc="left", fontsize=8)
        ax.grid(alpha=0.25, lw=0.4)
    ax_b.legend(frameon=False, fontsize=5.6, loc="lower right")
    if sp == "D1_heldout":
        fr = {m: np.mean(np.concatenate([[r["tnuc_err"] == 0 for r in d["sets"][sp]["rows"]] for d in PH[m]])) for m in det}
        put("tnucExactDu", "%d" % round(100 * fr["duhamel_uno"]))
        put("tnucExactBase", "%d" % round(100 * max(v for m, v in fr.items() if m != "duhamel_uno")))
    # (d) Cleveland dot plot over all metrics
    ax = fig.add_subplot(gs[1, :])
    V = np.array([[pmean(m, sp, k)[0] for m in det] for k, _ in PHYS_ROWS])
    R_ = V / np.nanmin(V, 1, keepdims=True)
    yy = np.arange(len(PHYS_ROWS))[::-1]
    for i in range(len(PHYS_ROWS)):
        ax.plot([1, np.nanmax(R_[i])], [yy[i]] * 2, color="0.85", lw=0.8, zorder=1)
    for j, m in enumerate(det):
        ax.scatter(R_[:, j], yy, s=30 if m == "duhamel_uno" else 16, color=COL(m), marker=MRK(m) if m != "duhamel_uno" else "o",
                   edgecolor="k" if m == "duhamel_uno" else COL(m), lw=0.5, zorder=4 if m == "duhamel_uno" else 3,
                   label=LAB(m).replace(" (ours)", " (ours)") + ("" if m == "duhamel_uno" else " + recipe"))
    # paired test against the best BASELINE on each metric: per-trajectory seed means, bootstrap CI of the ratio of means
    rng = np.random.default_rng(0); sig = []
    def traj(m, k):
        return np.mean([[r[k] for r in d["sets"][sp]["rows"]] for d in PH[m]], 0)
    for i, (k, _) in enumerate(PHYS_ROWS):
        bb = min([m for m in det if m != "duhamel_uno"], key=lambda m: V[i, det.index(m)])
        x, y = traj("duhamel_uno", k), traj(bb, k)
        bs = [x[j].mean() / y[j].mean() for j in (rng.integers(0, len(x), len(x)) for _ in range(2000))]
        lo, hi = np.percentile(bs, [2.5, 97.5])
        sig.append("better" if hi < 1 else ("worse" if lo > 1 else "tie"))
        if sig[-1] != "better":
            ax.text(1.0, yy[i], "  %s %s" % ("tied with" if sig[-1] == "tie" else "worse than", LAB(bb)),
                    transform=ax.get_yaxis_transform(), fontsize=5.6, va="center", color="0.3")
    if sp == "D1_heldout":
        put("phSigBetter", "%d" % sig.count("better")); put("phSigTie", "%d" % sig.count("tie")); put("phSigWorse", "%d" % sig.count("worse"))
        put("phTieList", ", ".join(r.replace(r"$P/P_s$ histogram $W_1$", "polarisation histogram") for (k, r), s_ in zip(PHYS_ROWS, sig) if s_ == "tie"))
    ax.set_yticks(yy); ax.set_yticklabels([r for _, r in PHYS_ROWS], fontsize=6.4)
    ax.set_xscale("log"); ax.set_xlabel("error ÷ error of the best model on that metric (1 = best)")
    tk = [1, 1.5, 2, 3, 5, 8]; ax.set_xticks(tk); ax.set_xticklabels(["%g" % x for x in tk]); ax.minorticks_off()
    ax.axvline(1, color="k", lw=0.6); ax.grid(axis="x", alpha=0.25, lw=0.4, which="both")
    ax.set_title("(d) all physical metrics, %s (generative" % ("held-out D1" if sp == "D1_heldout" else "zero-shot D2") + " and transformer baselines are 10--3000$\\times$ off this scale)", loc="left", fontsize=8)
    ax.legend(ncol=4, frameon=False, fontsize=6.0, loc="upper center", bbox_to_anchor=(0.45, -0.22))
    fig.savefig(os.path.join(FIG, "physics_panels%s.pdf" % ("" if sp == "D1_heldout" else "_D2"))); plt.close(fig)


def fig_physics_d2():
    fig_physics("D2_ood", nums=False); fig_physics_panels("D2_ood")


def fig_spectrum():
    ms = [m for m in ["duhamel_uno", "ffno", "unet", "uno", "cno", "fno"] if m in PH and any("spec_err_t1" in d["sets"]["D1_heldout"] for d in PH[m])]
    if not ms:
        return warn("no error spectra")
    ts = [1, 10, 50, 400]
    fig, axs = plt.subplots(1, 4, figsize=(6.9, 1.95), sharey=True)
    for ax, t in zip(axs, ts):
        tp = None
        for m in ms:
            E = [d["sets"]["D1_heldout"] for d in PH[m] if "spec_err_t%d" % t in d["sets"]["D1_heldout"]]
            s = np.array([e["spec_err_t%d" % t] for e in E]).mean(0); tp = np.array(E[0]["spec_truepow_t%d" % t])
            k = np.arange(len(s))
            ax.plot(k[1:], s[1:], color=COL(m), ls=LS(m), lw=1.5 if m == "duhamel_uno" else 1.0, label=LAB(m))
        if tp is not None:
            ax.plot(np.arange(len(tp))[1:], tp[1:], color="0.55", lw=0.9, ls=(0, (1, 1)), label="true power (share)")
        ax.axvspan(8, 8 * np.sqrt(2), color="0.88", lw=0, zorder=0)
        ax.set_yscale("log"); ax.set_ylim(1e-7, 1); ax.set_title("frame %d" % t); ax.set_xlabel(r"wavenumber $|k|$"); ax.set_xlim(1, 45)
        ax.grid(alpha=0.25, lw=0.4)
    axs[0].set_ylabel("squared error per $|k|$ bin\n(fraction of true power)")
    axs[0].text(12.0, 2e-7, "FNO\ncut-off", fontsize=5.8, color="0.4")
    axs[-1].legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=False)
    fig.savefig(os.path.join(FIG, "spectrum.pdf")); plt.close(fig)
    for m, nm in (("duhamel_uno", "du"), ("fno", "fno"), ("ffno", "ffno")):
        if m in ms:
            E = [d["sets"]["D1_heldout"] for d in PH[m] if "spec_err_t1" in d["sets"]["D1_heldout"]]
            s = np.array([e["spec_err_t1"] for e in E]).mean(0)
            put("hiK%s" % nm, "%d" % round(100 * s[17:].sum() / s.sum()))


def fig_pareto():
    sp = jload(os.path.join(BENCH, "results_phys", "speed.json"))
    if not sp:
        return warn("no speed.json")
    name2m = {v[0].replace(" (ours)", ""): k for k, v in STY.items()}
    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    pts = []
    for lab, r in sp["models"].items():
        base = lab.split(" + ")[0].replace(" (ours)", "").replace(", learned linear", "")
        m = {"Duhamel-UNO": "duhamel_uno", "F-FNO": "ffno", "U-Net": "unet", "UNO": "uno", "CNextU-Net": "cnext", "CNO": "cno", "TFNO": "tfno",
             "FNO": "fno", "Duhamel": "duhamel", "Transolver": "transolver", "ACDM": "acdm", "Flow matching": "flowmatch", "PDE-Refiner": "refiner2"}.get(base)
        if m is None or "s_per_traj_b16" not in r:
            continue
        e = metric("tdgl", m, "D1_heldout", "rel_final")[0]
        if not np.isfinite(e):
            continue
        x = r["s_per_traj_b16"]; pts.append((m, x, e, r["s_per_traj_b1"]))
        ax.scatter(x, e, s=34 if m == "duhamel_uno" else 22, color=COL(m), marker="o" if m == "duhamel_uno" else "s", zorder=4, edgecolor="k" if m == "duhamel_uno" else COL(m), lw=0.5)
        off = {"fno": (-14, 4), "tfno": (-2, 6), "ffno": (5, -4), "unet": (5, -1), "uno": (-18, -7), "cnext": (5, 3), "cno": (5, -2),
               "duhamel_uno": (6, -3), "duhamel": (5, 2)}.get(m, (4, 2))
        ax.annotate(LAB(m).replace(" (ours)", "").replace(" (abl.)", "").replace("Duhamel, learned linear", "Duhamel, learned lin."), (x, e),
                    fontsize=5.4, xytext=off, textcoords="offset points", color="k" if m == "duhamel_uno" else "0.25",
                    fontweight="bold" if m == "duhamel_uno" else "normal")
    ax.axvline(sp["ferrox_s_median"], color="k", lw=0.8, ls="--"); ax.text(sp["ferrox_s_median"] * 0.93, 0.13, "FerroX\n(196 s)", fontsize=6, ha="right")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("s per 400-frame trajectory (A100, batch 16)"); ax.set_ylabel("rel. L2 at frame 400 (D1)")
    yt = [0.1, 0.15, 0.2, 0.3, 0.5, 1, 2, 3]; ax.set_yticks(yt); ax.set_yticklabels(["%g" % v for v in yt]); ax.minorticks_off()
    ax.grid(alpha=0.25, lw=0.4, which="major")
    fig.savefig(os.path.join(FIG, "pareto.pdf")); plt.close(fig)
    d = {p[0]: p for p in pts}
    if "duhamel_uno" in d:
        put("duSecBatch", "%.2f" % d["duhamel_uno"][1]); put("duSecOne", "%.1f" % d["duhamel_uno"][3])
        put("duSpeedup", "%d" % round(sp["ferrox_s_median"] / d["duhamel_uno"][1])); put("duSpeedupOne", "%d" % round(sp["ferrox_s_median"] / d["duhamel_uno"][3]))
    gens = [p for p in pts if p[0] in ("acdm", "flowmatch", "refiner2")]
    if gens and "duhamel_uno" in d:
        put("genSlowdown", "%d" % round(min(p[1] for p in gens) / d["duhamel_uno"][1]))


def fig_subsets():
    subs = ["interp", "extrap", "A_train", "B_alpha", "B_units", "C_shape", "D_all"]
    nice = ["D1\ninterp.", "D1\nextrap.", "D2-A", "D2-B$_\\alpha$", "D2-B$_{u}$", "D2-C", "D2-D"]
    ms = [m for m in ["duhamel_uno", "ffno", "unet", "uno", "cno", "tfno", "fno"] if runs("tdgl", m)]
    fig, ax = plt.subplots(figsize=(6.9, 1.9))
    w = 0.8 / len(ms)
    for i, m in enumerate(ms):
        mu, sd = [], []
        for s in subs:
            sp = "D1_heldout" if s in ("extrap", "interp") else "D2_ood"
            v = [((d["eval"].get(sp) or {}).get("subsets") or {}).get(s, {}).get("rel_final") for d in runs("tdgl", m)]
            a = mean_std(v); mu.append(a[0]); sd.append(a[1])
        x = np.arange(len(subs)) + (i - len(ms) / 2 + 0.5) * w
        ax.bar(x, mu, w * 0.92, yerr=sd, color=COL(m), label=LAB(m), error_kw=dict(lw=0.5, capsize=0), edgecolor="none")
    ax.set_xticks(range(len(subs))); ax.set_xticklabels(nice); ax.set_ylabel("rel. L2 at frame 400")
    ax.legend(ncol=4, frameon=False, loc="upper left", fontsize=6.2); ax.set_ylim(0, None); ax.grid(axis="y", alpha=0.25, lw=0.4)
    fig.savefig(os.path.join(FIG, "subsets.pdf")); plt.close(fig)


def fig_acch():
    ms = [m for m in ["OURS", "ffno", "cnext", "unet", "uno", "tfno", "fno"] if runs("ac", m) or runs("ch", m)]
    if not ms:
        return warn("no ac/ch")
    fig, axs = plt.subplots(1, 2, figsize=(6.9, 2.0), sharey=True)
    sps = ["id", "ood_units", "ood_inv"]; nice = ["in range", "units", "new physics"]
    w = 0.8 / len(ms)
    for ax, t, ttl in zip(axs, ("ac", "ch"), ("Allen–Cahn", "Cahn–Hilliard")):
        for i, m in enumerate(ms):
            vals = [metric(t, m, sp, "rel_final") for sp in sps]
            x = np.arange(3) + (i - len(ms) / 2 + 0.5) * w
            ax.bar(x, [v[0] for v in vals], w * 0.92, yerr=[v[1] for v in vals], color=COL(m), label=LAB(m), error_kw=dict(lw=0.5, capsize=0))
        ax.set_yscale("log"); ax.set_xticks(range(3)); ax.set_xticklabels(nice); ax.set_title(ttl); ax.grid(axis="y", alpha=0.25, lw=0.4, which="both")
    axs[0].set_ylabel("rel. L2, final frame")
    axs[1].legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False)
    fig.savefig(os.path.join(FIG, "acch.pdf")); plt.close(fig)


def fig_res():
    rr = {}
    for f in glob.glob(os.path.join(BENCH, "results_res", "ac_*.json")):
        d = jload(f)
        if d:
            rr.setdefault(d["model"], []).append(d)
    if not rr:
        return warn("no resolution-transfer results")
    # pick the val-selected configuration per model (same as the benchmark rows)
    sel = {}
    for m, ds in rr.items():
        cfgs = {d["tag"].split("_s")[0] for d in runs("ac", m)} if runs("ac", m) else set()
        keep = [d for d in ds if not cfgs or d["tag"].rsplit("_s", 1)[0] in cfgs]
        sel[m] = keep or ds
    ms = [m for m in [ours("ac"), "ffno", "cnext", "unet", "cno", "uno", "tfno", "fno", "duhamel_uno"] if m in sel]
    if "duhamel_uno_kmu" not in sel:
        ms = [m for m in ms if m != "duhamel_uno"]
    fig, axs = plt.subplots(1, 2, figsize=(5.4, 2.1), sharey=True)
    for ax, sp, ttl in zip(axs, ("id", "ood_inv"), ("in range", "new physics")):
        for m in ms:
            ys = []
            for n in ("32", "64", "128"):
                v = [((d["grids"].get(n) or {}).get(sp) or {}).get("rel_final") for d in sel[m]]
                ys.append(mean_std(v)[0])
            lab_ = {"duhamel_uno": "Duhamel-NO, grid-unit symbol (abl.)"}.get(m, "Duhamel-NO (ours)" if m == ours("ac") else LAB(m) + ("" if m.startswith("duhamel") else " + recipe"))
            ax.plot([32, 64, 128], ys, color=("#b2182b" if m == ours("ac") else COL(m)) if m != "duhamel_uno" else "#d6604d", marker=MRK(m),
                    ls=("-" if m == ours("ac") else LS(m)) if m != "duhamel_uno" else ":", ms=3.5, label=lab_, lw=1.6 if m == ours("ac") else 1.0)
        ax.set_xscale("log", base=2); ax.set_xticks([32, 64, 128]); ax.set_xticklabels(["32²", "64² (train)", "128²"])
        ax.set_yscale("log"); ax.set_title(ttl); ax.grid(alpha=0.25, lw=0.4, which="both")
    axs[0].set_ylabel("rel. L2, final frame")
    axs[1].legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False)
    fig.savefig(os.path.join(FIG, "resolution.pdf")); plt.close(fig)
    for m, nm in ((ours("ac"), "du"), ("duhamel_ou2", "ou"), ("fno", "fno"), ("ffno", "ffno")):
        if m in sel:
            for n in ("32", "64", "128"):
                put("res%s%s" % (nm, {"32": "A", "64": "B", "128": "C"}[n]),
                    fmt(mean_std([((d["grids"].get(n) or {}).get("id") or {}).get("rel_final") for d in sel[m]])[0]))


def fig_data_eff():
    pts = {}
    for f in glob.glob(os.path.join(BENCH, "results_final", "tdgl_*_s*_K*.json")):
        d = jload(f)
        if not d or not isinstance(d.get("eval"), dict):
            continue
        K = int(re.search(r"_K(\d+)$", d["tag"]).group(1))
        pts.setdefault(d["model"], {}).setdefault(K, []).append(d["eval"]["D1_heldout"]["rel_final"])
    if not pts:
        return warn("no data-efficiency runs yet")
    fig, ax = plt.subplots(figsize=(3.0, 2.2))
    for m in ["duhamel_uno", "ffno", "fno"]:
        if m not in pts:
            continue
        full = [(d["eval"]["D1_heldout"]["rel_final"]) for d in runs("tdgl", m)]
        P = dict(pts[m]); P[682] = full
        ks = sorted(P); mu = [np.mean(P[k]) for k in ks]; lo = [np.min(P[k]) for k in ks]; hi = [np.max(P[k]) for k in ks]
        ax.plot(ks, mu, color=COL(m), marker=MRK(m), ms=3.5, label=LAB(m) + ("" if m == "duhamel_uno" else " + recipe"))
        ax.fill_between(ks, lo, hi, color=COL(m), alpha=0.18, lw=0)
        for k in ks:
            put("de%s%s" % ({"duhamel_uno": "du", "ffno": "ffno", "fno": "fno"}[m], {32: "A", 128: "B", 682: "C"}.get(k, "X")), fmt(np.mean(P[k])))
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xticks([32, 128, 682]); ax.set_xticklabels(["32", "128", "682"])
    yt = [0.12, 0.15, 0.2, 0.25, 0.3]; ax.set_yticks(yt); ax.set_yticklabels(["%g" % v for v in yt]); ax.minorticks_off()
    ax.set_xlabel("training materials"); ax.set_ylabel("rel. L2 at frame 400 (D1)"); ax.legend(frameon=False, loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=2, fontsize=6); ax.grid(alpha=0.25, lw=0.4, which="both")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "data_eff.pdf")); plt.close(fig)


def fig_theory():
    pc = jload(os.path.join(ROOT, "ml_units", "figs_recipe", "probe_chaos.json"))
    cap = jload(os.path.join(HERE, "data", "theory_cap.json"))
    if not pc or not cap:
        return warn("theory inputs missing")
    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    lab = {"a_raw(biased)": ("(a) FNO with biases", "#404040", "x"), "a_units(biased)": ("(a) FNO with biases, units", "#7f7f7f", "+"),
           "b_recipe(rms-norm)": ("(b) rms-normalised FNO", "#2166ac", "s"), "c_duhamel(O(u^2))": (r"(c) Duhamel, $O(u^2)$ remainder", "#b2182b", "o")}
    for k, (l, c, mk) in lab.items():
        if k not in pc:
            continue
        e = [r["eps_over_U"] for r in pc[k]]; H = [max(r["H"], 1e-16) for r in pc[k]]
        ax.plot(e, H, marker=mk, color=c, ms=3.5, label=l)
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel(r"seed amplitude $\epsilon/U$"); ax.set_ylabel(r"non-linear output share $\mathcal{H}$")
    ax.set_title("Allen–Cahn, one step of linear-phase noise", fontsize=8); ax.legend(frameon=False, fontsize=6); ax.grid(alpha=0.25, lw=0.4, which="both")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "theory_H.pdf")); plt.close(fig)
    fig, ax = plt.subplots(figsize=(4.6, 2.9))
    cb = jload(os.path.join(HERE, "data", "theory_cap_bank.json"))
    fr = cb["frames"]
    ax.plot(fr, cb["cap_fno8"], "k--", lw=1.0, label=r"cap, FNO mode set")
    ax.plot(fr, cb["cap_box16"], "k:", lw=1.0, label=r"cap, 16-mode box")
    for m in ("fno", "tfno", "uno", "ffno", "duhamel_uno"):
        if m in PH:
            y = [pmean(m, "D1_heldout", "corr_t%d" % f_)[0] for f_ in fr]
            ax.plot(fr, y, color=COL(m), marker=MRK(m), ms=3.5, label=LAB(m) + (" + recipe" if m != "duhamel_uno" else ""))
    ax.set_xlabel("frame $n$"); ax.set_ylabel("pattern correlation"); ax.set_xscale("log"); ax.set_xticks(fr); ax.set_xticklabels([str(x) for x in fr])
    ax.set_title("early frames, held-out D1 materials", fontsize=8); ax.legend(frameon=False, fontsize=5.6, loc="lower right", bbox_to_anchor=(1.0, 0.02)); ax.grid(alpha=0.25, lw=0.4)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "theory.pdf")); plt.close(fig)
    put("capTone", "%.3f" % cb["cap_fno8"][0]); put("capBoxTone", "%.3f" % cb["cap_box16"][0]); put("capTten", "%.3f" % cb["cap_fno8"][-1])
    if "fno" in PH:
        put("fnoCorrTone", "%.3f" % pmean("fno", "D1_heldout", "corr_t1")[0])
        put("fnoCorrTtwo", "%.3f" % pmean("fno", "D1_heldout", "corr_t2")[0]); put("capTtwo", "%.3f" % cb["cap_fno8"][1])
        put("fnoCorrTthree", "%.3f" % pmean("fno", "D1_heldout", "corr_t3")[0]); put("capTthree", "%.3f" % cb["cap_fno8"][2])
    if "duhamel_uno" in PH:
        put("duCorrTone", "%.3f" % pmean("duhamel_uno", "D1_heldout", "corr_t1")[0])
    b = pc.get("b_recipe(rms-norm)"); c = pc.get("c_duhamel(O(u^2))")
    if b and c:
        put("chaosHrecipe", "%.3f" % b[0]["H"]); put("chaosRhoRecipe", "%.3f" % b[0]["rho"]); e_ = "%.0e" % c[0]["H"]; put("chaosHduh", "%s\\times10^{%d}" % (e_.split("e")[0], int(e_.split("e")[1])))


def fig_schematic():
    fig, ax = plt.subplots(figsize=(6.9, 1.95)); ax.set_xlim(0, 100); ax.set_ylim(-3, 27); ax.axis("off")
    def box(x, y, w, h, txt, fc, fs=6.3):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.3,rounding_size=1.0", fc=fc, ec="0.25", lw=0.7))
        ax.text(x + w / 2, y + h / 2, txt, ha="center", va="center", fontsize=fs, linespacing=1.25)
    def arr(x0, y0, x1, y1):
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=7, lw=0.8, color="0.2"))
    box(0.3, 9.5, 8.5, 7, "$P_n$", "#f7f7f7", 8)
    box(11.5, 8, 14, 10, "material units\n$v=P/P_s(\\alpha,\\beta,\\gamma)$\n$\\theta=\\Delta t/\\lambda(\\alpha)$", "#fde0dd")
    box(28.5, 16, 33.5, 10, "exact linear propagator $e^{\\theta L}v$\nall Fourier modes; per $|k|$ a $Z{\\times}Z$ generator\nmeasured from training trajectories,\ninterpolated in $\\log|\\alpha|$", "#fbb4ae", 5.6)
    box(28.5, 0, 33.5, 10, "neural remainder $N(v)$\nUNO on $[\\,v/\\|v\\|,\\ \\log\\|v\\|\\,]$ + material code\nread-out initialised at zero\n(untrained model = linear propagator)", "#b3cde3", 5.6)
    box(65, 1.5, 13, 7, "$\\theta\\,\\varphi_1(\\theta L)$\nfilter", "#ccebc5")
    ax.text(82, 13, "$+$", fontsize=11, ha="center", va="center")
    box(86, 9.5, 13.5, 7, "$v_{n+1}\\to P_{n+1}$", "#f7f7f7", 7)
    arr(9.2, 13, 11.1, 13); arr(25.9, 15, 28.1, 20); arr(25.9, 11, 28.1, 6); arr(62.4, 5, 64.6, 5)
    arr(62.4, 21, 80.5, 14.2); arr(78.4, 5, 80.5, 11.8); arr(83.5, 13, 85.6, 13)
    ax.text(50, -2.6, r"$v_{n+1}=e^{\theta L}v_n+\theta\,\varphi_1(\theta L)\,N(v_n)$,  $\varphi_1(z)=(e^z-1)/z$:  one exponential-Euler step of the Duhamel formula",
            ha="center", fontsize=7)
    fig.savefig(os.path.join(FIG, "schematic.pdf")); plt.close(fig)


def fig_hyst():
    """closed P-V loops: the relaxed state at the end of every voltage step (quasi-static sweep), truth vs models;
    the virgin branch (0 -> +Vmax from the unpoled field) is drawn lightly."""
    p = os.path.join(BENCH, "results_phys", "hyst_loops_closed.npz")
    closed = os.path.exists(p)
    if not closed:
        p = os.path.join(BENCH, "results_phys", "hyst_loops.npz")
        if not os.path.exists(p):
            return warn("no hysteresis loops")
    z = np.load(p, allow_pickle=True)
    n = len([k for k in z.files if k.startswith("V_")])
    ms = [m for m in ("cno", "duhamel_uno", "fno") if "%s_0" % m in z.files]
    lab = {"ours_fno_pre": "C-FNO, D1-pretrained (ours)", "cno": "CNO + recipe", "duhamel_uno": "Duhamel-NO (ours)", "fno": "FNO + recipe"}
    col = {"ours_fno_pre": "#b2182b", "cno": "#8073ac", "duhamel_uno": "#e66101", "fno": "#404040"}
    mk = {"ours_fno_pre": "o", "cno": "s", "duhamel_uno": "D", "fno": "x"}
    nc = min(n, 3); nr = int(np.ceil(min(n, 6) / nc))
    fig, axs = plt.subplots(nr, nc, figsize=(6.9, 2.0 * nr), squeeze=False)
    for i in range(min(n, nr * nc)):
        ax = axs[i // nc, i % nc]; V = z["V_%d" % i]
        ends = np.r_[np.nonzero(np.diff(V))[0], len(V) - 1]                  # last frame of every voltage step = relaxed state
        Ve = V[ends]
        vmax = np.max(V); first_top = int(np.argmax(Ve >= vmax - 1e-9))         # end of the virgin branch
        def draw(P, **kw):
            Pe = P[ends]
            ax.plot(Ve[:first_top + 1], Pe[:first_top + 1], lw=0.6, alpha=0.35, color=kw["color"], ls=kw.get("ls", "-"))
            ax.plot(Ve[first_top:], Pe[first_top:], lw=kw.get("lw", 1.0), color=kw["color"], ls=kw.get("ls", "-"),
                    marker=kw.get("marker"), ms=kw.get("ms", 0), mfc="none", mew=0.6, label=kw.get("label"))
        draw(z["true_%d" % i], color="k", lw=1.8, label="FerroX (truth)")
        for m in ms:
            draw(z["%s_%d" % (m, i)], color=col[m], lw=0.9, ls="-" if m in ("duhamel_uno",) else "--",
                 marker=mk[m], ms=2.2, label=lab[m])
        ax.set_title(str(z["name_%d" % i]).replace("h_e100_", "").split("_s")[0].replace("_V", ",  $V_{\\max}$ = ") + " V", fontsize=6.5)
        ax.axhline(0, color="0.8", lw=0.5); ax.axvline(0, color="0.8", lw=0.5); ax.grid(alpha=0.2, lw=0.4)
        if i % nc == 0:
            ax.set_ylabel(r"$\langle P_z\rangle / P_{\mathrm{norm}}$")
        if i // nc == nr - 1:
            ax.set_xlabel("top-electrode voltage (V)")
    h, l = axs[0, 0].get_legend_handles_labels()
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.legend(h, l, loc="lower center", ncol=5, frameon=False, fontsize=6.8)
    fig.savefig(os.path.join(FIG, "hyst_loops.pdf")); plt.close(fig)
    put("hystFigClosed", "1" if closed else "0")


def misc_numbers():
    b = jload(os.path.join(BENCH, "results_canon", "alpha_bank_tdgl_val.json"))
    if b:
        put("bankCorrTone", "%.3f" % b["corr_t1_8"][0]); put("bankRelTone", "%.3f" % b["rel_t1_8"][0]); put("bankCorrTeight", "%.3f" % b["corr_t1_8"][-1])
    # ablations on TDGL
    for m, nm in (("duhamel_uno_poly", "Poly"), ("duhamel_uno_k0", "Kzero"), ("uno", "Uno"), ("duhamel", "Duh"), ("duhamel_uno_rou2", "Rtwo")):
        c, r = metric("tdgl", m, "D1_heldout", "corr_final"), metric("tdgl", m, "D1_heldout", "rel_final")
        put("abl%sCorr" % nm, fmt(c[0])); put("abl%sRel" % nm, fmt(r[0])); put("abl%sN" % nm, "%d" % r[2])
        put("abl%sRelTone" % nm, fmt(metric("tdgl", m, "D1_heldout", "rel_t1")[0]))
    # number of models / runs
    put("nTdglModels", "%d" % len([m for m in TDGL_MODELS if runs("tdgl", m)]))


def norecipe_numbers():
    """best TDGL validation corr@400 of any configuration WITHOUT our recipe (tuning + finals), and of the generative models"""
    best, bm = -1, None
    for f in glob.glob(os.path.join(BENCH, "results_tune", "tdgl_*.json")) + glob.glob(os.path.join(BENCH, "results_final", "tdgl_*.json")):
        d = jload(f)
        if not d or not isinstance(d.get("eval"), dict) or "recipe" in d.get("cfg", "") or d["model"].startswith(("duhamel", "ours")):
            continue
        v = (d["eval"].get("val") or {}).get("corr_final")
        if v is not None and np.isfinite(v) and v > best:
            best, bm = v, d["model"]
    if bm:
        put("noRecipeBestVal", "%.2f" % best); put("noRecipeBestModel", LAB(bm))


def table_recipe():
    """recipe vs the best STANDARD configuration of each backbone at the same (4k-step) tuning budget, TDGL validation"""
    B = {}
    for f in glob.glob(os.path.join(BENCH, "results_tune", "tdgl_*.json")):
        d = jload(f)
        if not d or not isinstance(d.get("eval"), dict):
            continue
        v = d["eval"].get("val") or {}
        if v.get("corr_final") is None:
            continue
        k = "recipe" if "recipe" in d["cfg"] else "std"
        sc = v["corr_final"] - 0.1 * min(v["rel_final"], 10)            # the benchmark's validation score
        if sc > B.setdefault(d["model"], {}).get(k, (-9,))[0]:
            B[d["model"]][k] = (sc, v["corr_final"], v["rel_final"], d["cfg"])
    ms = [m for m in ["ffno", "unet", "uno", "cnext", "cno", "tfno", "fno", "transolver"] if "recipe" in B.get(m, {}) and "std" in B[m]]
    if not ms:
        return warn("no tuning results")
    def cfgn(c):
        c = c.split("_", 1)[1]
        return {"rel_inst": "rel.\\ loss + per-sample norm.", "rel": "rel.\\ loss", "mse": "MSE", "mse_inst": "MSE + per-sample norm."}.get(c, c)
    L = [r"\begin{tabular}{lcccc}", r"\toprule", r" & \multicolumn{2}{c}{best standard configuration} & \multicolumn{2}{c}{with our recipe} \\",
         r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}", r"backbone & corr@400 & rel.\,L2@400 & corr@400 & rel.\,L2@400 \\", r"\midrule"]
    for m in ms:
        s_, r_ = B[m]["std"], B[m]["recipe"]
        L.append("%s & %.3f & %.2f & %.3f & %.2f \\\\" % (LAB(m) + ("$^{*}$" if cfgn(s_[3]) != "rel.\\ loss + per-sample norm." else ""), s_[1], s_[2], r_[1], r_[2]))
    L += [r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(TAB, "recipe.tex"), "w").write("\n".join(L))
    std = [B[m]["std"] for m in ms if m != "transolver"]; rec = [B[m]["recipe"] for m in ms if m != "transolver"]
    put("stdCorrMax", "%.2f" % max(x[1] for x in std)); put("stdRelMin", "%.1f" % min(x[2] for x in std)); put("stdRelMax", "%.1f" % max(x[2] for x in std))
    put("recCorrMin", "%.2f" % min(x[1] for x in rec)); put("recCorrMax", "%.2f" % max(x[1] for x in rec))
    put("recRelMin", "%.2f" % min(x[2] for x in rec)); put("recRelMax", "%.2f" % max(x[2] for x in rec))


def table_models():
    """appendix: per task, the validation-selected configuration, parameter count, seeds and training time of every model"""
    def cfgn(d):
        c = d.get("cfg", "")
        lr = re.match(r"lr([0-9.e-]+)", c).group(1) if c.startswith("lr") else "--"
        rest = c.split("_", 1)[1] if "_" in c else ""
        rest = {"recipe": "our recipe", "rel": "rel. L2", "mse": "MSE", "rel_inst": "rel. L2 + per-sample norm.", "mse_inst": "MSE + per-sample norm.",
                "scale-global": "global noise scale", "scale-inst": "per-sample noise scale"}.get(rest, rest)
        return lr, rest
    L = [r"\begin{tabular}{llrrrll}", r"\toprule", r"task & model & params (M) & seeds & train (min) & lr & configuration \\", r"\midrule"]
    for t, tn in (("tdgl", "FerroX TDGL"), ("hyst", "hysteresis"), ("ac", "Allen--Cahn"), ("ch", "Cahn--Hilliard")):
        first = True
        for m in ["duhamel_uno", "duhamel_ou2", "ffno", "unet", "uno", "cnext", "cno", "tfno", "fno", "transolver", "poseidon", "acdm", "flowmatch", "refiner2"]:
            ds = runs(t, m)
            if not ds:
                continue
            p = ds[0].get("params"); tm = mean_std([d.get("train_minutes") for d in ds])[0]
            lr, rest = cfgn(ds[0])
            var = (" [" + ds[0]["variant"] + "]") if ds[0].get("variant") else ""
            L.append("%s & %s & %s & %d & %s & %s & %s%s \\\\" % (tn if first else "", LAB(m), "%.2f" % (p / 1e6) if p else "--", len(ds),
                                                         "%.0f" % tm if np.isfinite(tm) else "--", lr, rest, var))
            first = False
        L.append(r"\midrule")
    L[-1] = r"\bottomrule"; L.append(r"\end{tabular}")
    open(os.path.join(TAB, "models.tex"), "w").write("\n".join(L))


def table_ablation():
    L = [r"\begin{tabular}{lccc}", r"\toprule",
         r"FerroX TDGL (same 9.5M-parameter UNO backbone) & corr@400 (D1) & rel.\,L2@400 (D1) & rel.\,L2, frame 1 \\", r"\midrule"]
    for m, lab in (("uno", "UNO + recipe (no Duhamel step)"), ("duhamel_uno", r"+ Duhamel step, measured operator (\DU{})"),
                   ("duhamel_uno_rou2", r"\DU{} with an $O(u^2)$ remainder instead of UNO"),
                   ("duhamel_uno_poly", "+ Duhamel step, learned conditioned symbol"), ("duhamel_uno_k0", r"\DU{}, uniform mode kept instead of damped"),
                   ("duhamel", r"Duhamel, learned generator + small $O(u^2)$ network")):
        if not runs("tdgl", m):
            continue
        c, r, t1 = metric("tdgl", m, "D1_heldout", "corr_final"), metric("tdgl", m, "D1_heldout", "rel_final"), metric("tdgl", m, "D1_heldout", "rel_t1")
        L.append("%s & %s & %s & %s \\\\" % (lab, cell(*c), cell(*r), cell(*t1)))
    L += [r"\midrule", r"Allen--Cahn / Cahn--Hilliard & rel.\,L2, in range & rel.\,L2, new physics & rel.\,L2, frame 1 (in range) \\", r"\midrule"]
    for t in ("ac", "ch"):
        o2 = "duhamel_uno_kmu_rou2" if runs(t, "duhamel_uno_kmu_rou2") else "duhamel_ou2"
        for m, lab in (("duhamel_uno_kmu", r"UNO remainder"), (o2, r"$O(u^2)$ remainder"),
                       ("duhamel_uno", "UNO remainder, symbol in grid wavenumber units"), ("duhamel_uno_kmu_trunc", "UNO remainder, linear part truncated to 16 modes")):
            if m == ours(t):
                lab += r" (selected)"
            if not runs(t, m):
                continue
            a_, b_, c_ = metric(t, m, "id", "rel_final"), metric(t, m, "ood_inv", "rel_final"), metric(t, m, "id", "rel_t1")
            L.append("%s: %s & %s & %s & %s \\\\" % (t.upper(), lab, cell(*a_), cell(*b_), cell(*c_)))
    L += [r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(TAB, "ablation.tex"), "w").write("\n".join(L))
    for m, nm in (("duhamel_uno_kmu", "Kmu"), ("duhamel_uno", "Grid"), ("duhamel_uno_kmu_trunc", "Trunc"), ("duhamel_ou2", "Ou"), ("duhamel_ou2_free", "Free"),
                  ("duhamel_uno_kmu_rou2", "Rou")):
        for t in ("ac", "ch"):
            put("abl%s%sId" % (nm, t.upper()), fmt(metric(t, m, "id", "rel_final")[0])); put("abl%s%sInv" % (nm, t.upper()), fmt(metric(t, m, "ood_inv", "rel_final")[0]))
            put("abl%s%sTone" % (nm, t.upper()), fmt(metric(t, m, "id", "rel_t1")[0], 3))


def _rank_fmt(vals, higher_better=False, nd=3):
    """bold the best, underline the second best"""
    fin = [(v, i) for i, v in enumerate(vals) if np.isfinite(v)]
    order = sorted(fin, reverse=higher_better)
    best = order[0][1] if order else -1; second = order[1][1] if len(order) > 1 else -1
    out = []
    for i, v in enumerate(vals):
        s = fmt(v, nd)
        out.append(r"\textbf{%s}" % s if i == best else (r"\underline{%s}" % s if i == second else s))
    return out


def table_compact():
    # FerroX TDGL
    ms = [m for m in ["duhamel_uno", "ffno", "unet", "uno", "cnext", "cno", "tfno", "fno"] if runs("tdgl", m)]
    gen = [m for m in ["poseidon", "acdm", "flowmatch", "refiner2", "transolver"] if runs("tdgl", m)]
    gbest = max(gen, key=lambda m: metric("tdgl", m, "D1_heldout", "corr_final")[0]) if gen else None
    rows = ms + ([gbest] if gbest else [])
    cols = [("D1_heldout", "corr_final", True), ("D1_heldout", "rel_final", False), ("D2_ood", "rel_final", False), ("D1_heldout", "rel_t1", False)]
    C = [[metric("tdgl", m, sp, k)[0] for m in rows] for sp, k, _ in cols]
    F = [_rank_fmt(c, hb) for c, (_, _, hb) in zip(C, cols)]
    L = [r"\begin{tabular}{lcccc}", r"\toprule",
         r"model & D1 corr.\ $\uparrow$ & D1 error $\downarrow$ & D2 error $\downarrow$ & frame-1 error $\downarrow$ \\", r"\midrule"]
    for i, m in enumerate(rows):
        lab = LAB(m) if m == "duhamel_uno" else (LAB(m) + " + recipe" if m in ms else "best generative / pretrained (%s)" % LAB(m))
        L.append(lab + " & " + " & ".join(F[c][i] for c in range(len(cols))) + r" \\")
        if m == "duhamel_uno" or (gbest and i == len(ms) - 1):
            L.append(r"\midrule")
    L += [r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(TAB, "tdgl_compact.tex"), "w").write("\n".join(L))
    # Allen-Cahn / Cahn-Hilliard: our model + deterministic baselines; in range and new physics only
    rows = ["OURS"] + [m for m in ["ffno", "cnext", "unet", "uno", "cno", "tfno", "fno"] if runs("ac", m)]
    cols = [("ac", "id"), ("ac", "ood_inv"), ("ch", "id"), ("ch", "ood_inv")]
    C = [[metric(t, m, sp, "rel_final")[0] for m in rows] for t, sp in cols]
    F = [_rank_fmt(c) for c in C]
    L = [r"\begin{tabular}{lcccc}", r"\toprule", r" & \multicolumn{2}{c}{Allen--Cahn} & \multicolumn{2}{c}{Cahn--Hilliard} \\",
         r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}", r"model & in range & new physics & in range & new physics \\", r"\midrule"]
    for i, m in enumerate(rows):
        lab = "Duhamel-NO (ours)" if m == "OURS" else LAB(m) + " + recipe"
        L.append(lab + " & " + " & ".join(F[c][i] for c in range(len(cols))) + r" \\")
        if i == 0:
            L.append(r"\midrule")
    L += [r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(TAB, "acch_compact.tex"), "w").write("\n".join(L))
    for t in ("tdgl", "ac", "ch"):
        put("rem%s" % t.upper(), remainder_of(ours(t) or "x").replace("O(u^2)", "$O(u^2)$"))


def write_numbers():
    L = ["%% written by make_figs.py -- do not edit"]
    for k in sorted(NUM):
        if not re.fullmatch(r"[A-Za-z]+", k):
            warn("macro name %s has non-letters" % k); continue
        L.append("\\newcommand{\\%s}{%s}" % (k, NUM[k]))
    open(os.path.join(HERE, "numbers.tex"), "w").write("\n".join(L) + "\n")


if __name__ == "__main__":
    for f in (table_tdgl, table_hyst, table_acch, fig_error_vs_time, fig_rollouts, fig_physics, fig_physics_d2, fig_physics_panels, fig_spectrum, fig_pareto,
              fig_subsets, fig_acch, fig_res, fig_data_eff, fig_theory, fig_schematic, fig_hyst, misc_numbers, norecipe_numbers, table_recipe, table_models, table_ablation, table_compact):
        try:
            f()
        except Exception as e:
            import traceback; traceback.print_exc(); warn("%s failed: %r" % (f.__name__, e))
    write_numbers()
    print("numbers:", len(NUM))
