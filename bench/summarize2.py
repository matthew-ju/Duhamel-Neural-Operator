#!/usr/bin/env python3
"""bench/summarize2.py -> bench/RESULTS_v2.md : the tuned benchmark (results_final/), one table per task.
Every model = its best configuration on the VALIDATION split (selection/<task>.json), retrained 20k steps;
mean +- std over seeds where >1. FerroX TDGL / hysteresis also broken down per subset (interp / extrap, D2 box,
voltage amplitude). Our earlier ml_units models (recipe, Duhamel) are listed as references (trained on the full
training set incl. the validation trajectories, separate code, same metric definitions)."""
import collections, glob, json, os, re
import numpy as np

H = os.path.dirname(os.path.abspath(__file__)); MU = os.path.join(H, "..", "ml_units")
NICE = dict(fno="FNO", tfno="TFNO", ffno="F-FNO", uno="UNO / E-UNO", unet="U-Net", cnext="CNextU-Net", cno="CNO",
            transolver="Transolver", poseidon="Poseidon-T (pretrained)", refiner2="PDE-Refiner (diffusion)",
            acdm="ACDM (diffusion)", flowmatch="Flow matching", ours_fno="OURS production FNO (all 745 train cfgs)",
            ours_switch5="OURS production + SSM frames 1-5", ours_fno_pre="OURS FNO, D1-pretrained (k=12)",
            ours_duhamel="OURS Duhamel (k=8)", duhamel="OURS Duhamel (multi-material, no hand-off)", duhamel_uno="OURS Duhamel-UNO (alpha-bank / learned symbol + UNO)",
            duhamel_ou2="OURS Duhamel O(u²) (k·mu symbol, bench harness)", duhamel_uno_poly="ABLATION Duhamel-UNO, learned symbol (no bank)",
            duhamel_uno_k0="ABLATION Duhamel-UNO, k=0 persistent")
FAMILY = dict(fno="neural operator", tfno="neural operator", ffno="neural operator", uno="neural operator", cno="neural operator",
              unet="U-Net", cnext="U-Net", transolver="transformer", poseidon="foundation model (transformer)",
              refiner2="diffusion", acdm="diffusion", flowmatch="flow matching", duhamel="ours", duhamel_uno="ours")
TITLE = {"tdgl": "FerroX TDGL, multi-material — train 683 D1 configs; test 256 held-out D1 (new materials + new noise) and 300 D2 (out of range, zero-shot)",
         "hyst": "FerroX hysteresis — train 90 loops; test held-out materials (30), new noise (34), closed loops (10)",
         "ac": "Allen–Cahn 2D", "ch": "Cahn–Hilliard 2D", "nsf": "Navier–Stokes forced 2D", "nsd": "Navier–Stokes decaying 2D",
         "rd": "Reaction–diffusion 1D (PDEBench)", "burgers": "Burgers 1D (PDEBench)"}
SPLITS = {"tdgl": ["D1_heldout", "D2_ood"], "hyst": ["test_material", "test_noise", "closed"]}


def vscore(d):
    v = (d.get("eval") or {}).get("val")
    if not v:
        return None
    return -v["loop_err"] if d["task"] == "hyst" else v["corr_final"] - 0.1 * min(v["rel_final"], 10)


def load(closed=False):
    """all full-length runs per (task, model), grouped by configuration; the reported configuration is the one with the
    best mean VALIDATION score of its full-length runs (test splits never used for the choice). Candidates include the
    pre-fix U-Net / CNextU-Net / CNO runs (no positional channels; CNO with BatchNorm) -- legitimate variants for the
    translation-invariant tasks."""
    G = collections.defaultdict(list)
    for f in glob.glob(os.path.join(H, "results_final", "*.json")):
        d = json.load(open(f))
        if isinstance(d.get("eval"), dict) and not d.get("smoke"):
            if re.search(r"_s\d+_K\d+$", d.get("tag", "")):             # data-efficiency runs (subset of training materials)
                continue
            if d.get("tag", "").endswith("_nounits"):                  # units ablation (BENCH_NOUNITS=1): read by make_units.py,
                continue                                               # never a candidate configuration of the benchmark
            if d.get("tag", "").endswith("_closed") != closed:          # closed-loop hysteresis runs are a separate benchmark
                continue
            d["variant"] = ""; G[(d["task"], d["model"], d.get("cfg", ""), "")].append(d)
    # Pre-fix runs (U-Net / CNextU-Net without coordinate channels, CNO with BatchNorm) are NOT candidates any more
    # (2026-10-09): their checkpoints were overwritten by the corrected runs under the same tags, so a selected pre-fix
    # run could not be re-evaluated, and the physical-quantity results were computed from the corrected checkpoints.
    # Set BENCH_PREFIX_VARIANTS=1 to restore the old behaviour.
    for f in ([] if closed or not os.environ.get("BENCH_PREFIX_VARIANTS") else glob.glob(os.path.join(H, "results_final_v2buggy", "*.json"))):
        d = json.load(open(f))
        if d["model"] in ("unet", "cnext", "cno") and isinstance(d.get("eval"), dict):
            var = "no pos-enc" + (", BatchNorm" if d["model"] == "cno" else "")
            d["variant"] = var; G[(d["task"], d["model"], d.get("cfg", ""), var)].append(d)
    R = {}
    for (t, m, cfg, var), ds in G.items():
        sc = [vscore(d) for d in ds]; sc = [x for x in sc if x is not None]
        key = np.mean(sc) if sc else -9e9
        if (t, m) not in R or key > R[(t, m)][0]:
            R[(t, m)] = (key, ds)
    return {k: v[1] for k, v in R.items()}


def ms(vals, fmt="%.3f"):
    vals = [v for v in vals if v is not None and np.isfinite(v)]
    if not vals:
        return "—"
    return (fmt % np.mean(vals)) + (("±" + fmt % np.std(vals)) if len(vals) > 1 else "")


def main():
    R = load(); out = ["# Tuned comparative benchmark (bench/, v2)\n",
                       "Each model: configurations tuned on the validation split (lr x loss x {global, per-sample norm, our recipe}; generative: "
                       "lr x noise scaling); full-length (20k-step) runs of the best candidates; the reported configuration = best mean "
                       "VALIDATION score among the full-length runs (test never used for any choice); top-3 per task + our recipe x 3 seeds. Cells: **final-frame relative L2 / "
                       "pattern correlation** (TDGL: corr@400 first). Hysteresis: **loop error / field corr**. mean ± std over seeds.\n"]
    for t in ["tdgl", "hyst", "ac", "ch", "nsf", "nsd", "rd", "burgers"]:
        sps = SPLITS.get(t, ["id", "ood_units", "ood_inv"])
        rows = []
        for (tt, m), ds in R.items():
            if tt != t:
                continue
            cells, key = [], None
            for sp in sps:
                if t == "hyst":
                    a = [d["eval"].get(sp, {}).get("loop_err") for d in ds]; b = [d["eval"].get(sp, {}).get("field_corr") for d in ds]
                elif t == "tdgl":
                    a = [d["eval"].get(sp, {}).get("corr_final") for d in ds]; b = [d["eval"].get(sp, {}).get("rel_final") for d in ds]
                else:
                    a = [d["eval"].get(sp, {}).get("rel_final") for d in ds]; b = [d["eval"].get(sp, {}).get("corr_final") for d in ds]
                cells.append("%s / %s" % (ms(a), ms(b)))
                if key is None:
                    v = [x for x in a if x is not None]
                    key = (-np.mean(v) if t == "tdgl" else np.mean(v)) if v else 9e9
            cfg = ds[0].get("cfg", "") + ((" [" + ds[0]["variant"] + "]") if ds[0].get("variant") else "")
            rows.append((key, m, cfg, len(ds), cells, ds))
        if not rows:
            continue
        out.append("\n## %s\n" % TITLE[t])
        out.append("| model | family | config (picked on val) | seeds | " + " | ".join(sps) + " |")
        out.append("|---|---|---|---|" + "---|" * len(sps))
        for key, m, cfg, n, cells, _ in sorted(rows, key=lambda r: r[0]):
            out.append("| %s | %s | %s | %d | %s |" % (NICE.get(m, m), FAMILY.get(m, "ours"), cfg if not m.startswith("ours") else "—", n, " | ".join(cells)))
        if t not in ("tdgl", "hyst"):
            ref = []
            sfx = "_ssmfix" if t in ("ac", "ch") else ""
            rp = sorted(glob.glob(os.path.join(MU, "results_recipe", "%s_recipe_s[0-9]%s.json" % (t, sfx))))
            if rp:
                ref.append(("OURS recipe (FNO, earlier run)", [json.load(open(p)) for p in rp], "corr"))
            pat = {"ac": "ac_duhamel_s[0-9].json", "ch": "ch_duhamel_s[0-9]_sub4.json", "nsd": "nsd_duhamel_s[0-9]_cons_diss_zclip.json"}.get(t)
            if pat:
                dp = sorted(glob.glob(os.path.join(MU, "results_duhamel", pat)))
                if dp:
                    ref.append(("OURS Duhamel O(u²) (earlier run)", [json.load(open(p)) for p in dp], "corr_final"))
            for lab, js, ck in ref:
                cells = ["%s / %s" % (ms([j[sp]["rel_final"] for j in js if sp in j]), ms([j[sp].get(ck) for j in js if sp in j])) for sp in sps]
                out.append("| %s | ours (reference) | — | %d | %s |" % (lab, len(js), " | ".join(cells)))
        # ---------- per-subset breakdown
        if t in ("tdgl", "hyst"):
            out.append("\n**Per-subset breakdown** (%s)\n" % ("corr@400 / rel. L2@400" if t == "tdgl" else "loop error / field corr"))
            subs = collections.OrderedDict()
            for _, m, cfg, n, cells, ds in sorted(rows, key=lambda r: r[0]):
                for sp in sps:
                    for g, v in (ds[0]["eval"].get(sp, {}).get("subsets") or {}).items():
                        subs.setdefault((sp, g), None)
            if not subs:
                out.append("(subset breakdown not yet available — re-evaluation pending)")
                continue
            cols = list(subs)
            out.append("| model | " + " | ".join("%s: %s" % (sp.replace("_heldout", "").replace("_ood", "").replace("test_", ""), g) for sp, g in cols) + " |")
            out.append("|---|" + "---|" * len(cols))
            for _, m, cfg, n, cells, ds in sorted(rows, key=lambda r: r[0]):
                cs = []
                for sp, g in cols:
                    if t == "tdgl":
                        a = [((d["eval"].get(sp, {}).get("subsets") or {}).get(g) or {}).get("corr_final") for d in ds]
                        b = [((d["eval"].get(sp, {}).get("subsets") or {}).get(g) or {}).get("rel_final") for d in ds]
                    else:
                        a = [((d["eval"].get(sp, {}).get("subsets") or {}).get(g) or {}).get("loop_err") for d in ds]
                        b = [((d["eval"].get(sp, {}).get("subsets") or {}).get(g) or {}).get("field_corr") for d in ds]
                    cs.append("%s / %s" % (ms(a), ms(b)))
                out.append("| %s | %s |" % (NICE.get(m, m), " | ".join(cs)))
            ns = {g: (ds0 := [r for r in rows][0][5][0]["eval"].get(sp, {}).get("subsets", {}).get(g, {})).get("n") for sp, g in cols}
            out.append("\nsubset sizes: " + ", ".join("%s=%s" % (g, n) for g, n in ns.items()))
    # ---------- canonical reproduction (harness sanity check)
    cj = sorted(glob.glob(os.path.join(H, "results_canon", "ns1e-3_*.json")), key=lambda f: (json.load(open(f))["epochs"], json.load(open(f))["test_rel_l2_full"]))
    if cj:
        out.append("\n## Sanity check: canonical FNO benchmark (2D Navier-Stokes, nu=1e-3, 10 -> 40 frames, Li et al. ICLR 2021 Table 1)\n")
        out.append("| model | params | epochs | test rel. L2 (full trajectory) | published |")
        out.append("|---|---|---|---|---|")
        for f in cj:
            d = json.load(open(f))
            out.append("| %s | %.2fM | %d | %.4f | %s |" % (d["model"], d["params"] / 1e6, d["epochs"], d["test_rel_l2_full"],
                                                    "%.4f" % d["published"] if d.get("published") else "—"))
    open(os.path.join(H, "RESULTS_v3.md"), "w").write("\n".join(out) + "\n")
    print("\n".join(out))


if __name__ == "__main__":
    main()
