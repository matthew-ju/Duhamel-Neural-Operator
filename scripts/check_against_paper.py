#!/usr/bin/env python3
"""scripts/check_against_paper.py -- compare what you regenerated with what the paper prints.

Checks every number of paper/numbers_duhamel.tex that the paper uses (53 in the main text, 24 more in the appendix) and
the ten tables the paper typesets, against the published copies in paper/reference/. Prints one line per mismatch and a
summary. Exit code 0 = everything matches."""
import os, re, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
P = os.path.join(ROOT, "paper")
MAIN = ["ablUnoRel", "ablUnoRelTone", "advDoneTone", "bankCorrTone", "capTone", "capTten", "capTthree", "capTtwo",
        "clTnucPhys", "clTnucRed", "deduA", "deffnoA", "duCorrDone", "duCorrDtwo", "duCorrTone", "duRelDone",
        "duRelDtwo", "duRelTone", "ffnoRelTone", "fnoCorrTone", "fnoCorrTthree", "fnoCorrTtwo", "fnoRelTone",
        "noiseAuditMax", "phRows", "phSigBetter", "phSigTie", "relGainDone", "relGainDtwo", "resduA", "resduC",
        "resffnoA", "resffnoC", "resfnoC", "spDuBone", "spDuBsixteen", "spDuParams", "spDuSpeedBone",
        "spDuSpeedBsixteen", "spFerroxHi", "spFerroxLo", "spFerroxMed", "stdCorrMax", "stdRelMax", "stdRelMin",
        "subDBunitsCorr", "subDCCorr", "subExtrapN", "subExtrapRel", "subInterpN", "subInterpRel",
        "tnucExactBase", "tnucExactDu"]
APPENDIX = ["aRateCrit", "aSatCrit", "aSatE", "aSatSlope", "advHystTone", "baseACId", "baseACInv", "clN", "clSatPhys",
            "clSatRed", "deduB", "hystCno", "hystDuno", "hystGapDu", "nuFnoOffBu", "nuFnoOnBu", "nuUnoOffBu",
            "nuUnoOnBu", "ouACId", "ouACInv", "ratioACInv", "satLawDtwoMean", "satLawMax", "satLawMean"]
TABLES = [("tdgl_compact.tex", "Table 1 (main results)"), ("physics_fig.tex", "Figure 5c (physical quantities)"),
          ("tmlr_ablation.tex", "ablation table"), ("app_tdgl_main.tex", "appendix: TDGL, all metrics"),
          ("app_tdgl_subsets.tex", "appendix: TDGL subsets"), ("app_models.tex", "appendix: models"),
          ("app_recipe.tex", "appendix: training setup"), ("app_units_new.tex", "appendix: change of units"),
          ("app_hyst.tex", "appendix: hysteresis"), ("ac_main.tex", "appendix: Allen-Cahn")]


def macros(path):
    return dict(re.findall(r"\\newcommand\{\\(\w+)\}\{([^}]*)\}", open(path).read()))


def lines(path):
    return open(path).read().rstrip("\n").split("\n") if os.path.exists(path) else ["<missing>"]


def main():
    bad = 0
    new, ref = macros(os.path.join(P, "numbers_duhamel.tex")), macros(os.path.join(P, "reference", "numbers_duhamel.tex"))
    for lab, names in (("main text", MAIN), ("appendix", APPENDIX)):
        n = 0
        for m in names:
            if new.get(m) != ref.get(m):
                print("MISMATCH  \\%-18s regenerated %-10s paper %s" % (m, new.get(m), ref.get(m))); n += 1
        print("numbers in the %s: %d / %d match" % (lab, len(names) - n, len(names)))
        bad += n
    for name, lab in TABLES:
        a, b = lines(os.path.join(P, "tables", name)), lines(os.path.join(P, "reference", name))
        diff = [(x, y) for x, y in zip(a, b) if x != y] + ([("<length>", "<length>")] if len(a) != len(b) else [])
        for x, y in diff:
            print("MISMATCH  %s\n   regenerated: %s\n   paper:       %s" % (lab, x.strip(), y.strip()))
        print("%-34s %s" % (lab + ":", "matches" if not diff else "%d rows differ" % len(diff)))
        bad += len(diff)
    print("\nALL MATCH" if not bad else "\n%d mismatches (retrained models differ in the third decimal; see docs/REPRODUCE.md)" % bad)
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
