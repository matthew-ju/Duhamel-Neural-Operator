#!/usr/bin/env python3
"""scripts/check_against_paper.py -- compare what you regenerated with what the paper prints.

Checks every number that appears in the main text (51 LaTeX macros, paper/numbers_duhamel.tex) and the two result tables
(paper/tables/tdgl_compact.tex = Table 1, paper/tables/app_physics_full.tex = Table 2) against the published copies in
paper/reference/. Prints one line per mismatch and a summary. Exit code 0 = everything matches."""
import os, re, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
P = os.path.join(ROOT, "paper")
MAIN = ["ablUnoRel", "ablUnoRelTone", "advDoneTone", "bankCorrTone", "baseACId", "baseACInv", "capTone", "clTnucPhys",
        "clTnucRed", "deduA", "deffnoA", "duCorrDone", "duCorrDtwo", "duCorrTone", "duRelDone", "duRelDtwo", "duRelTone",
        "ffnoRelTone", "fnoCorrTone", "fnoRelTone", "noiseAuditMax", "ouACId", "ouACInv", "phRows", "phSigBetter",
        "phSigTie", "ratioACInv", "relGainDone", "relGainDtwo", "resduC", "resffnoC", "resfnoC", "spDuBone",
        "spDuBsixteen", "spDuParams", "spDuSpeedBone", "spDuSpeedBsixteen", "spFerroxHi", "spFerroxLo", "spFerroxMed",
        "stdCorrMax", "stdRelMax", "stdRelMin", "subDBunitsCorr", "subDCCorr", "subExtrapN", "subExtrapRel",
        "subInterpN", "subInterpRel", "tnucExactBase", "tnucExactDu"]


def macros(path):
    return dict(re.findall(r"\\newcommand\{\\(\w+)\}\{([^}]*)\}", open(path).read()))


def main():
    bad = 0
    new, ref = macros(os.path.join(P, "numbers_duhamel.tex")), macros(os.path.join(P, "reference", "numbers_duhamel.tex"))
    for m in MAIN:
        if new.get(m) != ref.get(m):
            print("MISMATCH  \\%-18s regenerated %-10s paper %s" % (m, new.get(m), ref.get(m))); bad += 1
    print("numbers in the main text: %d / %d match" % (len(MAIN) - bad, len(MAIN)))
    for name, lab in (("tdgl_compact.tex", "Table 1"), ("app_physics_full.tex", "Table 2")):
        a = open(os.path.join(P, "tables", name)).read().split("\n")
        b = open(os.path.join(P, "reference", name)).read().split("\n")
        diff = [(x, y) for x, y in zip(a, b) if x != y] + ([("<length>", "<length>")] if len(a) != len(b) else [])
        for x, y in diff:
            print("MISMATCH  %s\n   regenerated: %s\n   paper:       %s" % (lab, x.strip(), y.strip()))
        print("%s: %s" % (lab, "matches" if not diff else "%d rows differ" % len(diff)))
        bad += len(diff)
    print("\nALL MATCH" if not bad else "\n%d mismatches (retrained models differ in the third decimal; see README)" % bad)
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
