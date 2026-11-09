#!/usr/bin/env python3
"""extract_sweep.py RUN_DIR OUT.npz [--every N] -- stack a voltage-sweep run: Pz frames (64,64,18,T),
the step of every frame, and the applied voltage in force when that frame was computed
(increments are applied AFTER the plotfile write at their step, so V(frame s) = last V set at a step < s)."""
import glob, os, re, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # data_generation/amrex_read.py
import amrex_read as ar
run, out = sys.argv[1], sys.argv[2]
every = int(sys.argv[sys.argv.index("--every") + 1]) if "--every" in sys.argv else 1
plts = sorted([p for p in glob.glob(os.path.join(run, "plt*")) if os.path.isdir(p)],
              key=lambda p: int(re.findall(r"(\d+)$", p)[0]))
steps = np.array([int(re.findall(r"(\d+)$", p)[0]) for p in plts])
keep = [i for i in range(len(plts)) if steps[i] % every == 0 or i == len(plts) - 1]
ev = [(int(a), float(b)) for a, b in re.findall(r"step = (\d+), Phi_Bc_hi = ([-\d.e+]+)", open(os.path.join(run, "run.log")).read())]
es, eV = np.array([e[0] for e in ev] or [10**9]), np.array([e[1] for e in ev] or [0.0])
V0 = 0.0                                                # starting voltage from the inputs file
for line in open(os.path.join(run, "inputs")):
    if line.split("=")[0].strip() == "Phi_Bc_hi":
        V0 = float(line.split("=")[1].split()[0])
def V_at(s):
    j = np.searchsorted(es, s, side="left") - 1          # last increment strictly before s
    return V0 if j < 0 else eV[j]
P = np.stack([ar.read_field(plts[i]) for i in keep], axis=-1).astype(np.float32)
np.savez(out, P=P, step=steps[keep], V=np.array([V_at(steps[i]) for i in keep]), inc_step=es, inc_V=eV)
fe = P[:, :, 8:18]
print("%d frames, steps %d..%d, V %.3f..%.3f" % (len(keep), steps[keep][0], steps[keep][-1], min(V_at(s) for s in steps), max(V_at(s) for s in steps)))
