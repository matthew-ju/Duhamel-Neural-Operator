#!/usr/bin/env python3
"""writes queue/todo/T<task#>_<model>_<cfg>.cmd for the validation-tuning stage (4000 steps, scored on val only)."""
import os
H = os.path.dirname(os.path.abspath(__file__)); Q = os.path.join(H, "queue", "todo")
PRE = "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True "
TASKS = ["tdgl", "hyst", "ac", "ch", "nsf", "nsd", "rd", "burgers"]
DET = ["fno", "tfno", "ffno", "uno", "cno", "unet", "cnext", "transolver"]
GEN = ["refiner2", "acdm", "flowmatch"]
n = 0
for ti, t in enumerate(TASKS):
    domain = t in ("tdgl", "hyst", "ac", "ch")
    for m in DET:
        cfgs = [("--lr %g --loss %s" % (lr, l)) for lr in (1e-3, 3e-4) for l in ("mse", "rel")]
        cfgs += ["--lr 1e-3 --wrap"] + (["--lr 3e-4 --wrap"] if domain else [])
        for c in cfgs:
            name = ("T%d_%s_%s" % (ti, m, c.replace("--", "").replace(" ", "_"))).replace(".", "p")
            open(os.path.join(Q, name + ".cmd"), "w").write(PRE + "python -u bench/run.py --mode tune --task %s --model %s %s\n" % (t, m, c)); n += 1
    for m in GEN:
        for lr in (1e-3, 3e-4):
            for sc in ("global", "inst"):
                name = ("T%d_%s_lr%g_%s" % (ti, m, lr, sc)).replace(".", "p")
                open(os.path.join(Q, name + ".cmd"), "w").write(PRE + "python -u bench/run.py --mode tune --task %s --model %s --lr %g --scale %s\n" % (t, m, lr, sc)); n += 1
    if t not in ("rd", "burgers"):
        for lr in (1e-4, 3e-5):
            for l in ("mse", "rel"):
                name = ("T%d_poseidon_lr%g_%s" % (ti, lr, l)).replace(".", "p")
                open(os.path.join(Q, name + ".cmd"), "w").write(
                    "source $SCRATCH/venv_poseidon/bin/activate && "
                    "python -u bench/run.py --mode tune --task %s --model poseidon --lr %g --loss %s --steps 2000\n" % (t, lr, l)); n += 1
print(n, "tune jobs")
