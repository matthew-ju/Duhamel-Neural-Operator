#!/usr/bin/env python3
"""Generic FerroX parameter-sweep tooling: patch AMReX inputs, extract runs.

Built for the material-parameter sweeps (alpha first, then beta/gamma/Phi_bc;
FE thickness needs pipeline-shape changes too -- see the fe10 caveat in
incubation_propagator.py). The FerroX runs ARE the ground truth; the hybrid
emulator (incubation propagator + fm_fno3d_v4_zc) is evaluated against them
afterwards.

Subcommands
-----------
gen-inputs   Patch arbitrary `key = value` lines in an AMReX inputs template.
             Every --set key must already exist in the template (typo guard).
             Multi-token values work quoted: --set "FE_lo=-16.e-9 -16.e-9 4.0e-9"
extract-run  Stack a run dir's plt* files -> (64,64,18,T) float32 .npy (Pz).
assemble     Stack per-seed extracted .npy files for one config ->
             (N,64,64,10,T) fe10 array (z 8:18 = the 10 FE layers, matching
             batch3d_fe10_b000X.npy), plus a sidecar .json manifest.

Pure NumPy/CPU; safe on a login node (extract is file I/O bound).
"""
import argparse
import glob
import json
import os
import re
import shutil
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "data", "ST_Ferro"))


def gen_inputs(args):
    with open(args.template) as f:
        lines = f.readlines()

    overrides = {}
    for kv in args.set:
        if "=" not in kv:
            sys.exit(f"gen-inputs: bad --set '{kv}' (need key=value)")
        k, v = kv.split("=", 1)
        overrides[k.strip()] = v.strip()

    seen = set()
    out_lines = []
    for line in lines:
        m = re.match(r"^(\s*)([A-Za-z_][\w.]*)\s*=", line)
        if m and m.group(2) in overrides:
            key = m.group(2)
            out_lines.append(f"{m.group(1)}{key} = {overrides[key]}\n")
            seen.add(key)
        else:
            out_lines.append(line)

    missing = set(overrides) - seen
    if missing:
        sys.exit(f"gen-inputs: keys not found in template: {sorted(missing)} "
                 "(refusing to add new keys -- fix the name or the template)")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        f.writelines(out_lines)
    print(f"gen-inputs: wrote {args.out} ({len(seen)} keys patched)")


def extract_run(args):
    import numpy as np
    import amrex_read as ar

    plts = sorted(glob.glob(os.path.join(args.run_dir, "plt*")))
    plts = [p for p in plts if os.path.isdir(p)]
    if not plts:
        sys.exit(f"extract-run: no plt* dirs in {args.run_dir}")
    short = args.expect_frames and len(plts) != args.expect_frames
    if short and not args.pad_early_stop:
        sys.exit(f"extract-run: {len(plts)} plotfiles, expected "
                 f"{args.expect_frames} -- run probably died early, not extracting")

    frames = [ar.read_field(p, comp=args.comp) for p in plts]
    if short:
        # FerroX has its own internal steady-state detector: it stops the
        # integration on its own once the Poisson potential stops changing
        # between output intervals, well before nsteps if the material
        # settles fast. That is not a crash -- run.log shows the convergence
        # norm decaying smoothly over thousands of steps to ~1e-7 -- and a
        # settled field STAYS settled, so padding with the last frame is
        # physically correct, not a hack. Measured on the weak-alpha (ext)
        # design: 829/983 solves hit this early-stop (84%), all cleanly
        # converged (AMReX finalized, exit 0), none of them numerical
        # blow-ups. Rejecting them wasted the entire solve every time, since
        # a deterministic run reproduces the same early stop on every retry.
        # Only ext_grid's own runner passes --pad-early-stop; abg/alpha43/
        # betagamma keep the strict behaviour they were validated under.
        need = args.expect_frames - len(frames)
        print(f"extract-run: {len(plts)} plotfiles, expected {args.expect_frames} "
              f"-- steady state reached early, padding {need} frame(s) with the "
              f"settled final frame")
        frames = frames + [frames[-1]] * need
    traj = np.stack(frames, axis=-1).astype(np.float32)  # (64,64,18,T)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.save(args.out, traj)
    print(f"extract-run: {args.run_dir} -> {args.out}  shape={traj.shape}")

    if args.delete_plotfiles:
        for p in plts:
            shutil.rmtree(p)
        print(f"extract-run: deleted {len(plts)} plotfiles")


def assemble(args):
    import numpy as np

    files = sorted(glob.glob(os.path.join(args.dir, "seed_*.npy")))
    if not files:
        sys.exit(f"assemble: no seed_*.npy in {args.dir}")

    # Stream seed-by-seed into a disk-backed array rather than np.stack-ing the
    # whole set in RAM. np.stack materialises every FULL-z trajectory at once:
    # fine at T=801 (7 GB for 30 seeds) but 19.5 GB at T=2201, which the login
    # node OOM-kills. Peak memory here is one seed's z-slice instead.
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    probe = np.load(files[0], mmap_mode="r")
    shape = (len(files), probe.shape[0], probe.shape[1],
             args.z_hi - args.z_lo, probe.shape[3])
    fe = np.lib.format.open_memmap(args.out, mode="w+",
                                   dtype=np.float32, shape=shape)
    for i, f in enumerate(files):
        fe[i] = np.load(f, mmap_mode="r")[:, :, args.z_lo:args.z_hi, :]
    fe.flush()

    meta = {}
    for kv in args.meta:
        k, v = kv.split("=", 1)
        meta[k] = v
    manifest = {
        "out": os.path.abspath(args.out),
        "shape": list(fe.shape),
        "z_slice": [args.z_lo, args.z_hi],
        "seeds": [os.path.basename(f)[len("seed_"):-len(".npy")] for f in files],
        "source_files": [os.path.abspath(f) for f in files],
        **meta,
    }
    mpath = os.path.splitext(args.out)[0] + ".json"
    with open(mpath, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"assemble: {len(files)} seeds -> {args.out}  shape={fe.shape}  "
          f"({fe.nbytes / 1e9:.2f} GB)  manifest={mpath}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("gen-inputs")
    g.add_argument("--template", required=True)
    g.add_argument("--out", required=True)
    g.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    g.set_defaults(fn=gen_inputs)

    e = sub.add_parser("extract-run")
    e.add_argument("--run-dir", required=True)
    e.add_argument("--out", required=True)
    e.add_argument("--comp", type=int, default=2, help="field component (2 = Pz)")
    e.add_argument("--expect-frames", type=int, default=0,
                   help="abort if plotfile count differs (0 = don't check)")
    e.add_argument("--delete-plotfiles", action="store_true")
    e.add_argument("--pad-early-stop", action="store_true",
                   help="a short trajectory is FerroX's own steady-state exit, "
                        "not a crash -- pad with the settled final frame "
                        "instead of refusing to extract")
    e.set_defaults(fn=extract_run)

    a = sub.add_parser("assemble")
    a.add_argument("--dir", required=True, help="dir containing seed_*.npy")
    a.add_argument("--out", required=True)
    a.add_argument("--z-lo", type=int, default=8)
    a.add_argument("--z-hi", type=int, default=18)
    a.add_argument("--meta", action="append", default=[], metavar="KEY=VALUE")
    a.set_defaults(fn=assemble)

    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
