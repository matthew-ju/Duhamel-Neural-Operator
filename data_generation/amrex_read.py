#!/usr/bin/env python3
"""Minimal AMReX single-level plotfile reader (no yt needed).

Reads FerroX plotfiles: 4 boxes of 32x32x18, 18 components,
8-byte IEEE doubles.  Component order (from Header): Px,Py,Pz,Phi,PoissonRHS,
Ex,Ey,Ez,holes,electrons,...  -> Pz is component index 2, Phi is 3.
"""
import os, re
import numpy as np

# Defaults for the original ST_Ferro stack.  read_field no longer uses NX/NY/NZ
# -- it infers the domain from each plotfile -- but they are kept because other
# callers import them.
NX = NY = 64
NZ = 18
NCOMP = 18


def _parse_cell_h(level_dir):
    with open(os.path.join(level_dir, "Cell_H")) as f:
        txt = f.read()
    boxes = re.findall(r"\(\((\d+),(\d+),(\d+)\) \((\d+),(\d+),(\d+)\) \(\d+,\d+,\d+\)\)", txt)
    boxes = [tuple(map(int, b)) for b in boxes]
    offsets = [int(m) for m in re.findall(r"FabOnDisk: \S+ (\d+)", txt)]
    fname = re.findall(r"FabOnDisk: (\S+) \d+", txt)[0]
    return boxes, offsets, fname


def read_field(plt_dir, comp=2, endian="<"):
    """Return field `comp` as (nx, ny, nz) array [x, y, z].

    The domain size is INFERRED from the plotfile's own box list rather than
    taken from the NX/NY/NZ constants.  Those constants describe one particular
    stack (64x64x18) and silently break on any other: a 48 nm ferroelectric is
    nz = 52, and the old code raised

        ValueError: could not broadcast input array from shape (32,32,44)
                    into shape (32,32,18)

    AFTER the solver had already run, so the compute was spent and thrown away.
    Inferring reproduces (64,64,18) exactly for every existing plotfile, so
    this is backward compatible with the abg/alpha43/betagamma stacks.
    """
    level_dir = os.path.join(plt_dir, "Level_0")
    boxes, offsets, fname = _parse_cell_h(level_dir)
    data_path = os.path.join(level_dir, fname)
    nx = max(b[3] for b in boxes) + 1
    ny = max(b[4] for b in boxes) + 1
    nz = max(b[5] for b in boxes) + 1
    full = np.zeros((nx, ny, nz), dtype=np.float64)
    with open(data_path, "rb") as fh:
        for (i0, j0, k0, i1, j1, k1), off in zip(boxes, offsets):
            nx, ny, nz = i1 - i0 + 1, j1 - j0 + 1, k1 - k0 + 1
            fh.seek(off)
            # skip the ASCII FAB header line
            line = b""
            while not line.endswith(b"\n"):
                line += fh.read(1)
            ncell = nx * ny * nz * NCOMP
            buf = np.frombuffer(fh.read(ncell * 8), dtype=endian + "f8")
            # AMReX FAB: component-major, then column-major (i fastest)
            arr = buf.reshape(NCOMP, nz, ny, nx)      # [c, k, j, i]
            sub = np.transpose(arr[comp], (2, 1, 0))  # -> [i, j, k] = [x, y, z]
            full[i0:i1+1, j0:j1+1, k0:k1+1] = sub
    return full



if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        sys.exit("usage: python amrex_read.py PLOTFILE_DIR   (prints the shape and range of Pz)")
    pz = read_field(sys.argv[1], comp=2)
    print("Pz", pz.shape, "range [%.4f, %.4f]" % (pz.min(), pz.max()))
