#!/usr/bin/env python3
"""build_train_cache.py -- $SCRATCH/bench_cache/tdgl_train_fp16.npy, the (745, 401, 10, 64, 64) float16 memmap of the D1
training trajectories (ferroelectric layers, in units of P_NORM) that bench/tasks.py TDGLTask reads. Same trajectory
order as TDGLTask (geo_data.index_d1(("useed",)), split == "train"), same frames as geo_data.frames(t, 0, 401).
  python release/build_train_cache.py            # build
  python release/build_train_cache.py --check 5  # compare 5 trajectories with an existing cache
"""
import argparse, os, sys
import numpy as np
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "geo"))
import geo_data as gd  # noqa: E402

ap = argparse.ArgumentParser(); ap.add_argument("--check", type=int, default=0); a = ap.parse_args()
trs = [t for t in gd.index_d1(("useed",)) if t.split == "train"]
path = os.path.join(os.environ["SCRATCH"], "bench_cache", "tdgl_train_fp16.npy")
if a.check:
    U = np.load(path, mmap_mode="r")
    assert U.shape[0] == len(trs), (U.shape, len(trs))
    for i in np.linspace(0, len(trs) - 1, a.check).astype(int):
        x = gd.frames(trs[i], 0, 401)[:, 0].astype(np.float16)
        print(i, trs[i].tag, "max |diff| =", float(np.abs(x.astype(np.float32) - U[i].astype(np.float32)).max()))
    sys.exit(0)
os.makedirs(os.path.dirname(path), exist_ok=True)
U = np.lib.format.open_memmap(path + ".tmp", mode="w+", dtype=np.float16, shape=(len(trs), 401, 10, 64, 64))
for i, t in enumerate(trs):
    U[i] = gd.frames(t, 0, 401)[:, 0].astype(np.float16)
    if i % 50 == 0:
        print(i, "/", len(trs), flush=True)
U.flush(); del U; os.replace(path + ".tmp", path)
print("wrote", path)
