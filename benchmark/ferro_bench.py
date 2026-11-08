"""ferro_bench.py -- load the ferroelectric TDGL benchmark and score predictions the way the paper does.

    import ferro_bench as fb
    meta = fb.metadata("d1")                         # list of dicts, one per trajectory
    P = fb.load("d1", meta[0])                       # (401, 10, 64, 64) float32, C/m^2 (memory-mapped)
    fb.pattern_corr(pred[400], P[400]); fb.rel_l2(pred[400], P[400])

Protocol of the paper (Section 3): roll every prediction out from the t=0 field alone (no solver frame), and report the
pattern correlation and relative L2 error at frame 1 and frame 400, averaged over trajectories. Train on the D1 rows with
split == "train" (select hyperparameters on "val"), test on D1 "test_interior" + "test_edge" and on all of D2.
Only numpy is required.
"""
import csv, os

import numpy as np

ROOT = os.environ.get("FERRO_BENCH_ROOT", os.path.dirname(os.path.abspath(__file__)))
_NUM = {"seed", "nsteps", "plot_int", "paper_eval"}


def metadata(dataset, split=None, root=ROOT):
    """Rows of <dataset>/metadata.csv (dataset: 'd1', 'd2' or 'hysteresis'), optionally filtered by split."""
    with open(os.path.join(root, dataset, "metadata.csv")) as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k, v in r.items():
            if k in _NUM:
                r[k] = int(v)
            elif k not in ("traj_id", "file", "split", "region", "material"):
                r[k] = float(v)
    return [r for r in rows if split is None or r["split"] in ([split] if isinstance(split, str) else split)]


def load(dataset, row, root=ROOT):
    """D1/D2: Pz array (T, Z, H, W). Hysteresis: dict with P (T, Z, H, W), step (T,), V (T,) in volts."""
    path = os.path.join(root, dataset, row["file"])
    if path.endswith(".npy"):
        return np.load(path, mmap_mode="r")
    d = np.load(path)
    return {k: d[k] for k in d.files}


def subsample_hysteresis(loop, every=20):
    """The paper uses the frames saved every 20 solver steps."""
    keep = loop["step"] % every == 0
    return {k: (v[keep] if k in ("P", "step", "V") else v) for k, v in loop.items()}


def pattern_corr(a, b):
    """Pearson correlation of two fields over all voxels."""
    a = np.asarray(a, np.float64).ravel(); b = np.asarray(b, np.float64).ravel()
    a = a - a.mean(); b = b - b.mean()
    return float((a @ b) / np.sqrt((a @ a) * (b @ b)))


def rel_l2(pred, true):
    """||pred - true||_2 / ||true||_2 over all voxels."""
    pred = np.asarray(pred, np.float64); true = np.asarray(true, np.float64)
    return float(np.linalg.norm(pred - true) / np.linalg.norm(true))


def score(pred, true, frames=(1, 400)):
    """pred, true: (T, Z, H, W) rollouts from the same t=0 field. Returns {'corr@t': .., 'rel@t': ..}."""
    out = {}
    for t in frames:
        out["corr@%d" % t] = pattern_corr(pred[t], true[t])
        out["rel@%d" % t] = rel_l2(pred[t], true[t])
    return out


if __name__ == "__main__":
    for ds in ("d1", "d2", "hysteresis"):
        if os.path.exists(os.path.join(ROOT, ds, "metadata.csv")):
            m = metadata(ds)
            print(ds, len(m), "trajectories; splits:", sorted({r["split"] for r in m}))
    m = metadata("d1", "test_interior") or metadata("d1")
    P = load("d1", m[0])
    print("persistence baseline (predict the t=0 field):", score(np.broadcast_to(P[:1], P.shape), P))
