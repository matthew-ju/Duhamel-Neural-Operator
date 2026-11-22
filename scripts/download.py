#!/usr/bin/env python3
"""scripts/download.py -- download the dataset and the trained models through their anonymous links.

The anonymous Hugging Face mirror has no `hf download` and its file browser lists at most 1000 files per folder (D1 has
1001), so this script reads the file list from the dataset's SHA256SUMS (for the data) or from the mirror's folder
listing (for the models), downloads only what you ask for, verifies every file's SHA-256, and skips files that are
already complete. It needs only the Python standard library.

    # data: everything (about 120 GB), or one dataset, or only some splits, or only the metadata
    python scripts/download.py data --out ./ferro_data
    python scripts/download.py data --out ./ferro_data --dataset d2
    python scripts/download.py data --out ./ferro_data --dataset d1 --split test_interior test_edge
    python scripts/download.py data --out ./ferro_data --metadata-only
    python scripts/download.py data --out ./ferro_data --dataset d1 --split test_interior --limit 2   # a quick try

    # trained models and measured Fourier SSMs (about 5 GB) into the working directory the code uses
    python scripts/download.py models --out "$SCRATCH"
    python scripts/download.py models --out "$SCRATCH" --include tdgl_duhamel_uno

Run it again after an interruption: complete files are skipped and partial ones are restarted.
"""
import argparse
import csv
import hashlib
import json
import os
import sys
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

DATA_LINK = "https://anonymous-hf.com/a/58mhxpc51v6u/"     # D1, D2, hysteresis
MODEL_LINK = "https://anonymous-hf.com/a/9z48lcxh6uv0/"   # 109 checkpoints + Fourier SSMs
DATASETS = ("d1", "d2", "hysteresis")
UA = {"User-Agent": "Mozilla/5.0 (compatible; duhamel-no-download/1.0)"}  # the mirror rejects the default urllib agent


def api(link):
    """https://anonymous-hf.com/a/<id>/ (or just <id>) -> https://anonymous-hf.com/api/a/<id>"""
    link = link.strip().rstrip("/")
    ident = link.split("/a/")[-1] if "/a/" in link else link
    if not ident or "MODEL_LINK_ID" in ident:
        sys.exit("no anonymous link for this repository yet: pass --link https://anonymous-hf.com/a/<id>/")
    return "https://anonymous-hf.com/api/a/" + ident


def get(url, retries=4):
    for k in range(retries):
        try:
            return urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=120)
        except Exception as e:  # noqa: BLE001 -- network errors of every kind are retried
            if k == retries - 1:
                raise RuntimeError("%s: %s" % (url, e)) from e
            time.sleep(2 ** k)


def resolve(base, path):
    return base + "/resolve/" + urllib.parse.quote(path)


def tree(base, path=""):
    """Every file under `path` of the mirror, recursively: [(path, size, sha256 or None)]."""
    entries = json.load(get(base + "/tree/" + urllib.parse.quote(path)))
    if len(entries) >= 1000:
        print("warning: %s lists %d entries, the mirror's limit; some files may be missing" % (path or "/", len(entries)))
    out = []
    for e in entries:
        if e["type"] == "directory":
            out += tree(base, e["path"])
        else:
            out.append((e["path"], e.get("size"), (e.get("lfs") or {}).get("oid")))
    return out


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


def complete(dest, size, digest):
    if not os.path.exists(dest) or (size is not None and os.path.getsize(dest) != size):
        return False
    return digest is None or sha256(dest) == digest


def fetch(base, path, dest, size=None, digest=None):
    """Download one file to dest (via dest.part), verify it, return 'skip' or 'ok'."""
    if complete(dest, size, digest):
        return "skip"
    os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
    part = dest + ".part"
    for attempt in range(3):
        h = hashlib.sha256()
        with get(resolve(base, path)) as r, open(part, "wb") as f:
            for block in iter(lambda: r.read(1 << 22), b""):
                f.write(block)
                h.update(block)
        if (size is None or os.path.getsize(part) == size) and (digest is None or h.hexdigest() == digest):
            os.replace(part, dest)
            return "ok"
        print("checksum mismatch for %s, retrying (%d/3)" % (path, attempt + 1))
    os.remove(part)
    raise RuntimeError("could not download %s intact" % path)


def run(base, jobs, out, workers):
    """jobs: [(remote path, size, sha256)] -> files under out/, in parallel."""
    sizes = [s for _, s, _ in jobs]
    gb = ", %.1f GB" % (sum(sizes) / 1e9) if None not in sizes else ""
    print("%d files%s -> %s" % (len(jobs), gb, os.path.abspath(out)))
    done = skipped = 0
    t0 = time.time()
    with ThreadPoolExecutor(workers) as pool:
        futs = {pool.submit(fetch, base, p, os.path.join(out, p), s, d): p for p, s, d in jobs}
        for i, fut in enumerate(as_completed(futs), 1):
            skipped += fut.result() == "skip"
            done += 1
            if i % 10 == 0 or i == len(jobs):
                print("  %d / %d files (%d already complete), %.0f s" % (i, len(jobs), skipped, time.time() - t0))
    print("done: every file verified")


def data(args):
    base = api(args.link or DATA_LINK)
    out = args.out
    os.makedirs(out, exist_ok=True)
    # small files first: checksums, loader, licence, the metadata tables and the FerroX inputs
    fetch(base, "SHA256SUMS", os.path.join(out, "SHA256SUMS"))
    sums = {}
    for line in open(os.path.join(out, "SHA256SUMS")):
        digest, path = line.split(None, 1)
        sums[path.strip()] = digest
    for p in ("README.md",):
        fetch(base, p, os.path.join(out, p))
    small = [p for p in sums if "/data/" not in p and (p.split("/")[0] not in DATASETS or p.split("/")[0] in args.dataset)]
    run(base, [(p, None, sums[p]) for p in small], out, args.workers)
    if args.metadata_only:
        return
    jobs = []
    for ds in args.dataset:
        rows = list(csv.DictReader(open(os.path.join(out, ds, "metadata.csv"))))
        if args.split:
            rows = [r for r in rows if r["split"] in args.split]
        if args.limit:
            rows = rows[:args.limit]
        jobs += [(ds + "/" + r["file"], None, sums[ds + "/" + r["file"]]) for r in rows]
    if not jobs:
        sys.exit("no trajectory matches --dataset %s --split %s" % (args.dataset, args.split))
    run(base, jobs, out, args.workers)


def models(args):
    base = api(args.link or MODEL_LINK)
    files = [f for f in tree(base) if f[0].startswith(("bench_ckpt/", "bench_cache/"))]
    if args.include:
        files = [f for f in files if f[0].startswith("bench_cache/") or any(s in f[0] for s in args.include)]
    run(base, files, args.out, args.workers)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="what")
    d = sub.add_parser("data", help="D1, D2 and hysteresis trajectories")
    d.add_argument("--out", default="ferro_data", help="target folder (default ./ferro_data)")
    d.add_argument("--dataset", nargs="+", choices=DATASETS, default=list(DATASETS))
    d.add_argument("--split", nargs="+", help="only these splits, e.g. test_interior test_edge (see metadata.csv)")
    d.add_argument("--limit", type=int, help="at most this many trajectories per dataset (e.g. 2, to try things out)")
    d.add_argument("--metadata-only", action="store_true", help="only metadata.csv, SHA256SUMS, loader and inputs")
    m = sub.add_parser("models", help="trained checkpoints (bench_ckpt/) and Fourier SSMs (bench_cache/)")
    m.add_argument("--out", default=os.environ.get("SCRATCH", "."), help="working directory (default $SCRATCH)")
    m.add_argument("--include", nargs="+", help="only checkpoints whose name contains one of these strings")
    for p in (d, m):
        p.add_argument("--link", help="anonymous link of the repository (default: the one of the paper)")
        p.add_argument("--workers", type=int, default=4, help="parallel downloads (default 4)")
    args = ap.parse_args()
    if args.what is None:
        ap.error("say what to download: data or models")
    data(args) if args.what == "data" else models(args)


if __name__ == "__main__":
    main()
