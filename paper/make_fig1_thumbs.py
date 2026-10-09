#!/usr/bin/env python3
"""make_fig1_thumbs.py -- field thumbnails for the TikZ Fig. 1 (figures/06_duhamel/fig1/*.png): FerroX top layer of one D1
held-out material (tag c0176, the median-error interior material of the rollout figure) at t = 0, 5, 20, 400, in units of
P_S with the colour scale of the rollout figure; plus the in-plane spectrum |P^(k)| at t = 10 (the growth ring)."""
import os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
BENCH = os.path.join(os.path.dirname(HERE), "bench"); sys.path.insert(0, BENCH)
import tasks as TK  # noqa: E402
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt  # noqa: E402
OUT = os.path.join(HERE, "figures", "06_duhamel", "fig1")
task = TK.make("tdgl")
tr = [t for t in task.eval_sets()["D1_heldout"]["trajs"] if t.tag == "c0176"][0]
u, _ = task.load_batch([tr]); u = u[0].numpy() * task.scale                 # (T, Z, H, W)
def save(img, name, cmap="RdBu_r", sym=True):
    fig = plt.figure(figsize=(1, 1), dpi=200); ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off")
    v = np.abs(img).max()
    ax.imshow(img, cmap=cmap, vmin=-v if sym else 0, vmax=v, interpolation="nearest")
    fig.savefig(os.path.join(OUT, name)); plt.close(fig)
for t in (0, 5, 20, 400):
    save(u[t, -1], "P_t%03d.png" % t)
S = np.abs(np.fft.fftshift(np.fft.fft2(u[10, -1])))[32 - 14:32 + 15, 32 - 14:32 + 15]          # low-wavenumber window
save(S, "spec_t010.png", cmap="magma", sym=False)
print("wrote", sorted(os.listdir(OUT)))

# Fourier-SSM icon: the per-mode growth factor of the measured Fourier SSM of this material, log e^{theta lambda_max(k)}
# (generators of the 8 training materials nearest in log|alpha|, as models.DuhamelUNO._bank_ops), on the same window
B = np.load(os.path.join(os.environ["SCRATCH"], "bench_cache", "alpha_bank_tdgl.npz"))
nn_ = np.argsort(np.abs(np.log(np.abs(B["alpha"])) - np.log(abs(float(tr.mat[0])))))[:8]
ok = B["valid"][nn_].astype(float); cnt = ok.sum(0)
G = (B["G"][nn_] * ok[..., None, None]).sum(0) / np.maximum(cnt, 1)[:, None, None]
lam = np.array([np.linalg.eigvals(g).real.max() for g in G]); lam[cnt == 0] = np.nan          # unresolved bins: decayed
kx = np.fft.fftfreq(64) * 64; KB = np.round(np.sqrt(kx[:, None] ** 2 + kx[None, :] ** 2)).astype(int)
lg = np.fft.fftshift(float(tr.theta) * lam[KB])[32 - 10:32 + 11, 32 - 10:32 + 11]   # |k| <= 10
v = 1.5 * np.nanmax(lg); lg = np.clip(np.nan_to_num(lg, nan=-v), -v, v)   # colour range set by the growing band
fig = plt.figure(figsize=(1, 1), dpi=200); ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off")
ax.imshow(lg, cmap="RdBu_r", vmin=-v, vmax=v, interpolation="nearest"); fig.savefig(os.path.join(OUT, "ssm_gain.png")); plt.close(fig)
print("theta %.4f  growth band |k| = %s" % (float(tr.theta), np.nonzero(lam > 0)[0].tolist()))
