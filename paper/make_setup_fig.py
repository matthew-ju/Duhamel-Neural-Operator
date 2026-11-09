#!/usr/bin/env python3
"""Figure 1 of the TMLR paper (the problem setup), drawn in TikZ/pgfplots from one real FerroX trajectory.

Step 1 (this script): export the data the TikZ source needs into figures/tmlr/setup_assets/
  - top/front/side faces of the film at the final frame (PNG), for the 3D device in panel (a)
  - snapshots of the top layer at the frames in THUMBS (PNG), for panel (c)
  - the Landau well f(P) of the same material and the rms amplitude per frame (.dat), for panels (b) and (c)
Step 2: compile figures/tmlr/setup_tikz.tex with pdflatex and copy the result to figures/tmlr/setup.pdf.

Run:  FERRO_DATA=./ferro_data python paper/make_setup_fig.py
Data: D1 validation trajectory c0000_s2000 of the public benchmark."""
import os, csv, shutil, subprocess
import numpy as np
import matplotlib
import matplotlib.image


HERE = os.path.dirname(os.path.abspath(__file__))                     # paper/
BENCH = os.path.join(os.environ.get("FERRO_DATA", "ferro_data"), "d1")   # the Hugging Face download (docs/REPRODUCE.md step 3)
TRAJ = "c0000_s2000"
FIG = os.path.join(HERE, "figures", "tmlr")
OUT = os.path.join(FIG, "setup_assets")
THUMBS = [0, 30, 60, 80, 400]
TOP = 9                                     # top layer of the film
UP = 8                                      # nearest-neighbour upsampling, keeps grid cells crisp in the PDF
CMAP = matplotlib.colormaps["RdBu_r"]

meta = {r["traj_id"]: r for r in csv.DictReader(open(os.path.join(BENCH, "metadata.csv")))}[TRAJ]
al, be, ga, Ps = (float(meta[k]) for k in ("alpha", "beta", "gamma", "P_s"))
P = np.load(os.path.join(BENCH, "data", TRAJ + ".npy"), mmap_mode="r")   # (T, 10 layers, 64, 64), C/m^2
T = P.shape[0]


def save_png(name, img, vmax, flip=True):
    """img[row, col] with row 0 at the bottom (flip=True), coloured on [-vmax, vmax]."""
    a = np.asarray(img, dtype=np.float64)
    if flip:
        a = a[::-1]
    rgb = (CMAP(np.clip(0.5 + 0.5 * a / vmax, 0, 1))[..., :3] * 255).astype(np.uint8)
    rgb = np.repeat(np.repeat(rgb, UP, axis=0), UP, axis=1)
    matplotlib.image.imsave(os.path.join(OUT, name), rgb)


os.makedirs(OUT, exist_ok=True)
F = np.asarray(P[T - 1])                                       # final state, (z, a, b)
save_png("top.png", F[TOP], Ps)                                # in-plane (a, b): rows a, columns b
save_png("front.png", F[:, 0, :], Ps)                          # x-z face at a = 0: rows z, columns b
save_png("side.png", F[:, :, -1], Ps)                          # y-z face at b = 63: rows z, columns a
for t in THUMBS:
    img = np.asarray(P[t, TOP])
    save_png("snap_%03d.png" % t, img, np.abs(img).max())     # each snapshot on its own colour range

cb = (CMAP(np.linspace(0, 1, 256))[:, :3] * 255).astype(np.uint8)[::-1, None, :]   # vertical colour bar, +1 on top
matplotlib.image.imsave(os.path.join(OUT, "cbar.png"), np.repeat(cb, 8, axis=1))

rms = np.sqrt((np.asarray(P, dtype=np.float64) ** 2).mean(axis=(1, 2, 3))) / Ps
np.savetxt(os.path.join(OUT, "rms.dat"), np.c_[np.arange(T), rms], fmt="%d %.6e", header="n rms_over_Ps", comments="")
for t in THUMBS:
    print("frame %3d  rms/Ps = %.4e" % (t, rms[t]))

pm = np.sqrt((-be + np.sqrt(be**2 - 4 * ga * al)) / (2 * ga))  # minimum of the bare Landau well
p = np.linspace(-1.42, 1.42, 241) * pm
f = al * p**2 / 2 + be * p**4 / 4 + ga * p**6 / 6
fmin = al * pm**2 / 2 + be * pm**4 / 4 + ga * pm**6 / 6
np.savetxt(os.path.join(OUT, "well.dat"), np.c_[p / pm, f / abs(fmin)], fmt="%.5f", header="p f", comments="")
print("Ps = %.3f, bare minimum = %.3f C/m^2" % (Ps, pm))

# step 2: compile the TikZ source
tex = os.path.join(FIG, "setup_tikz.tex")
if os.path.exists(tex):
    subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "setup_tikz.tex"], cwd=FIG, check=True,
                   stdout=subprocess.DEVNULL)
    shutil.copy(os.path.join(FIG, "setup_tikz.pdf"), os.path.join(FIG, "setup.pdf"))
    print("wrote", os.path.join(FIG, "setup.pdf"))
