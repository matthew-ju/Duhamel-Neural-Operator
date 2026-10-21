# Duhamel Neural Operator for Learning Phase-Field Dynamics

Code to reproduce the results of the AI4Mat @ NeurIPS 2026 workshop paper (`paper/paper.pdf`):
the ferroelectric (FerroX TDGL) experiments on datasets D1 and D2, and the Allen–Cahn experiments.

Duhamel-NO advances a phase field one frame at a time. A **Fourier SSM**, measured from simulations, moves every Fourier
mode by its exact linear factor. A **U-shaped neural operator** learns the nonlinear correction (paper, Section 4).

---

## What you can do, and how long it takes

| goal | steps | needs | time |
|---|---|---|---|
| **A. Check the paper's numbers** from the included results | 0 → 2 | a laptop is enough | 5 min |
| **B. Re-evaluate the released models** | 0 → 1 → 3 → 2 | 1 GPU, data + checkpoints download | ~3 h |
| **C. Retrain everything** | 0 → 1 → 4 → 3 → 2 | GPUs (57 runs × 20–60 min) | ~2 GPU-days |
| **D. Regenerate the data** | 6 | FerroX, GPU cluster with SLURM | ~1 node-day |

Start with **A**. It needs no data and no GPU, and it shows the whole pipeline end to end.

---

## Step 0. Install

```bash
git clone <this repository> duhamel-no && cd duhamel-no
conda create -n duhamel python=3.12 -y && conda activate duhamel
pip install -r requirements.txt
```

The paper used Python 3.12, PyTorch 2.11 (CUDA 12.9) and neuraloperator 2.0.0; `requirements.txt` pins every
package. On NERSC Perlmutter, `module load pytorch/2.11.0` provides all of them. To rebuild the Figure 1 diagram you
also need LaTeX with TikZ.

**Run every command from the repository root.**

## Step 1. Tell the code where the data live

All data, checkpoints and caches go in one directory, set by the environment variable `SCRATCH`:

```bash
export SCRATCH=/path/to/duhamel_data        # any directory with ~200 GB free
```

After downloading (step 3) or generating (steps 4–6) it looks like this:

```
$SCRATCH/
  ferrox_abg_useed/extracted/c0000_s2000.npy ...   D1: 1001 TDGL trajectories (one material each, own noise seed)
  ferrox_d2useed/extracted/o0000_s6000.npy ...     D2: 300 TDGL trajectories (materials outside the training range)
  ml_units/ac_train.npz, ac_id.npz, ...            Allen–Cahn (64^2) and ac32_*, ac128_* (zero-shot grids)
  bench_cache/alpha_bank_tdgl.npz                  the measured Fourier SSM (step 4a)
  bench_cache/tdgl_train_fp16.npy                  training cache (step 4a)
  bench_ckpt/<run>.pt                              trained models
```

Each TDGL file is an array of shape `(64, 64, 18, 401)`: in-plane grid, depth layers (layers 8–17 are the
ferroelectric film), frames. Its job line in `data/jobs_D1.txt` / `data/jobs_D2.txt` gives the material (α, β, γ),
time step and seed.

## Step 2. Regenerate the tables and figures, and compare with the paper

```bash
bash scripts/make_tables.sh
```

This rebuilds every number in the main text (`paper/numbers_duhamel.tex`), **Table 1**, **Table 2** and **Figure 3**
from the result files in `bench/results_*`. Then it runs `scripts/check_against_paper.py`, which compares them with the
published copies in `paper/reference/`. With the included results, the expected last line is:

```
ALL MATCH
```

Figures 1 and 2 show actual fields, so they need the data (step 3): `bash scripts/make_figures_from_data.sh`.

**Where each paper item comes from**

| paper | produced by | from |
|---|---|---|
| Table 1 | `paper_tools/make_figs.py` → `paper/make_duhamel_macros.py` | `bench/results_final/` |
| Table 2 | `paper/make_physics_table.py` | `bench/results_phys/` |
| Figure 1 | `paper/make_fig1_thumbs.py`, `paper/figures/06_duhamel/fig1/schematic_tikz.tex` | data |
| Figure 2 | `paper/make_rollout_fig.py` | data + checkpoint |
| Figure 3 | `paper/make_evt_fig.py` | `bench/results_phys/` (a,b), `bench/results_curves/` (c,d) |
| numbers in the text | `paper/make_duhamel_macros.py` | all of the above, plus `bench/results_res/` (Allen–Cahn grids) and `bench/results_tune/` (standard training) |

## Step 3. Download the data and the trained models

The data are on Hugging Face: **https://huggingface.co/datasets/matthewju/ferroelectric-tdgl-bench**
(DOI: [10.57967/hf/11169](https://doi.org/10.57967/hf/11169)). Download them and link them into the layout of step 1. The symlinks copy nothing.

```bash
pip install -U huggingface_hub   # provides the hf command
hf download matthewju/ferroelectric-tdgl-bench --repo-type dataset --local-dir ./ferro_data
#   only the test sets: add  --include "d1/metadata.csv" "d2/*"
bash scripts/link_hf_data.sh ./ferro_data
```

The released files hold only the 10 ferroelectric layers, as `(401, 10, 64, 64)` arrays. The loaders read both these
and the raw `(64, 64, 18, 401)` FerroX extracts. Trained checkpoints are not included;
step 4 retrains every model (they are written to `$SCRATCH/bench_ckpt/`).

Then re-evaluate the released models and rebuild the tables:

```bash
bash scripts/evaluate_all.sh      # writes bench/results_phys, results_res, results_curves (1 GPU, ~3 h)
bash scripts/make_tables.sh
```

## Step 4. Retrain (optional)

```bash
python release/build_train_cache.py       # 4a. training cache from D1 (bit-identical to ours)
bash scripts/train_all.sh                 # 4a. Fourier SSM, then 4b-4d. every run of the paper
bash scripts/evaluate_all.sh
bash scripts/make_tables.sh
```

`scripts/train_all.sh` has one line per training run: Duhamel-NO and the 7 baselines with up to 3 seeds each, on TDGL and
Allen–Cahn, plus the data-efficiency runs. Each line rebuilds the exact arguments of the run that produced the
published numbers, and is commented with the run name. The lines are independent: run them in parallel, one per GPU.
Each line saves `$SCRATCH/bench_ckpt/<run>.pt` and `bench/results_final/<run>.json`.

How the configurations were chosen: `bench/summarize2.py` picks, for every model, the configuration with the best mean
**validation** score of its full runs. Test data are never used for that choice. The candidate configurations came
from short tuning runs (`bench/gen_tune.py`, results in `bench/results_tune/`).

**Expected agreement.** GPU training is not bit-reproducible (cuDNN, TF32), so retrained numbers differ in the third
decimal and `check_against_paper.py` will report small mismatches. The paper averages three training seeds, which
agree to within about ±0.001 in correlation.

**Poseidon-T** (the "pretrained" row of Table 1) needs the Poseidon code and weights in `$SCRATCH/poseidon_src`
([camlab-ethz/poseidon](https://github.com/camlab-ethz/poseidon)). Skip its line if you don't need that row.

## Step 5. What the scripts run

- `bench/run.py`: training and evaluation of one model on one task (`--task tdgl` or `--task ac`)
- `bench/models.py`: Duhamel-NO (`DuhamelUNO`) and every baseline
- `bench/tasks.py`: data loading, material units, test sets
- `bench/fit_alpha_bank.py`: measures the Fourier SSM from the first frames of each training simulation (Section 4.3)
- `bench/physmetrics.py`: physical quantities of Table 2 and the per-frame errors of Figure 3a,b
- `bench/restransfer.py`, `bench/curves.py`: Allen–Cahn zero-shot grids and per-frame errors (Figure 3c,d)
- `bench/speed.py`: timings of Section 5.3. Run it alone on one GPU; other jobs on the same GPU inflate the timings.

## Step 6. Regenerate the data (optional)

**Allen–Cahn** (minutes, deterministic):

```bash
bash scripts/make_allen_cahn_data.sh
```

**TDGL D1 and D2** need the GPU phase-field solver [FerroX](https://github.com/AMReX-Microelectronics/FerroX)
(3D CUDA build) on a SLURM cluster. Each trajectory takes about 3 minutes on one A100.

```bash
export FERROX_EXE=/path/to/main3d.gnu.TPROF.MPI.CUDA.ex
salloc -N 1 --gpus 4 ...                                   # or put these lines in a batch script
bash scripts/run_ferrox.sh data/jobs_D1.txt $SCRATCH/ferrox_abg_useed
bash scripts/run_ferrox.sh data/jobs_D2.txt $SCRATCH/ferrox_d2useed
```

The materials were sampled by `abg_grid.py` (D1, Latin hypercube → `abg_grid.json`) and `ood_grid.py`
(D2 → `ood_grid.json`). The job lists in `data/` were written from those files. Every trajectory has its own noise seed
(D1: 2000 + i, D2: 6000 + i).

---

## The benchmark

D1, D2 and the hysteresis loops are released as an open benchmark for ferroelectric phase-field dynamics
(paper, Appendix F). The dataset card on Hugging Face describes the format, splits and protocol.
`benchmark/ferro_bench.py` needs only numpy. It loads trajectories and computes the paper's metrics:

```python
import sys; sys.path.insert(0, "benchmark"); import ferro_bench as fb
fb.ROOT = "./ferro_data"
test = fb.metadata("d1", ["test_interior", "test_edge"])     # 256 held-out materials
true = fb.load("d1", test[0])                                 # (401, 10, 64, 64), C/m^2
fb.score(my_rollout, true)                                    # corr and rel. L2 at frames 1 and 400
```

Protocol:
- Roll out from the frame-0 field alone, conditioned on (α, β, γ) and `dt_frame`.
- Train on D1 `train` and select hyperparameters on `val`.
- Report D1 `test_interior` + `test_edge`, and all of D2 zero-shot.

## Repository layout

```
paper/            paper PDF, scripts that write its numbers/tables/figures, published copies in paper/reference/
paper_tools/      table and figure code shared by those scripts
bench/            models, training, evaluation, metrics; results_*/ = the result files behind the paper
scripts/          the step scripts above
data/             FerroX job lists for D1 and D2
ferrox_inputs/    FerroX input template
abg_*.py, geo/, ml_units/, tdgl_*.py    physics laws, material units, data index, Allen–Cahn generator
release/          build_train_cache.py
benchmark/        ferro_bench.py: loader and metrics for the released benchmark
```

## Citation

```bibtex
@inproceedings{ju2026duhamel,
  title     = {Duhamel Neural Operator for Learning Phase-Field Dynamics},
  author    = {Ju, Matthew and Tang, Yingheng and Wang, Kang'an and Sha, Haozhi},
  booktitle = {AI for Accelerated Materials Design (AI4Mat) Workshop, NeurIPS},
  year      = {2026}
}

@misc{ju2026ferrobench,
  title     = {Ferroelectric {TDGL} Phase-Field Benchmark},
  author    = {Ju, Matthew and Tang, Yingheng and Wang, Kang'an and Sha, Haozhi},
  year      = {2026},
  publisher = {Hugging Face},
  doi       = {10.57967/hf/11169},
  url       = {https://huggingface.co/datasets/matthewju/ferroelectric-tdgl-bench}
}
```

## License

Data: CC BY 4.0 (see the dataset card). Code: <to be added after LBNL Intellectual Property Office approval>
