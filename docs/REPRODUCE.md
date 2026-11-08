# Reproducing the paper, step by step

| goal | steps | needs | time |
|---|---|---|---|
| **A. Check the paper's numbers** from the included results | 0 → 2 | a laptop is enough | 5 min |
| **B. Re-evaluate trained models** | 0 → 1 → 3 → 4 → 5 → 2 | 1 GPU, data download | ~3 h after training |
| **C. Retrain everything** | 0 → 1 → 3 → 4 → 5 → 2 | GPUs (57 runs × 20–60 min) | ~2 GPU-days |
| **D. Regenerate the data** | 6 | FerroX, GPU cluster with SLURM | ~1 node-day |

Trained checkpoints are not distributed, so B and C both start by training (step 4).
**Run every command from the repository root.**

## Step 0. Install

```bash
git clone https://github.com/matthew-ju/Duhamel-Neural-Operator.git && cd Duhamel-Neural-Operator
conda create -n duhamel python=3.12 -y && conda activate duhamel
pip install -r requirements.txt
```

The paper used Python 3.12, PyTorch 2.11 (CUDA 12.9) and neuraloperator 2.0.0; `requirements.txt` pins every
package. On NERSC Perlmutter, `module load pytorch/2.11.0` provides all of them. To rebuild the Figure 1 diagram you
also need LaTeX with TikZ.

## Step 1. Choose a working directory

All data, checkpoints and caches go in one directory, set by the environment variable `SCRATCH`:

```bash
export SCRATCH=/path/to/workdir        # any directory with ~200 GB free
```

After step 3 (or step 6) and step 4 it looks like this:

```
$SCRATCH/
  ferrox_abg_useed/extracted/c0000_s2000.npy ...   D1: 1001 TDGL trajectories (one material each, own noise seed)
  ferrox_d2useed/extracted/o0000_s6000.npy ...     D2: 300 TDGL trajectories (materials outside the training range)
  ml_units/ac_train.npz, ac_id.npz, ...            Allen–Cahn (64^2) and ac32_*, ac128_* (zero-shot grids)
  bench_cache/alpha_bank_tdgl.npz                  the measured Fourier SSM
  bench_cache/tdgl_train_fp16.npy                  training cache
  bench_ckpt/<run>.pt                              trained models
```

## Step 2. Regenerate the tables and figures, and compare with the paper

```bash
bash scripts/make_tables.sh
```

This rebuilds every number in the main text (`paper/numbers_duhamel.tex`), **Table 1**, **Table 2** and **Figure 3**
from the files in `results/`, then runs `scripts/check_against_paper.py`, which compares them with the published copies
in `paper/reference/`. With the included results the last line is `ALL MATCH`.

Figures 1 and 2 show actual fields, so they need the data and a trained Duhamel-NO:
`bash scripts/make_figures_from_data.sh`.

| paper | produced by | from |
|---|---|---|
| Table 1 | `paper/tools/make_figs.py` → `paper/make_duhamel_macros.py` | `results/final/` |
| Table 2 | `paper/make_physics_table.py` | `results/phys/` |
| Figure 1 | `paper/make_fig1_thumbs.py`, `paper/figures/06_duhamel/fig1/schematic_tikz.tex` | data |
| Figure 2 | `paper/make_rollout_fig.py` | data + checkpoint |
| Figure 3 | `paper/make_evt_fig.py` | `results/phys/` (a,b), `results/curves/` (c,d) |
| numbers in the text | `paper/make_duhamel_macros.py` | all of the above, plus `results/res/`, `results/tune/` and `results/analysis/` |

## Step 3. Download the data

The data are on Hugging Face: <https://huggingface.co/datasets/matthewju/ferroelectric-tdgl-bench>
(DOI [10.57967/hf/11169](https://doi.org/10.57967/hf/11169)). Download them and link them into `$SCRATCH`
(the symlinks copy nothing):

```bash
pip install -U huggingface_hub   # provides the hf command
hf download matthewju/ferroelectric-tdgl-bench --repo-type dataset --local-dir ./ferro_data
#   only the test sets: add  --include "d1/metadata.csv" --include "d2/*"
bash scripts/link_hf_data.sh ./ferro_data
```

The released files hold only the 10 ferroelectric layers, as `(401, 10, 64, 64)` arrays. The loaders read both these
and the raw `(64, 64, 18, 401)` FerroX extracts. Each trajectory's material (α, β, γ), time step and seed are in
`data_generation/jobs_D1.txt` / `jobs_D2.txt`.

The Allen–Cahn data are not downloaded; generate them (minutes, deterministic):

```bash
bash data_generation/make_allen_cahn_data.sh
```

## Step 4. Train

```bash
python duhamel_no/build_train_cache.py    # training cache from D1 (bit-identical to ours)
bash scripts/train_all.sh                 # the Fourier SSM, then every run of the paper
```

`scripts/train_all.sh` has one line per training run: Duhamel-NO and the 7 baselines with up to 3 seeds each, on TDGL
and Allen–Cahn, plus the data-efficiency runs. Each line has the exact arguments of the run behind the published
numbers and is commented with the run name. After the first line (`fit_alpha_bank.py`) the lines are independent: run
them in parallel, one per GPU. Each saves `$SCRATCH/bench_ckpt/<run>.pt` and `results/final/<run>.json`.

How the configurations were chosen: `duhamel_no/summarize2.py` picks, for every model, the configuration with the best
mean **validation** score of its full runs. Test data are never used for that choice. The candidates came from short
tuning runs (`duhamel_no/gen_tune.py`, results in `results/tune/`).

**Poseidon-T** (the "pretrained" row of Table 1) needs the Poseidon code and weights in `$SCRATCH/poseidon_src`
([camlab-ethz/poseidon](https://github.com/camlab-ethz/poseidon)). Skip its line if you don't need that row.

## Step 5. Evaluate

```bash
bash scripts/evaluate_all.sh      # writes results/phys, results/res, results/curves (1 GPU, ~3 h)
bash scripts/make_tables.sh
```

**Expected agreement.** GPU training is not bit-reproducible (cuDNN, TF32), so retrained numbers differ in the third
decimal and `check_against_paper.py` will report small mismatches. The paper averages three training seeds, which
agree to within about ±0.001 in correlation.

## Step 6. Regenerate the TDGL data (optional)

D1 and D2 need the GPU phase-field solver [FerroX](https://github.com/AMReX-Microelectronics/FerroX)
(3D CUDA build) on a SLURM cluster. Each trajectory takes about 3 minutes on one A100.

```bash
export FERROX_EXE=/path/to/main3d.gnu.TPROF.MPI.CUDA.ex
salloc -N 1 --gpus 4 ...                                   # or put these lines in a batch script
bash data_generation/run_ferrox.sh data_generation/jobs_D1.txt $SCRATCH/ferrox_abg_useed
bash data_generation/run_ferrox.sh data_generation/jobs_D2.txt $SCRATCH/ferrox_d2useed
```

The materials were sampled by `data_generation/abg_grid.py` (D1, Latin hypercube → `abg_grid.json`) and
`data_generation/ood_grid.py` (D2 → `ood_grid.json`); the job lists were written from those files. Every trajectory has
its own noise seed (D1: 2000 + i, D2: 6000 + i).

## What each file in `duhamel_no/` does

| file | role |
|---|---|
| `run.py` | train and evaluate one model on one task (`--task tdgl` or `--task ac`) |
| `models.py` | Duhamel-NO (`DuhamelUNO`) and every baseline |
| `tasks.py` | data loading, material units, test sets |
| `fit_alpha_bank.py` | measures the Fourier SSM from the first frames of each training simulation (Section 4.3) |
| `physmetrics.py` | physical quantities of Table 2 and the per-frame errors of Figure 3a,b |
| `restransfer.py`, `curves.py` | Allen–Cahn zero-shot grids and per-frame errors (Figure 3c,d) |
| `speed.py` | timings of Section 5.3 (run it alone on one GPU) |
| `summarize2.py`, `gen_tune.py` | validation-based model selection and the tuning sweep |
| `build_train_cache.py` | builds the TDGL training cache |
| `abg_laws.py`, `abg_data.py`, `abg_model.py`, `geo_data.py`, `geo_model.py` | material laws (change of units), TDGL data index |
| `recipe.py`, `duhamel.py`, `train.py` | Allen–Cahn data loading and the Duhamel remainder used by `models.py` |
| `probe_chaos.py`, `trunc_cap.py` | the Wiener-chaos probe and Fourier-truncation cap (`results/analysis/`) |
