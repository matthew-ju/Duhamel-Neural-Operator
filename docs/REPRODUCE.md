# Reproducing the paper, step by step

| goal | steps | needs | time |
|---|---|---|---|
| **A. Check the paper's numbers** from the included results | 0 → 2 | a laptop is enough | 5 min |
| **B. Re-evaluate the released models** | 0 → 1 → 3 → 5 → 2 | 1 GPU, data + checkpoint download | ~6 h |
| **C. Retrain everything** | 0 → 1 → 3 → 4 → 5 → 2 | GPUs (139 runs, 105 GPU-hours in total) | ~4.5 GPU-days |
| **D. Regenerate the data** | 6 | FerroX, GPU cluster with SLURM | ~2 node-days |

**Run every command from the repository root.**

## Step 0. Install

```bash
# download the zip from https://anonymous.4open.science/r/Duhamel-Neural-Operator-8DF4/ ("Download Repository"), then:
unzip Duhamel-Neural-Operator-8DF4.zip -d Duhamel-Neural-Operator && cd Duhamel-Neural-Operator
conda create -n duhamel python=3.12 -y && conda activate duhamel
pip install -r requirements.txt
```

The paper used Python 3.12, PyTorch 2.11 (CUDA 12.9) and neuraloperator 2.0.0; `requirements.txt` pins every
package. Step 2 alone needs only `pip install numpy scipy matplotlib`.

## Step 1. Choose a working directory

All data, checkpoints and caches go in one directory, set by the environment variable `SCRATCH`:

```bash
export SCRATCH=/path/to/workdir        # any directory with ~200 GB free
```

After step 3 it looks like this:

```
$SCRATCH/
  ferrox_abg_useed/extracted/c0000_s2000.npy ...   D1: 1001 TDGL trajectories (one material each, own noise seed)
  ferrox_d2useed/extracted/o0000_s6000.npy ...     D2: 300 TDGL trajectories (materials outside the training range)
  ferrox_hyst_closed2/h_*.npz                      hysteresis: 164 voltage-sweep loops
  ferrox_hyst_closed/as_h/h_*.npz                  hysteresis: the 10 extra closed loops
  ml_units/ac_train.npz, ac_id.npz, ...            Allen–Cahn (64^2) and ac32_*, ac128_* (zero-shot grids)
  bench_cache/alpha_bank_{tdgl,hyst}.npz           the measured Fourier SSMs
  bench_cache/tdgl_train_fp16.npy                  training-set cache
  bench_ckpt/<run>.pt                              trained models
```

## Step 2. Regenerate the numbers and tables, and compare with the paper

```bash
bash scripts/make_tables.sh
```

This rebuilds, from the files in `results/`, every number the paper prints (`reproduce/numbers_duhamel.tex`: 53 in
the main text, 24 in the appendix) and the ten tables it typesets (`reproduce/tables/`). Then
`scripts/check_against_paper.py` compares them with the values printed in the paper (`reproduce/expected/`). With the
included results the last line is `ALL MATCH`.

| paper | produced by | from |
|---|---|---|
| Table 1 | `reproduce/tools/make_figs.py` → `reproduce/make_duhamel_macros.py` | `results/final/` |
| Figure 3c (physical quantities) | `reproduce/make_physics_table.py` → `reproduce/make_appendix_tables.py` | `results/phys/` |
| ablation table | `reproduce/tools/make_figs.py` → `reproduce/make_appendix_tables.py` | `results/final/` |
| appendix tables | `reproduce/tools/make_figs.py`, `make_units.py`, `reproduce/make_ac_table.py` → `reproduce/make_appendix_tables.py` | `results/` |
| numbers in the text | `reproduce/make_duhamel_macros.py` | all of the above |

## Step 3. Download the data and the trained models

The data are on Hugging Face, behind an anonymous link for the review: <https://anonymous-hf.com/a/58mhxpc51v6u/>.
The mirror has no `hf download`, and its file browser shows at most 1000 files per folder (D1 has 1001), so use
`scripts/download.py`. It needs only the Python standard library, takes the file list from the dataset's `SHA256SUMS`,
verifies every file and skips files that are already complete, so it can simply be rerun after an interruption:

```bash
python scripts/download.py data --out ./ferro_data                                   # everything, ~120 GB
python scripts/download.py data --out ./ferro_data --dataset d1 d2                   # TDGL only, ~85 GB
python scripts/download.py data --out ./ferro_data --dataset d1 --split test_interior test_edge   # D1 test set, 17 GB
python scripts/download.py data --out ./ferro_data --metadata-only                   # metadata, checksums, inputs
```

Then link the data into `$SCRATCH` (the symlinks copy nothing):

```bash
bash scripts/link_hf_data.sh ./ferro_data
python duhamel_no/build_train_cache.py    # training-set cache from D1 (bit-identical to ours)
```

The released files hold only the 10 ferroelectric layers, as `(401, 10, 64, 64)` arrays. The loaders read both these
and the raw `(64, 64, 18, 401)` FerroX extracts. Each trajectory's material (α, β, γ), time step and seed are in
`data_generation/jobs_D1.txt` / `jobs_D2.txt`, and in the `metadata.csv` of each dataset.

The 109 trained models and the measured Fourier SSMs (about 5 GB) are behind a second anonymous link,
<https://anonymous-hf.com/a/9z48lcxh6uv0/>. Their folders mirror `$SCRATCH`, so downloading them into `$SCRATCH` puts every file in place:

```bash
python scripts/download.py models --out "$SCRATCH"                             # all checkpoints and Fourier SSMs
python scripts/download.py models --out "$SCRATCH" --include tdgl_duhamel_uno  # only Duhamel-NO on D1/D2
```

The Allen–Cahn data are not downloaded; generate them (minutes, deterministic):

```bash
bash data_generation/make_allen_cahn_data.sh
```

## Step 4. Train (optional with the released models)

```bash
bash scripts/train_all.sh
```

`scripts/train_all.sh` first measures the Fourier SSMs (TDGL, hysteresis, and the 32/128-material subsets), then has one
line per trained model: Duhamel-NO and every baseline with up to three seeds, on TDGL, hysteresis and Allen–Cahn, plus
the ablation variants, the runs without the change of units and the data-efficiency runs. Each line has the exact
arguments of the run behind the published numbers and is commented with its run name. After the SSM lines, the lines
are independent: run them in parallel, one per GPU. Each saves `$SCRATCH/bench_ckpt/<run>.pt` and
`results/final/<run>.json`.

How the configurations were chosen: `duhamel_no/summarize2.py` picks, for every model, the configuration with the best
mean **validation** score of its full runs. Test data are never used for that choice. Lines marked "candidate" are the
configurations that lost this comparison; the paper reports the others. The candidates came from short tuning runs
(`duhamel_no/gen_tune.py`, results in `results/tune/`).

**Poseidon-T** needs the Poseidon code and weights in `$SCRATCH/poseidon_src`
([camlab-ethz/poseidon](https://github.com/camlab-ethz/poseidon)). Skip its line if you don't need that row.

## Step 5. Evaluate

```bash
bash scripts/evaluate_all.sh      # writes results/phys, results/res, results/curves (1 GPU, ~6 h)
bash scripts/make_tables.sh
```

To re-score one released model at its training-time evaluation, add `--eval_only` to its line in
`scripts/train_all.sh`; it loads `$SCRATCH/bench_ckpt/<run>.pt` and rewrites `results/final/<run>.json`.

**Expected agreement.** Evaluating the released checkpoints reproduces the published numbers. Retrained models do not
match bit for bit (GPU training is not deterministic: cuDNN, TF32), so their numbers differ in the third decimal and
`check_against_paper.py` will report small mismatches. The paper averages three training seeds, which agree to within
about ±0.001 in correlation.

## Step 6. Regenerate the data (optional)

All three datasets need the GPU phase-field solver [FerroX](https://github.com/AMReX-Microelectronics/FerroX)
(3D CUDA build) on a SLURM cluster.

**D1 and D2** (about 3 minutes per trajectory on one A100):

```bash
export FERROX_EXE=/path/to/main3d.gnu.TPROF.MPI.CUDA.ex
salloc -N 1 --gpus 4 ...                                   # or put these lines in a batch script
bash data_generation/run_ferrox.sh data_generation/jobs_D1.txt $SCRATCH/ferrox_abg_useed
bash data_generation/run_ferrox.sh data_generation/jobs_D2.txt $SCRATCH/ferrox_d2useed
```

The materials were sampled by `data_generation/abg_grid.py` (D1, Latin hypercube → `abg_grid.json`) and
`data_generation/ood_grid.py` (D2 → `ood_grid.json`); the job lists were written from those files. Every trajectory has
its own noise seed (D1: 2000 + i, D2: 6000 + i).

**Hysteresis** (quasi-static voltage sweeps, 20–120 minutes per loop on one A100):

```bash
bash data_generation/hysteresis/run_ferrox_sweep.sh data_generation/hysteresis/jobs_hysteresis.txt $SCRATCH/ferrox_hyst_closed2
bash data_generation/hysteresis/run_ferrox_sweep.sh data_generation/hysteresis/jobs_hysteresis_closed.txt $SCRATCH/ferrox_hyst_closed
mkdir -p $SCRATCH/ferrox_hyst_closed/as_h
for f in $SCRATCH/ferrox_hyst_closed/hc_*.npz; do ln -sfn "$f" "$SCRATCH/ferrox_hyst_closed/as_h/h_${f##*/hc_}"; done
```

The held-out materials of the hysteresis split are listed in `data_generation/hysteresis/hyst_split.json`.

## What each file in `duhamel_no/` does

| file | role |
|---|---|
| `run.py` | train and evaluate one model on one task (`--task tdgl`, `hyst` or `ac`) |
| `models.py` | Duhamel-NO (`DuhamelUNO`) and every baseline |
| `tasks.py` | data loading, material units, test sets |
| `fit_alpha_bank.py` | measures the Fourier SSM from the first frames of each training simulation |
| `physmetrics.py` | physical quantities, per-frame errors and example rollouts (Figure 5, appendix) |
| `restransfer.py`, `curves.py` | Allen–Cahn zero-shot grids, per-frame (Allen–Cahn) and per-step (hysteresis) error curves |
| `hyst_data.py`, `hystloops.py` | hysteresis loops: loading, loop metrics, and the P–V loops of the appendix figure |
| `speed.py` | timings (run it alone on one GPU) |
| `summarize2.py`, `gen_tune.py` | validation-based model selection and the tuning sweep |
| `build_train_cache.py` | builds the TDGL training-set cache |
| `abg_laws.py`, `abg_data.py`, `abg_model.py`, `geo_data.py`, `geo_model.py` | material laws (change of units), TDGL data index |
| `recipe.py`, `duhamel.py`, `train.py` | Allen–Cahn data loading and the Duhamel remainder used by `models.py` |
| `probe_chaos.py`, `trunc_cap.py` | the Wiener-chaos probe and the Fourier-truncation cap (`results/analysis/`) |
