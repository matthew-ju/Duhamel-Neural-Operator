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
git clone https://github.com/matthew-ju/Duhamel-Neural-Operator.git && cd Duhamel-Neural-Operator
conda create -n duhamel python=3.12 -y && conda activate duhamel
pip install -r requirements.txt
```

The paper used Python 3.12, PyTorch 2.11 (CUDA 12.9) and neuraloperator 2.0.0; `requirements.txt` pins every
package. On NERSC Perlmutter, `module load pytorch/2.11.0` provides all of them. Figures 1 and 2 also need LaTeX with
TikZ.

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

## Step 2. Regenerate the numbers, tables and figures, and compare with the paper

```bash
bash scripts/make_tables.sh
```

This rebuilds, from the files in `results/`, every number the paper prints (`paper/numbers_duhamel.tex`: 53 in the
main text, 24 in the appendix), the ten tables it typesets (`paper/tables/`) and the figures drawn from results
(`paper/figures/`). Then `scripts/check_against_paper.py` compares the numbers and tables with the published copies in
`paper/reference/`. With the included results the last line is `ALL MATCH`.

The figures that show actual fields need the data and the trained models (steps 3 and 5):
`bash scripts/make_figures_from_data.sh`.

| paper | produced by | from |
|---|---|---|
| Figure 1 (problem setup) | `paper/make_setup_fig.py`, `paper/figures/tmlr/setup_tikz.tex` | data |
| Figure 2 (Duhamel-NO) | `paper/make_fig1_thumbs.py`, `paper/figures/06_duhamel/fig1/schematic_tikz.tex` | data |
| Figure 3 (truncation cap) | `paper/make_theory_fig.py` | `results/phys/`, `results/analysis/` |
| Figure 4 (rollouts) | `paper/make_rollout_fig.py` | data + checkpoint |
| Figure 5 (error per frame, physical quantities) | `paper/make_evt_ab_fig.py`, `paper/make_physics_table.py` → `make_appendix_tables.py` | `results/phys/` |
| Table 1 | `paper/tools/make_figs.py` → `paper/make_duhamel_macros.py` | `results/final/` |
| ablation table | `paper/tools/make_figs.py` → `paper/make_appendix_tables.py` | `results/final/` |
| appendix tables | `paper/tools/make_figs.py`, `make_units.py`, `paper/make_ac_table.py` → `paper/make_appendix_tables.py` | `results/` |
| appendix figures | `paper/tools/make_figs.py`, `fig_design.py`, `paper/make_hyst_curves_fig.py`; with data: `make_energy_fig.py`, `make_gallery_fig.py` | `results/`; data |
| numbers in the text | `paper/make_duhamel_macros.py` | all of the above |

## Step 3. Download the data and the trained models

The data are on Hugging Face: <https://huggingface.co/datasets/matthewju/ferroelectric-tdgl-bench>
(DOI [10.57967/hf/11169](https://doi.org/10.57967/hf/11169)). Download them and link them into `$SCRATCH`
(the symlinks copy nothing):

```bash
pip install -U huggingface_hub   # provides the hf command
hf download matthewju/ferroelectric-tdgl-bench --repo-type dataset --local-dir ./ferro_data
#   only the test sets: add  --include "d1/metadata.csv" --include "d2/*"
bash scripts/link_hf_data.sh ./ferro_data
python duhamel_no/build_train_cache.py    # training-set cache from D1 (bit-identical to ours)
```

The released files hold only the 10 ferroelectric layers, as `(401, 10, 64, 64)` arrays. The loaders read both these
and the raw `(64, 64, 18, 401)` FerroX extracts. Each trajectory's material (α, β, γ), time step and seed are in
`data_generation/jobs_D1.txt` / `jobs_D2.txt`, and in the `metadata.csv` of each dataset.

The trained models and the measured Fourier SSMs are at <https://huggingface.co/matthewju/duhamel-neural-operator>.
The repository mirrors `$SCRATCH`, so this puts every file in place:

```bash
hf download matthewju/duhamel-neural-operator --local-dir $SCRATCH
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
bash scripts/make_figures_from_data.sh
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
