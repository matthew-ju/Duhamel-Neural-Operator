#!/bin/bash
set -e
cd "$(dirname "$0")/.."            # repository root

# Step 4 of README: train every model of the paper (1 A100 per line, ~20-60 min each; lines are independent,
# so they can run in parallel, e.g. one per GPU). Each line writes $SCRATCH/bench_ckpt/<tag>.pt and
# bench/results_final/<tag>.json. Re-running a line overwrites that run.

# 4a. Fourier SSM measured from the training simulations (seconds)
python -u bench/fit_alpha_bank.py --task tdgl

# 4b. TDGL (Table 1, Table 2, Figures 2-3)
python -u bench/run.py --mode final --task tdgl --model duhamel_uno --seed 0 --lr 0.0003 --wrap   # -> tdgl_duhamel_uno_lr0.0003_recipe_s0
python -u bench/run.py --mode final --task tdgl --model duhamel_uno --seed 1 --lr 0.0003 --wrap   # -> tdgl_duhamel_uno_lr0.0003_recipe_s1
python -u bench/run.py --mode final --task tdgl --model duhamel_uno --seed 2 --lr 0.0003 --wrap   # -> tdgl_duhamel_uno_lr0.0003_recipe_s2
python -u bench/run.py --mode final --task tdgl --model ffno --seed 0 --lr 0.001 --wrap   # -> tdgl_ffno_lr0.001_recipe_s0
python -u bench/run.py --mode final --task tdgl --model ffno --seed 1 --lr 0.001 --wrap   # -> tdgl_ffno_lr0.001_recipe_s1
python -u bench/run.py --mode final --task tdgl --model ffno --seed 2 --lr 0.001 --wrap   # -> tdgl_ffno_lr0.001_recipe_s2
python -u bench/run.py --mode final --task tdgl --model unet --seed 0 --lr 0.0003 --wrap   # -> tdgl_unet_lr0.0003_recipe_s0
python -u bench/run.py --mode final --task tdgl --model uno --seed 0 --lr 0.001 --wrap   # -> tdgl_uno_lr0.001_recipe_s0
python -u bench/run.py --mode final --task tdgl --model uno --seed 1 --lr 0.001 --wrap   # -> tdgl_uno_lr0.001_recipe_s1
python -u bench/run.py --mode final --task tdgl --model uno --seed 2 --lr 0.001 --wrap   # -> tdgl_uno_lr0.001_recipe_s2
python -u bench/run.py --mode final --task tdgl --model cnext --seed 0 --lr 0.0003 --wrap   # -> tdgl_cnext_lr0.0003_recipe_s0
python -u bench/run.py --mode final --task tdgl --model cnext --seed 1 --lr 0.0003 --wrap   # -> tdgl_cnext_lr0.0003_recipe_s1
python -u bench/run.py --mode final --task tdgl --model cnext --seed 2 --lr 0.0003 --wrap   # -> tdgl_cnext_lr0.0003_recipe_s2
python -u bench/run.py --mode final --task tdgl --model cno --seed 0 --lr 0.001 --wrap   # -> tdgl_cno_lr0.001_recipe_s0
python -u bench/run.py --mode final --task tdgl --model cno --seed 1 --lr 0.001 --wrap   # -> tdgl_cno_lr0.001_recipe_s1
python -u bench/run.py --mode final --task tdgl --model cno --seed 2 --lr 0.001 --wrap   # -> tdgl_cno_lr0.001_recipe_s2
python -u bench/run.py --mode final --task tdgl --model tfno --seed 0 --lr 0.001 --wrap   # -> tdgl_tfno_lr0.001_recipe_s0
python -u bench/run.py --mode final --task tdgl --model tfno --seed 1 --lr 0.001 --wrap   # -> tdgl_tfno_lr0.001_recipe_s1
python -u bench/run.py --mode final --task tdgl --model tfno --seed 2 --lr 0.001 --wrap   # -> tdgl_tfno_lr0.001_recipe_s2
python -u bench/run.py --mode final --task tdgl --model fno --seed 0 --lr 0.001 --wrap   # -> tdgl_fno_lr0.001_recipe_s0
python -u bench/run.py --mode final --task tdgl --model fno --seed 1 --lr 0.001 --wrap   # -> tdgl_fno_lr0.001_recipe_s1
python -u bench/run.py --mode final --task tdgl --model fno --seed 2 --lr 0.001 --wrap   # -> tdgl_fno_lr0.001_recipe_s2
# Poseidon-T needs the Poseidon source and weights in $SCRATCH/poseidon_src (see README, optional)
python -u bench/run.py --mode final --task tdgl --model poseidon --seed 0 --lr 0.0001 --loss rel   # -> tdgl_poseidon_lr0.0001_rel_s0

# 4c. Allen-Cahn (Figure 3c,d and Section 5.4)
python -u bench/run.py --mode final --task ac --model duhamel_uno_kmu --seed 0 --lr 0.001 --wrap   # -> ac_duhamel_uno_kmu_lr0.001_recipe_s0
python -u bench/run.py --mode final --task ac --model duhamel_uno_kmu --seed 1 --lr 0.001 --wrap   # -> ac_duhamel_uno_kmu_lr0.001_recipe_s1
python -u bench/run.py --mode final --task ac --model duhamel_uno_kmu --seed 2 --lr 0.001 --wrap   # -> ac_duhamel_uno_kmu_lr0.001_recipe_s2
python -u bench/run.py --mode final --task ac --model ffno --seed 0 --lr 0.001 --wrap   # -> ac_ffno_lr0.001_recipe_s0
python -u bench/run.py --mode final --task ac --model ffno --seed 1 --lr 0.001 --wrap   # -> ac_ffno_lr0.001_recipe_s1
python -u bench/run.py --mode final --task ac --model ffno --seed 2 --lr 0.001 --wrap   # -> ac_ffno_lr0.001_recipe_s2
python -u bench/run.py --mode final --task ac --model unet --seed 0 --lr 0.001 --wrap   # -> ac_unet_lr0.001_recipe_s0
python -u bench/run.py --mode final --task ac --model uno --seed 0 --lr 0.001 --wrap   # -> ac_uno_lr0.001_recipe_s0
python -u bench/run.py --mode final --task ac --model uno --seed 1 --lr 0.001 --wrap   # -> ac_uno_lr0.001_recipe_s1
python -u bench/run.py --mode final --task ac --model uno --seed 2 --lr 0.001 --wrap   # -> ac_uno_lr0.001_recipe_s2
python -u bench/run.py --mode final --task ac --model cnext --seed 0 --lr 0.0003 --wrap   # -> ac_cnext_lr0.0003_recipe_s0
python -u bench/run.py --mode final --task ac --model cnext --seed 1 --lr 0.0003 --wrap   # -> ac_cnext_lr0.0003_recipe_s1
python -u bench/run.py --mode final --task ac --model cnext --seed 2 --lr 0.0003 --wrap   # -> ac_cnext_lr0.0003_recipe_s2
python -u bench/run.py --mode final --task ac --model cno --seed 0 --lr 0.001 --wrap   # -> ac_cno_lr0.001_recipe_s0
python -u bench/run.py --mode final --task ac --model cno --seed 1 --lr 0.001 --wrap   # -> ac_cno_lr0.001_recipe_s1
python -u bench/run.py --mode final --task ac --model cno --seed 2 --lr 0.001 --wrap   # -> ac_cno_lr0.001_recipe_s2
python -u bench/run.py --mode final --task ac --model tfno --seed 0 --lr 0.001 --wrap   # -> ac_tfno_lr0.001_recipe_s0
python -u bench/run.py --mode final --task ac --model tfno --seed 1 --lr 0.001 --wrap   # -> ac_tfno_lr0.001_recipe_s1
python -u bench/run.py --mode final --task ac --model tfno --seed 2 --lr 0.001 --wrap   # -> ac_tfno_lr0.001_recipe_s2
python -u bench/run.py --mode final --task ac --model fno --seed 0 --lr 0.001 --wrap   # -> ac_fno_lr0.001_recipe_s0
python -u bench/run.py --mode final --task ac --model fno --seed 1 --lr 0.001 --wrap   # -> ac_fno_lr0.001_recipe_s1
python -u bench/run.py --mode final --task ac --model fno --seed 2 --lr 0.001 --wrap   # -> ac_fno_lr0.001_recipe_s2

# 4d. data efficiency (Limitations: 32 training materials)
BENCH_TAG=_K128 TDGL_SUBSET=128 python -u bench/fit_alpha_bank.py --task tdgl
BENCH_TAG=_K32 TDGL_SUBSET=32 python -u bench/fit_alpha_bank.py --task tdgl
BENCH_TAG=_K128 TDGL_SUBSET=128 BENCH_BANK=$SCRATCH/bench_cache/alpha_bank_tdgl_K128.npz python -u bench/run.py --mode final --task tdgl --model duhamel_uno --seed 0 --lr 0.0003 --wrap   # -> tdgl_duhamel_uno_lr0.0003_recipe_s0_K128
BENCH_TAG=_K32 TDGL_SUBSET=32 BENCH_BANK=$SCRATCH/bench_cache/alpha_bank_tdgl_K32.npz python -u bench/run.py --mode final --task tdgl --model duhamel_uno --seed 0 --lr 0.0003 --wrap   # -> tdgl_duhamel_uno_lr0.0003_recipe_s0_K32
BENCH_TAG=_K128 TDGL_SUBSET=128 BENCH_BANK=$SCRATCH/bench_cache/alpha_bank_tdgl_K128.npz python -u bench/run.py --mode final --task tdgl --model duhamel_uno --seed 1 --lr 0.0003 --wrap   # -> tdgl_duhamel_uno_lr0.0003_recipe_s1_K128
BENCH_TAG=_K32 TDGL_SUBSET=32 BENCH_BANK=$SCRATCH/bench_cache/alpha_bank_tdgl_K32.npz python -u bench/run.py --mode final --task tdgl --model duhamel_uno --seed 1 --lr 0.0003 --wrap   # -> tdgl_duhamel_uno_lr0.0003_recipe_s1_K32
BENCH_TAG=_K128 TDGL_SUBSET=128 python -u bench/run.py --mode final --task tdgl --model ffno --seed 0 --lr 0.001 --wrap   # -> tdgl_ffno_lr0.001_recipe_s0_K128
BENCH_TAG=_K32 TDGL_SUBSET=32 python -u bench/run.py --mode final --task tdgl --model ffno --seed 0 --lr 0.001 --wrap   # -> tdgl_ffno_lr0.001_recipe_s0_K32
BENCH_TAG=_K128 TDGL_SUBSET=128 python -u bench/run.py --mode final --task tdgl --model ffno --seed 1 --lr 0.001 --wrap   # -> tdgl_ffno_lr0.001_recipe_s1_K128
BENCH_TAG=_K32 TDGL_SUBSET=32 python -u bench/run.py --mode final --task tdgl --model ffno --seed 1 --lr 0.001 --wrap   # -> tdgl_ffno_lr0.001_recipe_s1_K32
BENCH_TAG=_K128 TDGL_SUBSET=128 python -u bench/run.py --mode final --task tdgl --model fno --seed 0 --lr 0.001 --wrap   # -> tdgl_fno_lr0.001_recipe_s0_K128
BENCH_TAG=_K32 TDGL_SUBSET=32 python -u bench/run.py --mode final --task tdgl --model fno --seed 0 --lr 0.001 --wrap   # -> tdgl_fno_lr0.001_recipe_s0_K32
BENCH_TAG=_K128 TDGL_SUBSET=128 python -u bench/run.py --mode final --task tdgl --model fno --seed 1 --lr 0.001 --wrap   # -> tdgl_fno_lr0.001_recipe_s1_K128
BENCH_TAG=_K32 TDGL_SUBSET=32 python -u bench/run.py --mode final --task tdgl --model fno --seed 1 --lr 0.001 --wrap   # -> tdgl_fno_lr0.001_recipe_s1_K32
