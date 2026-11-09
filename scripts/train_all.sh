#!/bin/bash
set -e
cd "$(dirname "$0")/.."            # repository root

# docs/REPRODUCE.md step 4: train every model behind the paper. One A100 per line, 20-60 min each (Poseidon-T and the
# generative baselines up to ~3 h). After the Fourier SSM fits, the lines are independent: run them in parallel, one per GPU.
# Each line writes $SCRATCH/bench_ckpt/<tag>.pt and results/final/<tag>.json. Lines marked "candidate" are the other
# configurations that duhamel_no/summarize2.py compared on the validation split; the paper reports the selected ones.

# Fourier SSMs measured from the training simulations (minutes)
python -u duhamel_no/fit_alpha_bank.py --task tdgl
python -u duhamel_no/fit_alpha_bank.py --task hyst
BENCH_TAG=_K128 TDGL_SUBSET=128 python -u duhamel_no/fit_alpha_bank.py --task tdgl
BENCH_TAG=_K32 TDGL_SUBSET=32 python -u duhamel_no/fit_alpha_bank.py --task tdgl

# FerroX TDGL, D1/D2: Duhamel-NO, the baselines and the ablation variants (Table 1, Figure 5, ablation table, appendix)
python -u duhamel_no/run.py --mode final --task tdgl --model duhamel_uno --seed 0 --lr 0.0003 --wrap   # -> tdgl_duhamel_uno_lr0.0003_recipe_s0
python -u duhamel_no/run.py --mode final --task tdgl --model duhamel_uno --seed 1 --lr 0.0003 --wrap   # -> tdgl_duhamel_uno_lr0.0003_recipe_s1
python -u duhamel_no/run.py --mode final --task tdgl --model duhamel_uno --seed 2 --lr 0.0003 --wrap   # -> tdgl_duhamel_uno_lr0.0003_recipe_s2
python -u duhamel_no/run.py --mode final --task tdgl --model duhamel_uno --seed 0 --lr 0.001 --wrap   # -> tdgl_duhamel_uno_lr0.001_recipe_s0  (candidate, not selected)
python -u duhamel_no/run.py --mode final --task tdgl --model duhamel_uno --seed 1 --lr 0.001 --wrap   # -> tdgl_duhamel_uno_lr0.001_recipe_s1  (candidate, not selected)
python -u duhamel_no/run.py --mode final --task tdgl --model duhamel_uno --seed 2 --lr 0.001 --wrap   # -> tdgl_duhamel_uno_lr0.001_recipe_s2  (candidate, not selected)
python -u duhamel_no/run.py --mode final --task tdgl --model ffno --seed 0 --lr 0.001 --wrap   # -> tdgl_ffno_lr0.001_recipe_s0
python -u duhamel_no/run.py --mode final --task tdgl --model ffno --seed 1 --lr 0.001 --wrap   # -> tdgl_ffno_lr0.001_recipe_s1
python -u duhamel_no/run.py --mode final --task tdgl --model ffno --seed 2 --lr 0.001 --wrap   # -> tdgl_ffno_lr0.001_recipe_s2
python -u duhamel_no/run.py --mode final --task tdgl --model unet --seed 0 --lr 0.0003 --wrap   # -> tdgl_unet_lr0.0003_recipe_s0
python -u duhamel_no/run.py --mode final --task tdgl --model unet --seed 0 --lr 0.001 --wrap   # -> tdgl_unet_lr0.001_recipe_s0  (candidate, not selected)
python -u duhamel_no/run.py --mode final --task tdgl --model unet --seed 1 --lr 0.001 --wrap   # -> tdgl_unet_lr0.001_recipe_s1  (candidate, not selected)
python -u duhamel_no/run.py --mode final --task tdgl --model unet --seed 2 --lr 0.001 --wrap   # -> tdgl_unet_lr0.001_recipe_s2  (candidate, not selected)
python -u duhamel_no/run.py --mode final --task tdgl --model uno --seed 0 --lr 0.001 --wrap   # -> tdgl_uno_lr0.001_recipe_s0
python -u duhamel_no/run.py --mode final --task tdgl --model uno --seed 1 --lr 0.001 --wrap   # -> tdgl_uno_lr0.001_recipe_s1
python -u duhamel_no/run.py --mode final --task tdgl --model uno --seed 2 --lr 0.001 --wrap   # -> tdgl_uno_lr0.001_recipe_s2
python -u duhamel_no/run.py --mode final --task tdgl --model cnext --seed 0 --lr 0.0003 --wrap   # -> tdgl_cnext_lr0.0003_recipe_s0
python -u duhamel_no/run.py --mode final --task tdgl --model cnext --seed 1 --lr 0.0003 --wrap   # -> tdgl_cnext_lr0.0003_recipe_s1
python -u duhamel_no/run.py --mode final --task tdgl --model cnext --seed 2 --lr 0.0003 --wrap   # -> tdgl_cnext_lr0.0003_recipe_s2
python -u duhamel_no/run.py --mode final --task tdgl --model cnext --seed 0 --lr 0.001 --wrap   # -> tdgl_cnext_lr0.001_recipe_s0  (candidate, not selected)
python -u duhamel_no/run.py --mode final --task tdgl --model cno --seed 0 --lr 0.0003 --wrap   # -> tdgl_cno_lr0.0003_recipe_s0  (candidate, not selected)
python -u duhamel_no/run.py --mode final --task tdgl --model cno --seed 0 --lr 0.001 --wrap   # -> tdgl_cno_lr0.001_recipe_s0
python -u duhamel_no/run.py --mode final --task tdgl --model cno --seed 1 --lr 0.001 --wrap   # -> tdgl_cno_lr0.001_recipe_s1
python -u duhamel_no/run.py --mode final --task tdgl --model cno --seed 2 --lr 0.001 --wrap   # -> tdgl_cno_lr0.001_recipe_s2
python -u duhamel_no/run.py --mode final --task tdgl --model tfno --seed 0 --lr 0.001 --wrap   # -> tdgl_tfno_lr0.001_recipe_s0
python -u duhamel_no/run.py --mode final --task tdgl --model tfno --seed 1 --lr 0.001 --wrap   # -> tdgl_tfno_lr0.001_recipe_s1
python -u duhamel_no/run.py --mode final --task tdgl --model tfno --seed 2 --lr 0.001 --wrap   # -> tdgl_tfno_lr0.001_recipe_s2
python -u duhamel_no/run.py --mode final --task tdgl --model fno --seed 0 --lr 0.001 --wrap   # -> tdgl_fno_lr0.001_recipe_s0
python -u duhamel_no/run.py --mode final --task tdgl --model fno --seed 1 --lr 0.001 --wrap   # -> tdgl_fno_lr0.001_recipe_s1
python -u duhamel_no/run.py --mode final --task tdgl --model fno --seed 2 --lr 0.001 --wrap   # -> tdgl_fno_lr0.001_recipe_s2
# Poseidon-T needs the Poseidon code and weights in $SCRATCH/poseidon_src (optional)
python -u duhamel_no/run.py --mode final --task tdgl --model poseidon --seed 0 --lr 0.0001 --loss rel --steps 10000   # -> tdgl_poseidon_lr0.0001_rel_s0
python -u duhamel_no/run.py --mode final --task tdgl --model acdm --seed 0 --lr 0.001 --scale global   # -> tdgl_acdm_lr0.001_scale-global_s0
python -u duhamel_no/run.py --mode final --task tdgl --model flowmatch --seed 0 --lr 0.001 --scale global   # -> tdgl_flowmatch_lr0.001_scale-global_s0
python -u duhamel_no/run.py --mode final --task tdgl --model refiner2 --seed 0 --lr 0.001 --scale global   # -> tdgl_refiner2_lr0.001_scale-global_s0
python -u duhamel_no/run.py --mode final --task tdgl --model transolver --seed 0 --lr 0.001 --wrap   # -> tdgl_transolver_lr0.001_recipe_s0
python -u duhamel_no/run.py --mode final --task tdgl --model duhamel_uno_rou2 --seed 0 --lr 0.0003 --wrap   # -> tdgl_duhamel_uno_rou2_lr0.0003_recipe_s0  (candidate, not selected)
python -u duhamel_no/run.py --mode final --task tdgl --model duhamel_uno_rou2 --seed 1 --lr 0.0003 --wrap   # -> tdgl_duhamel_uno_rou2_lr0.0003_recipe_s1  (candidate, not selected)
python -u duhamel_no/run.py --mode final --task tdgl --model duhamel_uno_rou2 --seed 2 --lr 0.0003 --wrap   # -> tdgl_duhamel_uno_rou2_lr0.0003_recipe_s2  (candidate, not selected)
python -u duhamel_no/run.py --mode final --task tdgl --model duhamel_uno_rou2 --seed 0 --lr 0.001 --wrap   # -> tdgl_duhamel_uno_rou2_lr0.001_recipe_s0
python -u duhamel_no/run.py --mode final --task tdgl --model duhamel_uno_rou2 --seed 1 --lr 0.001 --wrap   # -> tdgl_duhamel_uno_rou2_lr0.001_recipe_s1
python -u duhamel_no/run.py --mode final --task tdgl --model duhamel_uno_rou2 --seed 2 --lr 0.001 --wrap   # -> tdgl_duhamel_uno_rou2_lr0.001_recipe_s2
python -u duhamel_no/run.py --mode final --task tdgl --model duhamel_uno_poly --seed 0 --lr 0.0003 --wrap   # -> tdgl_duhamel_uno_poly_lr0.0003_recipe_s0
python -u duhamel_no/run.py --mode final --task tdgl --model duhamel_uno_poly --seed 1 --lr 0.0003 --wrap   # -> tdgl_duhamel_uno_poly_lr0.0003_recipe_s1
python -u duhamel_no/run.py --mode final --task tdgl --model duhamel_uno_poly --seed 2 --lr 0.0003 --wrap   # -> tdgl_duhamel_uno_poly_lr0.0003_recipe_s2
python -u duhamel_no/run.py --mode final --task tdgl --model duhamel_uno_k0 --seed 0 --lr 0.0003 --wrap   # -> tdgl_duhamel_uno_k0_lr0.0003_recipe_s0
python -u duhamel_no/run.py --mode final --task tdgl --model duhamel --seed 0 --lr 0.001 --wrap   # -> tdgl_duhamel_lr0.001_recipe_s0
python -u duhamel_no/run.py --mode final --task tdgl --model duhamel --seed 1 --lr 0.001 --wrap   # -> tdgl_duhamel_lr0.001_recipe_s1
python -u duhamel_no/run.py --mode final --task tdgl --model duhamel --seed 2 --lr 0.001 --wrap   # -> tdgl_duhamel_lr0.001_recipe_s2

# change-of-units ablation (appendix): the same backbones without material units
BENCH_TAG=_nounits BENCH_NOUNITS=1 python -u duhamel_no/run.py --mode final --task tdgl --model uno --seed 0 --lr 0.001 --wrap   # -> tdgl_uno_lr0.001_recipe_s0_nounits  (candidate, not selected)
BENCH_TAG=_nounits BENCH_NOUNITS=1 python -u duhamel_no/run.py --mode final --task tdgl --model uno --seed 1 --lr 0.001 --wrap   # -> tdgl_uno_lr0.001_recipe_s1_nounits  (candidate, not selected)
BENCH_TAG=_nounits BENCH_NOUNITS=1 python -u duhamel_no/run.py --mode final --task tdgl --model uno --seed 2 --lr 0.001 --wrap   # -> tdgl_uno_lr0.001_recipe_s2_nounits  (candidate, not selected)
BENCH_TAG=_nounits BENCH_NOUNITS=1 python -u duhamel_no/run.py --mode final --task tdgl --model fno --seed 0 --lr 0.001 --wrap   # -> tdgl_fno_lr0.001_recipe_s0_nounits  (candidate, not selected)
BENCH_TAG=_nounits BENCH_NOUNITS=1 python -u duhamel_no/run.py --mode final --task tdgl --model fno --seed 1 --lr 0.001 --wrap   # -> tdgl_fno_lr0.001_recipe_s1_nounits  (candidate, not selected)
BENCH_TAG=_nounits BENCH_NOUNITS=1 python -u duhamel_no/run.py --mode final --task tdgl --model fno --seed 2 --lr 0.001 --wrap   # -> tdgl_fno_lr0.001_recipe_s2_nounits  (candidate, not selected)

# data efficiency (appendix): 32 and 128 training materials
BENCH_TAG=_K128 TDGL_SUBSET=128 BENCH_BANK=$SCRATCH/bench_cache/alpha_bank_tdgl_K128.npz python -u duhamel_no/run.py --mode final --task tdgl --model duhamel_uno --seed 0 --lr 0.0003 --wrap   # -> tdgl_duhamel_uno_lr0.0003_recipe_s0_K128  (candidate, not selected)
BENCH_TAG=_K32 TDGL_SUBSET=32 BENCH_BANK=$SCRATCH/bench_cache/alpha_bank_tdgl_K32.npz python -u duhamel_no/run.py --mode final --task tdgl --model duhamel_uno --seed 0 --lr 0.0003 --wrap   # -> tdgl_duhamel_uno_lr0.0003_recipe_s0_K32  (candidate, not selected)
BENCH_TAG=_K128 TDGL_SUBSET=128 BENCH_BANK=$SCRATCH/bench_cache/alpha_bank_tdgl_K128.npz python -u duhamel_no/run.py --mode final --task tdgl --model duhamel_uno --seed 1 --lr 0.0003 --wrap   # -> tdgl_duhamel_uno_lr0.0003_recipe_s1_K128  (candidate, not selected)
BENCH_TAG=_K32 TDGL_SUBSET=32 BENCH_BANK=$SCRATCH/bench_cache/alpha_bank_tdgl_K32.npz python -u duhamel_no/run.py --mode final --task tdgl --model duhamel_uno --seed 1 --lr 0.0003 --wrap   # -> tdgl_duhamel_uno_lr0.0003_recipe_s1_K32  (candidate, not selected)
BENCH_TAG=_K128 TDGL_SUBSET=128 python -u duhamel_no/run.py --mode final --task tdgl --model ffno --seed 0 --lr 0.001 --wrap   # -> tdgl_ffno_lr0.001_recipe_s0_K128  (candidate, not selected)
BENCH_TAG=_K32 TDGL_SUBSET=32 python -u duhamel_no/run.py --mode final --task tdgl --model ffno --seed 0 --lr 0.001 --wrap   # -> tdgl_ffno_lr0.001_recipe_s0_K32  (candidate, not selected)
BENCH_TAG=_K128 TDGL_SUBSET=128 python -u duhamel_no/run.py --mode final --task tdgl --model ffno --seed 1 --lr 0.001 --wrap   # -> tdgl_ffno_lr0.001_recipe_s1_K128  (candidate, not selected)
BENCH_TAG=_K32 TDGL_SUBSET=32 python -u duhamel_no/run.py --mode final --task tdgl --model ffno --seed 1 --lr 0.001 --wrap   # -> tdgl_ffno_lr0.001_recipe_s1_K32  (candidate, not selected)
BENCH_TAG=_K128 TDGL_SUBSET=128 python -u duhamel_no/run.py --mode final --task tdgl --model fno --seed 0 --lr 0.001 --wrap   # -> tdgl_fno_lr0.001_recipe_s0_K128  (candidate, not selected)
BENCH_TAG=_K32 TDGL_SUBSET=32 python -u duhamel_no/run.py --mode final --task tdgl --model fno --seed 0 --lr 0.001 --wrap   # -> tdgl_fno_lr0.001_recipe_s0_K32  (candidate, not selected)
BENCH_TAG=_K128 TDGL_SUBSET=128 python -u duhamel_no/run.py --mode final --task tdgl --model fno --seed 1 --lr 0.001 --wrap   # -> tdgl_fno_lr0.001_recipe_s1_K128  (candidate, not selected)
BENCH_TAG=_K32 TDGL_SUBSET=32 python -u duhamel_no/run.py --mode final --task tdgl --model fno --seed 1 --lr 0.001 --wrap   # -> tdgl_fno_lr0.001_recipe_s1_K32  (candidate, not selected)

# hysteresis loops (appendix)
BENCH_TAG=_closed python -u duhamel_no/run.py --mode final --task hyst --model duhamel_uno --seed 0 --lr 0.001 --wrap   # -> hyst_duhamel_uno_lr0.001_recipe_s0_closed
BENCH_TAG=_closed python -u duhamel_no/run.py --mode final --task hyst --model duhamel_uno --seed 1 --lr 0.001 --wrap   # -> hyst_duhamel_uno_lr0.001_recipe_s1_closed
BENCH_TAG=_closed python -u duhamel_no/run.py --mode final --task hyst --model duhamel_uno --seed 2 --lr 0.001 --wrap   # -> hyst_duhamel_uno_lr0.001_recipe_s2_closed
BENCH_TAG=_closed python -u duhamel_no/run.py --mode final --task hyst --model ffno --seed 0 --lr 0.001 --wrap   # -> hyst_ffno_lr0.001_recipe_s0_closed
BENCH_TAG=_closed python -u duhamel_no/run.py --mode final --task hyst --model unet --seed 0 --lr 0.0003 --loss rel   # -> hyst_unet_lr0.0003_rel_s0_closed
BENCH_TAG=_closed python -u duhamel_no/run.py --mode final --task hyst --model uno --seed 0 --lr 0.001 --wrap   # -> hyst_uno_lr0.001_recipe_s0_closed
BENCH_TAG=_closed python -u duhamel_no/run.py --mode final --task hyst --model cnext --seed 0 --lr 0.0003 --wrap   # -> hyst_cnext_lr0.0003_recipe_s0_closed
BENCH_TAG=_closed python -u duhamel_no/run.py --mode final --task hyst --model cnext --seed 1 --lr 0.0003 --wrap   # -> hyst_cnext_lr0.0003_recipe_s1_closed
BENCH_TAG=_closed python -u duhamel_no/run.py --mode final --task hyst --model cnext --seed 2 --lr 0.0003 --wrap   # -> hyst_cnext_lr0.0003_recipe_s2_closed
BENCH_TAG=_closed python -u duhamel_no/run.py --mode final --task hyst --model cno --seed 0 --lr 0.0003 --wrap   # -> hyst_cno_lr0.0003_recipe_s0_closed
BENCH_TAG=_closed python -u duhamel_no/run.py --mode final --task hyst --model cno --seed 1 --lr 0.0003 --wrap   # -> hyst_cno_lr0.0003_recipe_s1_closed
BENCH_TAG=_closed python -u duhamel_no/run.py --mode final --task hyst --model cno --seed 2 --lr 0.0003 --wrap   # -> hyst_cno_lr0.0003_recipe_s2_closed
BENCH_TAG=_closed python -u duhamel_no/run.py --mode final --task hyst --model tfno --seed 0 --lr 0.001 --loss mse   # -> hyst_tfno_lr0.001_mse_s0_closed
BENCH_TAG=_closed python -u duhamel_no/run.py --mode final --task hyst --model fno --seed 0 --lr 0.001 --wrap   # -> hyst_fno_lr0.001_recipe_s0_closed

# Allen-Cahn (appendix)
python -u duhamel_no/run.py --mode final --task ac --model duhamel_uno_kmu --seed 0 --lr 0.0003 --wrap   # -> ac_duhamel_uno_kmu_lr0.0003_recipe_s0  (candidate, not selected)
python -u duhamel_no/run.py --mode final --task ac --model duhamel_uno_kmu --seed 1 --lr 0.0003 --wrap   # -> ac_duhamel_uno_kmu_lr0.0003_recipe_s1  (candidate, not selected)
python -u duhamel_no/run.py --mode final --task ac --model duhamel_uno_kmu --seed 2 --lr 0.0003 --wrap   # -> ac_duhamel_uno_kmu_lr0.0003_recipe_s2  (candidate, not selected)
python -u duhamel_no/run.py --mode final --task ac --model duhamel_uno_kmu --seed 0 --lr 0.001 --wrap   # -> ac_duhamel_uno_kmu_lr0.001_recipe_s0
python -u duhamel_no/run.py --mode final --task ac --model duhamel_uno_kmu --seed 1 --lr 0.001 --wrap   # -> ac_duhamel_uno_kmu_lr0.001_recipe_s1
python -u duhamel_no/run.py --mode final --task ac --model duhamel_uno_kmu --seed 2 --lr 0.001 --wrap   # -> ac_duhamel_uno_kmu_lr0.001_recipe_s2
python -u duhamel_no/run.py --mode final --task ac --model duhamel_uno_kmu_rou2 --seed 0 --lr 0.0003 --wrap   # -> ac_duhamel_uno_kmu_rou2_lr0.0003_recipe_s0
python -u duhamel_no/run.py --mode final --task ac --model duhamel_uno_kmu_rou2 --seed 1 --lr 0.0003 --wrap   # -> ac_duhamel_uno_kmu_rou2_lr0.0003_recipe_s1
python -u duhamel_no/run.py --mode final --task ac --model duhamel_uno_kmu_rou2 --seed 2 --lr 0.0003 --wrap   # -> ac_duhamel_uno_kmu_rou2_lr0.0003_recipe_s2
python -u duhamel_no/run.py --mode final --task ac --model duhamel_uno_kmu_rou2 --seed 0 --lr 0.001 --wrap   # -> ac_duhamel_uno_kmu_rou2_lr0.001_recipe_s0  (candidate, not selected)
python -u duhamel_no/run.py --mode final --task ac --model duhamel_uno_kmu_rou2 --seed 1 --lr 0.001 --wrap   # -> ac_duhamel_uno_kmu_rou2_lr0.001_recipe_s1  (candidate, not selected)
python -u duhamel_no/run.py --mode final --task ac --model duhamel_uno_kmu_rou2 --seed 2 --lr 0.001 --wrap   # -> ac_duhamel_uno_kmu_rou2_lr0.001_recipe_s2  (candidate, not selected)
python -u duhamel_no/run.py --mode final --task ac --model duhamel_uno_kmu_trunc --seed 0 --lr 0.001 --wrap   # -> ac_duhamel_uno_kmu_trunc_lr0.001_recipe_s0
python -u duhamel_no/run.py --mode final --task ac --model duhamel_uno_kmu_trunc --seed 1 --lr 0.001 --wrap   # -> ac_duhamel_uno_kmu_trunc_lr0.001_recipe_s1
python -u duhamel_no/run.py --mode final --task ac --model duhamel_uno_kmu_trunc --seed 2 --lr 0.001 --wrap   # -> ac_duhamel_uno_kmu_trunc_lr0.001_recipe_s2
python -u duhamel_no/run.py --mode final --task ac --model duhamel_uno --seed 0 --lr 0.0003 --wrap   # -> ac_duhamel_uno_lr0.0003_recipe_s0  (candidate, not selected)
python -u duhamel_no/run.py --mode final --task ac --model duhamel_uno --seed 0 --lr 0.001 --wrap   # -> ac_duhamel_uno_lr0.001_recipe_s0
python -u duhamel_no/run.py --mode final --task ac --model duhamel_uno --seed 1 --lr 0.001 --wrap   # -> ac_duhamel_uno_lr0.001_recipe_s1
python -u duhamel_no/run.py --mode final --task ac --model duhamel_uno --seed 2 --lr 0.001 --wrap   # -> ac_duhamel_uno_lr0.001_recipe_s2
python -u duhamel_no/run.py --mode final --task ac --model duhamel_ou2 --seed 0 --lr 0.0003 --wrap   # -> ac_duhamel_ou2_lr0.0003_recipe_s0  (candidate, not selected)
python -u duhamel_no/run.py --mode final --task ac --model duhamel_ou2 --seed 1 --lr 0.0003 --wrap   # -> ac_duhamel_ou2_lr0.0003_recipe_s1  (candidate, not selected)
python -u duhamel_no/run.py --mode final --task ac --model duhamel_ou2 --seed 2 --lr 0.0003 --wrap   # -> ac_duhamel_ou2_lr0.0003_recipe_s2  (candidate, not selected)
python -u duhamel_no/run.py --mode final --task ac --model duhamel_ou2 --seed 0 --lr 0.001 --wrap   # -> ac_duhamel_ou2_lr0.001_recipe_s0
python -u duhamel_no/run.py --mode final --task ac --model duhamel_ou2 --seed 1 --lr 0.001 --wrap   # -> ac_duhamel_ou2_lr0.001_recipe_s1
python -u duhamel_no/run.py --mode final --task ac --model duhamel_ou2 --seed 2 --lr 0.001 --wrap   # -> ac_duhamel_ou2_lr0.001_recipe_s2
python -u duhamel_no/run.py --mode final --task ac --model duhamel_ou2_free --seed 0 --lr 0.001 --wrap   # -> ac_duhamel_ou2_free_lr0.001_recipe_s0
python -u duhamel_no/run.py --mode final --task ac --model duhamel_ou2_free --seed 1 --lr 0.001 --wrap   # -> ac_duhamel_ou2_free_lr0.001_recipe_s1
python -u duhamel_no/run.py --mode final --task ac --model duhamel_ou2_free --seed 2 --lr 0.001 --wrap   # -> ac_duhamel_ou2_free_lr0.001_recipe_s2
python -u duhamel_no/run.py --mode final --task ac --model duhamel_ou2_trunc --seed 0 --lr 0.001 --wrap   # -> ac_duhamel_ou2_trunc_lr0.001_recipe_s0
python -u duhamel_no/run.py --mode final --task ac --model duhamel_ou2_trunc --seed 1 --lr 0.001 --wrap   # -> ac_duhamel_ou2_trunc_lr0.001_recipe_s1
python -u duhamel_no/run.py --mode final --task ac --model duhamel_ou2_trunc --seed 2 --lr 0.001 --wrap   # -> ac_duhamel_ou2_trunc_lr0.001_recipe_s2
python -u duhamel_no/run.py --mode final --task ac --model ffno --seed 0 --lr 0.001 --wrap   # -> ac_ffno_lr0.001_recipe_s0
python -u duhamel_no/run.py --mode final --task ac --model ffno --seed 1 --lr 0.001 --wrap   # -> ac_ffno_lr0.001_recipe_s1
python -u duhamel_no/run.py --mode final --task ac --model ffno --seed 2 --lr 0.001 --wrap   # -> ac_ffno_lr0.001_recipe_s2
python -u duhamel_no/run.py --mode final --task ac --model unet --seed 0 --lr 0.0003 --wrap   # -> ac_unet_lr0.0003_recipe_s0  (candidate, not selected)
python -u duhamel_no/run.py --mode final --task ac --model unet --seed 1 --lr 0.0003 --wrap   # -> ac_unet_lr0.0003_recipe_s1  (candidate, not selected)
python -u duhamel_no/run.py --mode final --task ac --model unet --seed 2 --lr 0.0003 --wrap   # -> ac_unet_lr0.0003_recipe_s2  (candidate, not selected)
python -u duhamel_no/run.py --mode final --task ac --model unet --seed 0 --lr 0.001 --wrap   # -> ac_unet_lr0.001_recipe_s0
python -u duhamel_no/run.py --mode final --task ac --model uno --seed 0 --lr 0.001 --wrap   # -> ac_uno_lr0.001_recipe_s0
python -u duhamel_no/run.py --mode final --task ac --model uno --seed 1 --lr 0.001 --wrap   # -> ac_uno_lr0.001_recipe_s1
python -u duhamel_no/run.py --mode final --task ac --model uno --seed 2 --lr 0.001 --wrap   # -> ac_uno_lr0.001_recipe_s2
python -u duhamel_no/run.py --mode final --task ac --model cnext --seed 0 --lr 0.0003 --wrap   # -> ac_cnext_lr0.0003_recipe_s0
python -u duhamel_no/run.py --mode final --task ac --model cnext --seed 1 --lr 0.0003 --wrap   # -> ac_cnext_lr0.0003_recipe_s1
python -u duhamel_no/run.py --mode final --task ac --model cnext --seed 2 --lr 0.0003 --wrap   # -> ac_cnext_lr0.0003_recipe_s2
python -u duhamel_no/run.py --mode final --task ac --model cnext --seed 0 --lr 0.001 --wrap   # -> ac_cnext_lr0.001_recipe_s0  (candidate, not selected)
python -u duhamel_no/run.py --mode final --task ac --model cnext --seed 1 --lr 0.001 --wrap   # -> ac_cnext_lr0.001_recipe_s1  (candidate, not selected)
python -u duhamel_no/run.py --mode final --task ac --model cnext --seed 2 --lr 0.001 --wrap   # -> ac_cnext_lr0.001_recipe_s2  (candidate, not selected)
python -u duhamel_no/run.py --mode final --task ac --model cno --seed 0 --lr 0.0003 --wrap   # -> ac_cno_lr0.0003_recipe_s0  (candidate, not selected)
python -u duhamel_no/run.py --mode final --task ac --model cno --seed 1 --lr 0.0003 --wrap   # -> ac_cno_lr0.0003_recipe_s1  (candidate, not selected)
python -u duhamel_no/run.py --mode final --task ac --model cno --seed 2 --lr 0.0003 --wrap   # -> ac_cno_lr0.0003_recipe_s2  (candidate, not selected)
python -u duhamel_no/run.py --mode final --task ac --model cno --seed 0 --lr 0.001 --wrap   # -> ac_cno_lr0.001_recipe_s0
python -u duhamel_no/run.py --mode final --task ac --model cno --seed 1 --lr 0.001 --wrap   # -> ac_cno_lr0.001_recipe_s1
python -u duhamel_no/run.py --mode final --task ac --model cno --seed 2 --lr 0.001 --wrap   # -> ac_cno_lr0.001_recipe_s2
python -u duhamel_no/run.py --mode final --task ac --model tfno --seed 0 --lr 0.001 --wrap   # -> ac_tfno_lr0.001_recipe_s0
python -u duhamel_no/run.py --mode final --task ac --model tfno --seed 1 --lr 0.001 --wrap   # -> ac_tfno_lr0.001_recipe_s1
python -u duhamel_no/run.py --mode final --task ac --model tfno --seed 2 --lr 0.001 --wrap   # -> ac_tfno_lr0.001_recipe_s2
python -u duhamel_no/run.py --mode final --task ac --model fno --seed 0 --lr 0.001 --wrap   # -> ac_fno_lr0.001_recipe_s0
python -u duhamel_no/run.py --mode final --task ac --model fno --seed 1 --lr 0.001 --wrap   # -> ac_fno_lr0.001_recipe_s1
python -u duhamel_no/run.py --mode final --task ac --model fno --seed 2 --lr 0.001 --wrap   # -> ac_fno_lr0.001_recipe_s2
