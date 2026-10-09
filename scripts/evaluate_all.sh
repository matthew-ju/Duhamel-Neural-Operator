#!/bin/bash
set -e
cd "$(dirname "$0")/.."            # repository root

# Step 3/5 of README: evaluate trained checkpoints beyond the scores written at training time.

# 5a. physical quantities + per-frame errors, TDGL (Table 2, Figure 3a,b): bench/results_phys/<tag>.json
python -u bench/physmetrics.py --model duhamel_uno --seed 0 --lr 0.0003 --wrap
python -u bench/physmetrics.py --model duhamel_uno --seed 1 --lr 0.0003 --wrap
python -u bench/physmetrics.py --model duhamel_uno --seed 2 --lr 0.0003 --wrap
python -u bench/physmetrics.py --model ffno --seed 0 --lr 0.001 --wrap
python -u bench/physmetrics.py --model ffno --seed 1 --lr 0.001 --wrap
python -u bench/physmetrics.py --model ffno --seed 2 --lr 0.001 --wrap
python -u bench/physmetrics.py --model unet --seed 0 --lr 0.0003 --wrap
python -u bench/physmetrics.py --model uno --seed 0 --lr 0.001 --wrap
python -u bench/physmetrics.py --model uno --seed 1 --lr 0.001 --wrap
python -u bench/physmetrics.py --model uno --seed 2 --lr 0.001 --wrap
python -u bench/physmetrics.py --model cnext --seed 0 --lr 0.0003 --wrap
python -u bench/physmetrics.py --model cnext --seed 1 --lr 0.0003 --wrap
python -u bench/physmetrics.py --model cnext --seed 2 --lr 0.0003 --wrap
python -u bench/physmetrics.py --model cno --seed 0 --lr 0.001 --wrap
python -u bench/physmetrics.py --model cno --seed 1 --lr 0.001 --wrap
python -u bench/physmetrics.py --model cno --seed 2 --lr 0.001 --wrap
python -u bench/physmetrics.py --model tfno --seed 0 --lr 0.001 --wrap
python -u bench/physmetrics.py --model tfno --seed 1 --lr 0.001 --wrap
python -u bench/physmetrics.py --model tfno --seed 2 --lr 0.001 --wrap
python -u bench/physmetrics.py --model fno --seed 0 --lr 0.001 --wrap
python -u bench/physmetrics.py --model fno --seed 1 --lr 0.001 --wrap
python -u bench/physmetrics.py --model fno --seed 2 --lr 0.001 --wrap

# 5b. Allen-Cahn zero-shot grids 32^2 / 128^2: bench/results_res/<tag>.json
python -u bench/restransfer.py --model duhamel_uno_kmu --seed 0 --lr 0.001 --wrap
python -u bench/restransfer.py --model duhamel_uno_kmu --seed 1 --lr 0.001 --wrap
python -u bench/restransfer.py --model duhamel_uno_kmu --seed 2 --lr 0.001 --wrap
python -u bench/restransfer.py --model ffno --seed 0 --lr 0.001 --wrap
python -u bench/restransfer.py --model ffno --seed 1 --lr 0.001 --wrap
python -u bench/restransfer.py --model ffno --seed 2 --lr 0.001 --wrap
python -u bench/restransfer.py --model unet --seed 0 --lr 0.001 --wrap
python -u bench/restransfer.py --model uno --seed 0 --lr 0.001 --wrap
python -u bench/restransfer.py --model uno --seed 1 --lr 0.001 --wrap
python -u bench/restransfer.py --model uno --seed 2 --lr 0.001 --wrap
python -u bench/restransfer.py --model cnext --seed 0 --lr 0.0003 --wrap
python -u bench/restransfer.py --model cnext --seed 1 --lr 0.0003 --wrap
python -u bench/restransfer.py --model cnext --seed 2 --lr 0.0003 --wrap
python -u bench/restransfer.py --model cno --seed 0 --lr 0.001 --wrap
python -u bench/restransfer.py --model cno --seed 1 --lr 0.001 --wrap
python -u bench/restransfer.py --model cno --seed 2 --lr 0.001 --wrap
python -u bench/restransfer.py --model tfno --seed 0 --lr 0.001 --wrap
python -u bench/restransfer.py --model tfno --seed 1 --lr 0.001 --wrap
python -u bench/restransfer.py --model tfno --seed 2 --lr 0.001 --wrap
python -u bench/restransfer.py --model fno --seed 0 --lr 0.001 --wrap
python -u bench/restransfer.py --model fno --seed 1 --lr 0.001 --wrap
python -u bench/restransfer.py --model fno --seed 2 --lr 0.001 --wrap

# 5c. Allen-Cahn per-frame curves (Figure 3c,d): bench/results_curves/<tag>.json
python -u bench/curves.py --task ac

# 5d. speed (Section 5.3; run alone on ONE dedicated GPU, otherwise timings are inflated): bench/results_phys/speed.json
python -u bench/speed.py

