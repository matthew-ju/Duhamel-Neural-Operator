#!/bin/bash
set -e
cd "$(dirname "$0")/.."            # repository root

# docs/REPRODUCE.md step 5: evaluate the trained models beyond the scores written at training time.

# physical quantities, per-frame errors and example rollouts, TDGL (Figure 5, appendix): results/phys/<tag>.json
python -u duhamel_no/physmetrics.py --model duhamel_uno --seed 0 --lr 0.0003 --wrap
python -u duhamel_no/physmetrics.py --model duhamel_uno --seed 1 --lr 0.0003 --wrap
python -u duhamel_no/physmetrics.py --model duhamel_uno --seed 2 --lr 0.0003 --wrap
python -u duhamel_no/physmetrics.py --model ffno --seed 0 --lr 0.001 --wrap
python -u duhamel_no/physmetrics.py --model ffno --seed 1 --lr 0.001 --wrap
python -u duhamel_no/physmetrics.py --model ffno --seed 2 --lr 0.001 --wrap
python -u duhamel_no/physmetrics.py --model unet --seed 0 --lr 0.0003 --wrap
python -u duhamel_no/physmetrics.py --model uno --seed 0 --lr 0.001 --wrap
python -u duhamel_no/physmetrics.py --model uno --seed 1 --lr 0.001 --wrap
python -u duhamel_no/physmetrics.py --model uno --seed 2 --lr 0.001 --wrap
python -u duhamel_no/physmetrics.py --model cnext --seed 0 --lr 0.0003 --wrap
python -u duhamel_no/physmetrics.py --model cnext --seed 1 --lr 0.0003 --wrap
python -u duhamel_no/physmetrics.py --model cnext --seed 2 --lr 0.0003 --wrap
python -u duhamel_no/physmetrics.py --model cno --seed 0 --lr 0.001 --wrap
python -u duhamel_no/physmetrics.py --model cno --seed 1 --lr 0.001 --wrap
python -u duhamel_no/physmetrics.py --model cno --seed 2 --lr 0.001 --wrap
python -u duhamel_no/physmetrics.py --model tfno --seed 0 --lr 0.001 --wrap
python -u duhamel_no/physmetrics.py --model tfno --seed 1 --lr 0.001 --wrap
python -u duhamel_no/physmetrics.py --model tfno --seed 2 --lr 0.001 --wrap
python -u duhamel_no/physmetrics.py --model fno --seed 0 --lr 0.001 --wrap
python -u duhamel_no/physmetrics.py --model fno --seed 1 --lr 0.001 --wrap
python -u duhamel_no/physmetrics.py --model fno --seed 2 --lr 0.001 --wrap
python -u duhamel_no/physmetrics.py --model poseidon --seed 0 --lr 0.0001 --loss rel
python -u duhamel_no/physmetrics.py --model acdm --seed 0 --lr 0.001 --scale global
python -u duhamel_no/physmetrics.py --model flowmatch --seed 0 --lr 0.001 --scale global
python -u duhamel_no/physmetrics.py --model refiner2 --seed 0 --lr 0.001 --scale global
python -u duhamel_no/physmetrics.py --model transolver --seed 0 --lr 0.001 --wrap
python -u duhamel_no/physmetrics.py --model duhamel --seed 0 --lr 0.001 --wrap
python -u duhamel_no/physmetrics.py --model duhamel --seed 1 --lr 0.001 --wrap
python -u duhamel_no/physmetrics.py --model duhamel --seed 2 --lr 0.001 --wrap

# Allen-Cahn zero-shot grids 32^2 / 128^2: results/res/<tag>.json
python -u duhamel_no/restransfer.py --model duhamel_uno_kmu --seed 0 --lr 0.001 --wrap
python -u duhamel_no/restransfer.py --model duhamel_uno_kmu --seed 1 --lr 0.001 --wrap
python -u duhamel_no/restransfer.py --model duhamel_uno_kmu --seed 2 --lr 0.001 --wrap
python -u duhamel_no/restransfer.py --model ffno --seed 0 --lr 0.001 --wrap
python -u duhamel_no/restransfer.py --model ffno --seed 1 --lr 0.001 --wrap
python -u duhamel_no/restransfer.py --model ffno --seed 2 --lr 0.001 --wrap
python -u duhamel_no/restransfer.py --model unet --seed 0 --lr 0.001 --wrap
python -u duhamel_no/restransfer.py --model uno --seed 0 --lr 0.001 --wrap
python -u duhamel_no/restransfer.py --model uno --seed 1 --lr 0.001 --wrap
python -u duhamel_no/restransfer.py --model uno --seed 2 --lr 0.001 --wrap
python -u duhamel_no/restransfer.py --model cnext --seed 0 --lr 0.0003 --wrap
python -u duhamel_no/restransfer.py --model cnext --seed 1 --lr 0.0003 --wrap
python -u duhamel_no/restransfer.py --model cnext --seed 2 --lr 0.0003 --wrap
python -u duhamel_no/restransfer.py --model cno --seed 0 --lr 0.001 --wrap
python -u duhamel_no/restransfer.py --model cno --seed 1 --lr 0.001 --wrap
python -u duhamel_no/restransfer.py --model cno --seed 2 --lr 0.001 --wrap
python -u duhamel_no/restransfer.py --model tfno --seed 0 --lr 0.001 --wrap
python -u duhamel_no/restransfer.py --model tfno --seed 1 --lr 0.001 --wrap
python -u duhamel_no/restransfer.py --model tfno --seed 2 --lr 0.001 --wrap
python -u duhamel_no/restransfer.py --model fno --seed 0 --lr 0.001 --wrap
python -u duhamel_no/restransfer.py --model fno --seed 1 --lr 0.001 --wrap
python -u duhamel_no/restransfer.py --model fno --seed 2 --lr 0.001 --wrap

# per-frame / per-step error curves: results/curves/<tag>.json
python -u duhamel_no/curves.py --task ac
BENCH_TAG=_closed python -u duhamel_no/curves.py --task hyst

# P-V loops of the hysteresis figure: results/phys/hyst_loops_closed.npz
BENCH_TAG=_closed python -u duhamel_no/hystloops.py

# speed (run alone on ONE dedicated GPU, otherwise timings are inflated): results/phys/speed.json
python -u duhamel_no/speed.py
