#!/bin/bash
# scripts/make_allen_cahn_data.sh -- generate the Allen-Cahn data (Section 3) into $SCRATCH/ml_units/.
# Seeded, so the result is the same every time. One GPU, a few minutes.
set -e
cd "$(dirname "$0")/.."
: "${SCRATCH:?set SCRATCH to your data directory (see README step 1)}"
python ml_units/gen_ac_ch.py --system ac                                     # 64^2: train, in range, new units, new physics
python ml_units/gen_ac_ch.py --system ac --n 32  --splits id,ood_units,ood_inv  # zero-shot 32^2 grid
python ml_units/gen_ac_ch.py --system ac --n 128 --splits id,ood_units,ood_inv  # zero-shot 128^2 grid
ls -la "$SCRATCH/ml_units/" | grep -E " ac(32|128)?_"
