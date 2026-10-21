#!/bin/bash
# scripts/link_hf_data.sh HF_DIR -- point the code at the benchmark downloaded from Hugging Face.
#   HF_DIR: the folder from `huggingface-cli download ... --local-dir HF_DIR` (contains d1/, d2/, hysteresis/).
# Creates symlinks in the layout of README step 1, so no data are copied. The loaders read the released
# (T, 10, 64, 64) arrays directly (geo/geo_data.py frames, abg_data.py frame).
set -euo pipefail
HF=$(cd "$1" && pwd)
: "${SCRATCH:?set SCRATCH (README step 1)}"
link() { mkdir -p "$2"; for f in "$1"/*; do ln -sfn "$f" "$2/$(basename "$f")"; done; echo "$2: $(ls "$2" | wc -l) files"; }
link "$HF/d1/data" "$SCRATCH/ferrox_abg_useed/extracted"
link "$HF/d2/data" "$SCRATCH/ferrox_d2useed/extracted"
[[ -d "$HF/hysteresis/data" ]] && link "$HF/hysteresis/data" "$SCRATCH/ferrox_hyst_benchmark"
