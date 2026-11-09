#!/bin/bash
# scripts/link_hf_data.sh HF_DIR -- point the code at the benchmark downloaded from Hugging Face.
#   HF_DIR: the folder from `hf download ... --local-dir HF_DIR` (contains d1/, d2/, hysteresis/).
# Creates symlinks in the layout of docs/REPRODUCE.md step 1, so no data are copied. The loaders read the released
# (T, 10, 64, 64) arrays directly (duhamel_no/geo_data.py, abg_data.py, hyst_data.py).
set -euo pipefail
HF=$(cd "$1" && pwd)
: "${SCRATCH:?set SCRATCH (docs/REPRODUCE.md, step 1)}"
link() { mkdir -p "$2"; for f in "$1"/*; do ln -sfn "$f" "$2/$(basename "$f")"; done; echo "$2: $(ls "$2" | wc -l) files"; }
link "$HF/d1/data" "$SCRATCH/ferrox_abg_useed/extracted"
link "$HF/d2/data" "$SCRATCH/ferrox_d2useed/extracted"
if [[ -d "$HF/hysteresis/data" ]]; then                     # hysteresis: h_* loops, and the extra closed loops hc_* (read as h_*)
  mkdir -p "$SCRATCH/ferrox_hyst_closed2" "$SCRATCH/ferrox_hyst_closed/as_h"
  for f in "$HF"/hysteresis/data/h_*.npz; do ln -sfn "$f" "$SCRATCH/ferrox_hyst_closed2/$(basename "$f")"; done
  for f in "$HF"/hysteresis/data/hc_*.npz; do b=$(basename "$f"); ln -sfn "$f" "$SCRATCH/ferrox_hyst_closed/as_h/h_${b#hc_}"; done
  echo "$SCRATCH/ferrox_hyst_closed2: $(ls "$SCRATCH/ferrox_hyst_closed2" | wc -l) loops, closed set: $(ls "$SCRATCH/ferrox_hyst_closed/as_h" | wc -l)"
fi
