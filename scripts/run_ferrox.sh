#!/bin/bash
# scripts/run_ferrox.sh JOBFILE OUTDIR -- simulate every material of a job list with FerroX (README step 6, optional).
#   JOBFILE lines: NAME ALPHA BETA GAMMA NSTEPS PLOT_INT DT SEED PHI_HI     (data/jobs_D1.txt, data/jobs_D2.txt)
#   Output:       OUTDIR/extracted/NAME.npy   (the format the training code reads)
# Needs: FERROX_EXE (FerroX 3D CUDA build) and SLURM (srun). Uses 4 GPUs of one node; run it on several nodes at once
# to go faster: workers claim jobs with an atomic mkdir, so they never run the same material twice.
set -uo pipefail
JOBFILE=$1; WORK=$2
REPO="$(cd "$(dirname "$0")/.." && pwd)"
: "${FERROX_EXE:?set FERROX_EXE to the FerroX executable (main3d.*.CUDA.ex)}"
TPL=${FERROX_TEMPLATE:-$REPO/ferrox_inputs/inputs_mfim_Noeb}
PY=${PYTHON:-python}
cd "$REPO"
mkdir -p "$WORK/runs" "$WORK/extracted" "$WORK/claims"
LOG=$WORK/run.log
for C in "$WORK"/claims/*/; do                     # reap claims older than 40 min whose job never finished
  [[ -d "$C" ]] || continue
  B=$(basename "$C")
  [[ -f "$WORK/extracted/$B.npy" ]] && continue
  [[ -n "$(find "$C" -maxdepth 0 -mmin +40 2>/dev/null)" ]] && rm -rf "$C" "$WORK/runs/$B"
done

run_one() {
  local N=$1 AL=$2 BE=$3 GA=$4 NS=$5 PI=$6 DT=$7 SD=$8 V=$9
  local OUT="$WORK/extracted/$N.npy" RD="$WORK/runs/$N"
  [[ -f "$OUT" ]] && return 0
  mkdir "$WORK/claims/$N" 2>/dev/null || return 0
  rm -rf "$RD"; mkdir -p "$RD"
  $PY ferrox_param_sweep.py gen-inputs --template "$TPL" --out "$RD/inputs" \
      --set "alpha=$AL" --set "beta=$BE" --set "gamma=$GA" --set "random_seed=$SD" \
      --set "nsteps=$NS" --set "plot_int=$PI" --set "dt=$DT" --set "Phi_Bc_hi=$V" >> "$LOG" 2>&1 \
      || { echo "[FAIL gen] $N" >> "$LOG"; rm -rf "$WORK/claims/$N"; return 1; }
  ( cd "$RD" && timeout 1500 srun --exact -n1 --gpus=1 --cpus-per-task=16 --cpu-bind=none "$FERROX_EXE" inputs > run.log 2>&1 ) \
      || { echo "[FAIL solve] $N" >> "$LOG"; rm -rf "$WORK/claims/$N"; return 1; }
  $PY ferrox_param_sweep.py extract-run --run-dir "$RD" --out "$OUT" --delete-plotfiles >> "$LOG" 2>&1 \
      && echo "[ok] $N" >> "$LOG"
}

mapfile -t JOBS < <(grep -v '^#' "$JOBFILE" | grep -v '^\s*$')
worker() { local i; for ((i=0; i<${#JOBS[@]}; i++)); do read -r N AL BE GA NS PI DT SD V <<< "${JOBS[$i]}"; run_one "$N" "$AL" "$BE" "$GA" "$NS" "$PI" "$DT" "$SD" "$V"; done; }
for g in 0 1 2 3; do worker & sleep 2; done
wait
echo "done: $(ls "$WORK"/extracted/*.npy 2>/dev/null | wc -l) trajectories in $WORK/extracted" | tee -a "$LOG"
