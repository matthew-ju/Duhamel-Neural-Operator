#!/bin/bash
# data_generation/hysteresis/run_ferrox_sweep.sh JOBFILE OUTDIR -- simulate the hysteresis loops with FerroX (optional).
#   JOBFILE lines: NAME ALPHA BETA GAMMA DT EPS_DE V_MAX SEED   (jobs_hysteresis.txt: 164 loops, jobs_hysteresis_closed.txt: 10)
# Each loop is a quasi-static voltage sweep 0 -> +Vmax -> -Vmax -> +Vmax in steps of 0.125 V from the unpoled film; a new step
# starts once the potential solve converges (phi_tolerance 1e-4). Run inside a SLURM allocation with 4 GPUs; four workers
# take loops from the job list until it is empty. Each loop takes 20-120 minutes on one A100.
# Output: OUTDIR/<NAME>.npz with P (64, 64, 18, T), step, V (scripts/link_hf_data.sh shows where the code expects them).
set -u
JOBFILE=$1; W=$2
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
: "${FERROX_EXE:?set FERROX_EXE to the FerroX executable (main3d.*.CUDA.ex)}"
TPL=$REPO/data_generation/hysteresis/inputs_sweep_template
PY=${PYTHON:-python}
mkdir -p "$W/runs" "$W/claims"
one() {
  local N=$1 AL=$2 BE=$3 GA=$4 DT=$5 EPS=$6 VM=$7 SD=$8
  [ -f "$W/$N.npz" ] && return 0
  mkdir "$W/claims/$N" 2>/dev/null || return 0
  local RD=$W/runs/$N; rm -rf "$RD"; mkdir -p "$RD"
  local NV=$(python3 -c "print(int(round($VM/0.125))*5+1)")
  $PY "$REPO/data_generation/ferrox_param_sweep.py" gen-inputs --template "$TPL" --out "$RD/inputs" --set alpha=$AL --set beta=$BE \
     --set gamma=$GA --set dt=$DT --set random_seed=$SD --set epsilon_de=$EPS --set voltage_sweep=1 --set Phi_Bc_hi=0.0 \
     --set Phi_Bc_hi_max=$VM --set phi_tolerance=1e-4 --set Phi_Bc_inc=0.125 --set num_Vapp_max=$NV --set inc_step=1500 \
     --set nsteps=90000 --set plot_int=20 > "$RD/gen.log" 2>&1 || { rm -rf "$W/claims/$N"; return 1; }
  ( cd "$RD" && timeout 7800 srun --exact -n1 --gpus=1 --cpus-per-task=16 --cpu-bind=none "$FERROX_EXE" inputs > run.log 2>&1 ) \
     || { echo "[FAIL solve] $N" >> "$W/run.log"; rm -rf "$W/claims/$N"; return 1; }
  $PY "$REPO/data_generation/hysteresis/extract_sweep.py" "$RD" "$W/$N.npz" >> "$W/run.log" 2>&1 && rm -rf "$RD"/plt* \
     && echo "[ok] $N" >> "$W/run.log"
}
mapfile -t JOBS < <(grep -v '^#' "$JOBFILE" | grep -v '^\s*$')
worker() { for l in "${JOBS[@]}"; do read -r N AL BE GA DT EPS VM SD <<< "$l"; one $N $AL $BE $GA $DT $EPS $VM $SD; done; }
for g in 0 1 2 3; do worker & sleep 3; done
wait
echo "done: $(ls "$W"/*.npz 2>/dev/null | wc -l) loops in $W" | tee -a "$W/run.log"
