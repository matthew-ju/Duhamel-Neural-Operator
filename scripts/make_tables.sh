#!/bin/bash
# scripts/make_tables.sh -- regenerate every number, every table and the result-based figures of the paper from the
# result files in results/, then compare them with the published values. No GPU, no data download; about 2 minutes.
set -e
cd "$(dirname "$0")/.."
LOG=make_tables.log
echo "[1/4] shared tables, numbers and appendix figures (paper/tools/)"
(cd paper/tools && python make_figs.py > ../../$LOG 2>&1 && python make_units.py >> ../../$LOG 2>&1 \
                && python fig_design.py >> ../../$LOG 2>&1)
echo "[2/4] numbers and tables of the paper (paper/)"
python paper/make_duhamel_macros.py >> $LOG 2>&1
python paper/make_physics_table.py >> $LOG 2>&1
python paper/make_ac_table.py >> $LOG 2>&1
python paper/make_appendix_tables.py >> $LOG 2>&1
echo "      wrote paper/numbers_duhamel.tex and paper/tables/*.tex"
echo "[3/4] figures drawn from the result files"
python paper/make_theory_fig.py >> $LOG 2>&1          # Figure 3: truncation cap
python paper/make_evt_ab_fig.py >> $LOG 2>&1          # Figure 5a,b: error at every frame
python paper/make_hyst_curves_fig.py >> $LOG 2>&1     # appendix: hysteresis error curves
echo "      wrote paper/figures/ (tmlr/theory_cap.pdf, 06_duhamel/error_vs_time_ab.pdf, appendix/*)"
echo "[4/4] compare with the paper"
python scripts/check_against_paper.py
