#!/bin/bash
# scripts/make_tables.sh -- regenerate every number and every table of the paper from the result files in results/, then
# compare them with the values the paper prints (reproduce/expected/). No GPU, no data download; under a minute.
# Needs only numpy, scipy and matplotlib.
set -e
cd "$(dirname "$0")/.."
LOG=make_tables.log
echo "[1/3] shared tables and numbers (reproduce/tools/)"
(cd reproduce/tools && python make_figs.py > ../../$LOG 2>&1 && python make_units.py >> ../../$LOG 2>&1 \
                    && python fig_design.py >> ../../$LOG 2>&1)
echo "[2/3] numbers and tables of the paper (reproduce/)"
python reproduce/make_duhamel_macros.py >> $LOG 2>&1
python reproduce/make_physics_table.py >> $LOG 2>&1
python reproduce/make_ac_table.py >> $LOG 2>&1
python reproduce/make_appendix_tables.py >> $LOG 2>&1
echo "      wrote reproduce/numbers_duhamel.tex and reproduce/tables/*.tex"
echo "[3/3] compare with the paper"
python scripts/check_against_paper.py
