#!/bin/bash
# scripts/make_tables.sh -- regenerate every number, both tables and Figure 3 of the paper from the result files in
# bench/results_*/, then compare them with the published values. No GPU, no data download; about 2 minutes.
set -e
cd "$(dirname "$0")/.."
echo "[1/3] shared tables and numbers (paper_tools/)"
(cd paper_tools && python make_figs.py > ../make_tables.log 2>&1 && python make_units.py >> ../make_tables.log 2>&1)
echo "[2/3] paper numbers, Table 1, Table 2, Figure 3 (paper/)"
(cd paper && python make_duhamel_macros.py >> ../make_tables.log 2>&1 \
          && python make_physics_table.py >> ../make_tables.log 2>&1 \
          && python make_evt_fig.py >> ../make_tables.log 2>&1)
echo "      wrote paper/numbers_duhamel.tex, paper/tables/tdgl_compact.tex (Table 1),"
echo "            paper/tables/app_physics_full.tex (Table 2), paper/figures/06_duhamel/error_vs_time.pdf (Figure 3)"
echo "[3/3] compare with the paper"
python scripts/check_against_paper.py
