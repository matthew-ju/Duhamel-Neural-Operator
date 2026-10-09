#!/bin/bash
# scripts/make_figures_from_data.sh -- Figures 1 and 2, which show actual fields and therefore need the TDGL data, the
# Fourier SSM file and (Figure 2) the Duhamel-NO checkpoint. One GPU, a few minutes.
set -e
cd "$(dirname "$0")/.."
echo "[1/2] Figure 2: Duhamel-NO rolled out on two test materials"
(cd paper && python make_rollout_fig.py --skip B_units --out rollouts_multi_2row)
echo "      wrote paper/figures/06_duhamel/rollouts_multi_2row.pdf"
echo "[2/2] Figure 1: field thumbnails + TikZ diagram"
(cd paper && python make_fig1_thumbs.py)
(cd paper/figures/06_duhamel/fig1 && pdflatex -interaction=nonstopmode schematic_tikz.tex > /dev/null)
echo "      wrote paper/figures/06_duhamel/fig1/schematic_tikz.pdf"
