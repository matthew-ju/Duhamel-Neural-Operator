#!/bin/bash
# scripts/make_figures_from_data.sh -- the figures that show actual fields, and therefore need the downloaded data
# (docs/REPRODUCE.md step 3), the Fourier SSM and trained checkpoints (step 4), and the example rollouts that
# scripts/evaluate_all.sh saves (step 5). One GPU, about ten minutes. Figures 1 and 2 also need LaTeX with TikZ.
#   FERRO_DATA: the Hugging Face download (default ./ferro_data)
set -e
cd "$(dirname "$0")/.."
export FERRO_DATA=${FERRO_DATA:-$PWD/ferro_data}
echo "[1/5] Figure 1: problem setup (one D1 trajectory, TikZ)"
python paper/make_setup_fig.py
echo "[2/5] Figure 2: Duhamel-NO diagram (field thumbnails + TikZ)"
(cd paper && python make_fig1_thumbs.py)
(cd paper/figures/06_duhamel/fig1 && pdflatex -interaction=nonstopmode schematic_tikz.tex > /dev/null)
echo "[3/5] Figure 4: Duhamel-NO rolled out on two test materials"
(cd paper && python make_rollout_fig.py --skip B_units --out rollouts_multi_2row)
echo "[4/5] appendix: free-energy landscapes of those materials"
python paper/make_energy_fig.py
echo "[5/5] appendix: rollout galleries, every model on one D1 and one D2 trajectory"
python paper/make_gallery_fig.py
echo "wrote paper/figures/tmlr/setup.pdf, 06_duhamel/fig1/schematic_tikz.pdf, 06_duhamel/rollouts_multi_2row.pdf,"
echo "      appendix/energy_landscapes.pdf, appendix/gallery_tdgl_d1.pdf, appendix/gallery_tdgl_d2.pdf"
