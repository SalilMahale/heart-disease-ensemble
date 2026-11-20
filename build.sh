#!/usr/bin/env bash
# Rebuild every deliverable from source, in order.
# Usage:  bash build.sh [--all]      --all also re-runs the experiments (~8 min)
set -e
cd "$(dirname "$0")"

if [ "$1" = "--all" ]; then
  echo "== running experiments =="
  FORCE=1 python3 src/heart_ensemble.py
fi

echo "== assembling text (results -> JSON) =="
python3 docs/content_report.py
python3 docs/content_paper.py

echo "== LaTeX =="
python3 docs/render_tex.py
python3 docs/render_paper_ieee.py
python3 docs/render_paper_tex.py
for d in report_latex:ML_MiniProject_Report paper_ieee:ML_MiniProject_Paper_IEEE paper_lncs:ML_MiniProject_Paper_LNCS; do
  dir="${d%%:*}"; f="${d##*:}"
  (cd "$dir" && pdflatex -interaction=nonstopmode "$f.tex" >/dev/null && pdflatex -interaction=nonstopmode "$f.tex" >/dev/null)
done

echo "== Word =="
node docs/render_docx.js docs/report_content.json deliverables/ML_MiniProject_Report.docx report
node docs/render_docx.js docs/paper_content.json  deliverables/ML_MiniProject_Paper_IEEE.docx paper2col
node docs/render_docx.js docs/lit_content.json    deliverables/ML_MiniProject_Literature_Review.docx paper

echo "== collecting =="
cp report_latex/ML_MiniProject_Report.pdf      deliverables/
cp paper_ieee/ML_MiniProject_Paper_IEEE.pdf    deliverables/
cp paper_lncs/ML_MiniProject_Paper_LNCS.pdf    deliverables/
rm -f ./*/*.aux ./*/*.log ./*/*.out ./*/*.toc ./*/*.lof ./*/*.lot
rm -rf src/__pycache__ docs/__pycache__

echo "== done =="
ls -la deliverables
