# Prediction of Heart Disease using Ensemble Learning (MPBE re-implementation)

**Salil Mahale** — Roll 53, PID 232085, BE CMPN B, Batch 3, St. Francis Institute of Technology, University of Mumbai.
Machine Learning [CSL 701] mini project, 2026-27. Guide: **Dr. Nazneen Ansari**.
Base paper: Muthulakshmi, Parveen and Rajeswari, "Prediction of Heart Disease using Ensemble Learning", *Indian Journal of Science and Technology* 16(20):1469-1476, 2023.

---

## 1. Deliverables (`deliverables/`)

| File | What it is | Pages |
|---|---|---|
| `ML_MiniProject_Report.docx` / `.pdf` | Chapter-style mini project report, ML report format 2026 | 30 |
| `ML_MiniProject_Paper_IEEE.docx` / `.pdf` | Conference paper, IEEE two-column (primary) | 8 |
| `ML_MiniProject_Paper_LNCS.pdf` | Same paper in Springer LNCS single-column style (alternative) | 11 |
| `ML_MiniProject_Literature_Review.docx` | Summaries of the base paper and all 20 reviewed papers | 8 |

All four are generated from the same text and the same result files, so their numbers can never disagree.
After opening the report in Word, right-click the Table of Contents and choose **Update Field** to fill in page numbers.

## 2. Folder layout

```
build.sh                  rebuild everything (see section 4)
requirements.txt          Python dependencies
README.md                 this file
PROJECT_HANDOFF.md        full context for anyone picking the project up

src/heart_ensemble.py     the pipeline: loading, Pearson selection, MPBE class, 14-model zoo,
                          10-fold CV, hold-out, ablation, all figures
src/gui_tkinter.py        desktop predictor (trains MPBE once, predicts from typed values)
notebook/heart_ensemble.ipynb   the same pipeline as a Colab-runnable notebook

data/                     five raw UCI files + cleveland_prepared.csv, combined_prepared.csv
results/                  cv_results_*.csv, cv_folds_*.csv, holdout_results_*.csv,
                          rf_importance_*.csv, summary_*.json
figures/                  every figure used in the documents
assets/sfit_logo.png      college logo used on the report cover

docs/refs.py              the 25 references and the 20 literature-review entries
docs/content_report.py    text of the report   -> docs/report_content.json
docs/content_paper.py     text of the paper    -> docs/paper_content.json
docs/lit_content.json     text of the literature review
docs/render_tex.py        JSON -> report_latex/   (report class, Times, 12/14 pt)
docs/render_paper_ieee.py JSON -> paper_ieee/     (IEEEtran, two columns)
docs/render_paper_tex.py  JSON -> paper_lncs/     (llncs, single column)
docs/render_docx.js       JSON -> .docx           (modes: report, paper, paper2col)

report_latex/             ML_MiniProject_Report.tex + PDF
paper_ieee/               ML_MiniProject_Paper_IEEE.tex + PDF, IEEEtran.cls included
paper_lncs/               ML_MiniProject_Paper_LNCS.tex + PDF, llncs.cls + splncs04.bst included
deliverables/             the six files listed in section 1
```

## 3. Running the code

```bash
pip install -r requirements.txt
python3 src/heart_ensemble.py      # about 8 minutes on a laptop CPU
                                   # results are cached; FORCE=1 recomputes everything
python3 src/gui_tkinter.py         # Tkinter prediction window (needs python3-tk)
```

## 4. Rebuilding the documents

```bash
bash build.sh          # text -> LaTeX -> PDFs -> Word -> deliverables/
bash build.sh --all    # the same, but re-runs the experiments first
```

Edit the text in `docs/content_report.py` or `docs/content_paper.py`, never in the `.tex` or `.docx`
files, because those are regenerated on every build. Result numbers are read from `results/`, so
re-running the experiments updates every document automatically.

## 5. Document conventions

* Report: Times New Roman, 12 pt body and code, 14 pt headings, justified, line spacing 1, no italics, black only, no headers or footers, keywords in bold.
* Paper: IEEE two-column, 10 pt Times, all tables and wide figures span both columns.
* Citations: IEEE numeric `[n]`, the same numbering in the report, the paper and the literature review.
* Figures are numbered `Figure <chapter>.<n>` in the report and `Fig. <n>` in the paper; tables likewise.

## 6. Headline results (seed 42, stratified 10-fold CV, scaling fitted inside each fold)

| Dataset | MPBE accuracy | MPBE ROC-AUC | Rank of MPBE | Best model |
|---|---|---|---|---|
| Cleveland (303 patients) | 82.46 % | 0.912 | 7 / 14, top nine within one standard deviation | Naive Bayes and Stacking, 83.78 % |
| Combined 5-source (918 patients) | 80.84 % | 0.884 | 1 / 14 | MPBE |

The 95.33 % of the base paper is not reproducible under a leakage-free protocol; the reasoning is in
chapter 6 of the report and section 5 of the paper.
