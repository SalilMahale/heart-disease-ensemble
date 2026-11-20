# PROJECT HANDOFF: ML Mini Project (Heart Disease, Ensemble Learning)

Read this fully before doing anything. Everything below is DONE unless marked TODO. Your job is polish, fixes the user asks for, and viva prep. Do not rebuild from scratch.

## Who and what
- Student: **Salil Mahale**, **BE CMPN B, Batch 3**, SFIT (Mumbai University). Solo project. Guide/teacher: **Dr. Nazneen Ansari** (guide only, NOT a co-author).
- Subject: Machine Learning [CSL 701], 2026-27. Class mini-project sheet row **#37**.
- Base paper: Muthulakshmi, Parveen, Rajeswari, "Prediction of Heart Disease using Ensemble Learning", IJST 16(20):1469-1476, 2023 (file IJST-2022-2279.pdf in project knowledge). Their model = **MPBE**: MinMax scaling + Pearson correlation feature selection + bagging (100 estimators) with DT/RF/NB/SVM base learners, mean-score aggregation, claimed 95.33% on UCI Cleveland.
- Report format: ML-REPORT-FORMAT-2026_Refined.pdf (chapters: Abstract, Lists, 1 Intro/Problem/Prereqs, 2 Lit review (20 papers), 3 Methodology (3.1 architecture + dataset/EDA, 3.3 algorithms), 4 Implementation, 5 Experimental setup (5.1-5.5), 6 Results, 7 GUI, 8 Code, 9 Conclusion, 10 References). Paper style reference: reference_for_our_own_paper.pdf (Jiu et al., Springer LNCS).

## User's standing formatting rules (apply to every docx)
Times New Roman; 12 pt body/tables/code, 14 pt headings; fully justified; **no italics**; black only; **no headers/footers**; bold keywords; line spacing 1; **no em dashes** (use commas or full stops). Emojis fine in chat, not in documents. He likes teacher-style, intuitive explanations and wants every formula component explained.

## Decisions already made with the user (do not reopen)
1. Two documents, each in LaTeX AND docx: chapter-style report + LNCS conference paper.
2. Datasets: UCI Cleveland (303) + combined 5-source (Cleveland, Hungarian, Switzerland, Long Beach VA, Statlog; 1190 raw -> 918 after de-dup; zero chol/BP treated as missing; median imputation).
3. Extensions: compare MPBE vs DT, RF, NB, SVM, LR, KNN, Bagging(DT), Voting soft/hard, Stacking(LR meta), AdaBoost, Gradient Boosting, XGBoost. Stratified 10-fold CV (seed 42), MinMax fitted inside each fold via Pipeline, plus 80:20 hold-out; metrics acc/prec/rec/F1/ROC-AUC; ablation on all 13 features.
4. Literature: 20 papers, 2018-2025, all verified via web search with real DOIs. IEEE numeric citations. Refs [2]-[7] come from the base paper's own reference list; [8]-[21] are external. [22]-[25] are dataset/tool refs.
5. No web app. Tkinter GUI only (done, with screenshot).
6. Paper: author is Salil alone; LNCS class; ~10-12 pages (currently 12, flexible by a couple of pages).

## Real results (seed 42) - quote these, never invent numbers
Cleveland 10-fold CV: MPBE 82.46 +/- 6.41 %, AUC 0.912 (2nd best AUC), rank 7/14 by accuracy; NB and Stacking 83.78 (tie, top), AdaBoost 83.45, Voting soft 82.84, LR 82.80, Voting hard 82.48, RF 82.17, SVM 82.15, XGB 81.85, GB 80.49, KNN 79.87, Bagging(DT) 79.83, DT 77.17.
Cleveland hold-out (61 test): RF 90.16 best; MPBE 86.89, recall 92.86, AUC 0.958.
Combined 10-fold CV: **MPBE first** 80.84 +/- 3.53 %, AUC 0.884 (ties Stacking); RF 80.73, GB 80.72, Voting hard 80.51, Stacking 80.40, ..., DT 70.81.
Combined hold-out (184 test): MPBE first 82.07, recall 85.29.
Ablation (MPBE, all 13 vs 11 selected): Cleveland 82.46 vs 82.46 (AUC 0.909 vs 0.912); Combined 81.05 vs 80.84. Feature selection drops chol+fbs (Cleveland), chol+restecg (combined).
Narrative: 95.33% is NOT reproducible under leakage-free CV (favourable single split + preprocessing before splitting); MPBE design is sound (heterogeneous bagging beats bagging one family by ~2.6 pts); MPBE is most stable across datasets.

## File layout
See README.md for the full tree and the build instructions. Short version:
- `build.sh` rebuilds everything: text JSON -> LaTeX -> PDFs -> Word -> `deliverables/`. `build.sh --all` re-runs the experiments first (~8 min; in a sandbox this exceeds a 300 s tool limit, so run it in the background or rely on the cached `results/`).
- Text lives ONLY in `docs/content_report.py`, `docs/content_paper.py` and `docs/lit_content.json`; the `.tex` and `.docx` files are generated and must never be hand-edited.
- Renderers: `render_tex.py` (report), `render_paper_ieee.py` (IEEE two-column, primary paper), `render_paper_tex.py` (LNCS alternative), `render_docx.js` (modes `report`, `paper`, `paper2col`).
- Block types both renderers understand: h1/h2/h3, p, bullets, numbered, table, figure, code, eq, pagebreak, references, with `**bold**` inline. In report mode h1 must read "Chapter N: ..." because that drives figure and table numbering.
- Deliverables: `ML_MiniProject_Report.docx/.pdf` (30 pp), `ML_MiniProject_Paper_IEEE.docx/.pdf` (8 pp, chosen format), `ML_MiniProject_Paper_LNCS.pdf` (11 pp, alternative), `ML_MiniProject_Literature_Review.docx` (8 pp).
- `assets/sfit_logo.png` goes on the report cover between the Subject line and "Under the guidance of".

## Known small issues / TODO candidates (only if the user asks or you have spare capacity)
- (Sept 19 session, part 2) Standardisation pass: IEEE two-column is now the primary paper and its .docx matches (two-column sections, wide tables and figures in their own full-width continuous sections); LNCS version kept as an alternative and switched to Times so all three PDFs share a font; folders and file names unified (`paper_lncs/`, `ML_MiniProject_Paper_LNCS`, `ML_MiniProject_Literature_Review.docx`); `build.sh` added; all LaTeX now compiles with zero errors; build artefacts and `__pycache__` removed from the bundle.
- (Sept 19 session, part 1) Report was compressed from 43 to 30 pages: chapter 2 now has one summary table of all 20 papers plus four thematic subsections (per paper detail lives in ML_MiniProject_Literature_Review.docx), figure widths reduced, three low information figures dropped (two fold spread box plots, the combined correlation heat map), two of four code listings replaced by a pointer to the source files, floats changed from [H] to [htbp], tighter front matter (TOC flows after the abstract; list of tables and abbreviations share a page in the docx). Figures are now 17, tables 12; in text figure cross references in chapter 3 were renumbered to match.
- Report docx TOC is a Word field: user must right-click -> Update Field after opening; a note is in the doc. LaTeX version has a real TOC.
- Code font in docx is TNR 12 per user's rule (he said "12 for actual text and codes"); LaTeX uses ttfamily in lstlisting. Ask if he wants monospace in docx.
- Report LaTeX compiles with zero errors; some overfull hbox warnings remain (long URLs/code lines), cosmetic.
- Paper email `salil.mahale@student.sfit.ac.in` is a placeholder guess; confirm with user.
- In the IEEE PDF, a few full-width floats spill past the bibliography because double-column floats can only sit at a page top. Salil has accepted this (some IEEE journal templates require exactly that). If he changes his mind: add `\clearpage` before `\begin{thebibliography}`, load `stfloats` with `\setcounter{dbltopnumber}{3}`, or demote the ROC and importance figures to single-column `figure`.
- The email/affiliation for the lit-summaries docx title block is reused from paper META with different labels (looks fine).
- Karadeniz 2021 [14] and Li 2020 [15] have no DOI in REFS because I couldn't verify one; leave as is or verify.
- Notebook was not executed; cell code equals the .py so it should run, but a smoke test in Colab is worthwhile.
- Possible viva additions: SHAP plots, threshold tuning for recall, per-source accuracy on the combined data.

## Memory system note
Project memory (overview.md) already records the topic, deliverables, decisions and datasets. If the user asks to remember anything new, write only under /projects/019f7b5d-fead-7259-964a-7c5aa1937de0/.

## Style for chatting with Salil
Warm, concise, teacher-like, emojis OK, correct over fast, ask before big changes, no em dashes, avoid "genuinely/honestly".
