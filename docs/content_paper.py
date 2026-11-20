"""Conference paper content, reusing the numbers computed in content_report.py."""
import os, sys, json
sys.path.insert(0, os.path.dirname(__file__))
from content_report import (pct, num, cvc, cvm, hoc, hom, sc, sm, mpbe_c, mpbe_m, mpbe_hc, mpbe_hm,
                            best_c, rank_c, abl_c, abl_m, FIG, ROOT, h1, h2, h3, p, bullets, numbered,
                            table, eq, figure)
from refs import REFS, LIT

META = {
    "title": "Prediction of Heart Disease using Ensemble Learning: A Reproducible Re-implementation and Multi-Dataset Evaluation of the MPBE Model",
    "author": "Salil Mahale",
    "affil": "Department of Computer Engineering, St. Francis Institute of Technology, University of Mumbai, Mumbai, India",
    "email": "salil.mahale@student.sfit.ac.in",
    "keywords": "Heart disease prediction, Ensemble learning, Bagging, Stacking, Boosting, Pearson correlation, Cross validation, UCI Cleveland",
}

ABSTRACT = (
 "Early prediction of heart disease from routine clinical attributes is a well studied classification problem, yet published accuracies on the "
 "UCI Cleveland dataset range from below 75% to above 97%, largely because of differing evaluation protocols. This paper re-implements the "
 "MinMax normalized Pearson coefficient Bagging Ensemble (MPBE) proposed by Muthulakshmi et al., which combines bagged Decision Tree, "
 "Random Forest, Naive Bayes and Support Vector Machine members, and evaluates it under a leakage free protocol: stratified 10-fold cross "
 "validation with scaling fitted inside each fold, on both the Cleveland dataset (303 patients) and a combined five source UCI dataset "
 "(918 unique patients). MPBE is compared against six single classifiers and six other ensembles (homogeneous bagging, soft and hard voting, "
 f"stacking, AdaBoost, gradient boosting and XGBoost). On Cleveland, MPBE reaches {pct(mpbe_c.accuracy)}% accuracy and ROC-AUC "
 f"{num(mpbe_c.roc_auc)}, statistically level with the best models, but the 95.33% reported originally is not reproduced. On the combined "
 f"dataset MPBE is the best model ({pct(mpbe_m.accuracy)}% accuracy, ROC-AUC {num(mpbe_m.roc_auc)}) and loses the least accuracy when the "
 "population becomes heterogeneous. An ablation shows the Pearson feature selection step changes accuracy by under 0.5 points. The results "
 "support heterogeneous bagging as a robust default for tabular clinical screening and argue for cross validated reporting over single split "
 "headline figures."
)

def cvrows(df, names):
    return [[r.Model, f"{pct(r.accuracy)} \u00b1 {pct(r.accuracy_std)}", pct(r.recall), pct(r.f1), num(r.roc_auc) or "n/a"]
            for _, r in df.iterrows() if r.Model in names]

ALL = list(cvc.Model)
B = []
A = B.append

A(h1("1 Introduction"))
A(p("Cardiovascular disease is the leading cause of death worldwide, and in India a quarter of heart related deaths occur before the age of "
    "forty [1]. Thirteen routinely collected attributes (age, sex, chest pain type, resting blood pressure, cholesterol, fasting blood sugar, "
    "resting ECG, maximum heart rate, exercise induced angina, ST depression, ST slope, fluoroscopy vessel count and thalassemia status) are "
    "cheap to obtain, and a classifier that predicts angiographic disease from them can act as a screening aid. The UCI Cleveland dataset [22] "
    "has been the standard benchmark for this task for three decades, and ensemble methods, which combine several learners to reduce variance "
    "or bias, are consistently reported to outperform single classifiers on it [8, 9, 14]."))
A(p("Muthulakshmi et al. [1] proposed the **MPBE** model: MinMax normalisation, Pearson correlation feature selection, and a bagging ensemble "
    "whose members are a Decision Tree, a Random Forest, a Naive Bayes classifier and an SVM, aggregated by mean score. They reported 95.33% "
    "accuracy. The paper leaves several details open (the correlation threshold, whether scaling was fitted before or after splitting, whether "
    "aggregation is soft or hard) and evaluates on a single dataset. Recent work has warned that many accuracies above 95% on Cleveland arise "
    "from optimistic protocols or duplicate rows [11, 21]. This paper therefore asks: (i) what accuracy does MPBE reach under a strict "
    "cross validated protocol, (ii) how does it compare with voting, stacking and boosting ensembles, and (iii) does its advantage transfer to a "
    "larger, heterogeneous population?"))
A(p("Our contributions are a fully specified, scikit-learn compatible re-implementation of MPBE; a fourteen model comparison on two datasets "
    "with accuracy, precision, recall, F1 and ROC-AUC; an ablation of the feature selection step; and an honest reconciliation of our numbers "
    "with the original claim. Code, data preparation and figures are released with the paper."))

A(h1("2 Related Work"))
A(p("Table 1 summarises the twenty studies reviewed. Three themes emerge. First, **ensembles beat weak learners by a wide margin but strong "
    "learners by a narrow one**: Latha and Jeeva [9] measured gains of up to 7.3 points from majority voting over weak classifiers, whereas "
    "Karadeniz et al. [14] and Chandrasekhar and Peddakrishna [18] found boosting, stacking and soft voting only marginally ahead of tuned "
    "logistic regression. Second, **feature selection helps modestly**: Haq et al. [5], Alim et al. [7] and Li et al. [15] all found that "
    "dropping cholesterol and fasting blood sugar costs nothing and occasionally helps. Third, **protocol dominates**: Ali et al. [11] reported "
    "100% for Random Forest on a Kaggle copy later shown to contain duplicates, Ahmad et al. [19] showed that grid search on a single split "
    "pushes several models towards 99%, and Adekoya et al. [21] and Mienye and Jere [16] argue for nested cross validation, calibration and "
    "explanation as the criteria that matter for clinical use. Multi dataset evaluation, as in Doppala et al. [6] and Asif et al. [17], is the "
    "approach adopted here."))
A(table("Summary of the reviewed studies", ["Ref", "Year", "Dataset", "Technique", "Reported result"],
        [[f"[{n}]", str(y), ds, tech, res] for (n, t, y, ds, tech, res, s1, s2) in LIT], [500, 500, 1900, 3600, 2500]))

A(h1("3 Methodology"))
A(h2("3.1 Datasets"))
A(p(f"**Cleveland.** 303 records, 13 predictors and the diagnosis num in 0 to 4, binarised to target = 1 if num > 0 ({sc['n_pos']} positive, "
    f"{sc['n_neg']} negative). Six cells are missing (ca, thal) and were median imputed."))
A(p(f"**Combined.** All five UCI sources were concatenated: Cleveland, Hungarian (294), Switzerland (123), Long Beach VA (200) [22] and Statlog "
    f"(270) [23], whose 1/2 label was remapped to 0/1. Cholesterol or blood pressure equal to zero was treated as missing, and exact duplicate "
    f"rows (Statlog overlaps Cleveland) were removed, leaving {sm['n_rows']} unique patients ({sm['n_pos']} positive). The non Cleveland "
    "sources have many missing values in slope, ca and thal, which were median imputed; this makes the combined set harder and more realistic."))
A(figure("eda_cleveland.png", "Class balance and age distribution, Cleveland", 0.85))
A(h2("3.2 Pearson Correlation Feature Selection"))
A(p("For each predictor x and the label y the Pearson coefficient is"))
A(eq(r"r_{xy} = \frac{\operatorname{Cov}(x,y)}{\sigma_x\,\sigma_y}", "r_xy = Cov(x, y) / (sigma_x sigma_y)"))
A(p("where Cov(x, y) measures how x and y vary together and the product of standard deviations normalises it to the range -1 to 1. Since [1] does "
    "not state a threshold, we make the rule explicit: a predictor is kept if |r_xy| is at least 0.10, and if two kept predictors have |r| above "
    f"0.80 the one less correlated with the label is dropped. On Cleveland this removes {', '.join(sc['dropped'])}; on the combined data "
    f"{', '.join(sm['dropped'])}. No pair of predictors exceeds 0.60, so the redundancy rule never fires (Fig. 2)."))
A(figure("target_corr_cleveland.png", "Correlation of each attribute with the label on Cleveland; grey bars are dropped", 0.85))
A(h2("3.3 MinMax Normalisation"))
A(p("Every kept attribute is mapped to 0 to 1 by x' = (x - x_min)/(x_max - x_min), with x_min and x_max computed **on the training fold only** "
    "and applied to the test fold. This is implemented as a scikit-learn Pipeline so that no statistic of the test fold reaches the model; it is "
    "the one place where our protocol deliberately differs from the description in [1]."))
A(h2("3.4 The MPBE Ensemble"))
A(p("Four bagging ensembles are built, one per base learner: Decision Tree (CART, Gini), Random Forest (50 trees), Gaussian Naive Bayes and "
    "RBF SVM with Platt probabilities. Each ensemble holds 100 estimators trained on bootstrap resamples of the training fold. The prediction is "
    "the mean of the four bagged probabilities, thresholded at 0.5:"))
A(eq(r"P_{\mathrm{MPBE}}(y=1\mid x)=\frac{1}{4}\sum_{k=1}^{4}\frac{1}{100}\sum_{b=1}^{100}P_{k,b}(y=1\mid x)",
     "P_MPBE(y=1|x) = (1/4) sum_k (1/100) sum_b P_kb(y=1|x)"))
A(p("The inner average over bootstrap models reduces the variance of each learner family; the outer average lets the families compensate for "
    "each other's errors. We read the 'mean score' aggregation of [1] as this soft average and include a hard voting ensemble for contrast."))
A(figure("architecture.png", "Pipeline of the re-implemented MPBE model and the comparison branch", 1.0))
A(h2("3.5 Comparison Models and Protocol"))
A(p("Six single models (DT, RF with 100 trees, NB, SVM, Logistic Regression, KNN with k = 7) and six ensembles are evaluated: homogeneous "
    "Bagging of 100 trees; soft and hard Voting over DT, RF, NB, SVM and LR; Stacking of DT, RF, NB and SVM with a logistic meta learner on "
    "5 fold out-of-fold predictions; AdaBoost (100 stumps, learning rate 0.5); Gradient Boosting (100 trees, depth 3, rate 0.1); and XGBoost "
    "[25] (100 trees, depth 3, rate 0.1, subsample 0.9). Hyperparameters are default or lightly set and identical on both datasets so that the "
    "comparison isolates the ensemble design from tuning. All models use StratifiedKFold(10, shuffle, seed 42); a stratified 80:20 hold-out with "
    "the same seed provides confusion matrices and ROC curves. Metrics are accuracy, precision, recall, F1 and ROC-AUC."))

A(h1("4 Results"))
A(h2("4.1 Cleveland"))
A(table("10-fold CV on Cleveland (11 selected attributes). Accuracy is mean and standard deviation over folds",
        ["Model", "Accuracy (%)", "Recall (%)", "F1 (%)", "AUC"], cvrows(cvc, ALL), [3000, 1700, 1200, 1100, 900]))
A(figure("cv_accuracy_cleveland.png", "10-fold CV accuracy with standard deviation on Cleveland (MPBE in red)", 0.85))
A(p(f"MPBE obtains {pct(mpbe_c.accuracy)} \u00b1 {pct(mpbe_c.accuracy_std)}% accuracy and AUC {num(mpbe_c.roc_auc)}, the second highest AUC "
    f"of the fourteen models, and ranks {rank_c}th by accuracy behind Naive Bayes and Stacking ({pct(best_c.accuracy)}%), AdaBoost, soft Voting and "
    "Logistic Regression. All of these lie within one fold standard deviation (5 to 7 points on folds of 30 patients) of each other. The "
    "informative gaps are below: the single Decision Tree (77.2%) and homogeneous tree Bagging (79.8%) trail MPBE by 5.3 and 2.6 points, so "
    "mixing four learner families is worth about 2.6 points over bagging one family, which is the central claim of [1]. On the hold-out split "
    f"MPBE scores {pct(mpbe_hc.accuracy)}% with recall {pct(mpbe_hc.recall)}% and AUC {num(mpbe_hc.roc_auc)} (Fig. 5); Random Forest is best "
    "there at 90.2%, illustrating how a 61 patient test set reorders models by one or two patients."))
A(figure("roc_cleveland.png", "ROC curves on the Cleveland hold-out split", 0.55))
A(h2("4.2 Combined Dataset"))
A(table("10-fold CV on the combined 5 source dataset (918 records)",
        ["Model", "Accuracy (%)", "Recall (%)", "F1 (%)", "AUC"], cvrows(cvm, ALL), [3000, 1700, 1200, 1100, 900]))
A(figure("cross_dataset.png", "10-fold CV accuracy of every model on both datasets", 0.85))
A(p(f"On the combined population MPBE is **first** by accuracy ({pct(mpbe_m.accuracy)} \u00b1 {pct(mpbe_m.accuracy_std)}%) and ties Stacking "
    f"for the best AUC ({num(mpbe_m.roc_auc)}); it is also first on the hold-out split ({pct(mpbe_hm.accuracy)}%, recall {pct(mpbe_hm.recall)}%). "
    "Fold standard deviations fall to about 3 points, so this ranking is more reliable than the Cleveland one. Every model loses accuracy on the "
    "harder data (Fig. 6), the Decision Tree most (to 70.8%) and MPBE least (about 1.6 points), which is the property one wants from a model "
    "meant to run across hospitals."))
A(figure("cm_combined.png", "Confusion matrices on the combined hold-out split", 0.95))
A(h2("4.3 Ablation of Feature Selection"))
A(table("MPBE with and without Pearson selection (10-fold CV)", ["Dataset", "Attributes", "Accuracy (%)", "F1 (%)", "AUC"], [
 ["Cleveland", "11 selected", pct(mpbe_c.accuracy), pct(mpbe_c.f1), num(mpbe_c.roc_auc)],
 ["Cleveland", "all 13", pct(abl_c['accuracy']), pct(abl_c['f1']), num(abl_c['roc_auc'])],
 ["Combined", "11 selected", pct(mpbe_m.accuracy), pct(mpbe_m.f1), num(mpbe_m.roc_auc)],
 ["Combined", "all 13", pct(abl_m['accuracy']), pct(abl_m['f1']), num(abl_m['roc_auc'])]], [1800, 1800, 1800, 1800, 1800]))
A(p("Dropping the two weakest attributes moves accuracy by less than half a point and AUC by at most 0.003. The selection step is harmless and "
    "yields a marginally cheaper screening model, but it is not the source of any accuracy gain, in agreement with [7, 9]."))
A(h2("4.4 Feature Importance"))
A(figure("importance_cleveland.png", "Random Forest Gini importance on Cleveland", 0.75))
A(p("Random Forest importances rank thal, thalach, cp, ca and oldpeak highest on Cleveland; on the combined data cp, age, thalach and oldpeak "
    "dominate because thal and ca are mostly imputed outside Cleveland. Chest pain type, maximum heart rate and exercise ST depression are "
    "therefore the attributes a screening protocol should record first, consistent with [15, 16]."))

A(h1("5 Discussion"))
A(p("Three conclusions follow. **(1) The 95.33% of [1] does not survive a leakage free protocol.** Under stratified 10-fold cross validation no "
    "model in this study exceeds 84% mean accuracy on Cleveland, in line with [8, 9, 10, 21]. The original figure is most plausibly a single "
    "favourable split with scaling and selection applied to the whole dataset. **(2) The design of MPBE is nevertheless sound.** Heterogeneous "
    "bagging beats bagging a single learner family by 2.6 points on Cleveland and 1.3 on the combined data, and it is the most stable model "
    "across the two populations. **(3) Ensemble gains over strong single learners are small on this data.** Logistic Regression and Naive Bayes "
    "are within a point of every ensemble on Cleveland; the practical value of MPBE appears when the data become heterogeneous and noisy, where "
    "variance reduction matters. Limitations are the modest dataset sizes, median imputation of attributes missing by design in three sources, "
    "and the absence of hyperparameter tuning, which was deliberate. Future work should tune the decision threshold for recall, add SHAP based "
    "explanations, and validate on a local Indian cohort."))

A(h1("6 Conclusion"))
A(p(f"We re-implemented the MPBE heterogeneous bagging ensemble and evaluated it against thirteen alternatives on two datasets under a strict "
    f"protocol. MPBE reaches {pct(mpbe_c.accuracy)}% accuracy (AUC {num(mpbe_c.roc_auc)}) on Cleveland and is the best model on a combined "
    f"918 patient dataset ({pct(mpbe_m.accuracy)}%, AUC {num(mpbe_m.roc_auc)}). The reported 95.33% could not be reproduced, the Pearson "
    "selection step contributes under half a point, and heterogeneous bagging is a robust default for tabular clinical screening. All code and "
    "figures are provided for verification."))
A({"type": "references", "items": REFS})

if __name__ == "__main__":
    json.dump({"meta": META, "abstract": ABSTRACT, "abbrev": [], "blocks": B},
              open(os.path.join(ROOT, "docs", "paper_content.json"), "w"), indent=1)
    print("paper blocks:", len(B))
