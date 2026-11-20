"""Report content. Blocks are rendered by render_docx.js and render_tex.py."""
import os, json, sys
import pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
from refs import REFS, LIT

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
FIG = os.path.join(ROOT, "figures")

META = {
    "title": "Prediction of Heart Disease using Ensemble Learning: A Reproducible Re-implementation and Extension of the MPBE Model",
    "short_title": "Heart Disease Prediction using Ensemble Learning",
    "student": "Salil Mahale",
    "roll": "53", "pid": "232085", "class": "BE CMPN B", "batch": "3",
    "guide": "Dr. Nazneen Ansari",
    "subject": "Machine Learning [CSL 701]",
    "dept": "Department of Computer Engineering",
    "college": "St. Francis Institute of Technology",
    "univ": "University of Mumbai",
    "year": "2026-2027",
    "logo": os.path.join(ROOT, "assets", "sfit_logo.png"),
}

cvc = pd.read_csv(f"{RES}/cv_results_cleveland.csv")
cvm = pd.read_csv(f"{RES}/cv_results_combined.csv")
hoc = pd.read_csv(f"{RES}/holdout_results_cleveland.csv")
hom = pd.read_csv(f"{RES}/holdout_results_combined.csv")
sc = json.load(open(f"{RES}/summary_cleveland.json"))
sm = json.load(open(f"{RES}/summary_combined.json"))

def pct(x): return "" if pd.isna(x) else f"{100*x:.2f}"
def num(x): return "" if pd.isna(x) else f"{x:.3f}"

def cv_table(df):
    rows = []
    for _, r in df.iterrows():
        rows.append([r.Model, f"{pct(r.accuracy)} \u00b1 {pct(r.accuracy_std)}", pct(r.precision), pct(r.recall), pct(r.f1), num(r.roc_auc) or "n/a"])
    return rows

def ho_table(df):
    return [[r.Model, pct(r.accuracy), pct(r.precision), pct(r.recall), pct(r.f1), num(r.roc_auc)] for _, r in df.iterrows()]

def row(df, name): return df[df.Model == name].iloc[0]
mpbe_c, mpbe_m = row(cvc, "MPBE (proposed, replicated)"), row(cvm, "MPBE (proposed, replicated)")
mpbe_hc, mpbe_hm = row(hoc, "MPBE (proposed, replicated)"), row(hom, "MPBE (proposed, replicated)")
best_c, best_m = cvc.iloc[0], cvm.iloc[0]
rank_c = int(cvc.reset_index(drop=True).index[cvc.Model == "MPBE (proposed, replicated)"][0]) + 1
rank_m = int(cvm.reset_index(drop=True).index[cvm.Model == "MPBE (proposed, replicated)"][0]) + 1
abl_c, abl_m = sc["mpbe_all13_cv"], sm["mpbe_all13_cv"]

ATTRS = [
 ["age", "Age in years", "Numeric"],
 ["sex", "Sex (1 = male, 0 = female)", "Binary"],
 ["cp", "Chest pain type (1 typical angina, 2 atypical, 3 non-anginal, 4 asymptomatic)", "Categorical"],
 ["trestbps", "Resting blood pressure in mm Hg", "Numeric"],
 ["chol", "Serum cholesterol in mg/dl", "Numeric"],
 ["fbs", "Fasting blood sugar greater than 120 mg/dl (1 = true)", "Binary"],
 ["restecg", "Resting ECG result (0 normal, 1 ST-T abnormality, 2 LV hypertrophy)", "Categorical"],
 ["thalach", "Maximum heart rate achieved", "Numeric"],
 ["exang", "Exercise induced angina (1 = yes)", "Binary"],
 ["oldpeak", "ST depression induced by exercise relative to rest", "Numeric"],
 ["slope", "Slope of the peak exercise ST segment (1 up, 2 flat, 3 down)", "Categorical"],
 ["ca", "Number of major vessels coloured by fluoroscopy (0 to 3)", "Numeric"],
 ["thal", "Thalassemia (3 normal, 6 fixed defect, 7 reversible defect)", "Categorical"],
 ["target", "1 = presence of heart disease (num > 0), 0 = absence", "Binary (label)"],
]

ABBREV = [
 ["AUC", "Area Under the ROC Curve"], ["CART", "Classification and Regression Tree"],
 ["CV", "Cross Validation"], ["DT", "Decision Tree"], ["EDA", "Exploratory Data Analysis"],
 ["GB", "Gradient Boosting"], ["KNN", "K-Nearest Neighbours"], ["LR", "Logistic Regression"],
 ["ML", "Machine Learning"], ["MPBE", "MinMax normalized Pearson coefficient Bagging Ensemble"],
 ["NB", "Naive Bayes"], ["RBF", "Radial Basis Function"], ["RF", "Random Forest"],
 ["ROC", "Receiver Operating Characteristic"], ["SVM", "Support Vector Machine"],
 ["UCI", "University of California, Irvine (ML Repository)"], ["XGBoost", "Extreme Gradient Boosting"],
]

ABSTRACT = (
 "Cardiovascular disease remains the leading cause of death worldwide, and inexpensive clinical attributes such as age, chest pain type, "
 "resting blood pressure and exercise test results are routinely available long before an angiogram is ordered. This mini project "
 "re-implements the **MPBE** model (MinMax normalized Pearson coefficient Bagging Ensemble) proposed by Muthulakshmi et al. [1] and "
 "extends it in three directions. First, the model is evaluated not only on the **UCI Cleveland** dataset (303 patients) used in the "
 "original work but also on a **combined five source dataset** (Cleveland, Hungarian, Switzerland, Long Beach VA and Statlog, 918 unique "
 "patients) to test generalisation. Second, the bagging ensemble is compared against **Voting, Stacking, AdaBoost, Gradient Boosting and "
 "XGBoost** as well as six single classifiers. Third, evaluation uses **stratified 10-fold cross validation** with scaling fitted inside "
 "each fold, and reports accuracy, precision, recall, F1 score and ROC-AUC rather than accuracy alone. "
 f"On Cleveland, MPBE reaches {pct(mpbe_c.accuracy)}% cross validated accuracy (ROC-AUC {num(mpbe_c.roc_auc)}) and {pct(mpbe_hc.accuracy)}% on a "
 f"20% hold-out split; the 95.33% reported in the original paper could not be reproduced under a leakage free protocol. On the combined dataset, "
 f"MPBE is the best of all fourteen models with {pct(mpbe_m.accuracy)}% accuracy and ROC-AUC {num(mpbe_m.roc_auc)}. The project concludes that "
 "heterogeneous bagging is a robust choice for tabular clinical data, that ensemble gains over strong single learners are modest, and that "
 "honest cross validated reporting matters more than a headline accuracy."
)

def figure(name, caption, width=0.8):
    return {"type": "figure", "path": f"{FIG}/{name}", "caption": caption, "width": width}

def h1(t): return {"type": "h1", "text": t}
def h2(t): return {"type": "h2", "text": t}
def h3(t): return {"type": "h3", "text": t}
def p(t): return {"type": "p", "text": t}
def bullets(items): return {"type": "bullets", "items": items}
def numbered(items): return {"type": "numbered", "items": items}
def table(caption, header, rows, widths=None): return {"type": "table", "caption": caption, "header": header, "rows": rows, "widths": widths}
def code(t, caption=None): return {"type": "code", "text": t, "caption": caption}
def eq(latex, plain): return {"type": "eq", "latex": latex, "plain": plain}
PB = {"type": "pagebreak"}

CODE_FILE = os.path.join(ROOT, "src", "heart_ensemble.py")
GUI_FILE = os.path.join(ROOT, "src", "gui_tkinter.py")

def code_excerpt(start_marker, end_marker):
    s = open(CODE_FILE).read()
    a = s.index(start_marker); b = s.index(end_marker, a)
    return s[a:b].rstrip()

BLOCKS = []
B = BLOCKS.append

# ---------------- Chapter 1 ----------------
B(h1("Chapter 1: Introduction"))
B(h2("1.1 Introduction"))
B(p("**Heart disease** is the single largest cause of mortality in India and globally. Roughly one in four heart related deaths in India occurs "
    "in people younger than forty, which makes **early prediction** from routinely collected clinical measurements a genuinely valuable problem [1]. "
    "The measurements involved (age, sex, chest pain type, resting blood pressure, cholesterol, fasting blood sugar, resting ECG, maximum heart rate, "
    "exercise induced angina, ST depression, ST slope, number of vessels seen on fluoroscopy and thalassemia status) are cheap, non-invasive or "
    "already part of a standard cardiac work-up, so a model that predicts the presence of disease from them can act as a screening aid before "
    "expensive angiography."))
B(p("**Machine learning** offers a family of classifiers that learn the mapping from these attributes to a diagnosis. No single classifier is "
    "best on every dataset, and individual models such as a decision tree are unstable: small changes in the training data produce very "
    "different trees. **Ensemble learning** addresses this by training several models and combining their decisions. **Bagging** (bootstrap "
    "aggregating) trains each member on a bootstrap resample of the training set and averages their votes, which reduces variance; **boosting** "
    "trains members sequentially, each focusing on the mistakes of the previous ones, which reduces bias; **stacking** learns how to weight the "
    "members with a second level model."))
B(p("Muthulakshmi, Parveen and Rajeswari [1] proposed the **MPBE** model, a bagging ensemble whose base learners are a Decision Tree, a Random "
    "Forest, a Naive Bayes classifier and a Support Vector Machine, trained on Pearson correlation selected and MinMax normalised features of the "
    "UCI Cleveland dataset, and reported 95.33% accuracy. This project takes that paper as its starting point, re-implements the method from "
    "the description given, and asks two questions the original paper does not answer: does the result hold under a strict cross validated "
    "protocol, and does it hold on a larger, more heterogeneous population?"))
B(h2("1.2 Problem Definition"))
B(p("Given a patient record of thirteen clinical attributes, predict the **binary label** target = 1 (presence of heart disease, i.e. more "
    "than 50% narrowing of a major vessel) or target = 0 (absence). The specific objectives of this project are:"))
B(numbered([
 "Re-implement the MPBE pipeline of [1] (MinMax scaling, Pearson correlation feature selection, four bagging ensembles of 100 estimators each, mean aggregation) in Python with scikit-learn.",
 "Evaluate it on the UCI Cleveland dataset (303 records) and on a combined dataset built from all five UCI heart disease sources (918 unique records after de-duplication).",
 "Compare MPBE with six single classifiers (DT, RF, NB, SVM, LR, KNN) and six alternative ensembles (homogeneous Bagging, soft Voting, hard Voting, Stacking, AdaBoost, Gradient Boosting, XGBoost).",
 "Use stratified 10-fold cross validation and an 80:20 stratified hold-out, reporting accuracy, precision, recall, F1 score, ROC-AUC and confusion matrices.",
 "Quantify the contribution of the Pearson feature selection step through an ablation run on all 13 attributes.",
 "Provide a small desktop interface that loads the trained model and predicts for a single patient.",
]))
B(h2("1.3 Prerequisites"))
B(p("The reader is assumed to know the basics of **supervised classification**, the train and test split, the confusion matrix and its derived "
    "metrics, and to have seen decision trees, Naive Bayes, SVM and logistic regression. The relevant ensemble concepts (bagging, boosting, "
    "stacking, voting) are summarised in chapter 3. The implementation requires Python 3.10 or later with scikit-learn, xgboost, pandas, numpy, "
    "matplotlib and seaborn; the exact versions are listed in chapter 5."))

# ---------------- Chapter 2 ----------------
B(PB); B(h1("Chapter 2: Literature Review"))
B(p("Twenty research papers on heart disease prediction with machine learning were reviewed, with an emphasis on **ensemble methods**, "
    "**feature selection** and **evaluation protocol**, since those are the three ingredients of the MPBE model. Six of them are the papers "
    "the base paper itself builds on [2]-[7]; the remaining fourteen are later or more widely cited works that give a fuller picture of what "
    "is achievable on the same data [8]-[21]. Table 2.1 records the dataset, technique and headline result of each. Longer summaries of every "
    "paper, together with its relevance to this project, are provided in the companion document ML_MiniProject_Literature_Review.docx."))
B(table("The twenty reviewed papers", ["Ref", "Year", "Dataset", "Technique", "Reported result"],
        [[f"[{n}]", str(y), ds, tech, res] for (n, t, y, ds, tech, res, s1, s2) in LIT], [500, 500, 1700, 3400, 2200]))
B(h2("2.1 Themes in the Reviewed Literature"))
B(h3("2.1.1 Ensemble Design"))
B(p("Latha and Jeeva [9] quantified the gain of each ensemble family over weak base classifiers on Cleveland: majority voting lifted accuracy "
    "by up to 7.26 points, bagging by 6.92 and boosting by 5.94. Karadeniz et al. [14] compared the same families under one preprocessing "
    "pipeline and found boosting and stacking marginally ahead. Mienye et al. [4] showed that the **aggregation rule** matters as much as the "
    "members: an accuracy weighted, aging combination of CART models reached 93% on Cleveland where plain voting did not. Alim et al. [7] and "
    "Alqahtani et al. [3] both built heterogeneous ensembles (RF, SVM, NB and tree based learners respectively) and reported 86.8% and 88.7%. "
    "The consistent message is that combining **different learner families** helps more than enlarging one family, which is precisely the "
    "design of MPBE, but that the gain over an already strong single learner is small."))
B(h3("2.1.2 Feature Selection"))
B(p("Haq et al. [5] compared Relief, mRMR and LASSO selectors across seven classifiers and found Logistic Regression on Relief selected "
    "features best at 89%. Li et al. [15] proposed a conditional mutual information selector and reached 92.37% with a linear SVM. Mohan et al. [8] "
    "searched over feature subsets to build the widely cited HRFLM hybrid at 88.7%. Across these studies the attributes dropped are almost "
    "always **cholesterol and fasting blood sugar**, the two with the weakest correlation to the diagnosis, and the accuracy change is within "
    "one or two points. Our Pearson filter reaches the same conclusion by a simpler route."))
B(h3("2.1.3 Evaluation Protocol and Reproducibility"))
B(p("The widest spread in the literature comes from protocol rather than algorithm. Ali et al. [11] reported 100% accuracy for Random Forest on "
    "a Kaggle copy of the data later shown to contain duplicate rows, so test patients also appeared in training. Ahmad et al. [19] ran every "
    "model twice, with and without GridSearchCV, and pushed several to near 99% on single splits. Shorewala [10], by contrast, used a careful "
    "protocol and reported only about 75% for the best stacked ensemble. Adekoya et al. [21] and Mienye and Jere [16] argue explicitly for "
    "nested cross validation, leakage control, calibration and SHAP based explanation as the criteria that decide clinical usefulness. This "
    "is the direction our evaluation follows."))
B(h3("2.1.4 Datasets and Generalisation"))
B(p("Doppala et al. [6] evaluated ensembles on Cleveland, Statlog and a Mendeley cardiovascular dataset and argued that a reliable model must "
    "agree across datasets. Asif et al. [17] merged three Kaggle datasets with common attributes before tuning, reaching 98.15% on an 80:20 "
    "split. Bharti et al. [13] showed that outlier removal and normalisation move results by several points on their own. Pal et al. [2], "
    "Shah et al. [12] and Chandrasekhar and Peddakrishna [18] provide the single dataset baselines, in the 82% to 90% range, against which our "
    "Cleveland numbers can be read. Narasimhan and Victor [20] represent the current direction of metaheuristic optimizers wrapped around a "
    "stacked ensemble."))
B(h2("2.2 Observations from the Review"))
B(bullets([
 "Almost every paper uses the 13 attribute UCI Cleveland data [22]; accuracies reported on it range from about 75% to above 97%, and the spread is explained mostly by protocol (single split versus cross validation, tuning, duplicate rows) rather than by the algorithm.",
 "Heterogeneous ensembles (bagging, stacking, soft voting) consistently outperform a single Decision Tree by 8 to 12 points but beat a well regularised single learner such as Logistic Regression or Naive Bayes by only 0 to 3 points [9], [14], [18].",
 "Feature selection helps slightly and mainly removes chol and fbs, which have the weakest correlation with the label [5], [7], [15].",
 "Recent work stresses leakage control, calibration and explainability over headline accuracy [16], [21]; this project follows that direction.",
]))

# ---------------- Chapter 3 ----------------
B(PB); B(h1("Chapter 3: Proposed Work / Methodology"))
B(h2("3.1 System Architecture"))
B(p("The system is a **sequential pipeline** with six stages: data loading, preprocessing (missing value imputation), Pearson correlation "
    "feature selection, MinMax scaling, ensemble training and evaluation. Scaling is placed inside a scikit-learn Pipeline so that it is fitted "
    "only on the training fold of each cross validation split; this is the one point where our design deliberately differs from the description "
    "in [1], which scales the whole dataset before splitting."))
B(figure("architecture.png", "Architecture of the proposed system (our re-implementation of MPBE with the added comparison branch)", 0.88))
B(h3("3.1.1 Description and Source of the Datasets"))
B(p("**Dataset 1, Cleveland.** The processed Cleveland file from the UCI Machine Learning Repository [22] contains 303 patient records and "
    "14 attributes (13 predictors and the diagnosis num in the range 0 to 4). Following [1] and nearly all prior work, num is binarised to "
    f"target = 1 if num > 0. The class balance is {sc['n_neg']} negatives and {sc['n_pos']} positives. Six cells are missing (four in ca, two in thal)."))
B(p("**Dataset 2, Combined.** To test generalisation we concatenated all five UCI heart disease sources: Cleveland (303), Hungarian (294), "
    "Switzerland (123), Long Beach VA (200) [22] and Statlog (270) [23]. Statlog codes the label as 1 or 2 and was remapped to 0 or 1. The "
    "Hungarian, Swiss and VA records were collected with less complete protocols, so several attributes are missing; a cholesterol or blood "
    f"pressure of 0 is physiologically impossible and was treated as missing. After removing exact duplicate rows the dataset holds {sm['n_rows']} "
    f"unique patients ({sm['n_pos']} positive, {sm['n_neg']} negative). Duplicate removal matters because Statlog overlaps with Cleveland; leaving "
    "duplicates in would let test patients appear in training folds [11]."))
B(table("Attributes of the heart disease datasets", ["Attribute", "Meaning", "Type"], ATTRS, [1400, 6000, 1600]))
B(h3("3.1.2 Data Loading and Exploration"))
B(p("Both datasets are read with pandas, the '?' marker is mapped to NaN, and missing values are filled with the **column median** computed "
    "on the whole dataset (this is the only preprocessing not done inside the CV fold; medians of a few cells cannot leak the label). "
    "Figures 3.2 to 3.4 summarise the exploratory analysis. The classes are roughly balanced in both datasets, so accuracy is a meaningful "
    "metric and no resampling is required. The age distribution shows the expected shift: patients with disease are older on average."))
B(figure("eda_cleveland.png", "Class balance and age distribution, Cleveland", 0.62))
B(figure("eda_combined.png", "Class balance and age distribution, Combined dataset", 0.62))
B(p("The correlation heat map of figure 3.4 shows that no pair of predictors is strongly correlated with each other (the largest absolute value is "
    "below 0.6), so the redundancy removal rule of the Pearson step never fires; the step acts purely as a relevance filter. The attributes "
    "most correlated with the label are thal, ca, exang, oldpeak, cp and thalach (negative) on Cleveland, and cp, exang, thalach and oldpeak "
    "on the combined data."))
B(figure("corr_cleveland.png", "Pearson correlation matrix, Cleveland (the combined dataset shows the same pattern)", 0.56))
B(h2("3.2 Feature Selection and Normalisation"))
B(p("The base paper computes the **Pearson correlation coefficient** between attributes and removes highly correlated ones, but does not "
    "state the threshold. We make the rule explicit. For every predictor x and the label y,"))
B(eq(r"r_{xy} = \frac{\operatorname{Cov}(x, y)}{\sigma_x \sigma_y} = \frac{\sum_i (x_i-\bar{x})(y_i-\bar{y})}{\sqrt{\sum_i (x_i-\bar{x})^2}\sqrt{\sum_i (y_i-\bar{y})^2}}",
     "r_xy = Cov(x, y) / (sigma_x * sigma_y)"))
B(p("where Cov(x, y) is the covariance (how much x and y move together), and the two standard deviations in the denominator rescale it to the "
    "range -1 to +1 so that r_xy does not depend on the units of x. A predictor is **kept** if |r_xy| is at least 0.10; if two kept predictors "
    "were correlated above 0.80 with each other, the one less correlated with the label would be dropped. "
    f"On Cleveland this removes {', '.join(sc['dropped'])} and keeps {len(sc['kept'])} attributes; on the combined data it removes "
    f"{', '.join(sm['dropped'])}. Figures 3.5 and 3.6 show the ranked correlations with the cut-off line."))
B(figure("target_corr_cleveland.png", "Correlation of each attribute with the label, Cleveland (grey bars are dropped)", 0.6))
B(figure("target_corr_combined.png", "Correlation of each attribute with the label, Combined dataset", 0.6))
B(p("**MinMax normalisation** maps every attribute to the interval 0 to 1:"))
B(eq(r"x' = \frac{x - x_{\min}}{x_{\max} - x_{\min}}", "x' = (x - x_min) / (x_max - x_min)"))
B(p("Here x_min and x_max are the minimum and maximum of that attribute **in the training fold**; subtracting x_min shifts the smallest value to 0 "
    "and dividing by the range stretches the largest to 1. This matters for the SVM and KNN members, whose distance computations would otherwise "
    "be dominated by large valued attributes such as cholesterol; tree based members are unaffected by it."))
B(h2("3.3 Algorithms"))
B(h3("3.3.1 Base Learners"))
B(bullets([
 "**Decision Tree (CART):** recursively splits the data on the attribute and threshold that most reduce the Gini impurity G = 1 - sum(p_k^2), where p_k is the fraction of class k in a node. Unpruned trees fit the training data perfectly and therefore have high variance.",
 "**Random Forest:** a bagging ensemble of trees where each split considers only a random subset of attributes, which decorrelates the trees. Fifty trees are used inside MPBE and one hundred as a stand-alone baseline.",
 "**Gaussian Naive Bayes:** applies Bayes' theorem P(y|x) proportional to P(y) times the product of P(x_j|y), assuming attributes are conditionally independent and normally distributed within each class.",
 "**Support Vector Machine (RBF kernel):** finds the maximum margin separating boundary in a kernel induced feature space; probability outputs are obtained by Platt scaling so that the SVM can take part in soft aggregation.",
 "**Logistic Regression and KNN (k = 7)** are added as extra single model baselines.",
]))
B(h3("3.3.2 The MPBE Ensemble (proposed model of [1], our implementation)"))
B(p("For each of the four base learners C1 (DT), C2 (RF), C3 (NB) and C4 (SVM), a **BaggingClassifier** with n = 100 estimators is built. "
    "Each estimator is trained on a **bootstrap sample** of the training fold (the same size as the fold, drawn with replacement, so about 63% "
    "of the distinct rows appear in any one sample). The bagged probability of disease from member k is the mean over its 100 estimators, and the "
    "final MPBE score is the mean over the four members:"))
B(eq(r"P_{\text{MPBE}}(y=1\mid x) = \frac{1}{4}\sum_{k=1}^{4}\;\frac{1}{100}\sum_{b=1}^{100} P_{k,b}(y=1\mid x), \qquad \hat{y} = \mathbb{1}\left[P_{\text{MPBE}} \ge 0.5\right]",
     "P_MPBE(y=1|x) = (1/4) * sum_k (1/100) * sum_b P_kb(y=1|x);  predict 1 if P_MPBE >= 0.5"))
B(p("The inner sum averages the 100 bootstrap models of one base learner (this is what reduces variance); the outer sum averages the four "
    "different learner families (this is what lets the strengths of one family cover the weaknesses of another). The indicator 1[.] turns the "
    "averaged probability into a class by thresholding at 0.5. The paper describes the aggregation as 'the mean score', which we read as this "
    "soft average rather than a hard majority vote; a hard voting ensemble is included separately for comparison."))
B(h3("3.3.3 Comparison Ensembles"))
B(bullets([
 "**Bagging (DT, 100):** a homogeneous bagging ensemble of decision trees; isolates the effect of using four learner families instead of one.",
 "**Voting (soft / hard):** DT, RF, NB, SVM and LR trained once each on the full training fold (no bootstrapping) and combined by averaging probabilities (soft) or by majority (hard).",
 "**Stacking:** DT, RF, NB and SVM as level 0 learners; their out-of-fold predictions (5 inner folds) become the input of a Logistic Regression meta learner that learns how much to trust each member.",
 "**AdaBoost (100 rounds, learning rate 0.5):** sequentially reweights training rows so that each new stump concentrates on rows misclassified so far; the final vote weights each stump by log((1 - err)/err).",
 "**Gradient Boosting (100 trees, depth 3, learning rate 0.1):** each new tree fits the negative gradient of the log loss of the current ensemble, i.e. the residual errors.",
 "**XGBoost (100 trees, depth 3, learning rate 0.1, subsample 0.9) [25]:** a regularised, second order implementation of gradient boosting with column subsampling.",
]))

# ---------------- Chapter 4 ----------------
B(PB); B(h1("Chapter 4: Implementation (Steps / Procedure)"))
B(numbered([
 "**Download** the five UCI files (processed.cleveland.data, processed.hungarian.data, processed.switzerland.data, processed.va.data, heart.dat) into data/.",
 "**Load** each file with the 14 canonical column names, convert '?' to NaN, binarise num into target, and for the combined dataset tag the source, remap Statlog labels, set zero cholesterol and blood pressure to NaN, concatenate and drop duplicate rows.",
 "**Impute** missing cells with the column median and save the prepared CSVs (cleveland_prepared.csv, combined_prepared.csv).",
 "**Explore**: plot class balance, age histograms and the full correlation matrix.",
 "**Select features** with the Pearson rule (|r| with label at least 0.10, inter-feature |r| below 0.80) and plot the ranked correlations.",
 "**Build the model zoo**: 14 models, each wrapped in Pipeline(MinMaxScaler, classifier) so scaling is fitted per fold.",
 "**Cross validate** every model with StratifiedKFold(10, shuffle=True, random_state=42) and record accuracy, precision, recall, F1 and ROC-AUC per fold.",
 "**Hold-out evaluation**: one stratified 80:20 split (random_state=42); fit every model, save confusion matrices, ROC curves and Random Forest feature importances.",
 "**Ablation**: repeat the MPBE cross validation on all 13 attributes to measure what the selection step contributes.",
 "**Repeat** steps 4 to 9 on the combined dataset and draw the cross dataset comparison chart.",
 "**GUI**: a Tkinter window loads the prepared data, trains MPBE once, and predicts for values typed by the user.",
]))
B(p("The whole run (both datasets, all models, all figures) takes roughly eight minutes on a laptop CPU because MPBE trains 400 estimators "
    "per fold and the bagged SVM with probability calibration is the slowest member. The script caches results so that figures can be redrawn "
    "without retraining."))

# ---------------- Chapter 5 ----------------
B(PB); B(h1("Chapter 5: Experimental Setup"))
B(h2("5.1 Parameters"))
B(table("Model parameters", ["Model", "Setting"], [
 ["MPBE", "4 x BaggingClassifier, n_estimators = 100, bootstrap = True; bases DT(default), RF(50 trees), GaussianNB, SVC(rbf, probability=True); soft mean aggregation"],
 ["Decision Tree", "Gini criterion, no depth limit (scikit-learn defaults)"],
 ["Random Forest", "100 trees, sqrt(features) per split"],
 ["SVM", "RBF kernel, C = 1, gamma = scale, Platt probabilities"],
 ["Logistic Regression", "L2 penalty, max_iter = 1000"],
 ["KNN", "k = 7, Euclidean distance"],
 ["Voting", "DT + RF + NB + SVM + LR, soft (probability mean) and hard (majority)"],
 ["Stacking", "DT, RF, NB, SVM level 0; Logistic Regression meta learner; 5 inner folds"],
 ["AdaBoost", "100 decision stumps, learning rate 0.5"],
 ["Gradient Boosting", "100 trees, max_depth 3, learning rate 0.1"],
 ["XGBoost", "100 trees, max_depth 3, learning rate 0.1, subsample 0.9, colsample_bytree 0.9, logloss"],
 ["Feature selection", "|r(x, y)| >= 0.10 keep; inter-feature |r| > 0.80 drop weaker"],
 ["Validation", "StratifiedKFold(10, shuffle) and stratified 80:20 hold-out; random_state = 42 everywhere"],
], [2400, 6600]))
B(h2("5.2 Tools")); B(p("Visual Studio Code / Jupyter for development, Git for versioning, a standard laptop CPU (no GPU is needed), and matplotlib for all figures. "
    "The desktop interface uses Tkinter, which ships with Python."))
B(h2("5.3 Libraries")); B(table("Libraries and versions", ["Library", "Version", "Used for"], [
 ["scikit-learn [24]", "1.8", "All classifiers except XGBoost, Pipeline, MinMaxScaler, StratifiedKFold, metrics"],
 ["xgboost [25]", "3.4", "XGBClassifier"], ["pandas", "3.0", "Data loading, cleaning, results tables"],
 ["numpy", "2.x", "Array operations, mean aggregation in MPBE"], ["matplotlib / seaborn", "3.x / 0.13", "Figures"],
 ["tkinter", "built in", "Desktop prediction window"],
], [2600, 1400, 5000]))
B(h2("5.4 Framework")); B(p("scikit-learn's **estimator API** (fit, predict, predict_proba) is the framework; MPBE is implemented as a class that follows "
    "it (ClassifierMixin, BaseEstimator) so it can be dropped into Pipeline and cross_validate exactly like a built in model."))
B(h2("5.5 Languages")); B(p("Python 3.12. LaTeX was used to typeset the paper and this report."))

# ---------------- Chapter 6 ----------------
B(PB); B(h1("Chapter 6: Evaluation and Results / Discussion"))
B(h2("6.1 Metrics"))
B(p("With TP, TN, FP, FN the entries of the confusion matrix: **Accuracy** = (TP + TN) / (TP + TN + FP + FN) is the fraction of correct "
    "predictions; **Precision** = TP / (TP + FP) is the fraction of predicted patients who really have disease; **Recall** (sensitivity) = TP / "
    "(TP + FN) is the fraction of real patients the model catches, which is the clinically important number; **F1** = 2 x Precision x Recall / "
    "(Precision + Recall) is their harmonic mean, which is low if either is low; **ROC-AUC** is the probability that a randomly chosen diseased "
    "patient receives a higher score than a randomly chosen healthy one, computed as the area under the curve of true positive rate against "
    "false positive rate over all thresholds. Cross validation figures are means over the 10 folds with the standard deviation shown for accuracy."))
B(h2("6.2 Results on the Cleveland Dataset"))
B(table("10-fold stratified cross validation, Cleveland (303 records, 11 selected attributes)",
        ["Model", "Accuracy (%)", "Precision (%)", "Recall (%)", "F1 (%)", "ROC-AUC"], cv_table(cvc), [2900, 1700, 1300, 1200, 1100, 900]))
B(figure("cv_accuracy_cleveland.png", "10-fold CV accuracy with standard deviation, Cleveland (MPBE in red)", 0.62))
B(p(f"MPBE reaches **{pct(mpbe_c.accuracy)}% accuracy** and the **second highest ROC-AUC ({num(mpbe_c.roc_auc)})** of all fourteen models, "
    f"ranking {rank_c}th by accuracy. The differences between the top nine models (from {pct(best_c.accuracy)}% down to about 82%) are well inside one "
    "standard deviation (about 5 to 7 points on folds of 30 patients), so on this dataset they are statistically indistinguishable. What is clear is "
    "the gap to the single Decision Tree (77.2%) and to homogeneous bagging of trees (79.8%): mixing four learner families is worth about 2.5 "
    "points over bagging one family, which is exactly the argument of the base paper. Naive Bayes and Stacking tie for first by accuracy, and "
    "AdaBoost, soft Voting and Logistic Regression are essentially level with MPBE."))
B(table("Stratified 80:20 hold-out, Cleveland (61 test patients)",
        ["Model", "Accuracy (%)", "Precision (%)", "Recall (%)", "F1 (%)", "ROC-AUC"], ho_table(hoc), [2900, 1700, 1300, 1200, 1100, 900]))
B(figure("roc_cleveland.png", "ROC curves on the hold-out split, Cleveland", 0.40))
B(figure("cm_cleveland.png", "Confusion matrices on the hold-out split, Cleveland", 0.70))
B(p(f"On the single hold-out split MPBE scores {pct(mpbe_hc.accuracy)}% with recall {pct(mpbe_hc.recall)}% and AUC {num(mpbe_hc.roc_auc)}: "
    f"it misses only {int(round((1-mpbe_hc.recall)*28))} of the 28 diseased patients. The 61 patient test set is small enough that a single "
    "patient is 1.6 points of accuracy, which is why the hold-out ranking (Random Forest first at 90.2%) differs from the cross validated one. "
    "This is also the most likely explanation of the 95.33% in [1]: a favourable single split, plus scaling and feature selection performed "
    "on the whole dataset before splitting. Under a leakage free 10-fold protocol no model in our study exceeds 84% mean accuracy, which is "
    "consistent with [8], [9], [10] and [21]."))
B(h2("6.3 Results on the Combined Dataset"))
B(table("10-fold stratified cross validation, Combined dataset (918 records, 11 selected attributes)",
        ["Model", "Accuracy (%)", "Precision (%)", "Recall (%)", "F1 (%)", "ROC-AUC"], cv_table(cvm), [2900, 1700, 1300, 1200, 1100, 900]))
B(figure("cv_accuracy_combined.png", "10-fold CV accuracy with standard deviation, Combined dataset", 0.62))
B(p(f"On the larger and noisier combined population **MPBE is the best model** by accuracy ({pct(mpbe_m.accuracy)}%) and ties for the best "
    f"ROC-AUC ({num(mpbe_m.roc_auc)}) with Stacking. The standard deviations shrink to about 3 points because each fold now holds 92 patients, so "
    "the ranking is more trustworthy than on Cleveland. Absolute accuracy is lower for every model because a third of the records come from "
    "hospitals whose protocols left slope, ca and thal mostly unrecorded (they were median imputed). The single Decision Tree drops to 70.8%, "
    "so the value of ensembling grows as the data gets harder: MPBE is 10 points above its weakest member here versus 5 points on Cleveland."))
B(table("Stratified 80:20 hold-out, Combined dataset (184 test patients)",
        ["Model", "Accuracy (%)", "Precision (%)", "Recall (%)", "F1 (%)", "ROC-AUC"], ho_table(hom), [2900, 1700, 1300, 1200, 1100, 900]))
B(figure("roc_combined.png", "ROC curves on the hold-out split, Combined dataset", 0.40))
B(figure("cm_combined.png", "Confusion matrices on the hold-out split, Combined dataset", 0.70))
B(p(f"MPBE is also first on the hold-out split ({pct(mpbe_hm.accuracy)}%, recall {pct(mpbe_hm.recall)}%). "
    "AdaBoost has the highest AUC (0.90) but slightly lower accuracy at the 0.5 threshold, which suggests that threshold tuning could give it back "
    "a point or two; we kept the default threshold for all models to keep the comparison fair."))
B(h2("6.4 Effect of Feature Selection (Ablation)"))
B(table("MPBE with and without Pearson feature selection (10-fold CV)", ["Dataset", "Attributes", "Accuracy (%)", "F1 (%)", "ROC-AUC"], [
 ["Cleveland", f"11 selected", pct(mpbe_c.accuracy), pct(mpbe_c.f1), num(mpbe_c.roc_auc)],
 ["Cleveland", "all 13", pct(abl_c['accuracy']), pct(abl_c['f1']), num(abl_c['roc_auc'])],
 ["Combined", f"11 selected", pct(mpbe_m.accuracy), pct(mpbe_m.f1), num(mpbe_m.roc_auc)],
 ["Combined", "all 13", pct(abl_m['accuracy']), pct(abl_m['f1']), num(abl_m['roc_auc'])],
], [1800, 1800, 1800, 1800, 1800]))
B(p("Removing the two weakest attributes changes accuracy by less than half a point in either direction and raises AUC by 0.003 on Cleveland. "
    "The **Pearson step is therefore harmless but not decisive**; its practical benefit is a slightly cheaper model (two fewer inputs, which "
    "for a screening tool means two fewer measurements) rather than higher accuracy. This agrees with [9] and [7], where feature selection "
    "added at most one or two points."))
B(h2("6.5 Feature Importance"))
B(figure("importance_cleveland.png", "Random Forest Gini importance of the selected attributes, Cleveland", 0.55))
B(figure("importance_combined.png", "Random Forest Gini importance of the selected attributes, Combined dataset", 0.55))
B(p("On Cleveland the Random Forest relies most on thal, thalach, cp, ca and oldpeak; on the combined data cp, age, thalach and oldpeak "
    "dominate while thal and ca lose importance because they are largely imputed outside Cleveland. Chest pain type, maximum heart rate and "
    "exercise ST depression are therefore the attributes a clinician would want recorded first, which agrees with [15] and [16]."))
B(h2("6.6 Cross Dataset Comparison"))
B(figure("cross_dataset.png", "10-fold CV accuracy of every model on both datasets", 0.62))
B(p("Every model loses 1 to 7 points when moving from Cleveland to the combined population, and the loss is largest for the models that "
    "leaned on thal and ca. MPBE loses only about 1.6 points and moves from mid table to first place, which is the behaviour one wants from a "
    "model meant for deployment across hospitals: **stable rather than spectacular**."))

# ---------------- Chapter 7 ----------------
B(PB); B(h1("Chapter 7: Graphical User Interface"))
B(p("A minimal desktop interface was written with **Tkinter** (src/gui_tkinter.py). On start it loads cleveland_prepared.csv, applies the same "
    "Pearson selection, fits Pipeline(MinMaxScaler, MPBE) once (about 20 seconds), and then shows one entry field per selected attribute with the "
    "Cleveland median as default. Pressing Predict displays the ensemble probability of disease and the class, coloured green or red. No web server "
    "or browser is required; the window is shown in figure 7.1."))
B(figure("gui_screenshot.png", "Tkinter prediction window with an example patient", 0.38))

# ---------------- Chapter 8 ----------------
B(PB); B(h1("Chapter 8: Code"))
B(h2("8.1 Libraries and Functions Used"))
B(table("Main functions in heart_ensemble.py", ["Function / class", "Purpose"], [
 ["load_cleveland(), load_combined()", "Read the UCI files, binarise the label, clean impossible zeros, remove duplicates"],
 ["preprocess(df)", "Median imputation; returns the missing value count per column"],
 ["pearson_select(df, 0.10, 0.80)", "Pearson relevance filter and redundancy rule; returns kept and dropped attributes"],
 ["class MPBE", "The proposed ensemble: four bagging members, soft mean aggregation; scikit-learn compatible"],
 ["build_models()", "The 14 model zoo, each wrapped in Pipeline(MinMaxScaler, clf)"],
 ["evaluate_cv(), evaluate_holdout()", "Stratified 10-fold CV and 80:20 hold-out with all metrics, ROC data, confusion matrices and importances"],
 ["fig_*()", "All figures used in this report"],
 ["run_dataset(df, tag, title)", "Runs the whole procedure of chapter 4 for one dataset and writes results/summary_<tag>.json"],
], [3200, 5800]))
B(h2("8.2 Core Code Listing"))
B(p("The complete script is about 500 lines. The two listings below define the method itself: the Pearson selection rule and the MPBE "
    "ensemble class. The full file, the Colab notebook and the GUI are submitted alongside this report."))
B(code(code_excerpt("def pearson_select", "# --------------------------------------------------------------------------\n# 3."), "Pearson correlation feature selection"))
B(code(code_excerpt("class MPBE", "# --------------------------------------------------------------------------\n# 4."), "The MPBE ensemble class"))
B(p("The model zoo (build_models), the cross validation routine (evaluate_cv), the hold-out routine and the figure functions are omitted here "
    "for brevity and are included in full in the submitted source file src/heart_ensemble.py, in the Colab notebook "
    "notebook/heart_ensemble.ipynb and in the GUI file src/gui_tkinter.py."))

# ---------------- Chapter 9 ----------------
B(PB); B(h1("Chapter 9: Conclusion"))
B(p(f"This project re-implemented the MPBE heterogeneous bagging ensemble of [1] and evaluated it honestly. On the Cleveland dataset MPBE "
    f"achieves {pct(mpbe_c.accuracy)}% cross validated accuracy and an ROC-AUC of {num(mpbe_c.roc_auc)}, level with the strongest single and ensemble "
    f"models and about 5 points above a single decision tree, but well short of the 95.33% claimed in the original paper, which we attribute to a "
    f"single favourable split and preprocessing applied before splitting. On the larger five source dataset MPBE is the best of fourteen models "
    f"({pct(mpbe_m.accuracy)}% accuracy, AUC {num(mpbe_m.roc_auc)}) and loses the least accuracy when the population becomes more heterogeneous, "
    "which makes it a sound default for tabular clinical screening. The Pearson feature selection step is cheap and harmless but adds under half "
    "a point. Future work could tune the decision threshold for recall, add SHAP explanations for clinicians, and validate on a local Indian cohort."))

# ---------------- Chapter 10 ----------------
B(PB); B(h1("Chapter 10: References"))
B({"type": "references", "items": REFS})

if __name__ == "__main__":
    out = {"meta": META, "abstract": ABSTRACT, "abbrev": ABBREV, "blocks": BLOCKS}
    json.dump(out, open(os.path.join(ROOT, "docs", "report_content.json"), "w"), indent=1)
    print("blocks:", len(BLOCKS))
