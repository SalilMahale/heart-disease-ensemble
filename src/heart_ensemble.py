"""
Heart Disease Prediction using Ensemble Learning
-------------------------------------------------
Re-implementation and extension of the MPBE model
(MinMax normalized - Pearson coefficient Bagging Ensemble) proposed by
Muthulakshmi, Parveen and Rajeswari (IJST, 2023), with our own additions:

  * two datasets: UCI Cleveland (303 rows) and a combined 5-source
    heart dataset (Cleveland + Hungarian + Switzerland + Long Beach VA + Statlog)
  * extra ensembles: Voting, Stacking, AdaBoost, Gradient Boosting, XGBoost
  * fuller evaluation: accuracy, precision, recall, F1, ROC-AUC, confusion
    matrices, stratified 10-fold cross-validation with a fixed seed

Author : Salil Mahale, BE CMPN B, SFIT
Subject: Machine Learning (CSL 701), 2026-27
"""

import os
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import StratifiedKFold, train_test_split, cross_validate
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (RandomForestClassifier, BaggingClassifier,
                              VotingClassifier, StackingClassifier,
                              AdaBoostClassifier, GradientBoostingClassifier)
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix,
                             roc_curve)
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")
SEED = 42
FORCE = os.environ.get("FORCE", "0") == "1"   # set FORCE=1 to recompute everything
np.random.seed(SEED)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
FIG = os.path.join(ROOT, "figures")
RES = os.path.join(ROOT, "results")
for d in (FIG, RES):
    os.makedirs(d, exist_ok=True)

COLS = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach",
        "exang", "oldpeak", "slope", "ca", "thal", "num"]
FEATURES = COLS[:-1]

plt.rcParams.update({"font.family": "serif", "font.size": 11})


# --------------------------------------------------------------------------
# 1. DATA LOADING
# --------------------------------------------------------------------------
def load_cleveland():
    """UCI Cleveland: 303 rows, 13 features, target num in {0..4} -> binary."""
    df = pd.read_csv(os.path.join(DATA, "processed.cleveland.data"),
                     names=COLS, na_values="?")
    df["target"] = (df["num"] > 0).astype(int)
    df = df.drop(columns="num")
    return df


def load_combined():
    """
    Combined 5-source dataset (1190 rows before de-duplication):
    Cleveland, Hungarian, Switzerland, Long Beach VA (UCI processed files)
    + Statlog (UCI, space separated, target 1/2).
    Zeros in trestbps / chol are physiologically impossible -> treated as missing.
    """
    frames = []
    for src in ["cleveland", "hungarian", "switzerland", "va"]:
        d = pd.read_csv(os.path.join(DATA, f"processed.{src}.data"),
                        names=COLS, na_values="?")
        d["source"] = src
        frames.append(d)
    st = pd.read_csv(os.path.join(DATA, "heart.dat"), sep=r"\s+", names=COLS)
    st["num"] = st["num"] - 1          # Statlog uses 1 = absent, 2 = present
    st["source"] = "statlog"
    frames.append(st)
    df = pd.concat(frames, ignore_index=True)
    df["target"] = (df["num"] > 0).astype(int)
    df = df.drop(columns="num")
    for c in ["trestbps", "chol"]:
        df.loc[df[c] == 0, c] = np.nan
    n_before = len(df)
    df = df.drop_duplicates(subset=FEATURES + ["target"]).reset_index(drop=True)
    print(f"[combined] {n_before} rows -> {len(df)} after removing duplicates")
    return df


def preprocess(df):
    """Median imputation for missing values (paper found none in Cleveland;
    Cleveland actually has 6 missing cells in ca/thal, the other sources many)."""
    df = df.copy()
    missing = df[FEATURES].isna().sum()
    for c in FEATURES:
        if df[c].isna().any():
            df[c] = df[c].fillna(df[c].median())
    return df, missing


# --------------------------------------------------------------------------
# 2. PEARSON CORRELATION FEATURE SELECTION  (paper's step, made explicit)
# --------------------------------------------------------------------------
def pearson_select(df, min_target_corr=0.10, max_inter_corr=0.80):
    """
    Keep a feature if |corr(feature, target)| >= min_target_corr.
    Additionally, if two kept features are correlated above max_inter_corr,
    drop the one less correlated with the target (redundancy removal,
    which is what the original paper means by 'highly correlated features
    are removed').
    """
    corr_t = df[FEATURES + ["target"]].corr()["target"].drop("target")
    keep = [f for f in FEATURES if abs(corr_t[f]) >= min_target_corr]
    inter = df[keep].corr().abs()
    dropped_redundant = []
    for i, a in enumerate(keep):
        for b in keep[i + 1:]:
            if a in dropped_redundant or b in dropped_redundant:
                continue
            if inter.loc[a, b] > max_inter_corr:
                weaker = a if abs(corr_t[a]) < abs(corr_t[b]) else b
                dropped_redundant.append(weaker)
    keep = [f for f in keep if f not in dropped_redundant]
    dropped = [f for f in FEATURES if f not in keep]
    return keep, dropped, corr_t


# --------------------------------------------------------------------------
# 3. THE MPBE MODEL (our implementation of the paper's proposed ensemble)
# --------------------------------------------------------------------------
class MPBE(ClassifierMixin, BaseEstimator):
    """
    MinMax normalized - Pearson coefficient Bagging Ensemble.
    Four bagging ensembles (100 bootstrap estimators each) are built, one per
    base learner (Decision Tree, Random Forest, Naive Bayes, SVM). The final
    prediction is the mean of the four bagged probability scores, i.e. soft
    aggregation, which is the 'aggregate by mean score' step in the paper.
    MinMax scaling and Pearson selection are applied before this class.
    """

    def __init__(self, n_estimators=100, random_state=SEED):
        self.n_estimators = n_estimators
        self.random_state = random_state

    def _bases(self):
        rs = self.random_state
        return {
            "dt": DecisionTreeClassifier(random_state=rs),
            "rf": RandomForestClassifier(n_estimators=50, random_state=rs),
            "nb": GaussianNB(),
            "svm": SVC(kernel="rbf", probability=True, random_state=rs),
        }

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        self.members_ = {}
        for name, base in self._bases().items():
            bag = BaggingClassifier(estimator=base,
                                    n_estimators=self.n_estimators,
                                    bootstrap=True,
                                    random_state=self.random_state,
                                    n_jobs=-1)
            self.members_[name] = bag.fit(X, y)
        return self

    def predict_proba(self, X):
        probs = [m.predict_proba(X) for m in self.members_.values()]
        return np.mean(probs, axis=0)

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]


# --------------------------------------------------------------------------
# 4. MODEL ZOO  (baselines from the paper + our extensions)
# --------------------------------------------------------------------------
def build_models():
    rs = SEED
    dt = DecisionTreeClassifier(random_state=rs)
    rf = RandomForestClassifier(n_estimators=100, random_state=rs)
    nb = GaussianNB()
    svm = SVC(kernel="rbf", probability=True, random_state=rs)
    lr = LogisticRegression(max_iter=1000, random_state=rs)
    knn = KNeighborsClassifier(n_neighbors=7)

    models = {
        # ---- individual baselines used in the reference paper ----
        "Decision Tree": dt,
        "Random Forest": rf,
        "Naive Bayes": nb,
        "SVM (RBF)": svm,
        # ---- extra baselines ----
        "Logistic Regression": lr,
        "KNN (k=7)": knn,
        # ---- the paper's proposed model, our implementation ----
        "MPBE (proposed, replicated)": MPBE(n_estimators=100, random_state=rs),
        # ---- our ensemble extensions ----
        "Bagging (DT, 100)": BaggingClassifier(estimator=clone(dt), n_estimators=100,
                                                random_state=rs, n_jobs=-1),
        "Voting (soft)": VotingClassifier(
            estimators=[("dt", clone(dt)), ("rf", clone(rf)), ("nb", clone(nb)),
                        ("svm", clone(svm)), ("lr", clone(lr))], voting="soft"),
        "Voting (hard)": VotingClassifier(
            estimators=[("dt", clone(dt)), ("rf", clone(rf)), ("nb", clone(nb)),
                        ("svm", clone(svm)), ("lr", clone(lr))], voting="hard"),
        "Stacking (LR meta)": StackingClassifier(
            estimators=[("dt", clone(dt)), ("rf", clone(rf)), ("nb", clone(nb)),
                        ("svm", clone(svm))],
            final_estimator=LogisticRegression(max_iter=1000), cv=5, n_jobs=-1),
        "AdaBoost": AdaBoostClassifier(n_estimators=100, learning_rate=0.5,
                                       random_state=rs),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=100,
                                                        learning_rate=0.1,
                                                        max_depth=3,
                                                        random_state=rs),
        "XGBoost": XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=3,
                                 subsample=0.9, colsample_bytree=0.9,
                                 eval_metric="logloss", random_state=rs,
                                 verbosity=0),
    }
    # every model is wrapped in a pipeline so MinMax scaling is fitted
    # inside each fold (no leakage)
    return {k: Pipeline([("scaler", MinMaxScaler()), ("clf", v)])
            for k, v in models.items()}


# --------------------------------------------------------------------------
# 5. EVALUATION
# --------------------------------------------------------------------------
SCORING = {"accuracy": "accuracy", "precision": "precision", "recall": "recall",
           "f1": "f1", "roc_auc": "roc_auc"}


def evaluate_cv(models, X, y, tag):
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=SEED)
    rows, fold_acc = [], {}
    for name, model in models.items():
        scoring = SCORING if "hard" not in name else {k: v for k, v in SCORING.items() if k != "roc_auc"}
        cv = cross_validate(model, X, y, cv=skf, scoring=scoring,
                            n_jobs=1 if "MPBE" in name or "Stacking" in name else -1)
        r = {"Model": name}
        for m in SCORING:
            r[m] = cv[f"test_{m}"].mean() if m in scoring else np.nan
            r[m + "_std"] = cv[f"test_{m}"].std() if m in scoring else np.nan
        rows.append(r)
        fold_acc[name] = cv["test_accuracy"]
        print(f"  [{tag}] {name:30s} acc={r['accuracy']:.4f}+/-{r['accuracy_std']:.4f} "
              f"f1={r['f1']:.4f} auc={r['roc_auc']:.4f}")
    res = pd.DataFrame(rows).sort_values("accuracy", ascending=False)
    res.to_csv(os.path.join(RES, f"cv_results_{tag}.csv"), index=False)
    pd.DataFrame(fold_acc).to_csv(os.path.join(RES, f"cv_folds_{tag}.csv"), index=False)
    return res, fold_acc


def evaluate_holdout(models, X, y, tag, features):
    """80/20 stratified split like the paper, for confusion matrices and ROC."""
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2,
                                          stratify=y, random_state=SEED)
    rows, rocs, cms = [], {}, {}
    for name, model in models.items():
        model.fit(Xtr, ytr)
        yp = model.predict(Xte)
        try:
            ys = model.predict_proba(Xte)[:, 1]
        except AttributeError:                 # hard voting has no proba
            ys = yp
        rows.append({"Model": name,
                     "accuracy": accuracy_score(yte, yp),
                     "precision": precision_score(yte, yp),
                     "recall": recall_score(yte, yp),
                     "f1": f1_score(yte, yp),
                     "roc_auc": roc_auc_score(yte, ys)})
        rocs[name] = roc_curve(yte, ys)
        cms[name] = confusion_matrix(yte, yp)
    res = pd.DataFrame(rows).sort_values("accuracy", ascending=False)
    res.to_csv(os.path.join(RES, f"holdout_results_{tag}.csv"), index=False)
    # feature importance from the RF inside the holdout fit
    rf = models["Random Forest"].named_steps["clf"]
    imp = pd.Series(rf.feature_importances_, index=features).sort_values(ascending=False)
    imp.to_csv(os.path.join(RES, f"rf_importance_{tag}.csv"))
    return res, rocs, cms, imp


# --------------------------------------------------------------------------
# 6. FIGURES
# --------------------------------------------------------------------------
def fig_eda(df, tag, title):
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.4))
    counts = df["target"].value_counts().sort_index()
    ax[0].bar(["No disease (0)", "Disease (1)"], counts.values,
              color=["#4C72B0", "#C44E52"])
    for i, v in enumerate(counts.values):
        ax[0].text(i, v + 2, str(v), ha="center")
    ax[0].set_title(f"Class distribution ({title})")
    ax[0].set_ylabel("Patients")
    sns.histplot(data=df, x="age", hue="target", bins=20, ax=ax[1],
                 palette=["#4C72B0", "#C44E52"], alpha=0.7)
    ax[1].set_title("Age distribution by class")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG, f"eda_{tag}.png"), dpi=200)
    plt.close()


def fig_corr(df, tag, title):
    plt.figure(figsize=(9, 7.5))
    sns.heatmap(df[FEATURES + ["target"]].corr(), annot=True, fmt=".2f",
                cmap="coolwarm", center=0, annot_kws={"size": 8},
                cbar_kws={"shrink": 0.8})
    plt.title(f"Pearson correlation matrix ({title})")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG, f"corr_{tag}.png"), dpi=200)
    plt.close()


def fig_target_corr(corr_t, keep, tag, title, thr):
    s = corr_t.reindex(corr_t.abs().sort_values(ascending=False).index)
    colors = ["#55A868" if f in keep else "#8C8C8C" for f in s.index]
    plt.figure(figsize=(8, 3.8))
    plt.bar(s.index, s.values, color=colors)
    plt.axhline(thr, ls="--", c="k", lw=0.8)
    plt.axhline(-thr, ls="--", c="k", lw=0.8)
    plt.ylabel("Pearson r with target")
    plt.title(f"Feature to target correlation ({title}); grey = dropped")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG, f"target_corr_{tag}.png"), dpi=200)
    plt.close()


def fig_cv_bar(res, tag, title):
    r = res.sort_values("accuracy")
    plt.figure(figsize=(9, 5.2))
    colors = ["#C44E52" if "MPBE" in m else "#4C72B0" for m in r["Model"]]
    plt.barh(r["Model"], r["accuracy"] * 100, xerr=r["accuracy_std"] * 100,
             color=colors, capsize=3)
    for i, (v, sd) in enumerate(zip(r["accuracy"], r["accuracy_std"])):
        plt.text((v + sd) * 100 + 0.4, i, f"{v*100:.2f}", va="center", fontsize=9)
    plt.xlabel("10-fold CV accuracy (%)")
    plt.xlim(max(0, r["accuracy"].min() * 100 - 8), 100)
    plt.title(f"Model comparison, 10-fold stratified CV ({title})")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG, f"cv_accuracy_{tag}.png"), dpi=200)
    plt.close()


def fig_cv_box(fold_acc, tag, title):
    df = pd.DataFrame(fold_acc) * 100
    order = df.mean().sort_values(ascending=False).index
    plt.figure(figsize=(10, 4.6))
    sns.boxplot(data=df[order], color="#DDE5F0", fliersize=3)
    plt.ylabel("Fold accuracy (%)")
    plt.xticks(rotation=40, ha="right", fontsize=9)
    plt.title(f"Spread of fold accuracies ({title})")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG, f"cv_box_{tag}.png"), dpi=200)
    plt.close()


def fig_roc(rocs, tag, title, subset):
    plt.figure(figsize=(5.6, 5))
    for name in subset:
        fpr, tpr, _ = rocs[name]
        auc = np.trapezoid(tpr, fpr)
        plt.plot(fpr, tpr, lw=1.6, label=f"{name} (AUC={auc:.3f})")
    plt.plot([0, 1], [0, 1], "k--", lw=0.8)
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title(f"ROC curves, 20% hold-out ({title})")
    plt.legend(fontsize=8, loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG, f"roc_{tag}.png"), dpi=200)
    plt.close()


def fig_cm(cms, tag, title, names):
    fig, axes = plt.subplots(1, len(names), figsize=(3.6 * len(names), 3.4))
    for ax, n in zip(np.atleast_1d(axes), names):
        sns.heatmap(cms[n], annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
                    xticklabels=["Pred 0", "Pred 1"], yticklabels=["True 0", "True 1"])
        ax.set_title(n, fontsize=10)
    fig.suptitle(f"Confusion matrices, 20% hold-out ({title})")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG, f"cm_{tag}.png"), dpi=200)
    plt.close()


def fig_importance(imp, tag, title):
    plt.figure(figsize=(7, 3.8))
    plt.bar(imp.index, imp.values, color="#8172B2")
    plt.ylabel("Gini importance")
    plt.title(f"Random Forest feature importance ({title})")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG, f"importance_{tag}.png"), dpi=200)
    plt.close()


def fig_cross_dataset(res_c, res_m):
    a = res_c.set_index("Model")["accuracy"] * 100
    b = res_m.set_index("Model")["accuracy"] * 100
    df = pd.DataFrame({"Cleveland": a, "Combined": b}).sort_values("Cleveland")
    plt.figure(figsize=(9, 5.2))
    y = np.arange(len(df))
    plt.barh(y - 0.2, df["Cleveland"], 0.4, label="Cleveland (303)", color="#4C72B0")
    plt.barh(y + 0.2, df["Combined"], 0.4, label="Combined 5-source", color="#DD8452")
    plt.yticks(y, df.index)
    plt.xlabel("10-fold CV accuracy (%)")
    plt.xlim(max(0, df.min().min() - 8), 100)
    plt.legend(loc="lower right")
    plt.title("Accuracy on both datasets")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG, "cross_dataset.png"), dpi=200)
    plt.close()


# --------------------------------------------------------------------------
# 7. MAIN
# --------------------------------------------------------------------------
def run_dataset(df, tag, title, thr=0.10):
    print(f"\n==== {title} ====")
    df, missing = preprocess(df)
    print("missing per column:\n", missing[missing > 0].to_string())
    keep, dropped, corr_t = pearson_select(df, min_target_corr=thr)
    print("kept features :", keep)
    print("dropped       :", dropped)
    fig_eda(df, tag, title)
    fig_corr(df, tag, title)
    fig_target_corr(corr_t, keep, tag, title, thr)

    X, y = df[keep].values, df["target"].values
    cv_path = os.path.join(RES, f"cv_results_{tag}.csv")
    if os.path.exists(cv_path) and not FORCE:        # resume support
        res_cv = pd.read_csv(cv_path)
        fold_acc = pd.read_csv(os.path.join(RES, f"cv_folds_{tag}.csv")).to_dict("list")
    else:
        models = build_models()
        res_cv, fold_acc = evaluate_cv(models, X, y, tag)
    fig_cv_bar(res_cv, tag, title)
    fig_cv_box(fold_acc, tag, title)

    ho_path = os.path.join(RES, f"holdout_results_{tag}.csv")
    if os.path.exists(ho_path) and not FORCE:
        res_ho = pd.read_csv(ho_path)
        imp = pd.read_csv(os.path.join(RES, f"rf_importance_{tag}.csv"), index_col=0).iloc[:, 0]
    else:
        models = build_models()
        res_ho, rocs, cms, imp = evaluate_holdout(models, X, y, tag, keep)
        fig_roc(rocs, tag, title, ["MPBE (proposed, replicated)", "Random Forest",
                                   "SVM (RBF)", "Stacking (LR meta)", "XGBoost",
                                   "Decision Tree"])
        fig_cm(cms, tag, title, ["MPBE (proposed, replicated)", "Random Forest", "XGBoost"])
        fig_importance(imp, tag, title)

    # also: full 13-feature run of MPBE to quantify the effect of selection
    sum_path = os.path.join(RES, f"summary_{tag}.json")
    if os.path.exists(sum_path) and not FORCE:
        return res_cv, res_ho
    models_all = build_models()
    Xall = df[FEATURES].values
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=SEED)
    cv_all = cross_validate(models_all["MPBE (proposed, replicated)"], Xall, y,
                            cv=skf, scoring=SCORING)
    ablation = {m: float(cv_all[f"test_{m}"].mean()) for m in SCORING}

    summary = {
        "n_rows": int(len(df)), "n_pos": int(y.sum()), "n_neg": int((y == 0).sum()),
        "missing": {k: int(v) for k, v in missing.items() if v > 0},
        "kept": keep, "dropped": dropped,
        "target_corr": {k: round(float(v), 4) for k, v in corr_t.items()},
        "mpbe_all13_cv": ablation,
        "rf_importance": {k: round(float(v), 4) for k, v in imp.items()},
    }
    with open(os.path.join(RES, f"summary_{tag}.json"), "w") as f:
        json.dump(summary, f, indent=2)
    return res_cv, res_ho


if __name__ == "__main__":
    clev = load_cleveland()
    comb = load_combined()
    clev.to_csv(os.path.join(DATA, "cleveland_prepared.csv"), index=False)
    comb.to_csv(os.path.join(DATA, "combined_prepared.csv"), index=False)

    res_c, ho_c = run_dataset(clev, "cleveland", "Cleveland")
    res_m, ho_m = run_dataset(comb.drop(columns="source"), "combined", "Combined")
    fig_cross_dataset(res_c, res_m)
    print("\nDone. Results in", RES, "figures in", FIG)
