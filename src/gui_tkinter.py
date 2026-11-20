"""
Tkinter front end for the MPBE heart disease model.
Run:  python src/gui_tkinter.py
Trains Pipeline(MinMaxScaler, MPBE) once on the prepared Cleveland data
(about 20 s), then predicts for the values typed in the entry boxes.
"""
import os, sys
import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from heart_ensemble import (load_cleveland, preprocess, pearson_select, MPBE,
                            Pipeline, MinMaxScaler, SEED)

HELP = {
    "age": "years", "sex": "1 male / 0 female", "cp": "1-4 chest pain type",
    "trestbps": "resting BP mm Hg", "chol": "cholesterol mg/dl", "fbs": "1 if > 120 mg/dl",
    "restecg": "0 / 1 / 2", "thalach": "max heart rate", "exang": "1 yes / 0 no",
    "oldpeak": "ST depression", "slope": "1 up / 2 flat / 3 down", "ca": "0-3 vessels",
    "thal": "3 normal / 6 fixed / 7 reversible",
}


def train():
    df, _ = preprocess(load_cleveland())
    keep, _, _ = pearson_select(df)
    model = Pipeline([("scaler", MinMaxScaler()),
                      ("clf", MPBE(n_estimators=100, random_state=SEED))])
    model.fit(df[keep].values, df["target"].values)
    return model, keep, df[keep].median()


def main():
    root = tk.Tk()
    root.title("Heart Disease Prediction - MPBE ensemble")
    root.resizable(False, False)
    frm = ttk.Frame(root, padding=14)
    frm.grid()
    ttk.Label(frm, text="Heart Disease Prediction (MPBE bagging ensemble)",
              font=("Times New Roman", 14, "bold")).grid(column=0, row=0, columnspan=3, pady=(0, 8))
    status = ttk.Label(frm, text="Training model on Cleveland data, please wait...")
    status.grid(column=0, row=1, columnspan=3)
    root.update()

    model, keep, med = train()
    status.config(text=f"Model ready. Enter {len(keep)} attributes and press Predict.")

    entries = {}
    for i, f in enumerate(keep, start=2):
        ttk.Label(frm, text=f).grid(column=0, row=i, sticky="w", padx=(0, 8))
        e = ttk.Entry(frm, width=10)
        e.insert(0, f"{med[f]:g}")
        e.grid(column=1, row=i, pady=2)
        ttk.Label(frm, text=HELP.get(f, ""), foreground="#555").grid(column=2, row=i, sticky="w", padx=(8, 0))
        entries[f] = e

    result = ttk.Label(frm, text="", font=("Times New Roman", 12, "bold"))
    result.grid(column=0, row=len(keep) + 3, columnspan=3, pady=(10, 0))

    def predict():
        try:
            x = [[float(entries[f].get()) for f in keep]]
        except ValueError:
            messagebox.showerror("Input error", "All fields must be numeric.")
            return
        prob = model.predict_proba(x)[0, 1]
        if prob >= 0.5:
            result.config(text=f"Heart disease LIKELY  (probability {prob:.2f})", foreground="#B22222")
        else:
            result.config(text=f"Heart disease UNLIKELY  (probability {prob:.2f})", foreground="#2E8B57")

    ttk.Button(frm, text="Predict", command=predict).grid(column=0, row=len(keep) + 2, columnspan=3, pady=(10, 0))
    root.mainloop()


if __name__ == "__main__":
    main()
