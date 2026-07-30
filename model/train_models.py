"""
BITS WILP — Machine Learning Assignment 2
Train five classification models on UCI Spambase and save artifacts.

Dataset: Spambase (UCI / OpenML) — binary spam vs non-spam
Min requirements met: 57 features (>=12), 4601 instances (>=500)
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = Path(__file__).resolve().parent


def load_spambase() -> tuple[pd.DataFrame, list[str]]:
    """Load UCI Spambase via OpenML (public repository)."""
    bundle = fetch_openml(name="spambase", version=1, as_frame=True, parser="auto")
    df = bundle.frame.copy()
    # OpenML uses column name 'class' for the label
    if "class" not in df.columns:
        raise RuntimeError("Unexpected Spambase schema: missing 'class' column")
    df = df.rename(columns={"class": "target"})
    df["target"] = df["target"].astype(int)
    feature_names = [c for c in df.columns if c != "target"]
    return df, feature_names


def metrics_dict(y_true, y_pred, y_proba) -> dict:
    return {
        "Accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "AUC": round(float(roc_auc_score(y_true, y_proba)), 4),
        "Precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "Recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "F1": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "MCC": round(float(matthews_corrcoef(y_true, y_pred)), 4),
    }


def build_estimators() -> dict:
    """Five models required by the assignment comparison table."""
    return {
        "Logistic Regression": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        max_iter=3000,
                        random_state=42,
                        solver="lbfgs",
                        class_weight="balanced",
                    ),
                ),
            ]
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=12,
            min_samples_leaf=8,
            random_state=42,
            class_weight="balanced",
        ),
        "kNN": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", KNeighborsClassifier(n_neighbors=9, weights="distance")),
            ]
        ),
        "Naive Bayes": GaussianNB(),
        "Random Forest (Ensemble)": RandomForestClassifier(
            n_estimators=200,
            max_depth=14,
            min_samples_leaf=4,
            random_state=42,
            n_jobs=1,
            class_weight="balanced_subsample",
        ),
    }


def safe_filename(name: str) -> str:
    return (
        name.lower()
        .replace(" ", "_")
        .replace("(", "")
        .replace(")", "")
        .replace("-", "")
    )


def main() -> None:
    df, feature_names = load_spambase()
    X = df[feature_names]
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    # Hold-out test CSV for Streamlit upload demos (guideline Step 3)
    test_df = X_test.copy()
    test_df["target"] = y_test.values
    test_path = ROOT / "test_data.csv"
    test_df.to_csv(test_path, index=False)
    print(f"Saved {test_path} | rows={len(test_df)} cols={test_df.shape[1]}")

    estimators = build_estimators()
    metrics_table: dict = {}
    reports: dict = {}
    target_names = ["non-spam (0)", "spam (1)"]

    for name, estimator in estimators.items():
        print(f"\n=== Training: {name} ===")
        estimator.fit(X_train, y_train)
        y_pred = estimator.predict(X_test)
        y_proba = estimator.predict_proba(X_test)[:, 1]
        m = metrics_dict(y_test, y_pred, y_proba)
        metrics_table[name] = m
        reports[name] = {
            "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
            "classification_report": classification_report(
                y_test, y_pred, target_names=target_names, output_dict=True
            ),
        }
        out = MODEL_DIR / f"{safe_filename(name)}.joblib"
        joblib.dump(estimator, out)
        print(m)
        print(f"Saved -> {out.name}")

    meta = {
        "dataset_name": "Spambase",
        "dataset_source": "UCI Machine Learning Repository / OpenML",
        "dataset_url": "https://archive.ics.uci.edu/dataset/94/spambase",
        "task": "Binary classification (spam vs non-spam)",
        "n_instances": int(len(df)),
        "n_features": len(feature_names),
        "feature_names": feature_names,
        "target_names": target_names,
        "class_distribution": {
            "non-spam (0)": int((y == 0).sum()),
            "spam (1)": int((y == 1).sum()),
        },
        "train_size": int(len(X_train)),
        "test_size": int(len(X_test)),
        "random_state": 42,
        "metrics": metrics_table,
        "reports": reports,
    }
    meta_path = MODEL_DIR / "metrics.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    table = pd.DataFrame(metrics_table).T
    table.to_csv(MODEL_DIR / "comparison_table.csv")
    print("\n===== COMPARISON TABLE =====")
    print(table.to_string())
    print(f"\nWinner by F1: {table['F1'].idxmax()}")
    print(f"Winner by MCC: {table['MCC'].idxmax()}")


if __name__ == "__main__":
    main()
