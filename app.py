"""
BITS WILP — ML Assignment 2
Interactive Streamlit demo for Spambase spam classification.

Required UI features (guideline Step 6):
  a) CSV upload (test data)
  b) Model selection dropdown
  c) Evaluation metrics display
  d) Confusion matrix / classification report
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
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

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "model"

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "kNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Random Forest (Ensemble)": "random_forest_ensemble.joblib",
}


@st.cache_resource
def load_models_and_meta():
    models = {}
    for label, fname in MODEL_FILES.items():
        path = MODEL_DIR / fname
        if path.exists():
            models[label] = joblib.load(path)
    meta = {}
    meta_path = MODEL_DIR / "metrics.json"
    if meta_path.exists():
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)
    return models, meta


def live_metrics(y_true, y_pred, y_proba) -> dict:
    return {
        "Accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "AUC": round(float(roc_auc_score(y_true, y_proba)), 4),
        "Precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "Recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "F1": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "MCC": round(float(matthews_corrcoef(y_true, y_pred)), 4),
    }


def draw_cm(cm, labels):
    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="YlOrRd",
        xticklabels=labels,
        yticklabels=labels,
        ax=ax,
        cbar=True,
    )
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("Actual label")
    ax.set_title("Confusion Matrix")
    fig.tight_layout()
    return fig


def main():
    st.set_page_config(
        page_title="Spambase Classifier | ML Assignment 2",
        page_icon="📧",
        layout="wide",
    )

    st.markdown(
        """
        <style>
        .main-title {font-size:1.8rem; font-weight:700; color:#1a365d; margin-bottom:0.2rem;}
        .sub {color:#4a5568; margin-bottom:1rem;}
        div[data-testid="stMetricValue"] {font-size:1.15rem;}
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.markdown('<p class="main-title">📧 Spambase Email Spam Classification</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="sub">BITS Pilani WILP · AIML/DSE ZG565 · Machine Learning Assignment 2</p>',
        unsafe_allow_html=True,
    )

    models, meta = load_models_and_meta()
    if not models:
        st.error("Trained models not found. Run: `python model/train_models.py`")
        st.stop()

    # ----- Sidebar -----
    with st.sidebar:
        st.header("⚙️ Controls")
        # Feature (b): model selection dropdown
        model_name = st.selectbox("Select classification model", list(models.keys()))
        st.divider()
        st.subheader("Dataset card")
        st.write(f"**Name:** {meta.get('dataset_name', 'Spambase')}")
        st.write(f"**Instances:** {meta.get('n_instances', '—')}")
        st.write(f"**Features:** {meta.get('n_features', '—')}")
        st.write(f"**Train / Test:** {meta.get('train_size')} / {meta.get('test_size')}")
        st.caption("Source: UCI Spambase · Binary (0=ham, 1=spam)")

    # ----- Feature (a): CSV upload -----
    st.header("1. Upload test data (CSV)")
    st.info(
        "Streamlit free tier has limited capacity — upload **test data only**. "
        "CSV must include the Spambase feature columns. Optional `target` column enables scoring."
    )
    uploaded = st.file_uploader("Choose test CSV", type=["csv"], key="test_csv")
    use_bundle = st.checkbox("Use bundled hold-out `test_data.csv`", value=uploaded is None)

    if uploaded is not None:
        df = pd.read_csv(uploaded)
        source_note = "Uploaded file"
    elif use_bundle and (ROOT / "test_data.csv").exists():
        df = pd.read_csv(ROOT / "test_data.csv")
        source_note = "Bundled test_data.csv"
    else:
        st.warning("Please upload a CSV or enable the bundled test set.")
        st.stop()

    st.caption(f"Data source: **{source_note}** · shape `{df.shape}`")
    st.dataframe(df.head(8), use_container_width=True)

    feature_names = meta.get("feature_names") or [c for c in df.columns if c != "target"]
    missing = [c for c in feature_names if c not in df.columns]
    if missing:
        st.error(f"Missing required features ({len(missing)}): {missing[:8]} ...")
        st.stop()

    X = df[feature_names]
    has_y = "target" in df.columns
    y_true = df["target"].astype(int).values if has_y else None

    model = models[model_name]
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)[:, 1]

    # ----- Predictions -----
    st.header(f"2. Predictions — {model_name}")
    preview = pd.DataFrame(
        {
            "Predicted": y_pred,
            "Label": np.where(y_pred == 1, "spam", "non-spam"),
            "P(spam)": np.round(y_proba, 4),
        }
    )
    if has_y:
        preview.insert(0, "Actual", y_true)
    st.dataframe(preview.head(25), use_container_width=True)

    # ----- Feature (c): metrics -----
    st.header("3. Evaluation metrics")
    if has_y:
        scores = live_metrics(y_true, y_pred, y_proba)
        cols = st.columns(6)
        for col, (k, v) in zip(cols, scores.items()):
            col.metric(k, f"{v:.4f}")

        st.subheader("Comparison table (training-time hold-out)")
        if meta.get("metrics"):
            cmp = pd.DataFrame(meta["metrics"]).T
            st.dataframe(
                cmp.style.highlight_max(axis=0, color="#c6f6d5"),
                use_container_width=True,
            )
            winner = cmp["F1"].idxmax()
            st.success(f"Overall winner on hold-out F1: **{winner}**")
    else:
        st.warning("`target` column not found — showing reference metrics from training only.")
        if model_name in meta.get("metrics", {}):
            ref = meta["metrics"][model_name]
            cols = st.columns(6)
            for col, (k, v) in zip(cols, ref.items()):
                col.metric(k, f"{v:.4f}")

    # ----- Feature (d): confusion matrix / report -----
    st.header("4. Confusion matrix & classification report")
    if has_y:
        labels = ["non-spam", "spam"]
        cm = confusion_matrix(y_true, y_pred)
        left, right = st.columns(2)
        with left:
            st.pyplot(draw_cm(cm, labels))
        with right:
            st.text(
                classification_report(y_true, y_pred, target_names=labels, digits=4)
            )
    else:
        st.info("Add a `target` column to the CSV to unlock confusion matrix scoring.")

    st.divider()
    st.caption(
        "AIML ZG565 Assignment 2 · Models: Logistic Regression · Decision Tree · "
        "kNN · Naive Bayes · Random Forest · Metrics: Accuracy, AUC, Precision, Recall, F1, MCC"
    )


if __name__ == "__main__":
    main()
