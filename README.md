# Machine Learning Assignment 2 — AIML ZG565 / DSE ZG565

**BITS Pilani · Work Integrated Learning Programmes**  
**Marks:** 15 · **Deadline:** 18-Aug-2026 23:59

---

## a. Problem Statement

Email spam detection is a classic binary classification problem. Given numerical features extracted from an email (word frequencies, character frequencies, and capital-run statistics), the goal is to classify each message as **spam** or **non-spam (ham)**.

In this assignment I:

1. Train **five** supervised classification models on the same public dataset  
2. Evaluate each model using Accuracy, AUC, Precision, Recall, F1, and MCC  
3. Build an interactive **Streamlit** web app to upload test data, select a model, and view metrics + confusion matrix  
4. Deploy the app on **Streamlit Community Cloud** and share evaluation links  

---

## b. Dataset Description

| Item | Detail |
|------|--------|
| **Dataset name** | Spambase |
| **Source** | UCI Machine Learning Repository (also available via OpenML) |
| **URL** | https://archive.ics.uci.edu/dataset/94/spambase |
| **Task type** | Binary classification |
| **Instances** | **4601** (≥ 500 ✓) |
| **Features** | **57** continuous predictors (≥ 12 ✓) |
| **Target** | `0` = non-spam · `1` = spam |
| **Class balance** | non-spam ≈ 2788 · spam ≈ 1813 |
| **Train / Test** | 75% / 25%, **stratified**, `random_state=42` |
| **Test rows saved** | 1151 rows in `test_data.csv` |

**Feature summary:**  
- 48 word-frequency attributes (e.g., `word_freq_make`, `word_freq_address`, …)  
- 6 character-frequency attributes (e.g., `char_freq_;`, `char_freq_$`, …)  
- 3 capital-letter statistics (`capital_run_length_average`, `longest`, `total`)

**Preprocessing notes:**  
- No missing values in the source data  
- **StandardScaler** applied inside pipelines for Logistic Regression and kNN  
- Decision Tree, Gaussian Naive Bayes, and Random Forest use unscaled features  

---

## c. GitHub Repository Link

**https://github.com/soniyajuneja000/ML_Assignment_2**

(Push from your machine after creating the empty repo on [GitHub Dashboard](https://github.com/dashboard) — see commands below.)

### Required repository structure (guideline Step 3)

```text
ML_Assignment_2/
├── app.py                 # Streamlit application
├── requirements.txt
├── README.md
├── test_data.csv          # Hold-out test set used in experiments
└── model/
    ├── train_models.py    # Training + evaluation script
    ├── metrics.json
    ├── comparison_table.csv
    ├── logistic_regression.joblib
    ├── decision_tree.joblib
    ├── knn.joblib
    ├── naive_bayes.joblib
    └── random_forest_ensemble.joblib
```

### Live Streamlit App Link
**https://mlassignment2-gmvr8wnoh6htxrwr5rrzt9.streamlit.app/**

---

## d. Models Used — Comparison Table

Evaluation on the **stratified hold-out test set** (1151 samples):

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---------------|----------|-----|-----------|--------|-----|-----|
| Logistic Regression | 0.9227 | 0.9732 | 0.9047 | 0.8987 | 0.9017 | 0.8380 |
| Decision Tree | 0.9027 | 0.9487 | 0.8800 | 0.8722 | 0.8761 | 0.7960 |
| kNN | 0.9140 | 0.9664 | 0.8971 | 0.8833 | 0.8901 | 0.8195 |
| Naive Bayes | 0.8306 | 0.9469 | 0.7133 | 0.9537 | 0.8162 | 0.6893 |
| Random Forest (Ensemble) | **0.9409** | **0.9850** | **0.9427** | 0.9053 | **0.9236** | **0.8759** |

### Observations on model performance

| ML Model Name | Observation about model performance |
|---------------|-------------------------------------|
| **Logistic Regression** | Strong linear baseline (Accuracy 0.9227, AUC 0.9732). Balanced Precision/Recall after `class_weight='balanced'` and scaling. Interpretable and stable, but slightly behind the ensemble on this high-dimensional sparse word-frequency space. |
| **Decision Tree** | Competitive (Accuracy 0.9027) but lowest among tree-free strong models on MCC (0.7960). Captures non-linear interactions; depth/`min_samples_leaf` limits reduce overfit, yet a single tree still has higher variance than Random Forest. |
| **kNN** | Good Accuracy (0.9140) and AUC (0.9664). Distance weighting helps, and StandardScaler is essential because feature scales differ (word freqs vs capital-run lengths). Still sensitive to local neighbourhood noise in 57-D space. |
| **Naive Bayes (Gaussian)** | Highest Recall (0.9537) but lowest Precision (0.7133) and Accuracy (0.8306). Independence assumption is violated by correlated word frequencies, causing more false positives (legitimate mail flagged as spam). AUC remains decent (0.9469). |
| **Random Forest (Ensemble)** | **Best overall** — highest Accuracy, AUC, Precision, F1, and MCC. Bagging of many trees reduces variance vs a single Decision Tree and handles mixed-scale numeric features well without mandatory scaling. |
| **Overall Winner** | **Random Forest (Ensemble)** — best Accuracy (0.9409), AUC (0.9850), F1 (0.9236), and MCC (0.8759) on the hold-out test set. |

---

## How to run locally

```bash
cd ML_Assignment_2
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python model/train_models.py       # (re)train + refresh test_data.csv / metrics
streamlit run app.py
```

## Streamlit features implemented (guideline Step 6)

| Feature | Status |
|---------|--------|
| (a) Dataset upload option (CSV) + bundled test data | ✅ |
| (b) Model selection dropdown | ✅ |
| (c) Display of evaluation metrics | ✅ |
| (d) Confusion matrix / classification report | ✅ |

## Deploy on Streamlit Community Cloud

1. Push this folder to GitHub  
2. Open https://streamlit.io/cloud → Sign in with GitHub  
3. **New App** → select repo → branch `main` → main file `app.py`  
4. Deploy → copy the live URL into this README and your submission PDF  

## BITS Virtual Lab note

Run the training script and/or Streamlit app on **BITS Virtual Lab**, capture **ONE** screenshot, and include it in the submission PDF (1 mark).

---

*This submission is prepared for BITS WILP ML Assignment 2. Dataset is public (UCI Spambase).*
