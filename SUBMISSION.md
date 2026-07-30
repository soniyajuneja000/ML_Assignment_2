# ML Assignment 2 — Submission PDF Content

**Course:** AIML ZG565 / DSE ZG565 — Machine Learning  
**Assignment:** 2 (15 Marks)  
**Deadline:** 18-Aug-2026 23:59 (SUBMIT — not draft)

Paste this content into your final PDF **in this exact order** (guideline Section 2).

---

## 1. GitHub Repository Link

```
https://github.com/soniyajuneja000/ML_Assignment_2
```

Repository includes: complete source (`app.py`, `model/train_models.py`), `requirements.txt`, `README.md`, `test_data.csv`, and saved model files under `model/`.

---

## 2. Live Streamlit App Link

```
https://<YOUR_APP_NAME>.streamlit.app
```

Deployed via Streamlit Community Cloud; opens an interactive frontend for test CSV upload, model selection, metrics, and confusion matrix.

---

## 3. Screenshot — BITS Virtual Lab

**[INSERT ONE SCREENSHOT HERE]**

Suggested capture on BITS Virtual Lab:

```bash
pip install -r requirements.txt
python model/train_models.py
# and/or
streamlit run app.py
```

---

## 4. GitHub README Content (Section 3 — Step 5)

### a. Problem Statement

Email spam detection is a binary classification task. Using 57 numeric features from emails (word/character frequencies and capital-run statistics), classify each message as spam or non-spam. This assignment trains five classifiers, reports Accuracy/AUC/Precision/Recall/F1/MCC, and deploys a Streamlit app for interactive evaluation on hold-out test data.

### b. Dataset Description [1 mark]

| Item | Detail |
|------|--------|
| Name | Spambase |
| Source | UCI ML Repository / OpenML |
| URL | https://archive.ics.uci.edu/dataset/94/spambase |
| Task | Binary classification |
| Instances | 4601 (≥ 500 ✓) |
| Features | 57 (≥ 12 ✓) |
| Target | 0 = non-spam, 1 = spam |
| Class counts | non-spam ≈ 2788; spam ≈ 1813 |
| Split | 75/25 stratified, random_state=42 |
| Test CSV | 1151 rows (`test_data.csv`) |

Preprocessing: no missing values; StandardScaler for Logistic Regression & kNN pipelines.

### c. GitHub Repository Link [1 mark]

Same as Section 1. All required files present (`app.py`, `requirements.txt`, `README.md`, `test_data.csv`, `model/`).

### d. Models Used — Comparison Table [5 marks]

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---------------|----------|-----|-----------|--------|-----|-----|
| Logistic Regression | 0.9227 | 0.9732 | 0.9047 | 0.8987 | 0.9017 | 0.8380 |
| Decision Tree | 0.9027 | 0.9487 | 0.8800 | 0.8722 | 0.8761 | 0.7960 |
| kNN | 0.9140 | 0.9664 | 0.8971 | 0.8833 | 0.8901 | 0.8195 |
| Naive Bayes | 0.8306 | 0.9469 | 0.7133 | 0.9537 | 0.8162 | 0.6893 |
| Random Forest (Ensemble) | 0.9409 | 0.9850 | 0.9427 | 0.9053 | 0.9236 | 0.8759 |

### Observations [3 marks]

| ML Model Name | Observation about model performance |
|---------------|-------------------------------------|
| Logistic Regression | Strong baseline (Acc 0.9227, AUC 0.9732). Balanced P/R with class weights + scaling; slightly behind RF on this feature space. |
| Decision Tree | Acc 0.9027; higher variance than RF. Depth/leaf limits help, but a single tree underperforms the ensemble. |
| kNN | Acc 0.9140, AUC 0.9664. Needs StandardScaler; distance weighting helps in 57-D space. |
| Naive Bayes | High Recall (0.9537) but low Precision (0.7133). Independence assumption fails for correlated word freqs → more false positives. |
| Random Forest (Ensemble) | Best Acc/AUC/F1/MCC. Bagging reduces tree variance; robust without mandatory scaling. |
| **Overall Winner** | **Random Forest (Ensemble)** |

---

## Marks mapping (for your checklist)

| Component | Marks |
|-----------|-------|
| Models + metrics + GitHub + README observations | 10 |
| Streamlit features (upload, dropdown, metrics, CM) | 4 |
| BITS Virtual Lab screenshot | 1 |
| **Total** | **15** |
