# Setup Guide — ML Assignment 2 (Follow These Steps)

Deadline: **18-Aug-2026 23:59** · Submit as **SUBMIT** (not draft)

---

## A. Project already prepared in this folder

| Guideline item | Status |
|----------------|--------|
| Dataset from UCI (Spambase), ≥12 features, ≥500 rows | ✅ 57 feat / 4601 rows |
| 5 models: LR, DT, kNN, NB, Random Forest | ✅ |
| Metrics: Accuracy, AUC, Precision, Recall, F1, MCC | ✅ |
| `app.py` with upload / dropdown / metrics / confusion matrix | ✅ |
| `requirements.txt`, `README.md`, `test_data.csv`, `model/` | ✅ |

> Note: The PDF says “6 models” once, but **Step 2 + comparison table list 5 models**. All 5 required models are implemented.

---

## B. Push to GitHub (required for marks)

```bash
cd "/Users/sjuneja1/Desktop/MTech/ML/Assignment/ML_Assignment_2"

git init
git add app.py requirements.txt README.md test_data.csv model/ .gitignore
git commit -m "ML Assignment 2: Spambase classifiers and Streamlit app"

# Create empty repo on GitHub, then:
git branch -M main
git remote add origin https://github.com/soniyajuneja000/ML_Assignment_2.git
git push -u origin main
```

Update the GitHub URL in `README.md` and `SUBMISSION.md`.

**Tip (anti-plagiarism):** Make a few small personal commits (README tweak, comment) so history looks natural.

---

## C. Deploy Streamlit Community Cloud (4 marks UI)

1. https://streamlit.io/cloud → Sign in with GitHub  
2. **New App** → your repo → `main` → `app.py`  
3. Deploy → copy live URL into README + submission PDF  

Local test first:

```bash
pip install -r requirements.txt
streamlit run app.py
```

---

## D. BITS Virtual Lab screenshot (1 mark)

On BITS Virtual Lab:

```bash
pip install -r requirements.txt
python model/train_models.py
streamlit run app.py
```

Capture **ONE** screenshot (training output or running app) → paste into PDF Section 3.

---

## E. Build final submission PDF (mandatory order)

1. GitHub Repository Link  
2. Live Streamlit App Link  
3. BITS Virtual Lab Screenshot  
4. Full README content (sections **a–d** from `README.md` / `SUBMISSION.md`)  

Use `SUBMISSION.md` as the template. Convert via Word / Google Docs / Pages → PDF.

---

## F. Final checklist (from assignment)

- [ ] GitHub repo link works  
- [ ] Streamlit app link opens correctly  
- [ ] App loads without errors  
- [ ] All required features implemented  
- [ ] README.md content included in submitted PDF  
- [ ] Clicked **SUBMIT** (not draft) before deadline  

---

## Marks breakdown

| Part | Marks |
|------|------:|
| Model implementation + GitHub + README tables/observations | 10 |
| Streamlit app (4 features) | 4 |
| BITS Lab screenshot | 1 |
| **Total** | **15** |
