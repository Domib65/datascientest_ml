# Datascientest ML — Streamlit App

A multi-page Streamlit app to reproduce our end-to-end ML workflow from the original Google Colab notebooks:
- **Preprocessing (Regression)**
- **Modeling (Regression)**
- **Classification Modeling**
- (Optional) EDA/Exploration pages

The app supports CSV/XLS/XLSX uploads, automatic delimiter/encoding detection, runtime toggles for fast experimentation, model export, and basic explainability (SHAP).

---

## ✨ Features

- **CSV/XLSX ingestion** with auto delimiter/encoding detection (override via UI)
- **Preprocessing**: cleaning, feature engineering (`avg_mass`, `power_to_weight`, `log_co2`)
- **Modeling (Regression)**: multiple baseline models, optional CV & tuning
- **Modeling (Classification)**: pipeline from raw → features → training + metrics
- **Runtime controls**: sampling, enable/disable heavy models (KNN/SVR), enable XGBoost, CV, tuning
- **Model export** (`.pkl` / `.joblib`) and **predictions export** (CSV)
- **Explainability**: SHAP summary for tree models

---

---

## 🧰 Requirements

- Python **3.12**
- (Option A) Poetry — recommended
- (Option B) `pip` + `venv`

**macOS + XGBoost**: install OpenMP once (needed by XGBoost)

    brew install libomp

---

## 🚀 Quick start

### Option A — with Poetry (recommended)

    poetry install
    poetry run streamlit run app.py

### Option B — with pip

    python -m venv .venv
    source .venv/bin/activate        # Windows: .venv\Scripts\activate
    pip install -r requirements.txt
    streamlit run app.py

---

## 🧪 Using the app

- **Upload** your dataset (CSV/XLS/XLSX).  
  If auto-detection guesses wrong, use the **Delimiter** and **Encoding** selectors at the top.
- **Regression Modeling**:
  - Choose features/target.
  - Use **Runtime settings**:
    - *Max training rows (sampling)* — keep it responsive for large datasets.
    - *Include heavy models (KNN/SVR)* — off by default; they’re slow on big data.
    - *Include XGBoost* — enable if available on your machine.
    - *5-fold CV* and *Hyperparameter tuning* — optional (off by default).
  - **Save model** to `./models/` and **download** it as `.pkl`.
  - **Predictions** can be downloaded as CSV.
- **Classification Modeling**:
  - End-to-end pipeline: mapping, cleaning, one-hot, baseline, metrics (confusion matrix, ROC/PR), optional tuning.

---

## 📦 Requirements file 

If you use Poetry 1.2+ with the export plugin:

    poetry self add poetry-plugin-export
    poetry export -f requirements.txt -o requirements.txt --without-hashes

Fallback:

    poetry run pip freeze > requirements.txt

Commit the `requirements.txt` so non-Poetry users can `pip install -r requirements.txt`.

---

## 🧱 Git setup

`.gitignore` (recommended):

    .venv/
    __pycache__/
    *.pyc
    .ipynb_checkpoints/
    data/
    models/
    .streamlit/secrets.toml
    .DS_Store

