import io, os, re
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import (
    confusion_matrix, classification_report, roc_curve, roc_auc_score,
    precision_recall_curve, average_precision_score
)
from sklearn.ensemble import RandomForestClassifier
import joblib

st.title("🧮 Classification – Modeling (Notebook → Page)")

st.markdown(
    "Lade Jahresdateien (CSV/XLS/XLSX) von **2006–2015**. "
    "Für CSV versuchen wir zuerst `latin1` + `;` (wie im Colab-Notebook)."
)

# ---------- Upload ----------
files = st.file_uploader(
    "Dateien auswählen (mehrfach)", type=["csv", "xls", "xlsx"], accept_multiple_files=True
)
if not files:
    st.info("Bitte eine oder mehrere Jahresdateien hochladen.")
    st.stop()

def read_any(uploaded_file) -> pd.DataFrame:
    name = uploaded_file.name.lower()
    if name.endswith(".csv"):
        try:
            return pd.read_csv(uploaded_file, delimiter=";", encoding="ISO-8859-1", header=0)
        except Exception:
            uploaded_file.seek(0)
            return pd.read_csv(uploaded_file)
    # xls/xlsx
    return pd.read_excel(uploaded_file)

def guess_year(name: str) -> int | None:
    m = re.search(r"20(0[6-9]|1[0-5])", name)
    return int(m.group()) if m else None

# Jahr pro Datei wählen
years = []
st.subheader("Jahreszuordnung")
for f in files:
    guess = guess_year(f.name) or 2015
    years.append(
        st.selectbox(
            f"Jahr für **{f.name}**",
            list(range(2006, 2016)),
            index=list(range(2006, 2016)).index(guess),
            key=f"yr_{f.name}",
        )
    )

# ---------- Mapping-Varianten (aus Notebook) ----------
column_mapping_variants = {
    "brand": ["lib_mrq_utac", "lib_mrq_doss", "mrq_utac", "marque"],
    "model_file": ["lib_mod_doss", "mod_utac", "modele version", "modèle dossier"],
    "fuel_type": ["cod_cbr", "energ", "carburant", "typ_cbr"],
    "hybrid": ["hybride", "hybride"],
    "admin_power": ["puiss_admin_98", "puiss_admin", "puissance administrative"],
    "max_power_kw": ["puiss_max", "puissance maximale (kw)", "puissance reelle"],
    "gearbox_type": ["typ_boite_nb_rapp", "boîte de vitesse", "bv"],
    "consumption_city": ["conso_urb", "conso_urb_93", "consommation urbaine (l/100km)", "urb"],
    "consumption_highway": ["conso_exurb", "consommation extra-urbaine (l/100km)", "ex-urb"],
    "consumption_combined": ["conso_mixte", "consommation mixte (l/100km)", "mixte"],
    "co2_emission": ["co2", "co2_mixte", "co2 (g/km)", "co2"],
}

def map_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = df.columns.astype(str).str.strip().str.lower()
    col_map = {}
    for std_col, variants in column_mapping_variants.items():
        for v in variants:
            v_clean = v.strip().lower()
            if v_clean in df.columns:
                col_map[v_clean] = std_col
                break
    df = df.rename(columns=col_map)
    # doppelte Spalten nach Mapping entfernen
    df = df.loc[:, ~df.columns.duplicated(keep="first")]
    return df

# ---------- Einlesen & Vereinheitlichen ----------
all_data = []
for f, yr in zip(files, years):
    try:
        df = read_any(f)
        df.columns = df.columns.astype(str).str.strip().str.lower()
        df = df.loc[:, ~df.columns.duplicated(keep="first")]
        df = map_columns(df)
        df["year"] = yr
        df = df.loc[:, ~df.columns.duplicated(keep="first")]
        all_data.append(df)
    except Exception as e:
        st.error(f"Fehler beim Laden **{f.name}**: {e}")

if not all_data:
    st.error("Keine Daten geladen.")
    st.stop()

df_all = pd.concat(all_data, ignore_index=True)
st.success(f"✅ Gesamtform: {df_all.shape}")
st.write("📋 Spalten:", df_all.columns.tolist())

# ---------- Relevante Spalten & Ziel ----------
relevant_columns = [
    "brand", "model_file", "fuel_type", "hybrid",
    "admin_power", "max_power_kw", "gearbox_type",
    "consumption_city", "consumption_highway", "consumption_combined",
    "co2_emission", "year",
]
missing_rel = [c for c in relevant_columns if c not in df_all.columns]
if missing_rel:
    st.warning(f"Folgende erwartete Spalten fehlen (werden übersprungen, ggf. Upload prüfen): {missing_rel}")

keep_cols = [c for c in relevant_columns if c in df_all.columns]
df = df_all.dropna(subset=keep_cols)[keep_cols].copy()

# Ziel: CO2 > 150 → 1
df["churn"] = (df["co2_emission"] > 150).astype(int)

# ---------- Typkonvertierungen / Cleaning ----------
for col in ["max_power_kw", "consumption_city", "consumption_highway", "consumption_combined"]:
    if col in df.columns:
        df[col] = df[col].astype(str).str.replace(",", ".").str.strip()
        df[col] = pd.to_numeric(df[col], errors="coerce")

for col in ["brand", "fuel_type", "hybrid", "gearbox_type"]:
    if col in df.columns:
        df[col] = df[col].astype(str).str.upper().str.replace(r"\s+", " ", regex=True).str.strip()

# Dubletten in brand vereinheitlichen
if "brand" in df.columns:
    df["brand"] = df["brand"].replace({
        "ALFA ROMEO": "ALFA-ROMEO",
        "ALFA ROMEO ": "ALFA-ROMEO",
        "MERCEDES BENZ": "MERCEDES-BENZ",
        "ROLLS ROYCE": "ROLLS-ROYCE",
        "JAGUAR LAND ROVER LIMITED ": "JAGUAR LAND ROVER LIMITED",
    })

# Leakage vermeiden: combined raus
if "consumption_combined" in df.columns:
    df.drop(columns=["consumption_combined"], inplace=True)

# Neue Features
if all(c in df.columns for c in ["consumption_city", "consumption_highway"]):
    df["city_to_highway_ratio"] = df["consumption_city"] / df["consumption_highway"]
    df["consumption_delta"] = df["consumption_city"] - df["consumption_highway"]

# ---------- One-Hot Encoding (gesamter df, wie im Notebook) ----------
df_encoded = df.copy()
for cat_col, prefix in [
    ("brand", "brand"),
    ("fuel_type", "fuel"),
    ("hybrid", "hybrid"),
    ("gearbox_type", "gearbox"),
]:
    if cat_col in df_encoded.columns:
        df_encoded = df_encoded.join(pd.get_dummies(df_encoded[cat_col], prefix=prefix))

# Finale Features
to_drop = [c for c in ["brand", "fuel_type", "hybrid", "gearbox_type", "model_file", "co2_emission", "churn"] if c in df_encoded.columns]
data = df_encoded.drop(columns=to_drop)
target = df["churn"].copy()

st.write("🔎 Feature-Matrix Shape:", data.shape)

# ---------- Train/Test Split ----------
X_train, X_test, y_train, y_test = train_test_split(
    data, target, test_size=0.2, random_state=321, stratify=target
)

st.write("Train/Test Shapes:", X_train.shape, X_test.shape)

# ---------- RandomForest Baseline ----------
st.subheader("🌲 RandomForest – Baseline")
clf = RandomForestClassifier(n_jobs=-1, random_state=321)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

# Confusion Matrix & Report
cm = confusion_matrix(y_test, y_pred)
st.write("**Confusion Matrix**")
st.dataframe(pd.DataFrame(cm, index=["True 0", "True 1"], columns=["Pred 0", "Pred 1"]))

st.write("**Accuracy:**", float(clf.score(X_test, y_test)))

st.write("**Classification Report**")
report = classification_report(y_test, y_pred, digits=4)
st.text(report)

# ROC
if hasattr(clf, "predict_proba"):
    y_probas = clf.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_probas)
    roc_auc = roc_auc_score(y_test, y_probas)
    fig = plt.figure(figsize=(7, 5))
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
    plt.plot([0, 1], [0, 1], "--", color="gray")
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.title("ROC Curve – Baseline RF"); plt.legend(); plt.grid(True, alpha=0.3)
    st.pyplot(fig); plt.close(fig)

# Klassenverteilung
unique, counts = np.unique(y_test, return_counts=True)
st.write("**Verteilung in y_test:**", {int(u): int(c) for u, c in zip(unique, counts)})

# ---------- Tuning (GridSearchCV) ----------
st.subheader("🎛️ GridSearchCV (RandomForest)")
run_grid = st.checkbox("GridSearchCV ausführen (kann dauern)", value=False)
if run_grid:
    param_grid = {
        "n_estimators": [100],
        "max_depth": [10, None],
        "min_samples_split": [2],
        "min_samples_leaf": [1, 5],
        "max_features": ["sqrt"],  # entspricht deinem Raster („sqrt“ war genutzt)
    }
    rf = RandomForestClassifier(random_state=321, n_jobs=-1)
    grid_search = GridSearchCV(
        estimator=rf,
        param_grid=param_grid,
        cv=3,
        scoring="f1",
        verbose=2,
        n_jobs=-1,
    )
    with st.spinner("GridSearch läuft…"):
        grid_search.fit(X_train, y_train)
    st.success("GridSearch fertig.")
    st.write("**Beste Parameter:**", grid_search.best_params_)
    st.write("**Bestes F1 (CV):**", float(grid_search.best_score_))
    best_rf = grid_search.best_estimator_
    y_pred_best = best_rf.predict(X_test)
    st.write("**Confusion Matrix (best):**")
    st.dataframe(pd.DataFrame(confusion_matrix(y_test, y_pred_best),
                              index=["True 0", "True 1"], columns=["Pred 0", "Pred 1"]))
    st.write("**Classification Report (best):**")
    st.text(classification_report(y_test, y_pred_best, digits=4))

    # Speichern & Download
    st.subheader("💾 Bestes Modell speichern")
    os.makedirs("models", exist_ok=True)
    if st.button("Bestes Modell in ./models/ speichern"):
        path = "models/best_randomforest_classifier.pkl"
        joblib.dump(best_rf, path)
        st.success(f"Gespeichert: {path}")

    buf_model = io.BytesIO()
    joblib.dump(best_rf, buf_model)
    st.download_button("📦 Bestes Modell herunterladen (.pkl)", buf_model.getvalue(),
                       file_name="best_randomforest_classifier.pkl")

# ---------- Subsampled Model + Feature Importances + PR/ROC ----------
st.subheader("🧪 Subsampled RF (30%) + Feature Importances + PR/ROC")
sub_size = st.slider("Train-Subsample", 0.1, 0.9, 0.3, 0.05)
X_train_sub, _, y_train_sub, _ = train_test_split(X_train, y_train, train_size=sub_size, random_state=42, stratify=y_train)

clf_sub = RandomForestClassifier(n_jobs=-1, random_state=42)
clf_sub.fit(X_train_sub, y_train_sub)

# Top-20 Feature Importances
importances = clf_sub.feature_importances_
indices = np.argsort(importances)[::-1][:20]
features = X_train.columns[indices]

fig = plt.figure(figsize=(10, 6))
plt.title("Top 20 Feature Importances (Subsampled RF)")
plt.barh(range(len(indices)), importances[indices][::-1], align="center")
plt.yticks(range(len(indices)), features[::-1])
plt.xlabel("Importance"); plt.tight_layout()
st.pyplot(fig); plt.close(fig)

# PR-Curve
y_probas_sub = clf_sub.predict_proba(X_test)[:, 1]
precision, recall, _ = precision_recall_curve(y_test, y_probas_sub)
ap = average_precision_score(y_test, y_probas_sub)

fig = plt.figure(figsize=(7, 5))
plt.plot(recall, precision, label=f"AP = {ap:.2f}")
plt.xlabel("Recall"); plt.ylabel("Precision")
plt.title("Precision-Recall Curve (Subsampled RF)")
plt.legend(); plt.grid(True, alpha=0.3)
st.pyplot(fig); plt.close(fig)

# ROC-Curve
fpr, tpr, _ = roc_curve(y_test, y_probas_sub)
roc_auc = roc_auc_score(y_test, y_probas_sub)
fig = plt.figure(figsize=(7, 5))
plt.plot(fpr, tpr, label=f"ROC AUC = {roc_auc:.2f}")
plt.plot([0, 1], [0, 1], "--", color="gray")
plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
plt.title("ROC Curve – Subsampled RF"); plt.legend(); plt.grid(True, alpha=0.3)
st.pyplot(fig); plt.close(fig)

# Speichern subsampled Modell
os.makedirs("models", exist_ok=True)
if st.button("Subsampled Modell in ./models/ speichern"):
    path = "models/random_forest_subsampled.joblib"
    joblib.dump(clf_sub, path)
    st.success(f"Gespeichert: {path}")

buf_model2 = io.BytesIO()
joblib.dump(clf_sub, buf_model2)
st.download_button("📦 Subsampled Modell herunterladen (.joblib)", buf_model2.getvalue(),
                   file_name="random_forest_subsampled.joblib")

# Predictions Download
preds_df = pd.DataFrame({"y_true": y_test, "y_pred": clf.predict(X_test)}).reset_index(drop=True)
st.download_button("📥 Predictions (Baseline RF) als CSV", preds_df.to_csv(index=False).encode("utf-8"),
                   "predictions_classification.csv", "text/csv")

# Info Dump (wie df.info())
st.subheader("ℹ️ DataFrame info()")
buf = io.StringIO(); df.info(buf); st.text(buf.getvalue())
