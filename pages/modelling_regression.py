import io
import numpy as np
import pandas as pd
import streamlit as st

from sklearn.model_selection import train_test_split, KFold, cross_val_score, RandomizedSearchCV
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, make_scorer

from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestRegressor, ExtraTreesRegressor,
    GradientBoostingRegressor, HistGradientBoostingRegressor
)
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR

# XGBoost optional robust importieren
try:
    from xgboost import XGBRegressor
    XGB_OK, XGB_ERR = True, ""
except Exception as e:
    XGB_OK, XGB_ERR = False, str(e)

import joblib
import matplotlib.pyplot as plt
import shap

st.title("📉 Regression – Modeling (Notebook → Page)")
st.markdown(
    "Upload **`cleaned_dataset_final.csv`** (or XLS/XLSX). "
    "CSV delimiter/encoding are auto-detected; override below if needed."
)

# ---- CSV Optionen (Override) ----
col1, col2 = st.columns(2)
sep_choice = col1.selectbox("Delimiter", ["auto", ",", ";", "\\t"], index=0)
enc_choice = col2.selectbox("Encoding", ["auto", "utf-8", "latin1", "ISO-8859-1"], index=0)

# ---------- Datei laden ----------
@st.cache_data(show_spinner=False)
def read_any(uploaded_file, sep_opt: str, enc_opt: str) -> pd.DataFrame:
    name = uploaded_file.name.lower()
    if name.endswith(".csv"):
        # manueller Override?
        if sep_opt != "auto" or enc_opt != "auto":
            uploaded_file.seek(0)
            sep = {"\\t": "\t"}.get(sep_opt, sep_opt) if sep_opt != "auto" else None
            enc = None if enc_opt == "auto" else enc_opt
            df = pd.read_csv(uploaded_file, sep=sep, encoding=enc)
        else:
            uploaded_file.seek(0)
            df = pd.read_csv(uploaded_file, sep=None, engine="python")  # sniff
        # Fallbacks, falls 1-Spalten-CSV
        if df.shape[1] == 1:
            uploaded_file.seek(0)
            try:
                df = pd.read_csv(uploaded_file, sep=",")
            except Exception:
                uploaded_file.seek(0)
                df = pd.read_csv(uploaded_file, sep=";")
        return df
    # Excel
    uploaded_file.seek(0)
    return pd.read_excel(uploaded_file)

up = st.file_uploader("Choose file", type=["csv", "xls", "xlsx"])
if not up:
    st.stop()

df = read_any(up, sep_choice, enc_choice)

# Preview
st.write("**Shape:**", df.shape)
buf = io.StringIO(); df.info(buf); st.expander("ℹ️ df.info()").text(buf.getvalue())
st.dataframe(df.head(), use_container_width=True)

# ---------- Feature/Ziel Auswahl ----------
# 'hybrid' normalisieren
if "hybrid" in df.columns:
    s = df["hybrid"].astype(str).str.strip().str.upper()
    df["hybrid"] = s.map({"OUI": 1, "NON": 0, "YES": 1, "NO": 0, "TRUE": 1, "FALSE": 0, "1": 1, "0": 0}).fillna(0).astype(int)

default_features = ["power_to_weight", "hybrid", "brand", "fuel", "gearbox"]
available = [c for c in default_features if c in df.columns]
selected_features = st.multiselect("Features (X)", options=list(df.columns), default=available)
target_col = st.selectbox("Target (y)", options=[c for c in df.columns if c != ""],
                          index=(list(df.columns).index("co2") if "co2" in df.columns else 0))

missing = [c for c in selected_features + [target_col] if c not in df.columns]
if missing or len(selected_features) == 0:
    st.error(f"Missing columns or empty feature set: {missing if missing else 'no features selected'}")
    st.stop()

X = df[selected_features].copy()
y = df[target_col].copy()

# ---------- Split ----------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, shuffle=True
)

# ---------- Encoding / Scaling ----------
cat_cols = [c for c in ["brand", "gearbox", "fuel"] if c in X.columns]
num_cols = [c for c in ["power_to_weight"] if c in X.columns]

if cat_cols:
    enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    X_train.loc[:, cat_cols] = enc.fit_transform(X_train[cat_cols])
    X_test.loc[:, cat_cols] = enc.transform(X_test[cat_cols])

if num_cols:
    scaler = StandardScaler()
    X_train.loc[:, num_cols] = scaler.fit_transform(X_train[num_cols])
    X_test.loc[:, num_cols] = scaler.transform(X_test[num_cols])

# ---- Fallback: alle verbleibenden object-Spalten encodieren + float32 cast ----
obj_cols = X_train.select_dtypes(include=["object"]).columns.tolist()
if obj_cols:
    st.warning(f"Auto-encoding object columns: {obj_cols}")
    _oe = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    X_train.loc[:, obj_cols] = _oe.fit_transform(X_train[obj_cols])
    X_test.loc[:, obj_cols]  = _oe.transform(X_test[obj_cols])

X_train = X_train.astype("float32")
X_test  = X_test.astype("float32")

# ---------- Runtime Settings ----------
st.subheader("⚙️ Runtime settings")
max_train = st.slider("Max training rows (sampling)", 5_000, int(len(X_train)), min(50_000, int(len(X_train))), step=5_000)
include_xgb = st.checkbox("Include XGBoost", value=XGB_OK)
include_heavy = st.checkbox("Include heavy models (KNN, SVR) — slow on big data", value=False)
run_cv = st.checkbox("Run 5-fold CV on top models", value=False)
run_tuning = st.checkbox("Run hyperparameter tuning", value=False)
n_iter = st.slider("RandomizedSearch iterations", 5, 100, 20, step=5, disabled=not run_tuning)

# Sampling (nur Train; Test bleibt voll)
if len(X_train) > max_train:
    rs = np.random.RandomState(42)
    idx = rs.choice(X_train.index, size=max_train, replace=False)
    Xtr, ytr = X_train.loc[idx], y_train.loc[idx]
else:
    Xtr, ytr = X_train, y_train
Xte, yte = X_test, y_test

st.success(f"Train/Test after sampling: {Xtr.shape} / {Xte.shape}")

# ---------- Baseline ----------
st.subheader("🔬 Baseline comparison (Train→Test)")
models = {
    "LinearRegression": LinearRegression(),
    "Ridge": Ridge(),
    "Lasso": Lasso(),
    "ElasticNet": ElasticNet(),
    "DecisionTree": DecisionTreeRegressor(random_state=42),
    "RandomForest": RandomForestRegressor(random_state=42, n_estimators=200, n_jobs=-1),
    "ExtraTrees": ExtraTreesRegressor(random_state=42, n_estimators=300, n_jobs=-1),
    "GradientBoosting": GradientBoostingRegressor(random_state=42),
    "HistGradientBoosting": HistGradientBoostingRegressor(random_state=42),
}
if include_xgb:
    if XGB_OK:
        models["XGBoost"] = XGBRegressor(
            objective="reg:squarederror", random_state=42,
            tree_method="hist", n_estimators=300, n_jobs=-1
        )
    else:
        st.info("XGBoost disabled: " + XGB_ERR)

if include_heavy:
    models["KNeighbors"] = KNeighborsRegressor()
    models["SVR"] = SVR()

results, total = [], len(models)
progress = st.progress(0.0)
status = st.empty()

for i, (name, model) in enumerate(models.items(), 1):
    status.write(f"Training {name} ({i}/{total}) …")
    model.fit(Xtr, ytr)
    preds = model.predict(Xte)
    r2 = r2_score(yte, preds)
    rmse = float(np.sqrt(mean_squared_error(yte, preds)))
    results.append({"Model": name, "R²": r2, "RMSE": rmse})
    progress.progress(i / total)

progress.empty(); status.empty()

results_df = pd.DataFrame(results).sort_values("R²", ascending=False)
st.dataframe(results_df, use_container_width=True)
st.download_button("📥 Baseline results (CSV)", results_df.to_csv(index=False).encode("utf-8"), "baseline_results.csv", "text/csv")

# ---------- Cross-Validation (optional) ----------
if run_cv:
    st.subheader("🧪 5-Fold Cross-Validation (ExtraTrees / RandomForest / XGBoost)")
    top_models = {
        "ExtraTrees": ExtraTreesRegressor(random_state=42, n_estimators=300, n_jobs=-1),
        "RandomForest": RandomForestRegressor(random_state=42, n_estimators=200, n_jobs=-1),
    }
    if include_xgb and XGB_OK:
        top_models["XGBoost"] = XGBRegressor(
            objective="reg:squarederror", random_state=42,
            tree_method="hist", n_estimators=300, n_jobs=-1
        )

    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    rmse_scorer = make_scorer(mean_squared_error, greater_is_better=False)

    cv_rows = []
    for name, mdl in top_models.items():
        r2_scores = cross_val_score(mdl, Xtr, ytr, cv=cv, scoring="r2", n_jobs=-1)
        neg_mse = cross_val_score(mdl, Xtr, ytr, cv=cv, scoring=rmse_scorer, n_jobs=-1)
        rmse_scores = np.sqrt(-neg_mse)
        cv_rows.append({
            "Model": name, "Mean R²": np.mean(r2_scores), "Std R²": np.std(r2_scores),
            "Mean RMSE": float(np.mean(rmse_scores)), "Std RMSE": float(np.std(rmse_scores)),
        })
    cv_df = pd.DataFrame(cv_rows).sort_values("Mean R²", ascending=False)
    st.dataframe(cv_df, use_container_width=True)
else:
    st.info("Cross-validation is disabled by default to keep things fast. Enable it above if needed.")

# ---------- Feature Importances ----------
st.subheader("🌲 Feature importances")
models_for_imp = [
    ("ExtraTrees", ExtraTreesRegressor(random_state=42, n_estimators=300, n_jobs=-1).fit(Xtr, ytr)),
    ("RandomForest", RandomForestRegressor(random_state=42, n_estimators=200, n_jobs=-1).fit(Xtr, ytr)),
]
if include_xgb and XGB_OK:
    models_for_imp.append((
        "XGBoost",
        XGBRegressor(objective="reg:squarederror", random_state=42, tree_method="hist", n_estimators=300, n_jobs=-1).fit(Xtr, ytr)
    ))

for name, model in models_for_imp:
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        order = np.argsort(importances)
        fig = plt.figure(figsize=(10, 6))
        plt.barh(np.array(Xtr.columns)[order], importances[order])
        plt.title(f"Feature Importances – {name}")
        plt.xlabel("Importance"); plt.ylabel("Features"); plt.tight_layout()
        st.pyplot(fig); plt.close(fig)

# ---------- Hyperparameter Tuning (optional) ----------
if run_tuning:
    st.subheader("🎛️ Hyperparameter tuning (RandomizedSearchCV)")
    param_grids = {
        "ExtraTrees": {
            "n_estimators": [100, 200, 300],
            "max_depth": [None, 10, 20, 30],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
            "max_features": ["auto", "sqrt", "log2"],
        },
        "RandomForest": {
            "n_estimators": [100, 200, 300],
            "max_depth": [None, 10, 20, 30],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
            "max_features": ["auto", "sqrt", "log2"],
        },
    }
    search_space = {
        "ExtraTrees": ExtraTreesRegressor(random_state=42, n_jobs=-1),
        "RandomForest": RandomForestRegressor(random_state=42, n_jobs=-1),
    }
    if include_xgb and XGB_OK:
        param_grids["XGBoost"] = {
            "n_estimators": [100, 200, 300],
            "max_depth": [3, 5, 7, 10],
            "learning_rate": [0.01, 0.05, 0.1, 0.2],
            "subsample": [0.6, 0.8, 1.0],
            "colsample_bytree": [0.6, 0.8, 1.0],
        }
        search_space["XGBoost"] = XGBRegressor(objective="reg:squarederror", random_state=42, tree_method="hist", n_jobs=-1)

    best_models, rows = {}, []
    for name, mdl in search_space.items():
        with st.spinner(f"Tuning {name}…"):
            search = RandomizedSearchCV(
                mdl,
                param_distributions=param_grids[name],
                n_iter=st.session_state.get("n_iter", 20) if run_tuning else 20,
                scoring=make_scorer(mean_squared_error, greater_is_better=False),
                cv=5,
                random_state=42,
                n_jobs=-1,
                verbose=0,
            )
            search.fit(Xtr, ytr)
        best = search.best_estimator_
        best_models[name] = best
        preds = best.predict(Xte)
        rmse = float(np.sqrt(mean_squared_error(yte, preds)))
        r2 = float(r2_score(yte, preds))
        rows.append({"Model": name, "Best Params": search.best_params_, "Test RMSE": rmse, "Test R²": r2})

    tuned_df = pd.DataFrame(rows).sort_values("Test R²", ascending=False)
    st.dataframe(tuned_df, use_container_width=True)
else:
    st.info("Tuning is disabled by default. Enable it above if needed.")

# ---------- Save & Predictions ----------
st.subheader("💾 Save & download")
# Wenn kein Tuning lief, nimm bestes Baseline-Modell
if "best_models" not in locals() or len(best_models) == 0:
    best_name = results_df.iloc[0]["Model"]
    st.write(f"Using best baseline model: **{best_name}**")
    # Refit auf allen Trainingsdaten (ohne Sampling) für bestmögliche Qualität
    refit_map = {k: v for k, v in models.items()}
    chosen = refit_map[best_name].fit(X_train, y_train)
else:
    best_name = st.selectbox("Which tuned model to save/use?", list(best_models.keys()))
    chosen = best_models[best_name]

if st.button("Save model to ./models"):
    path = f"models/best_{best_name.lower()}_model.pkl"
    joblib.dump(chosen, path)
    st.success(f"Saved: {path}")

buf_model = io.BytesIO()
joblib.dump(chosen, buf_model)
st.download_button("📦 Download model (.pkl)", buf_model.getvalue(), file_name=f"best_{best_name.lower()}_model.pkl")

preds_df = pd.DataFrame({"Actual": y_test, "Predicted": chosen.predict(X_test)}, index=y_test.index)
st.dataframe(preds_df.head(20), use_container_width=True)
st.download_button("📥 Predictions (CSV)", preds_df.to_csv(index=False).encode("utf-8"), "predictions_regression.csv", "text/csv")

# ---------- SHAP ----------
st.subheader("🔎 SHAP summary (sample of 100)")
try:
    X_sample = X_test.sample(min(100, len(X_test)), random_state=42)
    explainer = shap.TreeExplainer(chosen)
    shap_values = explainer.shap_values(X_sample)
    fig = plt.figure()
    shap.summary_plot(shap_values, X_sample, show=False)
    st.pyplot(fig, clear_figure=True)
    plt.close(fig)
except Exception as e:
    st.info(f"SHAP not available for this model/setup: {e}")
