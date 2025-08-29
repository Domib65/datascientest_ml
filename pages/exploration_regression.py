import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import io, re

st.title("🔍 Exploration – Full (Notebook → Page)")

st.markdown(
    "Lade mehrere Dateien (CSV/XLS/XLSX, Jahre 2012–2015). Für CSV wird zuerst "
    "`latin1` + `;` versucht – wie im Notebook."
)

# ---------- File upload ----------
files = st.file_uploader(
    "Dateien wählen (mehrfach)", type=["csv", "xls", "xlsx"], accept_multiple_files=True
)
if not files:
    st.info("Bitte Dateien hochladen.")
    st.stop()

# Jahr pro Datei wählen/prüfen
def guess_year(name: str) -> int:
    m = re.search(r"201[2-5]", name)
    return int(m.group()) if m else 2015

years = []
for f in files:
    yr = guess_year(f.name)
    years.append(st.selectbox(f"Jahr für **{f.name}**", [2012, 2013, 2014, 2015],
                              index=[2012, 2013, 2014, 2015].index(yr), key=f"name_{f.name}"))

# ---------- Helpers aus dem Notebook (hier direkt in der Page) ----------
@st.cache_data(show_spinner=False)
def read_any(uploaded_file) -> pd.DataFrame:
    name = uploaded_file.name.lower()
    if name.endswith(".csv"):
        # erst latin1 + ;, dann fallback
        try:
            return pd.read_csv(uploaded_file, encoding="latin1", sep=";")
        except Exception:
            uploaded_file.seek(0)
            return pd.read_csv(uploaded_file)
    # xls/xlsx
    return pd.read_excel(uploaded_file)

standard_columns = [
    "brand", "model", "cnit", "power_hp", "power_admin", "gearbox",
    "cons_urban", "cons_extra", "cons_mixed", "co2", "fuel", "hybrid",
    "mass_min", "mass_max", "Year"
]

rename_2012 = {
    "lib_mrq": "brand", "lib_mod_doss": "model", "cnit": "cnit",
    "puiss_max": "power_hp", "puiss_admin_98": "power_admin",
    "typ_boite_nb_rapp": "gearbox", "conso_urb": "cons_urban",
    "conso_exurb": "cons_extra", "conso_mixte": "cons_mixed",
    "co2": "co2", "typ_cbr": "fuel", "hybride": "hybrid",
    "masse_ordma_min": "mass_min", "masse_ordma_max": "mass_max",
    "Year": "Year",
}
rename_2013 = {
    "Marque": "brand", "Modèle dossier": "model", "CNIT": "cnit",
    "Puissance maximale (kW)": "power_hp", "Puissance administrative": "power_admin",
    "Boîte de vitesse": "gearbox",
    "Consommation urbaine (l/100km)": "cons_urban",
    "Consommation extra-urbaine (l/100km)": "cons_extra",
    "Consommation mixte (l/100km)": "cons_mixed",
    "CO2 (g/km)": "co2", "Carburant": "fuel", "Hybride": "hybrid",
    "masse vide euro min (kg)": "mass_min", "masse vide euro max (kg)": "mass_max",
    "Year": "Year",
}
rename_2014 = {
    "lib_mrq": "brand", "lib_mod_doss": "model", "cnit": "cnit",
    "puiss_max": "power_hp", "puiss_admin_98": "power_admin",
    "typ_boite_nb_rapp": "gearbox", "conso_urb": "cons_urban",
    "conso_exurb": "cons_extra", "conso_mixte": "cons_mixed",
    "co2": "co2", "cod_cbr": "fuel", "hybride": "hybrid",
    "masse_ordma_min": "mass_min", "masse_ordma_max": "mass_max",
    "Year": "Year",
}
rename_2015 = {
    "lib_mrq_doss": "brand", "lib_mod_doss": "model", "cnit": "cnit",
    "puiss_max": "power_hp", "puiss_admin": "power_admin",
    "typ_boite_nb_rapp": "gearbox", "conso_urb_93": "cons_urban",
    "conso_exurb": "cons_extra", "conso_mixte": "cons_mixed",
    "co2_mixte": "co2", "energ": "fuel", "hybride": "hybrid",
    "masse_ordma_min": "mass_min", "masse_ordma_max": "mass_max",
    "Year": "Year",
}
rename_maps = {2012: rename_2012, 2013: rename_2013, 2014: rename_2014, 2015: rename_2015}

float_columns = [
    "cons_urban", "cons_extra", "cons_mixed", "co2",
    "power_hp", "mass_min", "mass_max", "power_admin"
]
categorical_columns = ["brand", "model", "gearbox", "fuel", "hybrid"]

def clean_numerical_columns(df, float_cols):
    for col in float_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.replace(",", ".", regex=False)
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df

def clean_categorical_columns(df, cat_cols):
    for col in cat_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.upper()
    return df

def clean_brand_column(df):
    if "brand" in df.columns:
        df["brand"] = (
            df["brand"].astype(str).str.strip().str.lower()
            .str.replace("-", " ").str.replace("_", " ")
            .str.replace(r"\s+", " ", regex=True).str.title()
        )
    return df

def clean_df(df, column_mapping, float_cols, cat_cols):
    df = df[list(column_mapping.keys())].copy().rename(columns=column_mapping)
    df = clean_numerical_columns(df, float_cols)
    df = clean_categorical_columns(df, cat_cols)
    df = clean_brand_column(df)
    return df

# ---------- Notebook-Logik: Einlesen -> Standardisieren -> Mergen ----------
cleaned = []
for f, year in zip(files, years):
    df_raw = read_any(f)
    df_raw["Year"] = year
    cleaned.append(clean_df(df_raw, rename_maps[year], float_columns, categorical_columns))

merged_df = pd.concat(cleaned, ignore_index=True)
st.success(f"Merged shape: {merged_df.shape[0]:,} Zeilen × {merged_df.shape[1]} Spalten")
st.dataframe(merged_df.head(50), use_container_width=True)

# Info wie df.info()
with st.expander("ℹ️ DataFrame info()"):
    buf = io.StringIO()
    merged_df.info(buf)
    st.text(buf.getvalue())

# Download wie im Notebook (statt Drive)
st.download_button(
    "📥 merged_df_exploration.csv herunterladen",
    merged_df.to_csv(index=False).encode("utf-8"),
    file_name="merged_df_exploration.csv",
    mime="text/csv",
)

# ---------- Plots aus dem Notebook (Seaborn/Matplotlib) ----------
sns.set_theme()

fig = plt.figure(figsize=(10, 6))
sns.histplot(merged_df["co2"].dropna(), bins=50, kde=True)
plt.title("Distribution of CO2 Emissions (g/km)")
plt.xlabel("CO2 (g/km)"); plt.ylabel("Count"); plt.grid(True, alpha=0.3)
st.pyplot(fig); plt.close(fig)

fig = plt.figure(figsize=(10, 6))
sns.scatterplot(data=merged_df, x="power_hp", y="co2", alpha=0.3)
plt.title("CO2 Emissions vs. Horsepower")
plt.xlabel("Horsepower (power_hp)"); plt.ylabel("CO2 (g/km)"); plt.grid(True, alpha=0.3)
st.pyplot(fig); plt.close(fig)

merged_df["fuel"] = merged_df["fuel"].astype(str).str.strip().str.upper()
plot_df = merged_df[["fuel", "co2"]].dropna()

fig = plt.figure(figsize=(14, 6))
sns.boxplot(data=plot_df, x="fuel", y="co2", order=sorted(plot_df["fuel"].unique()))
plt.title("CO2 Emissions by Fuel Type")
plt.xlabel("Fuel Type"); plt.ylabel("CO2 (g/km)")
plt.xticks(rotation=45); plt.tight_layout()
st.pyplot(fig); plt.close(fig)

co2_yearly = merged_df.groupby("Year")["co2"].mean().reset_index()
fig = plt.figure(figsize=(8, 5))
sns.lineplot(data=co2_yearly, x="Year", y="co2", marker="o")
plt.title("Average CO2 Emissions Over Time")
plt.xlabel("Year"); plt.ylabel("Average CO2 (g/km)"); plt.grid(True, alpha=0.3)
st.pyplot(fig); plt.close(fig)

fig = plt.figure(figsize=(8, 6))
sns.heatmap(merged_df[float_columns].corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Matrix of Numeric Features")
st.pyplot(fig); plt.close(fig)
