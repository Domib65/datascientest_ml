import streamlit as st
import pandas as pd
import numpy as np
import io

st.title("⚙️ Regression – Preprocessing (Notebook → Page)")
st.markdown(
    "Upload your `merged_df_exploration.csv` (or XLS/XLSX). "
    "CSV delimiter/encoding are detected automatically; override via controls if needed."
)

# ---- CSV options (helpful if sniffing guesses wrong) ----
col1, col2 = st.columns(2)
sep_choice = col1.selectbox("Delimiter", ["auto", ",", ";", "\\t"], index=0)
enc_choice = col2.selectbox("Encoding", ["auto", "utf-8", "latin1", "ISO-8859-1"], index=0)

# ---------- Datei laden ----------
up = st.file_uploader("Select file", type=["csv", "xls", "xlsx"])
if not up:
    st.info("Bitte Datei hochladen.")
    st.stop()

def read_any(uploaded_file) -> pd.DataFrame:
    name = uploaded_file.name.lower()

    if name.endswith(".csv"):
        # 1) manual override
        if sep_choice != "auto" or enc_choice != "auto":
            uploaded_file.seek(0)
            sep = {"\\t": "\t"}.get(sep_choice, sep_choice) if sep_choice != "auto" else None
            enc = None if enc_choice == "auto" else enc_choice
            return pd.read_csv(uploaded_file, sep=sep, encoding=enc)

        # 2) auto-sniff delimiter (and fallback)
        uploaded_file.seek(0)
        try:
            # engine=python + sep=None snifft ',', ';', '\t', etc.
            return pd.read_csv(uploaded_file, sep=None, engine="python")
        except Exception:
            pass

        # 3) legacy fallbacks (wie Colab-Notebook)
        uploaded_file.seek(0)
        try:
            return pd.read_csv(uploaded_file, encoding="latin1", sep=";")
        except Exception:
            uploaded_file.seek(0)
            return pd.read_csv(uploaded_file)

    # Excel
    uploaded_file.seek(0)
    return pd.read_excel(uploaded_file)

df = read_any(up)

# ---------- Erste Checks ----------
st.write("**Shape:**", df.shape)
buf = io.StringIO(); df.info(buf)
st.expander("ℹ️ df.info() anzeigen").text(buf.getvalue())
st.subheader("Erste Zeilen")
st.dataframe(df.head(), use_container_width=True)

# ---------- Numeric cleaning VOR dropna ----------
numeric_cols = ["mass_min", "mass_max", "power_hp", "co2",
                "cons_urban", "cons_extra", "cons_mixed"]
for col in numeric_cols:
    if col in df.columns:
        df[col] = (
            df[col]
            .astype(str)
            .str.replace(",", ".", regex=False)   # Dezimal-Komma → Punkt
            .str.replace("\xa0", "", regex=False) # non-breaking space
            .str.strip()
        )
        df[col] = pd.to_numeric(df[col], errors="coerce")

# ---------- Missing-Tabelle ----------
missing_counts = df.isnull().sum()
missing_percent = (missing_counts / len(df)) * 100
missing_df = (
    pd.DataFrame({"Missing Count": missing_counts, "Missing %": missing_percent})
    .query("`Missing Count` > 0")
    .sort_values("Missing Count", ascending=False)
)
st.subheader("Fehlende Werte")
st.dataframe(missing_df, use_container_width=True) if not missing_df.empty else st.success("Keine fehlenden Werte gefunden.")

# ---------- Preprocessing wie im Notebook ----------
st.subheader("Bereinigung & Feature-Engineering")

# Pflichtspalten prüfen
required = {"mass_min", "mass_max", "power_hp", "co2"}
missing_req = sorted(list(required - set(df.columns)))
if missing_req:
    st.error(f"Fehlende Pflichtspalten: {missing_req}")
    st.stop()

# Nur auf Pflichtspalten droppen (nicht auf ALLE Spalten)
before_drop = len(df)
df = df.dropna(subset=list(required)).reset_index(drop=True)
st.write(f"**Zeilen nach `dropna()` auf Pflichtspalten**: {df.shape[0]} (−{before_drop - df.shape[0]})")

# Features
df["avg_mass"] = (df["mass_min"] + df["mass_max"]) / 2
df["power_to_weight"] = df["power_hp"] / df["avg_mass"]
df.loc[~np.isfinite(df["power_to_weight"]), "power_to_weight"] = np.nan
df["log_co2"] = np.log1p(df["co2"])

# Duplikate entfernen
before = df.shape[0]
df = df.drop_duplicates().reset_index(drop=True)
st.write(f"**Zeilen nach Entfernen von Duplikaten**: {df.shape[0]} (−{before - df.shape[0]})")

# Infos
buf2 = io.StringIO(); df.info(buf2)
st.expander("ℹ️ df.info() nach Preprocessing").text(buf2.getvalue())
st.subheader("Beschreibung (describe)")
st.dataframe(df.describe(include="all").T, use_container_width=True)

# ---------- Download ----------
st.download_button(
    "📥 cleaned_dataset_final.csv herunterladen",
    df.to_csv(index=False).encode("utf-8"),
    file_name="cleaned_dataset_final.csv",
    mime="text/csv",
)
