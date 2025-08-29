import streamlit as st

st.set_page_config(page_title="Datascientest ML", page_icon="🧪", layout="wide")

st.title("🧪 Datascientest ML – Streamlit App")
st.markdown(
    """
Welcome! Use the sidebar to navigate to **Preprocessing**, **Exploration**,  
**Regression Modeling**, and **Classification Modeling**.
"""
)

st.divider()
st.subheader("Quick start")
st.code("poetry run streamlit run app.py", language="bash")

with st.expander("Environment (versions)"):
    try:
        import sklearn
        import pandas as pd
        st.write(f"scikit-learn: **{sklearn.__version__}**, pandas: **{pd.__version__}**")
    except Exception:
        st.write("Packages not installed yet.")
