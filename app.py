# ================================
# ActuarAI Streamlit App
# + Upload CSV & Batch Prediction
# ================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from utils import build_features, to_dense_array

# ----------------
# Paths
# ----------------
ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "fraud_model.joblib"
COLS_PATH = ROOT / "models" / "expected_columns.joblib"

# ----------------
# Page config
# ----------------
st.set_page_config(
    page_title="ActuarAI – Fraud Detection",
    page_icon="🛡️",
    layout="wide",
)

# ----------------
# Load CSS
# ----------------
def load_css():
    css_path = ROOT / "style.css"
    if css_path.exists():
        with open(css_path) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css()

# ----------------
# Load model
# ----------------
@st.cache_resource
def load_assets():
    model = joblib.load(MODEL_PATH)
    expected_cols = joblib.load(COLS_PATH)
    return model, expected_cols

model, expected_cols = load_assets()

# ----------------
# Helpers
# ----------------

def align_columns(df: pd.DataFrame, expected_cols):
    """Ensure the uploaded CSV matches training features"""
    for col in expected_cols:
        if col not in df.columns:
            df[col] = np.nan

    df = df[expected_cols]
    return df


def predict_proba_df(df: pd.DataFrame):
    df_feat = build_features(df)
    df_feat = align_columns(df_feat, expected_cols)

    proba = model.predict_proba(df_feat)[:, 1]
    return proba


# ----------------
# UI
# ----------------

st.markdown("""
<div class="header">
    <h1>🛡️ ActuarAI – Détection de fraude assurance</h1>
    <p>Analyse en temps réel & prédiction batch par fichier CSV</p>
</div>
""", unsafe_allow_html=True)

mode = st.sidebar.radio(
    "Mode de prédiction",
    ["🧍 Prédiction individuelle", "📁 Upload CSV (Batch)"]
)

# ----------------
# MODE 1 — SINGLE
# ----------------
if mode == "🧍 Prédiction individuelle":
    st.subheader("🧍 Analyse d'un sinistre individuel")

    col1, col2, col3 = st.columns(3)

    with col1:
        months_as_customer = st.number_input("Months as customer", 0, 500, 120)
        age = st.number_input("Age", 18, 100, 35)
        policy_annual_premium = st.number_input("Annual Premium", 0.0, 5000.0, 1200.0)

    with col2:
        incident_type = st.selectbox("Incident Type", ["Single Vehicle Collision", "Multi-vehicle Collision", "Parked Car", "Vehicle Theft"])
        incident_severity = st.selectbox("Incident Severity", ["Minor Damage", "Major Damage", "Total Loss", "Trivial Damage"])
        authorities_contacted = st.selectbox("Authorities Contacted", ["Police", "Fire", "None", "Ambulance"])

    with col3:
        insured_sex = st.selectbox("Sex", ["MALE", "FEMALE"])
        insured_education_level = st.selectbox("Education", ["High School", "College", "Masters", "PhD"])
        number_of_vehicles_involved = st.slider("Vehicles involved", 1, 5, 1)

    if st.button("🔍 Analyser la fraude"):
        input_data = pd.DataFrame([{
            "months_as_customer": months_as_customer,
            "age": age,
            "policy_annual_premium": policy_annual_premium,
            "incident_type": incident_type,
            "incident_severity": incident_severity,
            "authorities_contacted": authorities_contacted,
            "insured_sex": insured_sex,
            "insured_education_level": insured_education_level,
            "number_of_vehicles_involved": number_of_vehicles_involved
        }])

        proba = predict_proba_df(input_data)[0]

        st.metric("Probabilité de fraude", f"{proba:.2%}")

        if proba > 0.7:
            st.error("🚨 Risque ÉLEVÉ de fraude")
        elif proba > 0.4:
            st.warning("⚠️ Risque MODÉRÉ de fraude")
        else:
            st.success("✅ Risque FAIBLE de fraude")

# ----------------
# MODE 2 — CSV
# ----------------
else:
    st.subheader("📁 Analyse batch via CSV")

    st.markdown("""
    Téléverse un fichier CSV contenant les mêmes colonnes que le dataset d'entraînement.
    Une colonne **fraud_probability** sera ajoutée au fichier.
    """)

    uploaded_file = st.file_uploader("Uploader un CSV", type=["csv"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file, sep=";")

        st.markdown("### Aperçu des données")
        st.dataframe(df.head())

        if st.button("⚡ Lancer la prédiction batch"):
            with st.spinner("Analyse en cours..."):
                proba = predict_proba_df(df)
                df_out = df.copy()
                df_out["fraud_probability"] = proba

            st.success("Prédiction terminée")

            st.markdown("### Résultat")
            st.dataframe(df_out.head())

            csv = df_out.to_csv(index=False).encode("utf-8")

            st.download_button(
                "📥 Télécharger le CSV enrichi",
                csv,
                "fraud_predictions.csv",
                "text/csv"
            )

# ----------------
# Footer
# ----------------
st.markdown("""
<hr>
<p style="text-align:center; color: #888;">
    ActuarAI • ML Fraud Detection • Streamlit Cloud
</p>
""", unsafe_allow_html=True)
