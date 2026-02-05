# ================================
# ActuarAI Streamlit App
# UI Premium (SaaS-style) + Single & Batch Prediction
# ================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

from utils import build_features

# Optional: premium gauge
try:
    import plotly.graph_objects as go  # type: ignore
    _HAS_PLOTLY = True
except Exception:
    _HAS_PLOTLY = False


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
    page_title="ActuarAI – Détection de fraude",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)



# ----------------
# Load CSS
# ----------------
def load_css() -> None:
    css_path = ROOT / "style.css"  # style.css au même niveau que app.py
    if css_path.exists():
        st.markdown(
            f"<style>{css_path.read_text(encoding='utf-8')}</style>",
            unsafe_allow_html=True
        )


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
def align_columns(df: pd.DataFrame, expected_cols_) -> pd.DataFrame:
    """Ensure the dataframe matches training features."""
    for col in expected_cols_:
        if col not in df.columns:
            df[col] = np.nan
    df = df[expected_cols_]
    return df


def predict_proba_df(df: pd.DataFrame) -> np.ndarray:
    df_feat = build_features(df)
    df_feat = align_columns(df_feat, expected_cols)
    proba = model.predict_proba(df_feat)[:, 1]
    return proba


def risk_bucket(p: float) -> str:
    if p >= 0.70:
        return "ÉLEVÉ"
    if p >= 0.40:
        return "MODÉRÉ"
    return "FAIBLE"


def badge_class(p: float) -> str:
    if p >= 0.70:
        return "badge-risk"
    if p >= 0.40:
        return "badge-warn"
    return "badge-ok"


def render_gauge(p: float):
    """Plotly gauge if available; otherwise fallback."""
    if _HAS_PLOTLY:
        fig = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=float(p * 100),
                number={"suffix": "%"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": "rgba(59,130,246,0.9)"},
                    "steps": [
                        {"range": [0, 40], "color": "rgba(34,197,94,0.18)"},
                        {"range": [40, 70], "color": "rgba(245,158,11,0.18)"},
                        {"range": [70, 100], "color": "rgba(239,68,68,0.18)"},
                    ],
                    "threshold": {
                        "line": {"color": "rgba(239,68,68,0.8)", "width": 4},
                        "thickness": 0.75,
                        "value": float(p * 100),
                    },
                },
                title={"text": "Risque de fraude"},
            )
        )
        fig.update_layout(height=260, margin=dict(l=10, r=10, t=50, b=10))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.progress(min(max(p, 0.0), 1.0))
        st.caption("Pour une jauge premium : `pip install plotly`.")

# ----------------
# Header (Hero)
# ----------------
st.markdown(
    """
<div class="header-hero">
  <div class="kpi">
    <div style="font-size:34px;">🛡️</div>
    <div>
      <div class="title">ActuarAI – Détection de fraude assurance</div>
      <div class="subtitle">Analyse en temps réel (single) + scoring batch (CSV) • UI SaaS</div>
    </div>
    <div style="margin-left:auto;">
      <span class="badge">Modèle chargé</span>
    </div>
  </div>
</div>
""",
    unsafe_allow_html=True,
)

# ⭐ AJOUTE ÇA ICI
mode = st.radio(
    "Mode",
    ["🧍 Prédiction individuelle", "📁 Upload CSV (Batch)"],
    horizontal=True
)

# ----------------
# Sidebar
# ----------------
with st.sidebar:
    st.markdown(
        """<div class="card">
    <div style="font-weight:800; font-size:16px;">⚙️ Paramètres</div>
    <div class="small">Choisis le mode en haut de la page.</div>
</div>""",
        unsafe_allow_html=True,
    )

    st.markdown(
        """<div class="card" style="margin-top:12px;">
    <div style="font-weight:800; font-size:16px;">💡 Conseils</div>
    <div class="small">• Vérifie le format CSV (séparateur, colonnes)<br>
    • Le score est une probabilité, pas une certitude<br>
    • Mets en revue les cas « MODÉRÉ/ÉLEVÉ »</div>
</div>""",
        unsafe_allow_html=True,
    )

# ----------------
# MAIN
# ----------------
if mode == "🧍 Prédiction individuelle":
    tabs = st.tabs(["🧍 Formulaire", "📌 Explications (modèle)"])

    with tabs[0]:
        st.markdown(
            """<div class="card">
            <div style="font-weight:800; font-size:18px;">🧾 Déclare un sinistre</div>
            <div class="small">Renseigne quelques informations, puis lance l'analyse.</div>
        </div>""",
            unsafe_allow_html=True,
        )
        st.write("")

        c1, c2, c3 = st.columns(3)

        with c1:
            st.markdown("### 👤 Profil")
            months_as_customer = st.number_input("Ancienneté (mois)", 0, 500, 120)
            age = st.number_input("Âge", 18, 100, 35)
            insured_sex = st.selectbox("Sexe", ["MALE", "FEMALE"])

        with c2:
            st.markdown("### 🧾 Contrat")
            policy_annual_premium = st.number_input("Prime annuelle (€)", 0.0, 5000.0, 1200.0)
            insured_education_level = st.selectbox("Éducation", ["High School", "College", "Masters", "PhD"])

        with c3:
            st.markdown("### 🚗 Incident")
            incident_type = st.selectbox(
                "Type d'incident",
                ["Single Vehicle Collision", "Multi-vehicle Collision", "Parked Car", "Vehicle Theft"],
            )
            incident_severity = st.selectbox(
                "Gravité",
                ["Minor Damage", "Major Damage", "Total Loss", "Trivial Damage"],
            )
            authorities_contacted = st.selectbox("Autorités contactées", ["Police", "Fire", "Ambulance", "None"])
            number_of_vehicles_involved = st.slider("Véhicules impliqués", 1, 5, 1)

        st.markdown("---")

        run = st.button("🛡️ Analyser le risque de fraude", use_container_width=True)

        if run:
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

            with st.spinner("Analyse en cours..."):
                proba = float(predict_proba_df(input_data)[0])

            bucket = risk_bucket(proba)
            bclass = badge_class(proba)

            res_c1, res_c2 = st.columns([1, 1])

            with res_c1:
                st.markdown(
                    f"""
<div class="result-card">
  <div class="badge {bclass}">Risque {bucket}</div>
  <h2 style="margin-top:10px;">Probabilité estimée</h2>
  <h1>{proba:.1%}</h1>
  <div class="small">Seuils: 40% / 70% • Utilise ce score pour prioriser les contrôles.</div>
</div>
""",
                    unsafe_allow_html=True,
                )

            with res_c2:
                st.markdown(
                    """<div class="card">
                    <div style="font-weight:800; font-size:16px;">📊 Visualisation</div>
                    <div class="small">Jauge du risque de fraude</div>
                </div>""",
                    unsafe_allow_html=True,
                )
                render_gauge(proba)

            if bucket == "ÉLEVÉ":
                st.error("🚨 Risque ÉLEVÉ : contrôle manuel recommandé + vérifications complémentaires.")
            elif bucket == "MODÉRÉ":
                st.warning("⚠️ Risque MODÉRÉ : analyse rapide recommandée (pièces, cohérence, historique).")
            else:
                st.success("✅ Risque FAIBLE : traitement standard recommandé.")

    with tabs[1]:
        st.markdown(
            """<div class="card">
            <div style="font-weight:800; font-size:18px;">📌 Comment lire le score ?</div>
            <div class="small">
              Le score est une <b>probabilité</b> estimée par un modèle supervisé entraîné sur un historique de sinistres.
              Il sert à prioriser les contrôles, pas à décider seul.
            </div>
        </div>""",
            unsafe_allow_html=True,
        )

        st.write("")
        st.markdown(
            """<div class="card">
            <div style="font-weight:800; font-size:16px;">🧠 Bonnes pratiques</div>
            <div class="small">
              • Surveille surtout les cas <b>MODÉRÉ</b> & <b>ÉLEVÉ</b><br>
              • Combine avec des règles métier (incohérences, doublons, zones, etc.)<br>
              • Mesure le ROI: fraude évitée vs coût de contrôle
            </div>
        </div>""",
            unsafe_allow_html=True,
        )

else:
    st.markdown(
        """<div class="card">
        <div style="font-weight:800; font-size:18px;">📁 Scoring batch (CSV)</div>
        <div class="small">Upload ton fichier, score les sinistres, puis télécharge le CSV enrichi.</div>
    </div>""",
        unsafe_allow_html=True,
    )

    st.write("")

    col_upload, col_sep = st.columns([2, 1])

    with col_upload:
        uploaded_file = st.file_uploader("Uploader un CSV", type=["csv"])

    with col_sep:
        sep = st.selectbox("Séparateur", [";", ",", "\t"], index=0)

    if uploaded_file:
        df = pd.read_csv(uploaded_file, sep=sep)

        # =====================
        # KPI CARDS
        # =====================
        n_rows = len(df)
        n_cols = df.shape[1]
        n_missing = int(df.isna().sum().sum())
        miss_rate = n_missing / (n_rows * n_cols) if n_rows and n_cols else 0

        k1, k2, k3, k4 = st.columns(4)

        with k1:
            st.markdown(f"""<div class="card">
            <div class="small">Lignes</div>
            <div style="font-size:26px; font-weight:900;">{n_rows:,}</div>
            </div>""".replace(",", " "), unsafe_allow_html=True)

        with k2:
            st.markdown(f"""<div class="card">
            <div class="small">Colonnes</div>
            <div style="font-size:26px; font-weight:900;">{n_cols}</div>
            </div>""", unsafe_allow_html=True)

        with k3:
            st.markdown(f"""<div class="card">
            <div class="small">Valeurs manquantes</div>
            <div style="font-size:26px; font-weight:900;">{n_missing:,}</div>
            </div>""".replace(",", " "), unsafe_allow_html=True)

        with k4:
            st.markdown(f"""<div class="card">
            <div class="small">Taux manquant</div>
            <div style="font-size:26px; font-weight:900;">{miss_rate:.1%}</div>
            </div>""", unsafe_allow_html=True)

        st.write("")

        # =====================
        # TABS UI
        # =====================
        tab1, tab2, tab3 = st.tabs(["👀 Aperçu", "🧹 Qualité", "⚡ Scoring"])

        # -------- Aperçu
        with tab1:
            st.dataframe(df.head(20), use_container_width=True, height=400)

        # -------- Qualité données
        with tab2:
            miss_by_col = (df.isna().mean().sort_values(ascending=False) * 100).round(1)
            miss_table = miss_by_col.reset_index()
            miss_table.columns = ["Colonne", "% manquant"]

            st.dataframe(miss_table.head(20), use_container_width=True)

        # -------- Scoring
        with tab3:
            threshold_warn = st.slider("Seuil MODÉRÉ", 0.10, 0.90, 0.40)
            threshold_high = st.slider("Seuil ÉLEVÉ", 0.20, 0.99, 0.70)

            run_batch = st.button("🛡️ Lancer le scoring batch", use_container_width=True)

            if run_batch:

                prog = st.progress(0)

                prog.progress(20)
                proba = predict_proba_df(df)

                prog.progress(70)
                df_out = df.copy()
                df_out["fraud_probability"] = np.clip(proba, 0, 1)

                def risk(p):
                    if p >= threshold_high:
                        return "ÉLEVÉ"
                    if p >= threshold_warn:
                        return "MODÉRÉ"
                    return "FAIBLE"

                df_out["fraud_risk"] = df_out["fraud_probability"].apply(risk)

                prog.progress(100)
                st.success("✅ Scoring terminé")

                # KPI Résultat
                r1, r2, r3 = st.columns(3)

                with r1:
                    st.metric("Score moyen", f"{df_out['fraud_probability'].mean():.1%}")

                with r2:
                    st.metric("% ÉLEVÉ", f"{(df_out['fraud_risk']=='ÉLEVÉ').mean():.1%}")

                with r3:
                    st.metric("% MODÉRÉ", f"{(df_out['fraud_risk']=='MODÉRÉ').mean():.1%}")

                st.write("")

                # Distribution
                if _HAS_PLOTLY:
                    import plotly.express as px
                    fig = px.histogram(df_out, x="fraud_probability", nbins=30)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.bar_chart(df_out["fraud_probability"])

                st.write("")

                # Top risques
                st.markdown("### 🚨 Cas prioritaires")
                st.dataframe(
                    df_out.sort_values("fraud_probability", ascending=False).head(50),
                    use_container_width=True
                )

                # Download
                csv = df_out.to_csv(index=False).encode("utf-8")

                st.download_button(
                    "📥 Télécharger le CSV enrichi",
                    csv,
                    "fraud_predictions.csv",
                    "text/csv",
                    use_container_width=True,
                )



st.markdown(
    """
<hr>
<p style="text-align:center; color: rgba(229,231,235,0.6); font-size: 12px;">
  ActuarAI • ML Fraud Detection • Streamlit
</p>
""",
    unsafe_allow_html=True,
)
