import streamlit as st
import pandas as pd
import numpy as np
import joblib

from utils import build_features, to_dense_array  # important pour joblib.load

from pathlib import Path

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "fraud_model.joblib"
COLS_PATH  = ROOT / "models" / "expected_columns.joblib"


# ---------- Helpers ----------
@st.cache_resource
def load_assets():
    model = joblib.load(MODEL_PATH)
    expected_cols = joblib.load(COLS_PATH)
    return model, expected_cols

def load_css(path="style.css"):
    try:
        with open(path, "r", encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        pass

def pct(x: float) -> str:
    return f"{x*100:.1f}%"

def risk_label(p: float, t: float):
    # 3 niveaux pour un rendu UI plus pro
    if p < t * 0.75:
        return "Faible", "badge badge-ok"
    elif p < t:
        return "Modéré", "badge badge-warn"
    else:
        return "Élevé", "badge badge-risk"

def clean_unknown(v):
    return np.nan if v == "?" else v

def select_with_unknown(label, options, default=None):
    opts = list(options)
    if "?" not in opts:
        opts.append("?")
    idx = 0
    if default is not None and default in opts:
        idx = opts.index(default)
    return st.selectbox(label, opts, index=idx)

# ---------- Page ----------
st.set_page_config(page_title="ActuarAI – Fraud Detection", page_icon="🛡️", layout="wide")
load_css()

# Sidebar settings
st.sidebar.markdown("## ⚙️ Paramètres")
threshold = st.sidebar.slider("Seuil de décision", 0.01, 0.99, 0.14, 0.01)
show_debug = st.sidebar.toggle("Mode debug (voir features)", value=False)

st.sidebar.markdown("---")
st.sidebar.markdown("### ℹ️ À propos")
st.sidebar.write("ActuarAI – Détection de fraude en assurance auto.")
st.sidebar.caption("Modèle: HistGradientBoosting + preprocessing sklearn.")

# Header
st.markdown(
    """
    <div class="card">
      <div class="kpi">
        <div style="font-size:34px">🛡️</div>
        <div>
          <div class="title">ActuarAI</div>
          <div class="subtitle">Vehicle Insurance Fraud Detection — score en temps réel</div>
        </div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Load model
try:
    model, expected_cols = load_assets()
except Exception as e:
    st.error("Modèle introuvable. Lance d’abord : `python train.py`")
    st.code(str(e))
    st.stop()

st.markdown("")

# Layout: form left, result right
left, right = st.columns([1.25, 1])

with left:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("## 🧾 Nouvelle déclaration")

    with st.form("claim_form"):
        st.markdown("### 👤 Assuré & Police")
        c1, c2, c3 = st.columns(3)

        with c1:
            months_as_customer = st.number_input("months_as_customer", 0, 1000, 120)
            age = st.slider("age", 18, 100, 35)
            insured_sex = st.selectbox("insured_sex", ["MALE", "FEMALE"])

        with c2:
            policy_state = st.selectbox("policy_state", ["OH", "IN", "IL"])
            policy_csl = st.selectbox("policy_csl", ["100/300", "250/500", "500/1000"])
            policy_deductable = st.selectbox("policy_deductable", [500, 1000, 2000])

        with c3:
            policy_annual_premium = st.number_input("policy_annual_premium", min_value=0.0, value=1200.0)
            umbrella_limit = st.selectbox("umbrella_limit", [0, 2000000, 4000000, 5000000, 6000000])
            policy_bind_date = st.date_input("policy_bind_date", value=pd.to_datetime("2014-06-01"))

        c4, c5, c6 = st.columns(3)
        with c4:
            insured_education_level = st.selectbox(
                "insured_education_level",
                ["High School", "Associate", "College", "Masters", "MD", "PhD", "JD"]
            )
        with c5:
            insured_occupation = st.selectbox(
                "insured_occupation",
                ["adm-clerical","armed-forces","craft-repair","exec-managerial","farming-fishing",
                 "handlers-cleaners","machine-op-inspct","other-service","priv-house-serv",
                 "prof-specialty","protective-serv","sales","tech-support","transport-moving"]
            )
        with c6:
            insured_relationship = st.selectbox(
                "insured_relationship",
                ["husband","wife","own-child","unmarried","other-relative","not-in-family"]
            )

        insured_hobbies = st.selectbox(
            "insured_hobbies",
            ["chess","cross-fit","bungie-jumping","camping","exercise","golf","hiking",
             "kayaking","movies","paintball","polo","reading","skydiving","sleeping",
             "video-games","yachting","dancing","board-games","basketball","base-jumping"]
        )

        st.markdown("<hr/>", unsafe_allow_html=True)
        st.markdown("### 🚗 Sinistre & Véhicule")
        d1, d2, d3 = st.columns(3)

        with d1:
            incident_date = st.date_input("incident_date", value=pd.to_datetime("2015-01-15"))
            incident_type = st.selectbox("incident_type", ["Single Vehicle Collision","Multi-vehicle Collision","Vehicle Theft","Parked Car"])
            incident_severity = st.selectbox("incident_severity", ["Minor Damage","Major Damage","Total Loss","Trivial Damage"])

        with d2:
            collision_type = select_with_unknown("collision_type", ["Rear Collision","Side Collision","Front Collision","Vehicle Theft"], default="Rear Collision")
            authorities_contacted = st.selectbox("authorities_contacted", ["Police","Fire","Ambulance","Other"])
            incident_hour_of_the_day = st.slider("incident_hour_of_the_day", 0, 23, 12)

        with d3:
            incident_state = st.selectbox("incident_state", ["NY","SC","WV","VA","NC","PA","OH"])
            incident_city = st.selectbox("incident_city", ["Arlington","Columbus","Hillsdale","Northbrook","Riverwood","Springfield","Northbend"])
            number_of_vehicles_involved = st.selectbox("number_of_vehicles_involved", [1,2,3,4])

        e1, e2, e3, e4 = st.columns(4)
        with e1:
            property_damage = select_with_unknown("property_damage", ["YES","NO"], default="NO")
        with e2:
            police_report_available = select_with_unknown("police_report_available", ["YES","NO"], default="NO")
        with e3:
            bodily_injuries = st.selectbox("bodily_injuries", [0,1,2])
        with e4:
            witnesses = st.selectbox("witnesses", [0,1,2,3])

        st.markdown("### 💰 Montants")
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            total_claim_amount = st.number_input("total_claim_amount", min_value=0, value=15000)
        with m2:
            injury_claim = st.number_input("injury_claim", min_value=0, value=2000)
        with m3:
            property_claim = st.number_input("property_claim", min_value=0, value=2000)
        with m4:
            vehicle_claim = st.number_input("vehicle_claim", min_value=0, value=11000)

        st.markdown("### 🧩 Véhicule")
        v1, v2, v3 = st.columns(3)
        with v1:
            auto_make = st.selectbox(
                "auto_make",
                ["Accura","Audi","BMW","Chevrolet","Dodge","Ford","Honda","Jeep","Mercedes","Nissan","Saab","Suburu","Toyota","Volkswagen"]
            )
        with v2:
            auto_model = st.selectbox(
                "auto_model",
                ["92x","A3","A4","A5","Accord","C300","C400","Camry","Civic","Corolla",
                 "CRV","E400","Escape","Fusion","Highlander","Impreza","Jetta","Legacy","Malibu",
                 "Maxima","MDX","Mustang","Neon","Passat","Prius","RAM","RSX","Silverado","Tahoe",
                 "TL","TLX","Tundra","Wrangler","X5","X6","3 Series","5 Series","7 Series","M5"]
            )
        with v3:
            auto_year = st.number_input("auto_year", min_value=1980, max_value=2026, value=2010)

        st.markdown("### 📈 Capital (optionnel)")
        k1, k2 = st.columns(2)
        with k1:
            capital_gains = st.number_input("capital-gains", min_value=0, value=0)
        with k2:
            capital_loss = st.number_input("capital-loss", min_value=0, value=0)

        submit = st.form_submit_button("⚡ Calculer le risque")

    st.markdown("</div>", unsafe_allow_html=True)

# Result panel
with right:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("## 📌 Résultat")

    if not submit:
        st.info("Renseigne le formulaire puis clique sur **Calculer le risque**.")
        st.markdown('<div class="small">Astuce : tu peux ajuster le seuil dans la sidebar.</div>', unsafe_allow_html=True)
    else:
        # Build raw row matching dataset schema (minimal required + placeholders)
        raw = {
            "months_as_customer": months_as_customer,
            "age": age,
            "policy_number": np.nan,
            "policy_bind_date": pd.to_datetime(policy_bind_date).strftime("%Y-%m-%d"),
            "policy_state": policy_state,
            "policy_csl": policy_csl,
            "policy_deductable": policy_deductable,
            "policy_annual_premium": policy_annual_premium,
            "umbrella_limit": umbrella_limit,
            "insured_zip": np.nan,
            "insured_sex": insured_sex,
            "insured_education_level": insured_education_level,
            "insured_occupation": insured_occupation,
            "insured_hobbies": insured_hobbies,
            "insured_relationship": insured_relationship,
            "capital-gains": capital_gains,
            "capital-loss": capital_loss,
            "incident_date": pd.to_datetime(incident_date).strftime("%Y-%m-%d"),
            "incident_type": incident_type,
            "collision_type": clean_unknown(collision_type),
            "incident_severity": incident_severity,
            "authorities_contacted": authorities_contacted,
            "incident_state": incident_state,
            "incident_city": incident_city,
            "incident_location": np.nan,
            "incident_hour_of_the_day": incident_hour_of_the_day,
            "number_of_vehicles_involved": number_of_vehicles_involved,
            "property_damage": clean_unknown(property_damage),
            "bodily_injuries": bodily_injuries,
            "witnesses": witnesses,
            "police_report_available": clean_unknown(police_report_available),
            "total_claim_amount": total_claim_amount,
            "injury_claim": injury_claim,
            "property_claim": property_claim,
            "vehicle_claim": vehicle_claim,
            "auto_make": auto_make,
            "auto_model": auto_model,
            "auto_year": auto_year,
            "_c39": np.nan,
        }

        df_raw = pd.DataFrame([raw])
        df_feat = build_features(df_raw)

        for col in expected_cols:
            if col not in df_feat.columns:
                df_feat[col] = np.nan
        df_feat = df_feat[expected_cols]

        proba = float(model.predict_proba(df_feat)[0, 1])

        level, badge_cls = risk_label(proba, threshold)
        decision = "🚨 Fraude probable" if proba >= threshold else "✅ Fraude peu probable"

        st.markdown(
            f"""
            <div class="kpi">
              <div>
                <div class="{badge_cls}">{level} risque</div>
                <div style="font-size:28px; font-weight:800; margin-top:10px;">{pct(proba)} </div>
                <div class="small">Probabilité estimée de fraude</div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.progress(min(max(proba, 0.0), 1.0))

        st.markdown("<hr/>", unsafe_allow_html=True)
        if proba >= threshold:
            st.error(decision)
        else:
            st.success(decision)

        st.caption(f"Seuil actuel : {threshold:.2f} • (Ton seuil F1 recommandé = 0.14)")

        if show_debug:
            st.markdown("### 🧪 Debug")
            st.write("**Raw input (avant feature engineering)**")
            st.dataframe(df_raw, use_container_width=True)
            st.write("**Features envoyées au modèle**")
            st.dataframe(df_feat, use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("")
st.caption("© ActuarAI — Demo de scoring fraude (projet académique).")

