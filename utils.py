import numpy as np
import pandas as pd

def to_dense_array(x):
    return x.toarray() if hasattr(x, "toarray") else x

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    # Drop colonnes identifiantes/inutiles (comme dans ton PDF)
    to_drop = ["policy_number", "incident_location", "insured_zip", "_c39"]
    df = df.drop(columns=[c for c in to_drop if c in df.columns]).copy()

    # Remplacer "?" par NaN (colonnes concernées dans ton notebook)
    cols_with_qm = ["property_damage", "police_report_available", "collision_type"]
    for col in cols_with_qm:
        if col in df.columns:
            df[col] = df[col].replace("?", np.nan)

    # Dates -> datetime
    if "policy_bind_date" in df.columns:
        df["policy_bind_date"] = pd.to_datetime(df["policy_bind_date"], errors="coerce")
    if "incident_date" in df.columns:
        df["incident_date"] = pd.to_datetime(df["incident_date"], errors="coerce")

    # Features dérivées dates
    if "policy_bind_date" in df.columns:
        df["policy_bind_year"] = df["policy_bind_date"].dt.year
        df["policy_bind_month"] = df["policy_bind_date"].dt.month

    if "incident_date" in df.columns:
        df["incident_year"] = df["incident_date"].dt.year
        df["incident_month"] = df["incident_date"].dt.month
        df["incident_day"] = df["incident_date"].dt.day

    if "policy_bind_date" in df.columns and "incident_date" in df.columns:
        df["days_between_bind_and_incident"] = (df["incident_date"] - df["policy_bind_date"]).dt.days

    # On supprime les dates brutes
    df = df.drop(columns=[c for c in ["policy_bind_date", "incident_date"] if c in df.columns])

    return df
