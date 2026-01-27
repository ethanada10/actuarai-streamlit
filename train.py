import os
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.ensemble import HistGradientBoostingClassifier

from utils import to_dense_array

DATA_PATH = "data/insurance_fraud_dataset.csv"
MODEL_PATH = "models/fraud_model.joblib"
COLS_PATH = "models/expected_columns.joblib"


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    # Drop colonnes identifiantes / inutiles
    to_drop = ["policy_number", "incident_location", "insured_zip", "_c39"]
    df = df.drop(columns=[c for c in to_drop if c in df.columns]).copy()

    # Remplacer "?" par NaN
    cols_with_qm = ["property_damage", "police_report_available", "collision_type"]
    for col in cols_with_qm:
        if col in df.columns:
            df[col] = df[col].replace("?", np.nan)

    # Dates -> features
    if "policy_bind_date" in df.columns:
        df["policy_bind_date"] = pd.to_datetime(df["policy_bind_date"], errors="coerce")
    if "incident_date" in df.columns:
        df["incident_date"] = pd.to_datetime(df["incident_date"], errors="coerce")

    if "policy_bind_date" in df.columns:
        df["policy_bind_year"] = df["policy_bind_date"].dt.year
        df["policy_bind_month"] = df["policy_bind_date"].dt.month

    if "incident_date" in df.columns:
        df["incident_year"] = df["incident_date"].dt.year
        df["incident_month"] = df["incident_date"].dt.month
        df["incident_day"] = df["incident_date"].dt.day

    if "policy_bind_date" in df.columns and "incident_date" in df.columns:
        df["days_between_bind_and_incident"] = (df["incident_date"] - df["policy_bind_date"]).dt.days

    df = df.drop(columns=[c for c in ["policy_bind_date", "incident_date"] if c in df.columns])
    return df


def main():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Dataset introuvable: {DATA_PATH}")

    # ✅ IMPORTANT: ton CSV est séparé par des ';'
    df = pd.read_csv(DATA_PATH, sep=";")

    df = build_features(df)

    if "fraud_reported" not in df.columns:
        raise ValueError(f"Colonne target 'fraud_reported' introuvable. Colonnes: {list(df.columns)}")

    y = (df["fraud_reported"] == "Y").astype(int)
    X = df.drop(columns=["fraud_reported"])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    num_cols = X_train.select_dtypes(include=["int64", "float64", "int32", "float32"]).columns.tolist()
    cat_cols = X_train.select_dtypes(include=["object", "string", "category"]).columns.tolist()

    numeric_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median"))
    ])

    categorical_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocess = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, num_cols),
            ("cat", categorical_pipe, cat_cols),
        ],
        remainder="drop",
    )

    to_dense = FunctionTransformer(to_dense_array)

    model = Pipeline(steps=[
        ("preprocess", preprocess),
        ("to_dense", to_dense),
        ("model", HistGradientBoostingClassifier(random_state=42))
    ])

    model.fit(X_train, y_train)

    os.makedirs("models", exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    joblib.dump(list(X_train.columns), COLS_PATH)

    print("✅ Modèle sauvegardé :", MODEL_PATH)
    print("✅ Colonnes attendues sauvegardées :", COLS_PATH)


if __name__ == "__main__":
    main()
