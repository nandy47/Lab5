import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder


def split_data(X, y):
    return train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)


def get_column_groups(X_train):
    numeric_cols = X_train.select_dtypes(include="number").columns.tolist()
    categorical_cols = X_train.select_dtypes(include=["object", "category", "string"]).columns.tolist()
    if "Pclass" in numeric_cols:
        numeric_cols.remove("Pclass")
        categorical_cols.append("Pclass")
    return numeric_cols, categorical_cols


def build_preprocessor(num_cols, cat_cols):
    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    return ColumnTransformer([
        ("num", num_pipeline, num_cols),
        ("cat", cat_pipeline, cat_cols),
    ])


def build_pipeline(num_cols, cat_cols, model):
    return Pipeline([
        ("preprocessor", build_preprocessor(num_cols, cat_cols)),
        ("model", model),
    ])


def engineer(df):
    df = df.copy()
    df["has_cabin"] = df["Cabin"].notna().astype(int)
    df["family_size"] = df["SibSp"] + df["Parch"] + 1
    df["is_alone"] = (df["family_size"] == 1).astype(int)
    titles = df["Name"].str.extract(r",\s*([^\.]+)\.", expand=False)
    df["Title"] = titles.where(titles.isin(["Mr", "Miss", "Mrs", "Master"]), "Other")
    df["Fare"] = df["Fare"].replace(0, np.nan)
    return df.drop(columns=["Name", "Ticket", "Cabin", "home.dest"], errors="ignore")