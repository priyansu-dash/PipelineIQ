"""
preprocessor.py
---------------
Builds a leakage-free sklearn Pipeline from a ProfileReport.
Handles: missing imputation, encoding, scaling, constant col removal,
datetime feature extraction, and optional log transform for skewed features.
"""

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import (
    StandardScaler, LabelEncoder, OrdinalEncoder, FunctionTransformer
)
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin

from src.profiler.profiler import ProfileReport


# ---------------------------------------------------------------------------
# Custom Transformers
# ---------------------------------------------------------------------------

class DatetimeFeatureExtractor(BaseEstimator, TransformerMixin):
    """
    Extracts year, month, day, dayofweek, hour from datetime columns.
    Drops original datetime column afterward.
    """
    def __init__(self, datetime_cols: list):
        self.datetime_cols = datetime_cols

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        for col in self.datetime_cols:
            if col not in X.columns:
                continue
            dt = pd.to_datetime(X[col], errors="coerce")
            X[f"{col}_year"] = dt.dt.year
            X[f"{col}_month"] = dt.dt.month
            X[f"{col}_day"] = dt.dt.day
            X[f"{col}_dayofweek"] = dt.dt.dayofweek
            X[f"{col}_hour"] = dt.dt.hour
            X.drop(columns=[col], inplace=True)
        return X


class ConstantColumnDropper(BaseEstimator, TransformerMixin):
    """Drops columns with a single unique value (zero variance)."""
    def __init__(self, constant_cols: list):
        self.constant_cols = constant_cols

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        cols_to_drop = [c for c in self.constant_cols if c in X.columns]
        return X.drop(columns=cols_to_drop)


class LogTransformer(BaseEstimator, TransformerMixin):
    """
    Applies log1p to specified columns to reduce skew.
    Safe for non-negative data; clips negatives to 0 before transform.
    """
    def __init__(self, cols: list):
        self.cols = cols

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        for col in self.cols:
            if col in X.columns:
                X[col] = np.log1p(np.clip(X[col], 0, None))
        return X


class DataFramePassthrough(BaseEstimator, TransformerMixin):
    """Keeps DataFrame format through the pipeline for easier debugging."""
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X


# ---------------------------------------------------------------------------
# Pipeline Builder
# ---------------------------------------------------------------------------

def _get_numeric_pipeline(impute_strategy: str = "median") -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy=impute_strategy)),
        ("scaler", StandardScaler()),
    ])


def _get_categorical_pipeline(impute_strategy: str = "most_frequent") -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy=impute_strategy, fill_value="missing")),
        ("encoder", OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=-1
        )),
    ])


def build_preprocessor(
    df: pd.DataFrame,
    report: ProfileReport,
    skew_threshold: float = 2.0,
    apply_log_transform: bool = True,
) -> tuple[Pipeline, pd.DataFrame]:
    """
    Builds and returns a fitted sklearn preprocessing Pipeline.

    Parameters
    ----------
    df : pd.DataFrame
        Raw training dataframe (should NOT include target column).
    report : ProfileReport
        Output from profiler.profile().
    skew_threshold : float
        Columns with |skew| > threshold get log-transformed.
    apply_log_transform : bool
        Whether to apply log1p to skewed numeric columns.

    Returns
    -------
    pipeline : sklearn Pipeline (fitted on df)
    X_transformed : pd.DataFrame of transformed features
    """
    X = df.drop(columns=[report.target_col])

    # --- Step 1: Drop constant columns ---
    dropper = ConstantColumnDropper(report.constant_cols)
    X = dropper.transform(X)

    # --- Step 2: Extract datetime features ---
    datetime_cols = [c for c, t in report.feature_types.items() if t == "datetime" and c in X.columns]
    dt_extractor = DatetimeFeatureExtractor(datetime_cols)
    X = dt_extractor.transform(X)

    # --- Step 3: Log-transform skewed features ---
    if apply_log_transform:
        skewed_cols = [c for c, v in report.skewness.items() if abs(v) > skew_threshold and c in X.columns]
    else:
        skewed_cols = []
    log_transformer = LogTransformer(skewed_cols)
    X = log_transformer.transform(X)

    # --- Step 4: Identify final numeric and categorical cols ---
    numeric_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    categorical_cols = [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])]

    # --- Step 5: Build ColumnTransformer ---
    transformers = []
    if numeric_cols:
        transformers.append(("numeric", _get_numeric_pipeline(), numeric_cols))
    if categorical_cols:
        transformers.append(("categorical", _get_categorical_pipeline(), categorical_cols))

    col_transformer = ColumnTransformer(transformers=transformers, remainder="drop")

    pipeline = Pipeline([
        ("col_transform", col_transformer)
    ])

    X_transformed_arr = pipeline.fit_transform(X)

    # Reconstruct column names for interpretability
    out_cols = numeric_cols + categorical_cols
    X_transformed = pd.DataFrame(X_transformed_arr, columns=out_cols, index=X.index)

    # Encode target
    y = df[report.target_col].copy()
    if report.task_type == "classification":
        le = LabelEncoder()
        y = pd.Series(le.fit_transform(y.astype(str)), index=y.index, name=report.target_col)
    
    return pipeline, X_transformed, y


def get_preprocessing_summary(
    report: ProfileReport,
    skew_threshold: float = 2.0
) -> dict:
    """
    Returns a human-readable summary of what the preprocessor will do.
    Useful for displaying in the Streamlit dashboard before running.
    """
    skewed_cols = [c for c, v in report.skewness.items() if abs(v) > skew_threshold]
    datetime_cols = [c for c, t in report.feature_types.items() if t == "datetime"]
    numeric_cols = [c for c, t in report.feature_types.items() if t == "numeric"]
    categorical_cols = [c for c, t in report.feature_types.items() if t == "categorical"]

    return {
        "constant_cols_dropped": report.constant_cols,
        "datetime_cols_extracted": datetime_cols,
        "log_transformed_cols": skewed_cols,
        "numeric_cols_scaled": [c for c in numeric_cols if c not in skewed_cols],
        "categorical_cols_encoded": categorical_cols,
        "imputation": {
            "numeric": "median",
            "categorical": "most_frequent",
        }
    }


def print_preprocessing_summary(report: ProfileReport, skew_threshold: float = 2.0):
    """Pretty-print what the preprocessor will do."""
    summary = get_preprocessing_summary(report, skew_threshold)
    print("\n" + "=" * 55)
    print("PipelineIQ — Preprocessing Plan")
    print("=" * 55)

    if summary["constant_cols_dropped"]:
        print(f"Drop constant cols   : {summary['constant_cols_dropped']}")
    if summary["datetime_cols_extracted"]:
        print(f"Extract datetime     : {summary['datetime_cols_extracted']}")
    if summary["log_transformed_cols"]:
        print(f"Log-transform (skew) : {summary['log_transformed_cols']}")
    print(f"Scale (StandardScaler): {summary['numeric_cols_scaled']}")
    print(f"Encode (Ordinal)      : {summary['categorical_cols_encoded']}")
    print(f"Impute numeric        : {summary['imputation']['numeric']}")
    print(f"Impute categorical    : {summary['imputation']['categorical']}")
    print("=" * 55 + "\n")
