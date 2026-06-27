"""
profiler.py
-----------
Auto data profiling: types, missing values, skew, class balance, correlations.
Returns a structured ProfileReport dict consumed by the Streamlit dashboard.
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ProfileReport:
    n_rows: int
    n_cols: int
    target_col: str
    task_type: str                          # 'classification' | 'regression'
    feature_types: dict                     # col -> 'numeric' | 'categorical' | 'datetime'
    missing: dict                           # col -> % missing
    skewness: dict                          # numeric col -> skew value
    class_distribution: Optional[dict]     # only for classification
    high_cardinality_cols: list            # categorical cols with > threshold unique vals
    high_correlation_pairs: list           # list of (col1, col2, corr) tuples
    constant_cols: list                    # cols with a single unique value (useless)
    warnings: list[str] = field(default_factory=list)


def infer_task_type(series: pd.Series) -> str:
    """Classify target as regression or classification."""
    if series.dtype == object or series.nunique() <= 20:
        return "classification"
    return "regression"


def infer_feature_types(df: pd.DataFrame, target_col: str) -> dict:
    """Infer semantic type for each feature column."""
    types = {}
    for col in df.columns:
        if col == target_col:
            continue
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            types[col] = "datetime"
        elif pd.api.types.is_numeric_dtype(df[col]):
            types[col] = "numeric"
        else:
            # Try parsing as datetime
            try:
                pd.to_datetime(df[col], infer_datetime_format=True)
                types[col] = "datetime"
            except Exception:
                types[col] = "categorical"
    return types


def compute_missing(df: pd.DataFrame) -> dict:
    """Return % missing per column, only for cols with any missing."""
    missing_pct = (df.isnull().mean() * 100).round(2)
    return missing_pct[missing_pct > 0].to_dict()


def compute_skewness(df: pd.DataFrame, feature_types: dict) -> dict:
    """Return skewness for numeric columns."""
    numeric_cols = [c for c, t in feature_types.items() if t == "numeric"]
    skew = df[numeric_cols].skew().round(3)
    return skew[skew.abs() > 0.5].to_dict()   # only flag meaningfully skewed


def compute_class_distribution(series: pd.Series) -> dict:
    """Value counts as percentages for classification targets."""
    counts = series.value_counts(normalize=True).round(3) * 100
    return counts.to_dict()


def find_high_cardinality(df: pd.DataFrame, feature_types: dict, threshold: int = 50) -> list:
    """Categorical cols with more unique values than threshold."""
    cats = [c for c, t in feature_types.items() if t == "categorical"]
    return [c for c in cats if df[c].nunique() > threshold]


def find_high_correlations(df: pd.DataFrame, feature_types: dict, threshold: float = 0.9) -> list:
    """Pairs of numeric features with absolute correlation above threshold."""
    numeric_cols = [c for c, t in feature_types.items() if t == "numeric"]
    if len(numeric_cols) < 2:
        return []
    corr_matrix = df[numeric_cols].corr().abs()
    pairs = []
    cols = corr_matrix.columns.tolist()
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            val = corr_matrix.iloc[i, j]
            if val >= threshold:
                pairs.append((cols[i], cols[j], round(float(val), 3)))
    return pairs


def find_constant_cols(df: pd.DataFrame, target_col: str) -> list:
    """Columns with only one unique value — zero information."""
    return [c for c in df.columns if c != target_col and df[c].nunique() <= 1]


def build_warnings(report_data: dict) -> list[str]:
    """Generate human-readable warnings from profile data."""
    warnings = []

    # Missing data
    high_missing = {c: v for c, v in report_data["missing"].items() if v > 30}
    for col, pct in high_missing.items():
        warnings.append(f"'{col}' has {pct:.1f}% missing values — consider dropping or careful imputation.")

    # Class imbalance
    if report_data["task_type"] == "classification" and report_data["class_distribution"]:
        dist = report_data["class_distribution"]
        min_pct = min(dist.values())
        if min_pct < 10:
            warnings.append(f"Class imbalance detected — minority class at {min_pct:.1f}%. Consider SMOTE or class weights.")

    # High cardinality
    for col in report_data["high_cardinality_cols"]:
        warnings.append(f"'{col}' has high cardinality — may need target encoding or dropping.")

    # High correlations
    for c1, c2, corr in report_data["high_correlation_pairs"]:
        warnings.append(f"'{c1}' and '{c2}' are highly correlated ({corr}) — consider removing one.")

    # Constant columns
    for col in report_data["constant_cols"]:
        warnings.append(f"'{col}' is constant — will be dropped automatically.")

    # High skew
    high_skew = {c: v for c, v in report_data["skewness"].items() if abs(v) > 2}
    for col, skew in high_skew.items():
        warnings.append(f"'{col}' is highly skewed ({skew:.2f}) — log/sqrt transform recommended.")

    return warnings


def profile(df: pd.DataFrame, target_col: str) -> ProfileReport:
    """
    Main entry point. Returns a ProfileReport for the given DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Raw input dataframe including target column.
    target_col : str
        Name of the target/label column.

    Returns
    -------
    ProfileReport
    """
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in DataFrame.")

    task_type = infer_task_type(df[target_col])
    feature_types = infer_feature_types(df, target_col)
    missing = compute_missing(df)
    skewness = compute_skewness(df, feature_types)
    class_dist = compute_class_distribution(df[target_col]) if task_type == "classification" else None
    high_card = find_high_cardinality(df, feature_types)
    high_corr = find_high_correlations(df, feature_types)
    constant_cols = find_constant_cols(df, target_col)

    report_data = {
        "missing": missing,
        "task_type": task_type,
        "class_distribution": class_dist,
        "high_cardinality_cols": high_card,
        "high_correlation_pairs": high_corr,
        "constant_cols": constant_cols,
        "skewness": skewness,
    }

    warnings = build_warnings(report_data)

    return ProfileReport(
        n_rows=len(df),
        n_cols=len(df.columns),
        target_col=target_col,
        task_type=task_type,
        feature_types=feature_types,
        missing=missing,
        skewness=skewness,
        class_distribution=class_dist,
        high_cardinality_cols=high_card,
        high_correlation_pairs=high_corr,
        constant_cols=constant_cols,
        warnings=warnings,
    )


def print_report(report: ProfileReport):
    """Pretty-print a ProfileReport to stdout."""
    print("\n" + "=" * 55)
    print("PipelineIQ — Data Profile Report")
    print("=" * 55)
    print(f"  Rows          : {report.n_rows}")
    print(f"  Columns       : {report.n_cols}")
    print(f"  Target        : {report.target_col}")
    print(f"  Task Type     : {report.task_type.upper()}")
    print()

    print("  Feature Types:")
    for col, ftype in report.feature_types.items():
        print(f"    {col:<30} {ftype}")

    if report.missing:
        print("\n  Missing Values:")
        for col, pct in report.missing.items():
            print(f"    {col:<30} {pct:.1f}%")

    if report.skewness:
        print("\n  Skewed Features (|skew| > 0.5):")
        for col, skew in report.skewness.items():
            print(f"    {col:<30} {skew:.3f}")

    if report.class_distribution:
        print("\n  Class Distribution:")
        for cls, pct in report.class_distribution.items():
            print(f"    {str(cls):<30} {pct:.1f}%")

    if report.warnings:
        print("\n  Warnings:")
        for w in report.warnings:
            print(f"    {w}")

    print("=" * 55 + "\n")
