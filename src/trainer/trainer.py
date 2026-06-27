"""
trainer.py
----------
Trains 6 models on preprocessed data, evaluates them, and returns
a ranked leaderboard. Supports both classification and regression.

Models:
  Classification : LogisticRegression, RandomForest, XGBoost, LightGBM, SVM, KNN
  Regression     : Ridge, RandomForest, XGBoost, LightGBM, SVR, KNN
"""

import time
import numpy as np
import pandas as pd
from tqdm import tqdm
from dataclasses import dataclass, field
from typing import Any

from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR
from sklearn.calibration import CalibratedClassifierCV
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.model_selection import StratifiedKFold, KFold, cross_validate
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    mean_squared_error, mean_absolute_error, r2_score
)
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor


# ---------------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------------

@dataclass
class ModelResult:
    name: str
    task_type: str
    metrics: dict           # metric_name -> mean value across folds
    metrics_std: dict       # metric_name -> std across folds
    train_time_sec: float
    model: Any              # fitted sklearn estimator
    rank: int = 0


@dataclass
class Leaderboard:
    task_type: str
    primary_metric: str     # metric used for ranking
    results: list[ModelResult] = field(default_factory=list)

    def ranked(self) -> list[ModelResult]:
        """Return results sorted by primary metric descending (higher = better)."""
        reverse = self.primary_metric not in ("rmse", "mae")
        return sorted(self.results, key=lambda r: r.metrics[self.primary_metric], reverse=reverse)

    def to_dataframe(self) -> pd.DataFrame:
        rows = []
        for r in self.ranked():
            row = {"Model": r.name, "Train Time (s)": round(r.train_time_sec, 2)}
            for k, v in r.metrics.items():
                row[k] = round(v, 4)
            rows.append(row)
        return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Model Registry
# ---------------------------------------------------------------------------

CLASSIFICATION_MODELS = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000, random_state=42
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=100, random_state=42, n_jobs=-1
    ),
    "XGBoost": XGBClassifier(
        n_estimators=100, random_state=42,
        eval_metric="logloss", verbosity=0, n_jobs=-1
    ),
    "LightGBM": LGBMClassifier(
        n_estimators=100, random_state=42,
        verbose=-1, n_jobs=-1
    ),
    "SVM": CalibratedClassifierCV(SVC(random_state=42)),
    "KNN": KNeighborsClassifier(n_neighbors=5, n_jobs=-1),
}

REGRESSION_MODELS = {
    "Ridge Regression": Ridge(alpha=1.0),
    "Random Forest": RandomForestRegressor(
        n_estimators=100, random_state=42, n_jobs=-1
    ),
    "XGBoost": XGBRegressor(
        n_estimators=100, random_state=42, verbosity=0, n_jobs=-1
    ),
    "LightGBM": LGBMRegressor(
        n_estimators=100, random_state=42, verbose=-1, n_jobs=-1
    ),
    "SVR": SVR(),
    "KNN": KNeighborsRegressor(n_neighbors=5, n_jobs=-1),
}


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

CLASSIFICATION_SCORING = {
    "accuracy": "accuracy",
    "f1": "f1_weighted",
    "roc_auc": "roc_auc_ovr_weighted",
}

REGRESSION_SCORING = {
    "r2": "r2",
    "mae": "neg_mean_absolute_error",
    "rmse": "neg_root_mean_squared_error",
}

PRIMARY_METRICS = {
    "classification": "f1",
    "regression": "r2",
}

# Metrics where sklearn returns negated values
NEGATE_METRICS = {"mae", "rmse"}


def _fix_metric(name: str, value: float) -> float:
    """Convert negated sklearn metrics back to positive."""
    return -value if name in NEGATE_METRICS else value


# ---------------------------------------------------------------------------
# Core Training Function
# ---------------------------------------------------------------------------

def train_all(
    X: pd.DataFrame,
    y: pd.Series,
    task_type: str,
    cv_folds: int = 5,
    verbose: bool = True,
) -> Leaderboard:
    """
    Cross-validate all models and return a ranked Leaderboard.

    Parameters
    ----------
    X : pd.DataFrame
        Preprocessed feature matrix.
    y : pd.Series
        Target vector (label-encoded for classification).
    task_type : str
        'classification' or 'regression'.
    cv_folds : int
        Number of CV folds (default 5).
    verbose : bool
        Print progress to stdout.

    Returns
    -------
    Leaderboard
    """
    assert task_type in ("classification", "regression"), \
        f"task_type must be 'classification' or 'regression', got '{task_type}'"

    models = CLASSIFICATION_MODELS if task_type == "classification" else REGRESSION_MODELS
    scoring = CLASSIFICATION_SCORING if task_type == "classification" else REGRESSION_SCORING
    primary_metric = PRIMARY_METRICS[task_type]

    cv = (
        StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        if task_type == "classification"
        else KFold(n_splits=cv_folds, shuffle=True, random_state=42)
    )

    leaderboard = Leaderboard(task_type=task_type, primary_metric=primary_metric)

    if verbose:
        print("\n" + "=" * 55)
        print(f"  PipelineIQ — Training ({task_type.upper()}, {cv_folds}-fold CV)")
    model_iterator = tqdm(models.items(), disable=not verbose, desc="Training", leave=True, dynamic_ncols=True)
    for name, model in model_iterator:
        if verbose:
            model_iterator.set_postfix({"model": name})

        t0 = time.time()
        try:
            cv_results = cross_validate(
                model, X, y,
                cv=cv,
                scoring=scoring,
                return_train_score=False,
                n_jobs=1,
            )
            elapsed = time.time() - t0

            metrics = {}
            metrics_std = {}
            for metric_name in scoring:
                key = f"test_{metric_name}"
                raw = cv_results[key]
                metrics[metric_name] = _fix_metric(metric_name, float(np.mean(raw)))
                metrics_std[metric_name] = float(np.std(raw))

            # Fit on full data for later use (SHAP, predictions)
            model.fit(X, y)

            result = ModelResult(
                name=name,
                task_type=task_type,
                metrics=metrics,
                metrics_std=metrics_std,
                train_time_sec=elapsed,
                model=model,
            )
            leaderboard.results.append(result)

            if verbose:
                pm_val = metrics[primary_metric]
                model_iterator.set_postfix({
                    "model": name,
                    primary_metric: f"{pm_val:.4f}", 
                    "time": f"{elapsed:.1f}s"
                })

        except Exception as e:
            elapsed = time.time() - t0
            if verbose:
                print(f"  [FAILED] {name} — {e}")

    # Assign ranks
    ranked = leaderboard.ranked()
    for i, r in enumerate(ranked):
        r.rank = i + 1

    if verbose:
        print("\n" + "=" * 55)
        print(f"  Best Model: {ranked[0].name}")
        print(f"  {primary_metric} = {ranked[0].metrics[primary_metric]:.4f}")
        print("=" * 55 + "\n")

    return leaderboard


def get_best_model(leaderboard: Leaderboard) -> ModelResult:
    """Return the top-ranked ModelResult."""
    return leaderboard.ranked()[0]


def print_leaderboard(leaderboard: Leaderboard):
    """Pretty-print the full leaderboard table."""
    df = leaderboard.to_dataframe()
    print("\nLeaderboard:")
    print(df.to_string(index=False))
    print()
