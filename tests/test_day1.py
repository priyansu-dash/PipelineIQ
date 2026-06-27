"""
test_day1.py
------------
Basic sanity checks for profiler, preprocessor, and trainer.
Run with: pytest tests/test_day1.py -v
"""

import pytest
import numpy as np
import pandas as pd

from src.profiler.profiler import profile, ProfileReport
from src.preprocessor.preprocessor import build_preprocessor
from src.trainer.trainer import train_all, Leaderboard


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def classification_df():
    np.random.seed(42)
    return pd.DataFrame({
        "age":       np.random.randint(20, 70, 200),
        "income":    np.random.exponential(50000, 200),   # skewed
        "city":      np.random.choice(["Mumbai", "Delhi", "Bangalore"], 200),
        "score":     np.random.uniform(0, 1, 200),
        "constant":  [1] * 200,                            # should be dropped
        "target":    np.random.choice([0, 1], 200),
    })


@pytest.fixture
def regression_df():
    np.random.seed(0)
    return pd.DataFrame({
        "rooms":    np.random.randint(1, 10, 150),
        "area":     np.random.uniform(500, 5000, 150),
        "location": np.random.choice(["urban", "suburban", "rural"], 150),
        "price":    np.random.uniform(100000, 900000, 150),
    })


# ---------------------------------------------------------------------------
# Profiler Tests
# ---------------------------------------------------------------------------

class TestProfiler:
    def test_returns_profile_report(self, classification_df):
        report = profile(classification_df, "target")
        assert isinstance(report, ProfileReport)

    def test_detects_classification(self, classification_df):
        report = profile(classification_df, "target")
        assert report.task_type == "classification"

    def test_detects_regression(self, regression_df):
        report = profile(regression_df, "price")
        assert report.task_type == "regression"

    def test_detects_constant_col(self, classification_df):
        report = profile(classification_df, "target")
        assert "constant" in report.constant_cols

    def test_detects_skewed_col(self, classification_df):
        report = profile(classification_df, "target")
        assert "income" in report.skewness

    def test_feature_types_excludes_target(self, classification_df):
        report = profile(classification_df, "target")
        assert "target" not in report.feature_types

    def test_invalid_target_raises(self, classification_df):
        with pytest.raises(ValueError):
            profile(classification_df, "nonexistent_col")


# ---------------------------------------------------------------------------
# Preprocessor Tests
# ---------------------------------------------------------------------------

class TestPreprocessor:
    def test_returns_dataframe(self, classification_df):
        report = profile(classification_df, "target")
        _, X, y = build_preprocessor(classification_df, report)
        assert isinstance(X, pd.DataFrame)

    def test_drops_constant_col(self, classification_df):
        report = profile(classification_df, "target")
        _, X, _ = build_preprocessor(classification_df, report)
        assert "constant" not in X.columns

    def test_no_missing_after_preprocessing(self, classification_df):
        classification_df.loc[0:10, "age"] = np.nan
        report = profile(classification_df, "target")
        _, X, _ = build_preprocessor(classification_df, report)
        assert X.isnull().sum().sum() == 0

    def test_target_encoded_for_classification(self, classification_df):
        report = profile(classification_df, "target")
        _, _, y = build_preprocessor(classification_df, report)
        assert y.dtype in [np.int32, np.int64, int]

    def test_output_shape_matches(self, classification_df):
        report = profile(classification_df, "target")
        _, X, y = build_preprocessor(classification_df, report)
        assert len(X) == len(y) == len(classification_df)


# ---------------------------------------------------------------------------
# Trainer Tests
# ---------------------------------------------------------------------------

class TestTrainer:
    def test_returns_leaderboard(self, classification_df):
        report = profile(classification_df, "target")
        _, X, y = build_preprocessor(classification_df, report)
        lb = train_all(X, y, task_type="classification", cv_folds=3, verbose=False)
        assert isinstance(lb, Leaderboard)

    def test_leaderboard_has_6_models(self, classification_df):
        report = profile(classification_df, "target")
        _, X, y = build_preprocessor(classification_df, report)
        lb = train_all(X, y, task_type="classification", cv_folds=3, verbose=False)
        assert len(lb.results) == 6

    def test_primary_metric_classification(self, classification_df):
        report = profile(classification_df, "target")
        _, X, y = build_preprocessor(classification_df, report)
        lb = train_all(X, y, task_type="classification", cv_folds=3, verbose=False)
        assert lb.primary_metric == "f1"

    def test_ranked_descending(self, classification_df):
        report = profile(classification_df, "target")
        _, X, y = build_preprocessor(classification_df, report)
        lb = train_all(X, y, task_type="classification", cv_folds=3, verbose=False)
        ranked = lb.ranked()
        scores = [r.metrics["f1"] for r in ranked]
        assert scores == sorted(scores, reverse=True)

    def test_regression_pipeline(self, regression_df):
        report = profile(regression_df, "price")
        _, X, y = build_preprocessor(regression_df, report)
        lb = train_all(X, y, task_type="regression", cv_folds=3, verbose=False)
        assert lb.primary_metric == "r2"
        assert len(lb.results) == 6
