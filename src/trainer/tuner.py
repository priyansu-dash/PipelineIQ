"""
tuner.py
--------
Let Optuna loose to find the mathematically best hyperparameters for our winning model. 
Everything gets tracked silently in MLflow.
"""

import optuna
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, KFold, cross_val_score
from typing import Any

from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR
from sklearn.calibration import CalibratedClassifierCV
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor

from src.trainer.trainer import PRIMARY_METRICS, CLASSIFICATION_SCORING, REGRESSION_SCORING

def get_model_and_params(trial: optuna.Trial, model_name: str, task_type: str) -> Any:
    """Spins up a fresh model with Optuna's suggested parameters for this trial."""
    if task_type == "classification":
        if model_name == "Logistic Regression":
            c = trial.suggest_float("C", 1e-4, 10.0, log=True)
            return LogisticRegression(C=c, max_iter=1000, random_state=42)
        elif model_name == "Random Forest":
            n_estimators = trial.suggest_int("n_estimators", 50, 300)
            max_depth = trial.suggest_int("max_depth", 3, 15)
            return RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42, n_jobs=-1)
        elif model_name == "XGBoost":
            n_estimators = trial.suggest_int("n_estimators", 50, 300)
            max_depth = trial.suggest_int("max_depth", 3, 10)
            lr = trial.suggest_float("learning_rate", 1e-3, 0.3, log=True)
            return XGBClassifier(n_estimators=n_estimators, max_depth=max_depth, learning_rate=lr, 
                                 eval_metric="logloss", verbosity=0, random_state=42, n_jobs=-1)
        elif model_name == "LightGBM":
            n_estimators = trial.suggest_int("n_estimators", 50, 300)
            max_depth = trial.suggest_int("max_depth", 3, 10)
            lr = trial.suggest_float("learning_rate", 1e-3, 0.3, log=True)
            return LGBMClassifier(n_estimators=n_estimators, max_depth=max_depth, learning_rate=lr, 
                                  verbose=-1, random_state=42, n_jobs=-1)
        elif model_name == "SVM":
            c = trial.suggest_float("C", 1e-3, 10.0, log=True)
            return CalibratedClassifierCV(SVC(C=c, random_state=42))
        elif model_name == "KNN":
            n_neighbors = trial.suggest_int("n_neighbors", 3, 15)
            return KNeighborsClassifier(n_neighbors=n_neighbors, n_jobs=-1)
    else:
        if model_name == "Ridge Regression":
            alpha = trial.suggest_float("alpha", 1e-4, 10.0, log=True)
            return Ridge(alpha=alpha)
        elif model_name == "Random Forest":
            n_estimators = trial.suggest_int("n_estimators", 50, 300)
            max_depth = trial.suggest_int("max_depth", 3, 15)
            return RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42, n_jobs=-1)
        elif model_name == "XGBoost":
            n_estimators = trial.suggest_int("n_estimators", 50, 300)
            max_depth = trial.suggest_int("max_depth", 3, 10)
            lr = trial.suggest_float("learning_rate", 1e-3, 0.3, log=True)
            return XGBRegressor(n_estimators=n_estimators, max_depth=max_depth, learning_rate=lr, 
                                verbosity=0, random_state=42, n_jobs=-1)
        elif model_name == "LightGBM":
            n_estimators = trial.suggest_int("n_estimators", 50, 300)
            max_depth = trial.suggest_int("max_depth", 3, 10)
            lr = trial.suggest_float("learning_rate", 1e-3, 0.3, log=True)
            return LGBMRegressor(n_estimators=n_estimators, max_depth=max_depth, learning_rate=lr, 
                                 verbose=-1, random_state=42, n_jobs=-1)
        elif model_name == "SVR":
            c = trial.suggest_float("C", 1e-3, 10.0, log=True)
            epsilon = trial.suggest_float("epsilon", 1e-3, 1.0, log=True)
            return SVR(C=c, epsilon=epsilon)
        elif model_name == "KNN":
            n_neighbors = trial.suggest_int("n_neighbors", 3, 15)
            return KNeighborsRegressor(n_neighbors=n_neighbors, n_jobs=-1)
    
    raise ValueError(f"Unknown model_name: {model_name} for task_type: {task_type}")

def tune_model(
    model_name: str, 
    task_type: str, 
    X: pd.DataFrame, 
    y: pd.Series, 
    n_trials: int = 15,
    experiment_name: str = "PipelineIQ_Experiment"
) -> Any:
    """Fires up the Optuna search engine, logs to MLflow, and returns the optimized model."""
    
    mlflow.set_experiment(experiment_name)
    
    cv_folds = 5
    if task_type == "classification":
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        primary_metric_key = PRIMARY_METRICS["classification"]
        scoring = CLASSIFICATION_SCORING[primary_metric_key]
        direction = "maximize"
    else:
        cv = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
        primary_metric_key = PRIMARY_METRICS["regression"]
        scoring = REGRESSION_SCORING[primary_metric_key]
        direction = "maximize" if scoring == "r2" else "minimize"

    def objective(trial):
        model = get_model_and_params(trial, model_name, task_type)
        
        with mlflow.start_run(nested=True):
            mlflow.log_params(trial.params)
            mlflow.log_param("model_name", model_name)
            mlflow.log_param("task_type", task_type)
            
            scores = cross_val_score(model, X, y, cv=cv, scoring=scoring, n_jobs=1)
            score = np.mean(scores)
            
            mlflow.log_metric(f"cv_mean_{primary_metric_key}", score)
            mlflow.log_metric(f"cv_std_{primary_metric_key}", float(np.std(scores)))
            
        return score
    
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction=direction)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    
    print(f"Optuna tuning complete for {model_name}. Best trial {primary_metric_key}: {study.best_value:.4f}")
    
    best_model = get_model_and_params(study.best_trial, model_name, task_type)
    
    with mlflow.start_run(run_name=f"Best_{model_name}"):
        mlflow.log_params(study.best_params)
        mlflow.log_metric(f"best_cv_{primary_metric_key}", study.best_value)
        best_model.fit(X, y)
        mlflow.sklearn.log_model(best_model, "model")
        
    return best_model
