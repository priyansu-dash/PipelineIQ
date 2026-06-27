"""
explainer.py
------------
We use SHAP here to crack open the best models and see what makes them tick. 
It drops a summary plot and a waterfall plot into the outputs folder.
"""

import shap
import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
import warnings
from typing import Any, List

def generate_shap_explanations(
    model: Any, 
    X: pd.DataFrame, 
    output_dir: str = "outputs/shap", 
    max_display: int = 10,
    model_name: str = "model"
) -> List[str]:
    """
    Crunches the SHAP values and spits out the summary and waterfall charts.
    """
    os.makedirs(output_dir, exist_ok=True)
    warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
    
    # Downsample if the dataset is massive so we don't sit here waiting forever
    if len(X) > 500:
        X_sample = X.sample(500, random_state=42)
    else:
        X_sample = X

    is_tree = type(model).__name__ in [
        "RandomForestClassifier", "RandomForestRegressor", 
        "XGBClassifier", "XGBRegressor", 
        "LGBMClassifier", "LGBMRegressor"
    ]
    
    if is_tree:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer(X_sample)
    else:
        # For non-tree models, we have to fall back to the slower KernelExplainer
        X_background = shap.kmeans(X, min(10, len(X)))
        
        def predict_wrapper(X_in):
            if hasattr(model, "predict_proba"):
                preds = model.predict_proba(X_in)
                # Keep probability of positive class for binary classification
                return preds[:, 1] if len(preds.shape) == 2 else preds
            return model.predict(X_in)
            
        explainer = shap.KernelExplainer(predict_wrapper, X_background)
        # Grab a tiny slice for the kernel explainer to keep things snappy
        X_sample = X_sample[:50]
        shap_values = explainer(X_sample, silent=True)

    # Sometimes SHAP gives us 3D arrays for classification, we just want the positive class
    if len(shap_values.shape) == 3: # (samples, features, classes)
        shap_values = shap_values[:, :, 1]
    
    plt.figure()
    shap.summary_plot(shap_values, X_sample, max_display=max_display, show=False)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{model_name}_shap_summary.png"), dpi=300)
    plt.close()
    
    plt.figure()
    shap.plots.waterfall(shap_values[0], max_display=max_display, show=False)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{model_name}_shap_waterfall.png"), dpi=300)
    plt.close()
    
    print(f"SHAP explanations saved to {output_dir}/")
    
    # Figure out the absolute top features driving this model
    if hasattr(shap_values, "values"):
        mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
    else:
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        
    top_indices = np.argsort(mean_abs_shap)[::-1][:3]
    top_features = X_sample.columns[top_indices].tolist()
    
    return top_features
