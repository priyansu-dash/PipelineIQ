# 🧬 PipelineIQ
**Automated Machine Learning Benchmark and Tracking Dashboard**

PipelineIQ is a lightweight, fully automated end-to-end Machine Learning pipeline built for rapid prototyping. Drop in any tabular dataset, and the system will automatically handle data profiling, dynamic preprocessing, model training, hyperparameter tuning, and advanced model explainability—all orchestrated through a sleek Streamlit dashboard.

---

## 🚀 Features

- **Automated Data Profiling**: Instantly detects feature types, missing values, skewness, and class imbalance.
- **Dynamic Preprocessing**: Builds leakage-free Scikit-Learn pipelines that automatically impute, scale, encode, and log-transform data based on the profile.
- **Model Leaderboard**: Trains and evaluates 6 industry-standard models (Logistic Regression, Random Forest, XGBoost, LightGBM, SVM, KNN) using robust cross-validation.
- **Hyperparameter Tuning (Optuna)**: Automatically searches for the optimal parameters for the winning model.
- **Experiment Tracking (MLflow)**: Silently logs every metric, parameter, and trial locally.
- **Explainability (SHAP)**: Cracks open the winning model to generate Global Feature Importance and Local Prediction Waterfall plots.
- **LLM AI Summary**: Integrates with the **Groq API** to read the SHAP results and generate a jargon-free, stakeholder-ready plain English summary.
- **One-Click Export**: Bundles the fitted preprocessor and tuned model into a single `.pkl` file for instant deployment.

## 🛠️ Tech Stack

- **Core**: Python, Scikit-Learn, Pandas, NumPy
- **MLOps**: MLflow, Optuna
- **Explainability**: SHAP
- **UI & LLM**: Streamlit, Groq API (Llama 3.1)

## 💻 Quick Start

### 1. Install Dependencies
Make sure you have conda or an active Python environment.
```bash
pip install -r requirements.txt
```

### 2. Add API Key (Optional but recommended)
Rename the provided `.env.example` file to `.env` and paste your free Groq API key:
```env
GROQ_API_KEY=your_api_key_here
```

### 3. Run the Dashboard
```bash
streamlit run app.py
```
Upload any CSV (e.g., Titanic, Breast Cancer, House Prices) and hit **Run PipelineIQ**.

## 📁 Directory Structure
```
PipelineIQ/
├── app.py                   # Main Streamlit dashboard
├── src/
│   ├── profiler/            # Auto-detects data types & issues
│   ├── preprocessor/        # Auto-builds sklearn pipelines
│   └── trainer/             # Trains, tunes, and explains models
├── .env.example             # Template for API keys
└── .gitignore
```

---
*Built as a showcase for end-to-end Machine Learning architecture.*
