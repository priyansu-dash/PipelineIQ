import streamlit as st
import pandas as pd
import numpy as np
import os
import pickle
from PIL import Image
try:
    from groq import Groq
except ImportError:
    Groq = None

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from src.profiler.profiler import profile
from src.preprocessor.preprocessor import build_preprocessor
from src.trainer import train_all, get_best_model, tune_model, generate_shap_explanations

# Create the SHAP output dir if it's missing to avoid crashes
os.makedirs("outputs/shap_app", exist_ok=True)

st.set_page_config(page_title="PipelineIQ", layout="wide")

# Inject some custom CSS to make the UI pop
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    .stApp {
        background-color: #0b0f19;
        color: #e2e8f0;
    }
    
    h1, h2, h3 {
        color: #38bdf8 !important;
        font-weight: 700;
        letter-spacing: -0.5px;
    }
    
    /* Stylish Primary Button */
    .stButton > button {
        background: linear-gradient(135deg, #0284c7 0%, #38bdf8 100%);
        color: white;
        border-radius: 12px;
        border: none;
        padding: 0.6rem 2rem;
        font-weight: 600;
        box-shadow: 0 4px 14px 0 rgba(56, 189, 248, 0.39);
        transition: all 0.2s ease-in-out;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px 0 rgba(56, 189, 248, 0.45);
    }
    
    /* Metrics styling */
    [data-testid="stMetricValue"] {
        color: #10b981 !important; 
        font-weight: 700;
    }
    [data-testid="stMetricLabel"] {
        color: #94a3b8 !important;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    /* Upload Box */
    .stFileUploader > div > div {
        background-color: rgba(255, 255, 255, 0.03);
        border: 2px dashed #334155;
        border-radius: 12px;
        transition: all 0.3s;
    }
    .stFileUploader > div > div:hover {
        border-color: #38bdf8;
        background-color: rgba(56, 189, 248, 0.05);
    }
</style>
""", unsafe_allow_html=True)

st.title("PipelineIQ")
st.markdown("### Automated end-to-end Machine Learning benchmark and tracking dashboard.")

st.sidebar.markdown("### Groq AI Summary")

# Pull the Groq API key from .env locally or from Streamlit secrets in prod
groq_api_key = os.environ.get("GROQ_API_KEY")
if not groq_api_key and hasattr(st, "secrets") and "GROQ_API_KEY" in st.secrets:
    groq_api_key = st.secrets["GROQ_API_KEY"]

if groq_api_key:
    st.sidebar.success("AI Summarization is Active")
else:
    st.sidebar.info("AI Summarization is Disabled. Add GROQ_API_KEY to your .env file to enable.")

# Handle file upload
uploaded_file = st.file_uploader("Drop your dataset here (CSV)", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    
    st.markdown("---")
    col1, col2 = st.columns([1, 2])
    with col1:
        st.subheader("Data Configuration")
        target_col = st.selectbox("Select Target Column", df.columns, index=len(df.columns)-1)
    
    with col2:
        st.write(f"**Dataset Preview**: `{df.shape[0]}` rows, `{df.shape[1]}` columns")
        st.dataframe(df.head(4), use_container_width=True)
    
    if st.button("Run PipelineIQ"):
        # Run data profiling
        with st.spinner("Profiling Data..."):
            report = profile(df, target_col)
            
        st.markdown("---")
        st.subheader("Data Profile")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Task Type", report.task_type.upper())
        m2.metric("Features", report.n_cols - 1)
        
        missing_pct = sum(report.missing.values()) / len(report.missing) if report.missing else 0
        m3.metric("Avg Missing %", f"{missing_pct:.1f}%")
        
        imbalance = "N/A"
        if report.task_type == "classification" and report.class_distribution:
            imbalance = f"Min Class: {min(report.class_distribution.values()):.1f}%"
        m4.metric("Class Imbalance", imbalance)
        
        if report.warnings:
            for w in report.warnings:
                st.warning(f"{w}")
                
        # Build the preprocessing pipeline based on the profile
        with st.spinner("Building Preprocessing Pipeline..."):
            pipeline, X_transformed, y = build_preprocessor(df, report)
            
        # Train all baseline models and generate the leaderboard
        st.markdown("---")
        st.subheader("Model Leaderboard (3-fold CV)")
        with st.spinner("Training baseline models..."):
            leaderboard = train_all(X_transformed, y, task_type=report.task_type, cv_folds=3, verbose=False)
            st.dataframe(leaderboard.to_dataframe(), use_container_width=True)
            
        best_model_result = get_best_model(leaderboard)
        best_model_name = best_model_result.name
        
        st.info(f"Best Model: **{best_model_name}**")
        
        # Tune the best model using Optuna
        st.markdown("---")
        st.subheader(f"Optuna Tuning: {best_model_name}")
        with st.spinner("Running Optuna hyperparameter search..."):
            tuned_model = tune_model(
                model_name=best_model_name,
                task_type=report.task_type,
                X=X_transformed,
                y=y,
                n_trials=10,
                experiment_name="PipelineIQ_Streamlit"
            )
            st.success("Tuning complete! Parameters and metrics tracked in MLflow.")
            
        # Generate SHAP plots for interpretability
        st.markdown("---")
        st.subheader("Model Explainability (SHAP)")
        with st.spinner("Generating feature importance plots..."):
            output_dir = "outputs/shap_app"
            model_slug = best_model_name.replace(" ", "_")
            top_features = generate_shap_explanations(
                model=tuned_model,
                X=X_transformed,
                output_dir=output_dir,
                model_name=model_slug
            )
            
            try:
                summary_img = Image.open(f"{output_dir}/{model_slug}_shap_summary.png")
                waterfall_img = Image.open(f"{output_dir}/{model_slug}_shap_waterfall.png")
                
                s1, s2 = st.columns(2)
                with s1:
                    st.image(summary_img, caption="Global Feature Importance (Summary Plot)", use_container_width=True)
                with s2:
                    st.image(waterfall_img, caption="Local Prediction (Waterfall Plot)", use_container_width=True)
            except Exception as e:
                st.error(f"Could not load SHAP plots: {e}")
                
        if groq_api_key and Groq is not None:
            st.markdown("---")
            st.subheader("AI Conclusion (Powered by Groq)")
            with st.spinner("Analyzing results with Llama 3..."):
                try:
                    client = Groq(api_key=groq_api_key)
                    leaderboard_str = leaderboard.to_dataframe().head(3).to_string()
                    
                    prompt = f"""
                    You are an expert Data Scientist presenting the results of an automated machine learning pipeline to stakeholders.
                    The task was {report.task_type.upper()} predicting the target variable '{target_col}'.
                    
                    Here are the top 3 models from the leaderboard:
                    {leaderboard_str}
                    
                    The absolute best model found was: {best_model_name}.
                    
                    The top 3 most important features driving the model's predictions overall (from SHAP) are: {', '.join(top_features)}.
                    
                    Please provide a detailed, clear explanation for a layman covering these 3 points:
                    1. **Why the Best Model Won**: Explain why {best_model_name} likely emerged as the best model here compared to the others. (For example, Logistic Regression often wins if the data is highly linear or small; Tree models win on complex/non-linear data).
                    2. **Global Feature Importance (Summary Plot)**: Explain what this plot represents conceptually. Mention the top 3 features and how they generally drive the overall predictions for '{target_col}'.
                    3. **Local Prediction (Waterfall Plot)**: Explain what this plot represents. Tell them that it breaks down exactly how a *single* specific prediction was made, showing how each individual feature pushes the final prediction up or down from the average baseline.
                    
                    Format your answer nicely with bold text and bullet points for readability. Keep it engaging, professional, and avoid dense math jargon.
                    IMPORTANT: Do NOT include any conversational greetings or sign-offs (e.g. "Thank you for joining me today...", "Feel free to ask questions..."). Output ONLY the requested technical explanation.
                    """
                    chat_completion = client.chat.completions.create(
                        messages=[{"role": "user", "content": prompt}],
                        model="llama-3.1-8b-instant",
                    )
                    st.success(chat_completion.choices[0].message.content)
                except Exception as e:
                    st.error(f"Groq API Error: {e}")
                
        # Package the pipeline and model for download
        st.markdown("---")
        st.subheader("Export Ready")
        
        full_pipeline = {"preprocessor": pipeline, "model": tuned_model}
        pipeline_path = "best_pipeline.pkl"
        with open(pipeline_path, "wb") as f:
            pickle.dump(full_pipeline, f)
            
        with open(pipeline_path, "rb") as f:
            st.download_button(
                label="Download Full Pipeline (.pkl)",
                data=f,
                file_name=f"{model_slug}_pipelineiq.pkl",
                mime="application/octet-stream"
            )
