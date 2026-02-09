import joblib
import pandas as pd
import numpy as np
import streamlit as st

@st.cache_resource
def load_artifacts():
    """Loads all models and encoders efficiently."""
    try:
        models = {
            "crop_model": joblib.load("models/crop_model.pkl"),
            "fert_model": joblib.load("models/fertilizer_model.pkl"),
            "crop_enc": joblib.load("models/crop_encoder.pkl"),
            "fert_enc": joblib.load("models/fertilizer_encoder.pkl"),
            "profiles": pd.read_csv("data/crop_profiles.csv", index_col='label')
        }
        return models
    except FileNotFoundError as e:
        st.error(f"Artifact missing: {e}. Did you run 'src/training_pipeline.py'?")
        return None

def get_top_predictions(model, input_data, encoder, top_k=3):
    """
    Returns the top K classes and their probabilities.
    Args:
        model: Trained XGBoost model
        input_data: DataFrame row
        encoder: LabelEncoder
        top_k: Number of predictions to return
    Returns:
        List of tuples: [('Rice', 0.85), ('Maize', 0.10), ...]
    """
    # Get probabilities for all classes
    probs = model.predict_proba(input_data)[0]
    
    # Get indices of the top K probabilities (sorted descending)
    top_indices = probs.argsort()[-top_k:][::-1]
    
    # Decode class names and match with probabilities
    classes = encoder.inverse_transform(top_indices)
    scores = probs[top_indices]
    
    return list(zip(classes, scores))