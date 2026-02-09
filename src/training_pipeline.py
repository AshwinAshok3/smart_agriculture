import pandas as pd
import numpy as np
import joblib
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# --- CONFIGURATION ---
DATA_PATH = "../data/Crop_Yield_Fertilizer.csv"
MODEL_DIR = "../models/"
PROFILE_PATH = "../data/crop_profiles.csv"
RANDOM_STATE = 4345

def train_and_save():
    print("⏳ Loading Data...")
    try:
        df = pd.read_csv(DATA_PATH)
    except FileNotFoundError:
        print(f"❌ Error: {DATA_PATH} not found. Please put your CSV in the 'data' folder.")
        return

    # --- STEP 1: GENERATE IDEAL CROP PROFILES ---
    print("📊 Generating Ideal Soil Profiles...")
    # Calculate average soil conditions for each crop
    crop_profiles = df.groupby('label')[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']].mean()
    crop_profiles.to_csv(PROFILE_PATH)
    print(f"✅ Saved crop profiles to {PROFILE_PATH}")

    # --- STEP 2: TRAIN CROP MODEL ---
    print("🌱 Training Crop Model...")
    X_crop = df[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]
    y_crop = df['label']

    le_crop = LabelEncoder()
    y_crop_enc = le_crop.fit_transform(y_crop)

    # XGBoost with probability support (multi:softprob)
    crop_model = XGBClassifier(
        n_estimators=200, 
        learning_rate=0.05, 
        max_depth=5, 
        objective='multi:softprob', 
        random_state=RANDOM_STATE
    )
    crop_model.fit(X_crop, y_crop_enc)
    
    # Save Crop Artifacts
    joblib.dump(crop_model, f"{MODEL_DIR}crop_model.pkl")
    joblib.dump(le_crop, f"{MODEL_DIR}crop_encoder.pkl")
    print("✅ Crop Model Saved.")

    # --- STEP 3: TRAIN FERTILIZER MODEL ---
    print("🧪 Training Fertilizer Model...")
    # Note: We include 'label' (crop name) as a feature for fertilizer prediction
    X_fert = df[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall', 'label']]
    y_fert = df['fertilizer']

    # We need to encode the 'label' column inside X_fert because XGBoost expects numbers
    # However, for simplicity in this pipeline, we will use the same encoder we just made
    X_fert = X_fert.copy()
    X_fert['label'] = le_crop.transform(X_fert['label'])

    le_fert = LabelEncoder()
    y_fert_enc = le_fert.fit_transform(y_fert)

    fert_model = XGBClassifier(
        n_estimators=200, 
        learning_rate=0.05, 
        max_depth=5, 
        objective='multi:softprob', 
        random_state=RANDOM_STATE
    )
    fert_model.fit(X_fert, y_fert_enc)

    # Save Fertilizer Artifacts
    joblib.dump(fert_model, f"{MODEL_DIR}fertilizer_model.pkl")
    joblib.dump(le_fert, f"{MODEL_DIR}fertilizer_encoder.pkl")
    print("✅ Fertilizer Model Saved.")
    print("🚀 PIPELINE COMPLETE.")

if __name__ == "__main__":
    train_and_save()