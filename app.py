import streamlit as st
import pandas as pd
import joblib

# Load models
crop_model = joblib.load("models/crop_model.pkl")
fertilizer_model = joblib.load("models/fertilizer_model.pkl")

crop_encoder = joblib.load("models/crop_label_encoder.pkl")
fertilizer_encoder = joblib.load("models/fertilizer_label_encoder.pkl")

st.set_page_config(
    page_title="Smart Agriculture System",
    page_icon="🌱",
    layout="centered"
)

st.title("🌾 Smart Agriculture Decision Support System")

st.markdown("""
This system:
1. **Predicts the best crop** based on soil & climate  
2. **Recommends the most suitable fertilizer** for that crop  
""")

st.header("📥 Enter Soil & Climate Details")

N = st.number_input("Nitrogen (N)", 0, 200, 90)
P = st.number_input("Phosphorous (P)", 0, 200, 40)
K = st.number_input("Potassium (K)", 0, 200, 40)
temperature = st.number_input("Temperature (°C)", 0.0, 50.0, 25.0)
humidity = st.number_input("Humidity (%)", 0.0, 100.0, 80.0)
ph = st.number_input("Soil pH", 0.0, 14.0, 6.5)
rainfall = st.number_input("Rainfall (mm)", 0.0, 500.0, 200.0)

if st.button("🚀 Predict Crop & Fertilizer"):
    input_data = pd.DataFrame([{
        'N': N,
        'P': P,
        'K': K,
        'temperature': temperature,
        'humidity': humidity,
        'ph': ph,
        'rainfall': rainfall
    }])

    # Step 1: Crop prediction
    crop_pred_encoded = crop_model.predict(input_data)[0]
    crop_pred = crop_encoder.inverse_transform([crop_pred_encoded])[0]

    st.success(f"🌱 Recommended Crop: **{crop_pred.upper()}**")

    # Step 2: Fertilizer prediction
    fertilizer_input = input_data.copy()
    fertilizer_input['label'] = crop_pred

    fert_pred_encoded = fertilizer_model.predict(fertilizer_input)[0]
    fert_pred = fertilizer_encoder.inverse_transform([fert_pred_encoded])[0]

    st.success(f"🧪 Recommended Fertilizer: **{fert_pred}**")

    st.info("⚠️ Recommendation is advisory. Validate with local agronomist.")
