# 🌾 Agri-Intel: The Smart Crop Ecosystem

![Agri-Intel](https://img.shields.io/badge/Status-Active-brightgreen?style=flat-square) ![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=flat-square) ![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-red?style=flat-square)

**Agri-Intel** is a next-generation intelligent agriculture platform designed to empower farmers and agronomists with data-driven decision-making tools. By leveraging machine learning (XGBoost) and advanced data visualization, the system analyzes soil parameters to recommend the most suitable crops and optimal fertilizers, maximizing yield and profitability.

---

## 🚀 Key Features

### 1. 🤖 Intelligent Crop Recommendation
- **AI-Powered Analysis**: Uses an **XGBoost Classifier** to analyze 7 key soil and environmental parameters (N, P, K, Temperature, Humidity, pH, Rainfall).
- **Top-3 Ranking**: Provides not just the best crop, but also the 2nd and 3rd best alternatives with confidence scores.
- **Yield Estimation**: Estimates potential yield (kg/ha) based on historical data for the recommended crop.

### 2. 🧪 Smart Fertilizer Assistant
- **Tailored Recommendations**: Suggests the specific fertilizer type required for the selected crop and soil conditions.
- **Chemical Composition**: Visualizes the N-P-K breakdown of recommended fertilizers using interactive charts.
- **Supply Chain Resilience**: Offers backup fertilizer options in case of market shortages.

### 3. 📊 Visual Analytics Dashboard
- **Soil Health Gauge Charts**: Compare your soil's current nutrient levels against the *ideal* requirements for the target crop.
- **Profitability Modeling**: Interactive "Revenue Calculator" allows users to estimate gross revenue based on varying market prices and farm size.
- **Dynamic User Interface**: Clean, modern UI with a "Forest Green" theme, responsive layout, and intuitive sidebar controls.

---

## 📂 Project Structure

```bash
smart_agriculture/
├── app.py                      # 📱 Main Streamlit Application (Dashboard Entry Point)
├── requirements.txt            # 📦 Python Dependencies
├── README.md                   # 📄 Project Documentation
├── data/                       # 💾 Datasets & Static Data
│   ├── Crop_Yield_Fertilizer.csv   # Primary training dataset
│   ├── crop_profiles.csv           # Generated ideal crop profiles (Metadata)
│   └── UI_design_ideas.txt         # Design reference notes
├── models/                     # 🧠 Trained AI Models (Git-ignored in production)
│   ├── crop_model.pkl              # Trained XGBoost Crop Classifier
│   ├── fertilizer_model.pkl        # Trained XGBoost Fertilizer Classifier
│   ├── crop_encoder.pkl            # Label Encoder for Crops
│   └── fertilizer_encoder.pkl      # Label Encoder for Fertilizers
├── src/                        # 🛠️ Source Code & Utilities
│   ├── training_pipeline.py        # 🚀 Model Training Script (Run this first!)
│   └── utils.py                    # 🧩 Helper functions (Model loading, prediction logic)
└── training/                   # 📓 Research Notebooks (Deprecated/Experimental)
    ├── train_crop_model.ipynb      # Initial Jupyter analysis for crops
    └── train_fertilizer_model.ipynb # Initial Jupyter analysis for fertilizers
```

---

## 🛠️ Installation & Setup

Follow these steps to set up the project locally.

### 1. Clone the Repository
```bash
git clone <repository_url>
cd smart_agriculture
```

### 2. Create a Virtual Environment (Recommended)
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Mac/Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

<<<<<<< HEAD
python src/training_pipeline.py
=======
---
>>>>>>> cdb8ba9 (updated README file)

## ⚙️ How to Run

### Step 1: Train the Models 🧠
Before running the app, you need to generate the machine learning models. The system comes with a training pipeline that processes the data in `data/Crop_Yield_Fertilizer.csv` and saves the artifacts to the `models/` directory.

```bash
# Run this command from the root directory
python src/training_pipeline.py
```
*Expected Output:*
> ✅ Saved crop profiles...
> ✅ Crop Model Saved.
> ✅ Fertilizer Model Saved.
> 🚀 PIPELINE COMPLETE.

### Step 2: Launch the Dashboard 🚀
Once the models are ready, start the Streamlit application.

```bash
streamlit run app.py
<<<<<<< HEAD

=======
```
*The app will open automatically in your default browser at `http://localhost:8501`. If it does not, simply click the link displayed in your terminal.*

---

## 📈 Visualizations Explained

The dashboard includes several advanced charts to help interpret the data:

| Visualization | Description | Purpose |
| :--- | :--- | :--- |
| **Gauge Charts** | Circular dials showing N, P, K, and pH levels. | **Diagnostics**: Shows if your soil nutrients are "Low" (Red), "Optimal" (Green), or "High" (Red) compared to what the specific crop needs. |
| **Donut Chart** | A circular chart broken into segments. | **Analysis**: Displays the precise chemical composition (e.g., 46% Nitrogen for Urea) of the recommended fertilizer. |
| **Area Chart** | A filled line graph showing revenue trends. | **Forecasting**: Demonstrates how your estimated revenue changes as crop market prices fluctuate. |

---

## 💻 Technlogy Stack

*   **Frontend**: Streamlit (Python-based Web Framework)
*   **Machine Learning**: XGBoost (Extreme Gradient Boosting), Scikit-learn
*   **Data Processing**: Pandas, NumPy
*   **Visualization**: Plotly Express, Plotly Graph Objects
*   **Model Persistence**: Joblib

---

## 🤝 Contributing

1.  Fork the repository.
2.  Create a feature branch (`git checkout -b feature/NewFeature`).
3.  Commit your changes.
4.  Push to the branch and open a Pull Request.

---

**Developed for the Smart Agriculture Initiative.** 🌿
>>>>>>> cdb8ba9 (updated README file)
