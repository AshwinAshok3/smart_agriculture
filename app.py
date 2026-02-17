import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from src.utils import load_artifacts, get_top_predictions
import numpy as np

# ==============================================================================
# 1. PAGE CONFIGURATION & STATE MANAGEMENT
# ==============================================================================
# Sets the browser tab title, favicon, and layout to wide mode for a dashboard feel.
st.set_page_config(
    page_title="Agri-Intel | Eco-System",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# INITIALIZE SESSION STATE
# This ensures the app remembers the analysis results even if the user interacts with sliders.
if 'analyzed' not in st.session_state:
    st.session_state.analyzed = False
if 'results' not in st.session_state:
    st.session_state.results = None

# ==============================================================================
# 2. "MODERN AGRITECH" LIGHT THEME CSS
# ==============================================================================
# Overriding default styles to create a clean, white-and-green professional theme.
st.markdown("""
    <style>
    /* GLOBAL THEME */
    .stApp {
        background-color: #f8f9fa; /* Soft White */
        color: #2c3e50; /* Dark Slate Blue Text */
    }
    
    /* SIDEBAR */
    section[data-testid="stSidebar"] {
        background-color: #ffffff;
        border-right: 1px solid #e9ecef;
    }
    
    /* INPUT LABELS */
    .stSlider label, .stNumberInput label, .stSelectbox label {
        color: #2e7d32 !important; /* Forest Green */
        font-weight: 700;
        font-family: 'Segoe UI', sans-serif;
    }
    
    /* METRIC CARDS */
    div[data-testid="stMetric"] {
        background-color: #ffffff;
        border: 1px solid #e0e0e0;
        padding: 15px;
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
    }
    div[data-testid="stMetricLabel"] {
        color: #66bb6a !important; /* Light Green Label */
    }
    div[data-testid="stMetricValue"] {
        color: #1b5e20 !important; /* Dark Green Value */
    }
    
    /* TABS STYLING */
    .stTabs [data-baseweb="tab-list"] {
        gap: 20px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #ffffff;
        border-radius: 20px;
        color: #4caf50;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        padding: 10px 20px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #e8f5e9 !important;
        color: #1b5e20 !important;
        border: 1px solid #c8e6c9;
    }
    
    /* PROGRESS BAR COLOR */
    .stProgress > div > div > div > div {
        background-color: #4caf50;
    }
    </style>
    """, unsafe_allow_html=True)

# ==============================================================================
# 3. HELPER FUNCTIONS
# ==============================================================================

def create_gauge_chart(value, title, min_val, max_val, target_val):
    """
    Creates a visual Gauge Chart for soil nutrients.
    Shows the user's value relative to the 'Ideal' value for the crop.
    """
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = value,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': title, 'font': {'size': 16, 'color': "#2c3e50"}},
        gauge = {
            'axis': {'range': [min_val, max_val], 'tickwidth': 1, 'tickcolor': "#b0bec5"},
            'bar': {'color': "#2e7d32"}, # Forest Green Needle
            'bgcolor': "white",
            'borderwidth': 1,
            'bordercolor': "#e0e0e0",
            'steps': [
                {'range': [0, target_val*0.8], 'color': "#ffebee"}, # Low (Redish)
                {'range': [target_val*0.8, target_val*1.2], 'color': "#e8f5e9"}, # Optimal (Greenish)
                {'range': [target_val*1.2, max_val], 'color': "#ffebee"} # High (Redish)
            ],
            'threshold': {
                'line': {'color': "#c62828", 'width': 2},
                'thickness': 0.75,
                'value': target_val
            }
        }
    ))
    fig.update_layout(height=180, margin=dict(l=10,r=10,t=40,b=10), paper_bgcolor="rgba(0,0,0,0)", font={'color': "#2c3e50"})
    return fig

def get_fertilizer_composition(fert_name):
    """
    Returns chemical composition data for the Donut Chart.
    """
    compositions = {
        "Urea": {"Nitrogen": 46, "Phosphorus": 0, "Potassium": 0, "Filler": 54},
        "DAP": {"Nitrogen": 18, "Phosphorus": 46, "Potassium": 0, "Filler": 36},
        "MOP": {"Nitrogen": 0, "Phosphorus": 0, "Potassium": 60, "Filler": 40},
        "SSP": {"Nitrogen": 0, "Phosphorus": 16, "Potassium": 0, "Sulphur": 12, "Filler": 72},
        "Generic": {"Nitrogen": 20, "Phosphorus": 20, "Potassium": 20, "Filler": 40}
    }
    return compositions.get(fert_name, compositions["Generic"])

def get_heatmap(z, x, y, colorscale='Viridis'):
    """
    Creates a heatmap for sensitivity analysis.
    """
    fig = go.Heatmap(
        z=z,
        x=x,
        y=y,
        colorscale=colorscale,
        colorbar=dict(title="Revenue (₹)"),
        hovertemplate="Price: ₹%{x}<br>Yield: %{y} kg/ha<br>Revenue: ₹%{z:,.0f}<extra></extra>"
    )
    return fig

# ==============================================================================
# 4. LOAD ASSETS
# ==============================================================================
artifacts = load_artifacts()

if artifacts:
    # ==========================================================================
    # SIDEBAR: INPUTS
    # ==========================================================================
    with st.sidebar:
        st.image("https://cdn-icons-png.flaticon.com/512/628/628283.png", width=80)
        st.markdown("### 🌿 FIELD PARAMETERS")
        st.info("Input your soil test report data below.")
        st.markdown("---")
        
        # SLIDERS
        N = st.slider("Nitrogen Content (N)", 0, 140, 90)
        P = st.slider("Phosphorous Content (P)", 0, 145, 40)
        K = st.slider("Potassium Content (K)", 0, 205, 40)
        ph = st.number_input("Soil Acidity (pH)", 0.0, 14.0, 6.5, step=0.1)
        
        st.markdown("---")
        col_s1, col_s2 = st.columns(2)
        with col_s1:
            temp = st.number_input("Temp (°C)", -10.0, 60.0, 25.0)
            rainfall = st.number_input("Rainfall (mm)", 0.0, 500.0, 200.0)
        with col_s2:
            humid = st.number_input("Humidity (%)", 0.0, 100.0, 80.0)
            
        st.markdown("<br>", unsafe_allow_html=True)
        
        # ACTION BUTTON
        if st.button("🚀 GENERATE INSIGHTS", type="primary", use_container_width=True):
            st.session_state.analyzed = True
            
            # PREPARE DATA
            input_df = pd.DataFrame({'N': [N], 'P': [P], 'K': [K], 'temperature': [temp], 
                                     'humidity': [humid], 'ph': [ph], 'rainfall': [rainfall]})
            
            # GET TOP 3 CROP PREDICTIONS
            top_crops = get_top_predictions(artifacts['crop_model'], input_df, artifacts['crop_enc'], top_k=3)
            best_crop = top_crops[0][0]
            confidence = top_crops[0][1]

            # GET FERTILIZER PREDICTION (Based on Best Crop)
            fert_input = input_df.copy()
            encoded_crop_label = artifacts['crop_enc'].transform([best_crop])[0]
            fert_input['label'] = encoded_crop_label
            top_ferts = get_top_predictions(artifacts['fert_model'], fert_input, artifacts['fert_enc'], top_k=3)
            
            # SAVE TO SESSION STATE
            st.session_state.results = {
                'top_crops': top_crops,
                'top_ferts': top_ferts,
                'input_df': input_df
            }

    # ==========================================================================
    # MAIN DASHBOARD
    # ==========================================================================
    st.markdown("# 🌾 AGRI-INTEL")
    st.markdown("##### INTELLIGENT CROP & SOIL MANAGEMENT SYSTEM")
    st.divider()

    if st.session_state.analyzed and st.session_state.results:
        results = st.session_state.results
        # Extract top crops list: [('Rice', 0.8), ('Maize', 0.1), ('Jute', 0.05)]
        top_crops = results['top_crops'] 
        best_crop = top_crops[0][0]
        best_fert = results['top_ferts'][0][0]
        confidence = top_crops[0][1]
        
        try:
            ideal_profile = artifacts['profiles'].loc[best_crop]
        except KeyError:
            ideal_profile = None

        # --- HERO METRICS (Top 1 Crop Info) ---
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("BEST SUITED CROP", best_crop.upper())
        with col2:
            st.metric("AI CONFIDENCE", f"{confidence*100:.1f}%")
        with col3:
            st.metric("RECOMMENDED FERTILIZER", best_fert)
        with col4:
            est_yield = ideal_profile.get('yield', 5000) if ideal_profile is not None else 5000
            st.metric("YIELD ESTIMATE", f"{est_yield:,.0f} kg/ha")

        # --- [NEW SECTION] ALTERNATIVE CROP RANKING ---
        # This section explicitly lists the 2nd and 3rd best options as requested.
        st.markdown("<br>", unsafe_allow_html=True)
        st.subheader("📋 Alternative Crop Ranking")
        st.markdown("Other viable crops ranked by suitability for your current soil conditions.")
        
        # Create columns for the alternative crops
        alt_col1, alt_col2 = st.columns(2)
        
        # Display 2nd Best Crop
        if len(top_crops) > 1:
            crop_name_2 = top_crops[1][0]
            prob_2 = float(top_crops[1][1]) # Cast to float to avoid Streamlit error
            with alt_col1:
                st.info(f"🥈 **Rank 2: {crop_name_2}**")
                st.progress(prob_2, text=f"Suitability Score: {prob_2*100:.1f}%")
                
        # Display 3rd Best Crop
        if len(top_crops) > 2:
            crop_name_3 = top_crops[2][0]
            prob_3 = float(top_crops[2][1]) # Cast to float to avoid Streamlit error
            with alt_col2:
                st.info(f"🥉 **Rank 3: {crop_name_3}**")
                st.progress(prob_3, text=f"Suitability Score: {prob_3*100:.1f}%")

        st.markdown("<br>", unsafe_allow_html=True)

        # --- TABS (Detailed Analysis) ---
        tab_soil, tab_market, tab_fert = st.tabs(["🧪 SOIL HEALTH", "💰 PROFITABILITY", "🔬 NUTRIENT MATRIX"])

        # TAB 1: SOIL HEALTH
        with tab_soil:
            st.subheader("Field Diagnostics")
            st.markdown(f"Comparing soil vitals against optimal requirements for **{best_crop}**.")
            g1, g2, g3, g4 = st.columns(4)
            
            ideal_N = ideal_profile['N'] if ideal_profile is not None else 90
            ideal_P = ideal_profile['P'] if ideal_profile is not None else 40
            ideal_K = ideal_profile['K'] if ideal_profile is not None else 40
            
            with g1: st.plotly_chart(create_gauge_chart(N, "Nitrogen (N)", 0, 150, ideal_N), use_container_width=True)
            with g2: st.plotly_chart(create_gauge_chart(P, "Phosphorous (P)", 0, 150, ideal_P), use_container_width=True)
            with g3: st.plotly_chart(create_gauge_chart(K, "Potassium (K)", 0, 210, ideal_K), use_container_width=True)
            with g4: st.plotly_chart(create_gauge_chart(ph, "Soil pH", 0, 14, 6.5), use_container_width=True)

        # TAB 2: FINANCIAL MODEL
        with tab_market:
            st.subheader("💰 Profitability & Risk Analysis")
            st.markdown("Analyze how **Market Price** and **Yield Variation** affect your total revenue.")
            
            col_m1, col_m2 = st.columns([1, 2])
            with col_m1:
                st.info("Adjust the base parameters to simulate different scenarios.")
                
                # Interactive sliders
                base_price = st.slider(f"Base Price for {best_crop} (₹/kg)", 10, 200, 45)
                base_yield = est_yield # From crop profile
                farm_area = st.number_input("Farm Area (Acres)", 1.0, 100.0, 5.0)
                
                # Conversion: 1 Acre = 0.4047 Hectares
                area_in_ha = farm_area * 0.4047
                
                # Current Scenario Calculation
                current_revenue = base_yield * area_in_ha * base_price
                st.metric("ESTIMATED GROSS REVENUE", f"₹ {current_revenue:,.0f}", delta=None)
                
            with col_m2:
                # --- SENSITIVITY ANALYSIS HEATMAP ---
                # Generate ranges for Price (+/- 20%) and Yield (+/- 20%)
                price_range = np.linspace(base_price * 0.8, base_price * 1.2, 10)
                yield_range = np.linspace(base_yield * 0.8, base_yield * 1.2, 10)
                
                # Create Grid
                z_data = [] # Revenue matrix
                for y_val in yield_range:
                    row = []
                    for p_val in price_range:
                        rev = y_val * area_in_ha * p_val
                        row.append(rev)
                    z_data.append(row)
                
                # Create Heatmap
                fig_heat = go.Figure(data=get_heatmap(
                    z=z_data,
                    x=np.round(price_range, 1),
                    y=np.round(yield_range, 0),
                    colorscale='Greens'
                ))
                
                fig_heat.update_layout(
                    title="Revenue Sensitivity (Yield vs Price)",
                    xaxis_title="Market Price (₹/kg)",
                    yaxis_title="Yield (kg/ha)",
                    height=400
                )
                st.plotly_chart(fig_heat, use_container_width=True)

        # TAB 3: NUTRIENT MATRIX
        with tab_fert:
            st.subheader("🔬 Fertilizer Chemical Composition")
            st.markdown("Compare the nutrient profile of recommended fertilizers.")

            col_f1, col_f2 = st.columns([1, 2])
            
            with col_f1:
                st.info("Select a fertilizer to view its composition.")
                
                # Create a clearer list for the selectbox
                fert_options = [f[0] for f in results['top_ferts']]
                selected_fert_name = st.radio("Select Fertilizer:", fert_options)
                
                # Get probability for the selected one
                sel_prob = next(item[1] for item in results['top_ferts'] if item[0] == selected_fert_name)
                st.metric("Recommendation Score", f"{float(sel_prob)*100:.1f}%")

            with col_f2:
                # --- RADAR CHART for Composition ---
                comp_data = get_fertilizer_composition(selected_fert_name)
                
                # Prepare data for Radar Chart
                categories = list(comp_data.keys())
                values = list(comp_data.values())
                
                fig_radar = go.Figure()

                fig_radar.add_trace(go.Scatterpolar(
                    r=values,
                    theta=categories,
                    fill='toself',
                    name=selected_fert_name,
                    line_color='#2e7d32'
                ))

                fig_radar.update_layout(
                    polar=dict(
                        radialaxis=dict(
                            visible=True,
                            range=[0, 100]
                        )),
                    showlegend=False,
                    title=f"Nutrient Profile: {selected_fert_name}"
                )
                
                st.plotly_chart(fig_radar, use_container_width=True)

    else:
        # IDLE STATE
        st.markdown("""
        <div style='text-align: center; padding: 50px; opacity: 0.6;'>
            <h2 style='color: #2e7d32;'>READY TO ANALYZE</h2>
            <p>Please enter your field data in the sidebar to begin.</p>
        </div>
        """, unsafe_allow_html=True)
        
else:
    st.error("System Error: Models not found. Please run 'src/training_pipeline.py'.")