import streamlit as st
import pandas as pd
import joblib

# -----------------------------
# Load model + feature order
# -----------------------------
data = joblib.load("ml_project.jbl")

model = data["model"]
features = data["features"]
scaler = data["scaler"]

# -----------------------------
# Page config
# -----------------------------
st.set_page_config(
    page_title="AI-Based Tropical Cyclone Forecasting",
    page_icon="🌪️",
    layout="centered"
)

st.title("🌪️ AI-Based Tropical Cyclone Forecasting")
st.write("Adjust parameters to predict cyclone formation")

# -----------------------------
# Scrollable input UI
# -----------------------------
with st.container(height=420, border=True):

    sea_surface_temperature = st.slider("Sea Surface Temperature (°C)", 20.0, 35.0, 28.0)
    humidity = st.slider("Humidity (%)", 0.0, 100.0, 80.0)
    atmospheric_pressure = st.slider("Atmospheric Pressure (hPa)", 900.0, 1100.0, 1005.0)
    wind_shear = st.slider("Wind Shear", 0.0, 50.0, 15.0)
    latitude = st.slider("Latitude", -90.0, 90.0, 10.0)
    ocean_depth = st.slider("Ocean Depth (m)", 0.0, 6000.0, 2000.0)
    proximity_to_coastline = st.slider("Proximity to Coastline (km)", 0.0, 1000.0, 300.0)

# -----------------------------
# Predict button
# -----------------------------
if st.button("🔍 Predict Cyclone Formation"):

    # Create input dataframe
    input_data = pd.DataFrame([[
        sea_surface_temperature,
        humidity,
        atmospheric_pressure,
        wind_shear,
        latitude,
        ocean_depth,
        proximity_to_coastline
    ]])

    # FORCE correct column order (MOST IMPORTANT FIX)
    input_data.columns = features

    # Prediction
    prediction = model.predict(input_data)[0]

    # Output
    if prediction == 1:
        st.error("🌪️ Cyclone Likely to Form")
    else:
        st.success("☀️ Cyclone Not Likely to Form")

    # Probability (if supported)
    if hasattr(model, "predict_proba"):
        probability = model.predict_proba(input_data)[0][1] * 100
        st.write(f"### 🌡️ Probability: {probability:.2f}%")