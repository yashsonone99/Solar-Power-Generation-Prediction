import streamlit as st
import numpy as np
import joblib
import pandas as pd
import altair as alt
from io import BytesIO
import base64
import os

# ==================================================
# PAGE CONFIG
# ==================================================
st.set_page_config(
    page_title="Solar Power Generation Prediction",
    layout="wide"
)

# ==================================================
# BASE DIRECTORY
# ==================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BACKGROUND_PATH = os.path.join(BASE_DIR, "background.png")
MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")

# ==================================================
# BACKGROUND
# ==================================================
def set_bg(path):
    if not os.path.exists(path):
        return

    with open(path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode()

    st.markdown(
        f"""
        <style>
        .stApp {{
            background: url("data:image/png;base64,{encoded}") no-repeat center center fixed;
            background-size: cover;
        }}
        .stApp::before {{
            content: "";
            position: fixed;
            inset: 0;
            background: rgba(0,0,0,0.7);
            z-index: -1;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

set_bg(BACKGROUND_PATH)

# ==================================================
# LOAD MODEL
# ==================================================
if not os.path.exists(MODEL_PATH):
    st.error("model.pkl not found")
    st.stop()

model = joblib.load(MODEL_PATH)

# ==================================================
# TITLE
# ==================================================
st.title("☀️ Solar Power Generation Prediction")
st.markdown("---")

left, right = st.columns(2, gap="large")

# ==================================================
# LEFT – INPUTS
# ==================================================
with left:
    st.subheader("Input Environmental Conditions")

    distance = st.slider("Distance to Solar Noon", 0.0, 1.5, 0.25)
    temperature = st.slider("Temperature (°C)", 0, 50, 25)
    wind_direction = st.slider("Wind Direction (degrees)", 0, 360, 90)
    wind_speed = st.slider("Wind Speed (m/s)", 0.0, 20.0, 5.0)

    sky_cover = st.radio("Sky Cover", ["Clear", "Partly Cloudy", "Cloudy"], horizontal=True)

    visibility = st.slider("Visibility (km)", 0.0, 20.0, 10.0)
    humidity = st.slider("Humidity (%)", 0, 100, 50)
    avg_wind_speed = st.slider("Average Wind Speed (period)", 0.0, 50.0, 30.0)
    avg_pressure = st.slider("Average Pressure (period)", 950, 1050, 1013)

    # ---------------- FEATURE IMPORTANCE ----------------
    st.subheader("Feature Importance")

    fi_df = pd.DataFrame({
        "Feature": [
            "Solar Noon Distance", "Temperature", "Wind Direction",
            "Wind Speed", "Sky Cover", "Visibility",
            "Humidity", "Avg Wind Speed", "Avg Pressure"
        ],
        "Importance": model.feature_importances_
    }).sort_values(by="Importance", ascending=False)

    fi_chart = alt.Chart(fi_df).mark_bar(size=30).encode(
        x=alt.X("Feature:N", title=None,
                axis=alt.Axis(labelColor="white")),
        y=alt.Y("Importance:Q", title="Importance",
                axis=alt.Axis(labelColor="white")),
        color=alt.value("#4ade80")
    ).properties(
        height=320,
        background="#0f172a"
    )

    st.altair_chart(fi_chart, use_container_width=True)

# ==================================================
# RIGHT – OUTPUT
# ==================================================
with right:
    st.subheader("Predicted Solar Power Generation")

    if st.button("Predict Power Generated"):
        sky_map = {"Clear": 0, "Partly Cloudy": 2, "Cloudy": 4}

        X = np.array([[
            distance,
            temperature,
            wind_direction,
            wind_speed,
            sky_map[sky_cover],
            visibility,
            humidity,
            avg_wind_speed,
            avg_pressure
        ]])

        prediction = model.predict(X)[0]

        st.success(f"Predicted Power: {prediction:.2f} Joules")

        # ---------------- VISUALIZATION INSIGHTS ----------------
        st.subheader("Visualization Insights")

        viz_df = pd.DataFrame({
            "Feature": [
                "Humidity (%)",
                "Solar Noon Distance",
                "Temperature (°C)",
                "Visibility (km)",
                "Wind Speed (m/s)"
            ],
            "Value": [
                humidity,
                distance,
                temperature,
                visibility,
                wind_speed
            ]
        })

        viz_chart = alt.Chart(viz_df).mark_bar(size=40).encode(
            x=alt.X("Feature:N",
                    axis=alt.Axis(labelAngle=0, labelColor="white")),
            y=alt.Y("Value:Q",
                    axis=alt.Axis(labelColor="white")),
            color=alt.Color(
                "Feature:N",
                scale=alt.Scale(scheme="set2"),
                legend=None
            ),
            tooltip=["Feature", "Value"]
        ).properties(
            height=360,
            background="#020617"
        )

        labels = viz_chart.mark_text(
            dy=-8,
            color="white",
            fontSize=12
        ).encode(
            text=alt.Text("Value:Q", format=".2f")
        )

        st.altair_chart(viz_chart + labels, use_container_width=True)

        # ---------------- DOWNLOAD ----------------
        buffer = BytesIO()
        viz_df.to_csv(buffer, index=False)
        buffer.seek(0)

        st.download_button(
            "Download Prediction Report",
            buffer,
            "solar_power_prediction_report.csv",
            "text/csv"
        )
