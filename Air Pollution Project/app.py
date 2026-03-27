import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# ── Page config ────────────────────────────────────────────
st.set_page_config(
    page_title="Air Quality Predictor",
    page_icon="🌿",
    layout="centered",
)

# ── Custom CSS ─────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #f8fafb; }
    .stApp { max-width: 800px; margin: auto; }
    .result-box {
        background: linear-gradient(135deg, #e8f5e9, #f1f8e9);
        border-left: 5px solid #43a047;
        padding: 20px 24px;
        border-radius: 12px;
        margin-top: 20px;
    }
    .aqi-number {
        font-size: 3rem;
        font-weight: 800;
        color: #2e7d32;
    }
    .badge {
        display: inline-block;
        padding: 4px 14px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.9rem;
        margin-top: 6px;
    }
</style>
""", unsafe_allow_html=True)

# ── Load & train model (cached so it runs only once) ───────
@st.cache_resource
def load_model():
    df = pd.DataFrame({
        "city":             ["Ahmedabad","Ahmedabad","Ahmedabad",
                             "Surat","Surat","Surat",
                             "Vadodara","Vadodara","Vadodara",
                             "Rajkot","Rajkot","Rajkot"],
        "temperature":      [38,40,35, 33,36,34, 37,39,34, 32,35,33],
        "pollution":        [180,210,150, 130,160,145, 175,200,140, 120,155,135],
        "last_temperature": [36,38,33, 31,33,32, 35,37,32, 30,32,31],
        "last_pollution":   [170,180,140, 120,130,138, 165,175,130, 110,120,125],
    })
    X = df[["temperature", "last_temperature", "last_pollution"]]
    y = df["pollution"]
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    r2 = r2_score(y_test, model.predict(X_test))
    return model, scaler, df, r2

model, scaler, df, r2 = load_model()

# ── AQI helpers ────────────────────────────────────────────
def aqi_label(aqi):
    if aqi < 50:   return "Good",          "#43a047", "#e8f5e9"
    if aqi < 100:  return "Satisfactory",  "#7cb342", "#f1f8e9"
    if aqi < 200:  return "Moderate",      "#f9a825", "#fffde7"
    if aqi < 300:  return "Poor",          "#ef6c00", "#fff3e0"
    return             "Very Poor",        "#c62828", "#ffebee"

def predict(temp, last_temp, last_poll):
    data   = pd.DataFrame({"temperature":[temp],
                            "last_temperature":[last_temp],
                            "last_pollution":[last_poll]})
    scaled = scaler.transform(data)
    return round(model.predict(scaled)[0], 1)

# ── Header ─────────────────────────────────────────────────
st.title("🌿 Air Quality Predictor")
st.caption("Gujarat Cities · Linear Regression Model")
st.divider()

# ── Input form ─────────────────────────────────────────────
st.subheader("Enter Details")

col1, col2 = st.columns(2)
with col1:
    city = st.selectbox("🏙️ City",
        ["Ahmedabad", "Surat", "Vadodara", "Rajkot", "Other"])
    temperature = st.slider("🌡️ Today's Temperature (°C)", 25, 50, 35)

with col2:
    last_temp = st.slider("🌡️ Yesterday's Temperature (°C)", 25, 50, 33)
    last_poll = st.slider("💨 Yesterday's AQI", 50, 350, 140)

st.divider()

# ── Predict button ─────────────────────────────────────────
if st.button("🔍 Predict Air Quality", use_container_width=True, type="primary"):
    result            = predict(temperature, last_temp, last_poll)
    label, color, bg  = aqi_label(result)

    st.markdown(f"""
    <div class="result-box" style="background:{bg}; border-left-color:{color}">
        <p style="margin:0; font-size:1rem; color:#555;">Predicted AQI for <strong>{city}</strong></p>
        <div class="aqi-number" style="color:{color}">{result}</div>
        <span class="badge" style="background:{color}; color:white">{label}</span>
        <p style="margin-top:12px; font-size:0.85rem; color:#666;">
            Based on today's temp <b>{temperature}°C</b>,
            yesterday's temp <b>{last_temp}°C</b>,
            yesterday's AQI <b>{last_poll}</b>
        </p>
    </div>
    """, unsafe_allow_html=True)

    # ── Mini charts ────────────────────────────────────────
    st.markdown("### 📊 Visual Insights")

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
    fig.patch.set_facecolor("#f8fafb")

    # Chart 1 – City AQI comparison
    cities     = df.groupby("city")["pollution"].mean().sort_values()
    bar_colors = ["#ef9a9a" if c == city else "#90caf9" for c in cities.index]
    axes[0].barh(cities.index, cities.values, color=bar_colors, height=0.5)
    axes[0].axvline(100, color="red", linestyle="--", linewidth=1, alpha=0.6, label="Safe limit")
    axes[0].set_title("Avg AQI by City", fontweight="bold")
    axes[0].set_xlabel("AQI")
    axes[0].legend(fontsize=8)
    for i, v in enumerate(cities.values):
        axes[0].text(v + 1, i, str(round(v, 1)), va="center", fontsize=8)

    # Chart 2 – Temperature vs Pollution trend
    axes[1].scatter(df["temperature"], df["pollution"],
                    color="#90caf9", edgecolors="white", s=60, zorder=3)
    axes[1].scatter([temperature], [result],
                    color=color, s=120, zorder=4, label="Your input")
    z = np.polyfit(df["temperature"], df["pollution"], 1)
    x_line = np.linspace(df["temperature"].min(), df["temperature"].max(), 100)
    axes[1].plot(x_line, np.poly1d(z)(x_line), "--", color="#9c27b0", linewidth=1.5)
    axes[1].set_title("Temp vs Pollution", fontweight="bold")
    axes[1].set_xlabel("Temperature (°C)")
    axes[1].set_ylabel("AQI")
    axes[1].legend(fontsize=8)

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # ── Model accuracy badge ───────────────────────────────
    st.markdown(f"""
    <p style="text-align:center; color:#888; font-size:0.8rem; margin-top:8px;">
        Model R² accuracy: <strong>{round(r2*100,1)}%</strong>
    </p>
    """, unsafe_allow_html=True)

# ── AQI reference table ────────────────────────────────────
with st.expander("📖 AQI Reference Guide"):
    ref = pd.DataFrame({
        "AQI Range":  ["0–50", "51–100", "101–200", "201–300", "300+"],
        "Category":   ["Good", "Satisfactory", "Moderate", "Poor", "Very Poor"],
        "Health Tip": [
            "Air is clean. Enjoy outdoor activities.",
            "Acceptable quality. Sensitive groups take care.",
            "May cause discomfort. Limit prolonged exposure.",
            "Unhealthy for most people. Avoid outdoor activity.",
            "Hazardous. Stay indoors and wear a mask.",
        ]
    })
    st.dataframe(ref, hide_index=True, use_container_width=True)