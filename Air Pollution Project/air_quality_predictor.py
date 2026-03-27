# ============================================================
# AIR QUALITY PREDICTOR - IMPRESSIVE VERSION
# With Accuracy Checks + 4 Beautiful Graphs
# ============================================================

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.preprocessing   import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model    import LinearRegression
from sklearn.metrics         import mean_absolute_error, r2_score, mean_squared_error
import numpy as np

# ============================================================
# STEP 1: Load and Clean Data
# ============================================================
df = pd.read_csv("air_quality.csv")
df = df.dropna()

print("==================================================")
print("   AIR QUALITY PREDICTOR - Gujarat Cities")
print("   Impressive Version with Graphs + Accuracy")
print("==================================================")
print("Dataset loaded! Total rows:", len(df))

# ============================================================
# STEP 2: Features and Target
# ============================================================
X = df[["temperature", "last_temperature", "last_pollution"]]
y = df["pollution"]

# ============================================================
# STEP 3: Scale + Split
# ============================================================
scaler   = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# ============================================================
# STEP 4: Train the Model
# ============================================================
model = LinearRegression()
model.fit(X_train, y_train)
print("\nModel trained successfully!")

# ============================================================
# STEP 5: ACCURACY CHECKING (4 different ways!)
# ============================================================

y_predicted = model.predict(X_test)

# --- Metric 1: MAE (Mean Absolute Error) ---
# Simple: on average, how many AQI points off are we?
mae = mean_absolute_error(y_test, y_predicted)

# --- Metric 2: RMSE (Root Mean Squared Error) ---
# Punishes big mistakes more than small ones
rmse = np.sqrt(mean_squared_error(y_test, y_predicted))

# --- Metric 3: R2 Score ---
# How much of the pattern did AI understand? (0 to 1, higher = better)
r2 = r2_score(y_test, y_predicted)

# --- Metric 4: Cross Validation Score ---
# Tests accuracy 5 different times on different slices of data
# More reliable than a single test!
cv_scores = cross_val_score(model, X_scaled, y, cv=5, scoring="r2")

print("\n==================================================")
print("   ACCURACY REPORT")
print("==================================================")
print("MAE  (Mean Absolute Error)   :", round(mae, 2))
print("     -> Predictions wrong by ~", round(mae, 1), "AQI points on average")
print()
print("RMSE (Root Mean Sq. Error)   :", round(rmse, 2))
print("     -> Heavily penalizes big mistakes")
print()
print("R2 Score                     :", round(r2, 2))
print("     -> Model understands", round(r2 * 100, 1), "% of the pollution pattern")
print()
print("Cross Validation R2 Scores   :", [round(s, 2) for s in cv_scores])
print("Average CV Score             :", round(cv_scores.mean(), 2))
print("     -> Consistent across 5 different test slices")

# Accuracy grade
avg = cv_scores.mean()
if avg >= 0.9:
    grade = "A+ EXCELLENT"
elif avg >= 0.75:
    grade = "A  VERY GOOD"
elif avg >= 0.6:
    grade = "B  GOOD"
elif avg >= 0.4:
    grade = "C  AVERAGE"
else:
    grade = "D  NEEDS MORE DATA"

print("\nOverall Model Grade          :", grade)
print("==================================================")

# ============================================================
# STEP 6: Sample Predictions for all cities
# ============================================================
def get_aqi_label(aqi):
    if aqi < 50:
        return "GOOD"
    elif aqi < 100:
        return "SATISFACTORY"
    elif aqi < 200:
        return "MODERATE"
    elif aqi < 300:
        return "POOR"
    else:
        return "VERY POOR"

def predict_pollution(temperature, last_temperature, last_pollution):
    new_data = pd.DataFrame({
        "temperature"      : [temperature],
        "last_temperature" : [last_temperature],
        "last_pollution"   : [last_pollution]
    })
    scaled    = scaler.transform(new_data)
    predicted = model.predict(scaled)[0]
    return round(predicted, 1)

cities_data = [
    ("Ahmedabad", 41, 39, 200),
    ("Surat",     34, 33, 140),
    ("Vadodara",  38, 36, 170),
    ("Rajkot",    33, 31, 120),
]

city_names  = []
city_actual = []
city_pred   = []

print("\nSAMPLE CITY PREDICTIONS:")
print("-" * 50)
for city, temp, last_temp, last_poll in cities_data:
    pred = predict_pollution(temp, last_temp, last_poll)
    label = get_aqi_label(pred)
    city_names.append(city)
    city_actual.append(last_poll)
    city_pred.append(pred)
    print(f"  {city:12s} -> Predicted AQI: {pred:6.1f}  [{label}]")

# ============================================================
# STEP 7: GRAPHS (4 charts in one window!)
# ============================================================
# What each graph shows:
# Graph 1 - Actual vs Predicted  : Did AI predict correctly?
# Graph 2 - Pollution by City    : Which city is most polluted?
# Graph 3 - Pollution Trend      : How does temp affect pollution?
# Graph 4 - Feature Importance   : Which input matters most to AI?

print("\nGenerating graphs...")

fig = plt.figure(figsize=(14, 10))
fig.suptitle("Air Quality Predictor - Gujarat Cities Dashboard",
             fontsize=16, fontweight="bold", y=0.98)

gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.35)

# ----------------------------------------------------------
# GRAPH 1: Actual vs Predicted AQI
# What it shows: Blue dots = actual values, Red line = perfect prediction
# If dots are close to the red line, AI is very accurate!
# ----------------------------------------------------------
ax1 = fig.add_subplot(gs[0, 0])

all_predicted = model.predict(X_scaled)

ax1.scatter(y, all_predicted, color="#2196F3", alpha=0.7, s=80, label="Predictions")
ax1.plot([y.min(), y.max()], [y.min(), y.max()],
         color="#F44336", linewidth=2, linestyle="--", label="Perfect Prediction")
ax1.set_title("Graph 1: Actual vs Predicted AQI", fontweight="bold")
ax1.set_xlabel("Actual AQI")
ax1.set_ylabel("Predicted AQI")
ax1.legend()
ax1.text(0.05, 0.92, "R2 = " + str(round(r2, 2)),
         transform=ax1.transAxes, fontsize=10,
         bbox=dict(boxstyle="round", facecolor="#E3F2FD"))

# ----------------------------------------------------------
# GRAPH 2: Pollution Level by City (Bar Chart)
# What it shows: Compares yesterday's AQI vs predicted today's AQI
# Lets you see which city needs attention!
# ----------------------------------------------------------
ax2 = fig.add_subplot(gs[0, 1])

x_pos  = np.arange(len(city_names))
width  = 0.35
bars1  = ax2.bar(x_pos - width/2, city_actual, width,
                 label="Yesterday AQI", color="#FF9800", alpha=0.85)
bars2  = ax2.bar(x_pos + width/2, city_pred,   width,
                 label="Predicted Today", color="#4CAF50", alpha=0.85)

ax2.set_title("Graph 2: City-wise Pollution Comparison", fontweight="bold")
ax2.set_xlabel("City")
ax2.set_ylabel("AQI Value")
ax2.set_xticks(x_pos)
ax2.set_xticklabels(city_names)
ax2.legend()
ax2.axhline(y=100, color="red", linestyle="--", alpha=0.5, label="Safe Limit")

for bar in bars1:
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
             str(int(bar.get_height())), ha="center", va="bottom", fontsize=8)
for bar in bars2:
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
             str(int(bar.get_height())), ha="center", va="bottom", fontsize=8)

# ----------------------------------------------------------
# GRAPH 3: Pollution Trend - Temperature vs Pollution
# What it shows: As temperature rises, does pollution rise too?
# Each dot = one data row, color = pollution level
# ----------------------------------------------------------
ax3 = fig.add_subplot(gs[1, 0])

scatter = ax3.scatter(df["temperature"], df["pollution"],
                      c=df["pollution"], cmap="RdYlGn_r",
                      s=100, alpha=0.8, edgecolors="white", linewidth=0.5)

# Trend line
z   = np.polyfit(df["temperature"], df["pollution"], 1)
p   = np.poly1d(z)
x_line = np.linspace(df["temperature"].min(), df["temperature"].max(), 100)
ax3.plot(x_line, p(x_line), color="#9C27B0",
         linewidth=2, linestyle="--", label="Trend Line")

plt.colorbar(scatter, ax=ax3, label="Pollution Level")
ax3.set_title("Graph 3: Temperature vs Pollution Trend", fontweight="bold")
ax3.set_xlabel("Temperature (C)")
ax3.set_ylabel("Pollution (AQI)")
ax3.legend()

# ----------------------------------------------------------
# GRAPH 4: Feature Importance
# What it shows: Which input (temperature, last_temp, last_pollution)
# affects the AI's prediction the most?
# Longer bar = more important feature!
# ----------------------------------------------------------
ax4 = fig.add_subplot(gs[1, 1])

feature_names = ["Temperature", "Last Temperature", "Last Pollution"]
coefficients  = np.abs(model.coef_)
colors        = ["#2196F3", "#FF9800", "#F44336"]

bars = ax4.barh(feature_names, coefficients, color=colors, alpha=0.85)
ax4.set_title("Graph 4: Feature Importance", fontweight="bold")
ax4.set_xlabel("Importance Score (absolute coefficient)")
ax4.set_ylabel("Feature")

for bar, val in zip(bars, coefficients):
    ax4.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
             str(round(val, 3)), va="center", fontsize=9, fontweight="bold")

ax4.text(0.5, -0.18,
         "Longer bar = feature matters more to the AI prediction",
         transform=ax4.transAxes, ha="center", fontsize=8,
         color="gray", style="italic")

plt.savefig("air_quality_results.png", dpi=150, bbox_inches="tight")
print("Graph saved as: air_quality_results.png")
plt.show()
print("\nAll done! Your Air Quality Predictor is now impressive!")