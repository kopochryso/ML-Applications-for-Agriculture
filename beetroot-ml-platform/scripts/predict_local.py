import joblib
import pandas as pd

# -------- 1. Load trained model --------
MODEL_PATH = r"C:\Users\chryk\OneDrive\Έγγραφα\GitHub\ML-Applications-for-Agriculture\beetroot-ml-platform\models\root_yield_model.pkl"
model = joblib.load(MODEL_PATH)
print(f"Loaded model from {MODEL_PATH}")

# -------- 2. Define the final feature order used in training --------
FEATURE_ORDER = [
    "dps",
    "soil_moisture_30",
    "soil_moisture_60",
    "soil_moisture_90",
    "PAW_30",
    "PAW_60",
    "PAW_90",
    "LAI",
    "min_temp",
    "max_temp",
    "av_temp",
    "precipitation",
    "glob_radiation",
    "ET_grass",
    "irrigation_binary",  # this replaced 'irrigation' in training
]

# -------- 3. Build one or more NEW samples (no root_yield!) --------
# You can change these numbers to test different scenarios.
new_samples_raw = [
    {
        "dps": 120,
        "irrigation": "yes",
        "soil_moisture_30": 32.0,
        "soil_moisture_60": 28.0,
        "soil_moisture_90": 25.0,
        "PAW_30": 50.0,
        "PAW_60": 45.0,
        "PAW_90": 40.0,
        "LAI": 4.2,
        "min_temp": 12.0,
        "max_temp": 25.0,
        "av_temp": 18.0,
        "precipitation": 3.0,
        "glob_radiation": 19.0,
        "ET_grass": 3.0,
    },
    {
        "dps": 95,
        "irrigation": "no",
        "soil_moisture_30": 20.0,
        "soil_moisture_60": 19.0,
        "soil_moisture_90": 18.0,
        "PAW_30": 30.0,
        "PAW_60": 28.0,
        "PAW_90": 25.0,
        "LAI": 3.0,
        "min_temp": 10.0,
        "max_temp": 22.0,
        "av_temp": 16.0,
        "precipitation": 0.0,
        "glob_radiation": 15.0,
        "ET_grass": 4.0,
    },
]

df = pd.DataFrame(new_samples_raw)

# -------- 4. Apply the SAME preprocessing as training --------

# irrigation -> irrigation_binary
df["irrigation_binary"] = (
    df["irrigation"]
    .astype(str)
    .str.strip()
    .str.lower()
    .map({"yes": 1, "no": 0})
)

# Drop original irrigation column
df = df.drop(columns=["irrigation"])

# (Optional) If you expect missing numeric values, you could fill them with 0 or medians here.
# For now assume you provide complete numeric data.

# Reorder columns to match training
df = df[FEATURE_ORDER]

print("Input features for prediction:")
print(df)

# -------- 5. Predict --------
predictions = model.predict(df)

# -------- 6. Show results --------
for i, pred in enumerate(predictions):
    print(f"\nSample {i+1}: predicted root_yield = {pred:.2f}")
