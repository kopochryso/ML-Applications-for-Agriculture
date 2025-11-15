import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import joblib

# 1. Load data
field = pd.read_csv(
    r"C:\Users\chryk\Downloads\Dataset\8_field_data.csv",
    sep=";",
    encoding="latin1"
)
weather = pd.read_csv(
    r"C:\Users\chryk\Downloads\Dataset\9_weather_data.csv",
    sep=";",
    encoding="latin1"
)

# 2. Parse dates (both have a 'date' column)
for df in (field, weather):
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

field = field.dropna(subset=["date"])
weather = weather.dropna(subset=["date"])

# 3. Merge on year, site_no, date
data = pd.merge(
    field,
    weather,
    on=["year", "site_no", "date"],
    how="left",
    suffixes=("_field", "_weather")  # avoid x/y confusion
)

# After this, your columns will look like:
# year, site_no, trial, site_field, irrigation, plot, rep, date,
# dps_field, soil_moisture_30, ..., site_weather, dps_weather, min_temp, ...

# 4. Rename columns to something nicer
data = data.rename(columns={
    "site_field": "site",
    "dps_field": "dps",
    "site_weather": "site_weather",
    "dps_weather": "dps_weather"
})

# (optional) if dps_field and dps_weather are the same, you can drop dps_weather
# data = data.drop(columns=["dps_weather", "site_weather"])

# 5. Define target and features
target_col = "root_yield"

feature_cols = [
    "dps",
    "irrigation",
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
]

# 6. Drop rows with missing target
data = data.dropna(subset=[target_col])

# 7. Build X, y
X = data[feature_cols].copy()
y = data[target_col]

# 8. Encode irrigation yes/no into 0/1 (robust to case/whitespace)
X["irrigation_binary"] = (
    X["irrigation"]
    .astype(str)
    .str.strip()
    .str.lower()
    .map({"yes": 1, "no": 0})
)

# If your irrigation column isn't yes/no, print this to inspect:
# print(X["irrigation"].unique())

X = X.drop(columns=["irrigation"])

# 9. Fill numeric NaNs with column medians
X = X.fillna(X.median(numeric_only=True))

# 10. Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 11. Train model
model = RandomForestRegressor(
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

# 12. Evaluate
y_pred = model.predict(X_test)
print("R2:", r2_score(y_test, y_pred))
print("RMSE:", mean_squared_error(y_test, y_pred, squared=False))

# 13. Save model
joblib.dump(model, "root_yield_model.pkl")
print("Model saved to root_yield_model.pkl")
