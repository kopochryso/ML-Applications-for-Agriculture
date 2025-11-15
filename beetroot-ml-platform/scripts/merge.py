import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import joblib

# 1. Load data
field = pd.read_csv(r"C:\Users\chryk\Downloads\Dataset\8_field_data.csv",
                    sep=";", encoding="latin1")
weather = pd.read_csv(r"C:\Users\chryk\Downloads\Dataset\9_weather_data.csv",
                      sep=";", encoding="latin1")

# (optional but nice) parse dates if "date" is a string
for df in (field, weather):
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

# 2. Merge on year, site_no, date
data = pd.merge(field, weather, on=["year", "site_no", "date"], how="left")

# 3. Define target and features
target_col = "root_yield"
feature_cols = [
    "dps",
    "irrigation yes-no",
    "soil_moisture30",
    "soil_moisture60",
    "soil_moisture90",
    "PAW30",
    "PAW60",
    "PAW90",
    "LAI",
    "min_temp",
    "max_temp",
    "av_temp",
    "precipitation",
    "glob_radiation",
    "ET_grass"
]

# 4. Drop rows with missing target
data = data.dropna(subset=[target_col])

# 5. Build X, y
X = data[feature_cols].copy()
y = data[target_col]

# Encode irrigation yes/no into 0/1
X["irrigation_binary"] = (
    X["irrigation yes-no"]
    .astype(str)
    .str.strip()
    .str.lower()
    .map({"yes": 1, "no": 0})
)

X = X.drop(columns=["irrigation yes-no"])

# Fill numeric NaNs with column medians
X = X.fillna(X.median(numeric_only=True))

# 6. Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 7. Train model
model = RandomForestRegressor(
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

# 8. Evaluate
y_pred = model.predict(X_test)
print("R2:", r2_score(y_test, y_pred))
print("RMSE:", mean_squared_error(y_test, y_pred, squared=False))

# 9. Save model
joblib.dump(model, "root_yield_model.pkl")
print("Model saved to root_yield_model.pkl")
