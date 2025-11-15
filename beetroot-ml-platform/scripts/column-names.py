import pandas as pd

field = pd.read_csv(r"C:\Users\chryk\Downloads\Dataset\8_field_data.csv", sep=";", encoding="latin1")
weather = pd.read_csv(r"C:\Users\chryk\Downloads\Dataset\9_weather_data.csv", sep=";", encoding="latin1")

print("\nFIELD DATA COLUMNS:\n", field.columns.tolist())
print("\nWEATHER DATA COLUMNS:\n", weather.columns.tolist())

# Optional: try merging to see final columns
try:
    merged = pd.merge(field, weather, on=["year", "site_no", "date"], how="left")
    print("\nMERGED COLUMNS:\n", merged.columns.tolist())
except Exception as e:
    print("\nMERGE ERROR:", e)
