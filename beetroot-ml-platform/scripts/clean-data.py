import pandas as pd

field_df = pd.read_csv(r"C:\Users\chryk\Downloads\Dataset\8_field_data.csv", encoding = "latin1")
weather_df = pd.read_csv(r"C:\Users\chryk\Downloads\Dataset\9_weather_data.csv", encoding = "latin1")


#drop rows where date could not be parsed 
field_df = field_df.dropna()
weather_df = weather_df.dropna()

#remove doubles
field_df = field_df.drop_duplicates()
weather_df = weather_df.drop_duplicates()


