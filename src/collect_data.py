from pathlib import Path
import fastf1
import pandas as pd

# Create a cache folder if it does not exist
Path("cache").mkdir(exist_ok=True)

# Enable FastF1 caching
fastf1.Cache.enable_cache("cache")

# Retrieve the qualifying session
qualifying = fastf1.get_session(2025, "Mexico City", "Q")

# Load results without telemetry or lap data
qualifying.load(
    laps=False,
    telemetry=False,
    weather=False,
    messages=False
)

# Extract the qualifying results DataFrame
qualifying_results = qualifying.results

# print(qualifying_results.shape)
# print(qualifying_results.dtypes)
# print(qualifying_results.isna().sum())
# print(qualifying_results[["Abbreviation", "Position", "Q1", "Q2", "Q3"]])

race = fastf1.get_session(2025, "Mexico City", "R")
race.load(
    laps=False, 
    telemetry=False, 
    weather=False,
    messages=False
)

race_results = race.results

# print(race_results[
#     ["Abbreviation", "GridPosition", "Position", "Status", "Points"]
# ].sort_values(by="Position", ascending=True))
# print(race_results.shape)
# print(race_results.dtypes)
# print(race_results.isna().sum())

qualifying_results.to_csv(
    "data/raw/2025_mexico_qualifying.csv",
    index=False
)

race_results.to_csv(
    "data/raw/2025_mexico_race.csv",
    index=False
)