
import pandas as pd

qualifying_df = pd.read_csv("data/raw/historical_qualifying.csv")
race_df = pd.read_csv("data/raw/historical_races.csv")

# Columns that uniquely identify a driver in a race
keys = ["Year", "RoundNumber", "DriverId"]

# Compare the two datasets
comparison = pd.merge(
    qualifying_df[keys + ["Abbreviation"]],
    race_df[keys + ["Abbreviation", "Status", "GridPosition"]],
    on=keys,
    how="outer",
    indicator=True,
    suffixes=("_quali", "_race"),
    validate="one_to_one"
)

# Find drivers appearing in only one dataset
missing = comparison[comparison["_merge"] != "both"]

print("\nRecord counts:")
print(comparison["_merge"].value_counts())

print("\nUnmatched drivers:")
print(missing.to_string(index=False))
