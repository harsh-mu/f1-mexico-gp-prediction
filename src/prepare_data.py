import pandas as pd
import numpy as np

qualifying_df = pd.read_csv("data/raw/historical_qualifying.csv")
race_df = pd.read_csv("data/raw/historical_races.csv")

# Check for unmatched records between both qualifying and race
keys = ["Year", "RoundNumber", "DriverId"]

combined_df = pd.merge(qualifying_df, race_df, on=keys, how="inner", suffixes=["_quali", "_race"], validate="one_to_one")

# print(combined_df.head(100))

cleaned_df = combined_df.copy()

quali_time_cols = ["Q1_quali", "Q2_quali", "Q3_quali"]


cleaned_df[quali_time_cols] = cleaned_df[quali_time_cols].apply(pd.to_timedelta, errors="coerce")

# Convert Timedelta values into numerical seconds
cleaned_df[quali_time_cols] = cleaned_df[quali_time_cols].apply(lambda col: col.dt.total_seconds())

conditions = [
    cleaned_df["Q3_quali"].notna(),
    cleaned_df["Q2_quali"].notna(),
    cleaned_df["Q1_quali"].notna()
]

choices = [
    cleaned_df["Q3_quali"],
    cleaned_df["Q2_quali"],
    cleaned_df["Q1_quali"]
]

segments = ["Q3", "Q2", "Q1"]

cleaned_df["qualifying_time"] = np.select(
    conditions,
    choices,
    default=np.nan
)

cleaned_df["qualifying_segment"] = np.select(
    conditions,
    segments,
    default="Missing"
)

cleaned_df["fastest_q1"] = cleaned_df.groupby(["Year", "RoundNumber"])["Q1_quali"].transform("min")
cleaned_df["fastest_q2"] = cleaned_df.groupby(["Year", "RoundNumber"])["Q2_quali"].transform("min")
cleaned_df["fastest_q3"] = cleaned_df.groupby(["Year", "RoundNumber"])["Q3_quali"].transform("min")

segment_conditions = [
    cleaned_df["qualifying_segment"] == "Q3", 
    cleaned_df["qualifying_segment"] == "Q2",
    cleaned_df["qualifying_segment"] == "Q1"
]

segment_choices = [
    cleaned_df["fastest_q3"],
    cleaned_df["fastest_q2"], 
    cleaned_df["fastest_q1"]
]

cleaned_df["segment_fastest_time"] = np.select(
    segment_conditions,
    segment_choices,
    default=np.nan
)

cleaned_df["qualifying_gap_pct"] = ((cleaned_df["qualifying_time"] - cleaned_df["segment_fastest_time"]) / cleaned_df["segment_fastest_time"]) * 100

cleaned_df["driver_avg_finish_5"] = (
    cleaned_df.groupby(["DriverId"])["Position_race"].transform(
        lambda x: x.shift(1).rolling(window=5, min_periods=1).mean()
    )
)

cleaned_df["driver_avg_qualifying_5"] = (
    cleaned_df.groupby(["DriverId"])["Position_quali"].transform(
        lambda x: x.shift(1).rolling(window=5, min_periods=1).mean()
    )
)

cleaned_df["driver_finish_std_5"] = (
    cleaned_df.groupby("DriverId")["Position_race"].transform(
        lambda x: x.shift(1).rolling(window=5, min_periods = 2).std()
    )
)

print(cleaned_df.loc[
    cleaned_df["DriverId"] == "max_verstappen",
    [
        "Year",
        "RoundNumber",
        "Position_race",
        "Position_quali",
        "driver_avg_finish_5",
        "driver_avg_qualifying_5",
        "driver_finish_std_5"
    ]
].head(10))