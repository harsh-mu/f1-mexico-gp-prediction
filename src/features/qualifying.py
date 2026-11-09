
import numpy as np
import pandas as pd


def engineer_qualifying_features(df):
    """Calculate qualifying pace relative to the driver's final segment."""

    df = df.copy()

    # Convert qualifying times into numerical seconds
    quali_time_cols = ["Q1_quali", "Q2_quali", "Q3_quali"]

    df[quali_time_cols] = df[quali_time_cols].apply(
        pd.to_timedelta, errors="coerce"
    )

    df[quali_time_cols] = df[quali_time_cols].apply(
        lambda col: col.dt.total_seconds()
    )

    # Identify the last qualifying segment completed
    conditions = [
        df["Q3_quali"].notna(),
        df["Q2_quali"].notna(),
        df["Q1_quali"].notna()
    ]

    choices = [
        df["Q3_quali"],
        df["Q2_quali"],
        df["Q1_quali"]
    ]

    segments = ["Q3", "Q2", "Q1"]

    df["qualifying_time"] = np.select(
        conditions,
        choices,
        default=np.nan
    )

    df["qualifying_segment"] = np.select(
        conditions,
        segments,
        default="Missing"
    )

    # Determine the fastest lap in each segment
    group_cols = ["Year", "RoundNumber"]

    df["fastest_q1"] = (
        df.groupby(group_cols)["Q1_quali"].transform("min")
    )

    df["fastest_q2"] = (
        df.groupby(group_cols)["Q2_quali"].transform("min")
    )

    df["fastest_q3"] = (
        df.groupby(group_cols)["Q3_quali"].transform("min")
    )

    # Compare drivers against the fastest time in their segment
    segment_conditions = [
        df["qualifying_segment"] == "Q3",
        df["qualifying_segment"] == "Q2",
        df["qualifying_segment"] == "Q1"
    ]

    segment_choices = [
        df["fastest_q3"],
        df["fastest_q2"],
        df["fastest_q1"]
    ]

    df["segment_fastest_time"] = np.select(
        segment_conditions,
        segment_choices,
        default=np.nan
    )

    df["qualifying_gap_pct"] = (
        (df["qualifying_time"] - df["segment_fastest_time"])
        / df["segment_fastest_time"]
    ) * 100

    return df
