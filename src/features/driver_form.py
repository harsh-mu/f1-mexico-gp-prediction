import numpy as np
import pandas as pd


def average_finish(df):
    """Average finishing position over previous five races."""

    df = df.copy()

    df["driver_avg_finish_5"] = (
        df.groupby("DriverId")["Position_race"]
        .transform(
            lambda x: x.shift(1)
            .rolling(window=5, min_periods=1)
            .mean()
        )
    )

    return df


def average_qualifying(df):
    """Average qualifying position over previous five races."""

    df = df.copy()

    df["driver_avg_qualifying_5"] = (
        df.groupby("DriverId")["Position_quali"]
        .transform(
            lambda x: x.shift(1)
            .rolling(window=5, min_periods=1)
            .mean()
        )
    )

    return df


def finishing_consistency(df):
    """Standard deviation of finishing position over previous five races."""

    df = df.copy()

    df["driver_finish_std_5"] = (
        df.groupby("DriverId")["Position_race"]
        .transform(
            lambda x: x.shift(1)
            .rolling(window=5, min_periods=2)
            .std()
        )
    )

    return df


def dnf_rate(df):
    df = df.copy()

    status = df["Status_race"]

    finished = (
        status.isin(["Finished", "Lapped"])
        | status.str.match(r"^\+\d+ Laps?$", na=False)
    )

    exceptions = (
        status.isin(["Did not start", "Withdrew", "Disqualified"])
    )

    did_not_finish = status.notna() & ~(finished | exceptions)

    df["is_dnf"] = np.select(
        condlist=[finished, exceptions, did_not_finish], 
        choicelist=[0, np.nan, 1],
        default=np.nan
    )

    df["driver_dnf_rate_10"] = (
        df.groupby("DriverId")["is_dnf"].transform(lambda x: x.shift(1).rolling(window=10, min_periods=1).mean())
    )

    return df

def validate_chronological_order(df):
    """Ensure records are chronologically sorted."""

    expected_index = df.sort_values(
        ["Year", "RoundNumber", "DriverId"]
    ).index

    if not df.index.equals(expected_index):
        raise ValueError(
            "DataFrame must be sorted by Year, RoundNumber, and DriverId."
        )


def validate_driver_history(df):
    keys = ["Year", "RoundNumber", "DriverId"]

    if df.duplicated(subset=keys).any():
        raise ValueError(
            "Duplicate driver-race observations detected."
        )


def engineer_driver_features(df):
    """Run all completed driver-performance feature calculations."""
    validate_chronological_order(df)
    validate_driver_history(df)

    df = average_finish(df)
    df = average_qualifying(df)
    df = finishing_consistency(df)
    df = dnf_rate(df)

    return df
