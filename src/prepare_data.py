
from pathlib import Path

import pandas as pd

from features.build_features import build_features


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

MERGE_KEYS = ["Year", "RoundNumber", "DriverId"]


def load_data():
    """Load historical qualifying and race results."""

    qualifying_df = pd.read_csv(RAW_DIR / "historical_qualifying.csv")
    race_df = pd.read_csv(RAW_DIR / "historical_races.csv")

    return qualifying_df, race_df


def merge_data(qualifying_df, race_df):
    """Merge corresponding qualifying and race records."""

    combined_df = pd.merge(
        qualifying_df,
        race_df,
        on=MERGE_KEYS,
        how="inner",
        suffixes=("_quali", "_race"),
        validate="one_to_one"
    )

    return combined_df


def clean_data(df):
    """Apply basic cleaning and chronological sorting."""

    df = df.copy()

    df["Position_race"] = pd.to_numeric(
        df["Position_race"], errors="coerce"
    )

    df["Position_quali"] = pd.to_numeric(
        df["Position_quali"], errors="coerce"
    )

    df = df.sort_values(
        ["Year", "RoundNumber", "DriverId"]
    ).reset_index(drop=True)

    return df


def save_data(df):
    """Export the model-ready dataset."""

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    output_path = PROCESSED_DIR / "model_data.csv"
    df.to_csv(output_path, index=False)

    print(f"Saved processed data to: {output_path}")


def main():
    qualifying_df, race_df = load_data()

    combined_df = merge_data(qualifying_df, race_df)

    cleaned_df = clean_data(combined_df)

    featured_df = build_features(cleaned_df)

    save_data(featured_df)

    print(featured_df.loc[
        featured_df["DriverId"] == "max_verstappen",
        [
            "Year",
            "RoundNumber",
            "Status_race",
            "is_dnf",
            "driver_dnf_rate_10"
        ]
    ].head(15))






if __name__ == "__main__":
    main()
