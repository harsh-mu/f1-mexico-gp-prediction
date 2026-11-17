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
    """Merge qualifying and race records, validating event metadata."""

    combined_df = pd.merge(
        left=qualifying_df,
        right=race_df,
        on=MERGE_KEYS,
        how="inner",
        suffixes=("_quali", "_race"),
        validate="one_to_one"
    )

    # Validate and consolidate event metadata
    for col in ["EventName", "Location", "Country"]:
        quali_col = f"{col}_quali"
        race_col = f"{col}_race"

        if not combined_df[quali_col].equals(combined_df[race_col]):
            raise ValueError(f"Mismatched event metadata: {col}")

        combined_df[col] = combined_df[quali_col]

        combined_df = combined_df.drop(
            columns=[quali_col, race_col]
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


    print(
        featured_df.loc[
            (featured_df["DriverId"] == "max_verstappen")
            & (featured_df["Location"] == "Mexico City"),
            [
                "Year",
                "RoundNumber",
                "Position_race",
                "driver_circuit_avg_finish"
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
