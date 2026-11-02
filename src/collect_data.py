from pathlib import Path
from datetime import datetime
import os
import time

import fastf1
import pandas as pd

START_YEAR = 2022
END_YEAR = 2026
CUTOFF_DATE = pd.Timestamp("2026-11-01")

DATA_DIR = Path("data/raw")
CACHE_DIR = Path("cache")

QUALIFYING_FILE = DATA_DIR / "historical_qualifying.csv"
RACE_FILE = DATA_DIR / "historical_races.csv"
FAILED_FILE = DATA_DIR / "failed_rounds.csv"

MAX_RETRIES = 3

DATA_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)

fastf1.Cache.enable_cache(str(CACHE_DIR))


def load_session(year, round_number, session_type):
    session = fastf1.get_session(
        year,
        gp=int(round_number),
        identifier=session_type
    )

    session.load(
        laps=False,
        telemetry=False,
        weather=False,
        messages=False
    )

    results = session.results.copy()

    if results.empty:
        raise ValueError("Empty session results")

    results["Year"] = year
    results["RoundNumber"] = int(round_number)

    return results


def collect_qualifying(year, round_number):
    return load_session(year, round_number, "Q")


def collect_race(year, round_number):
    return load_session(year, round_number, "R")


def retry_request(function, *args):
    """
    Retry an API operation when it fails.
    """

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return function(*args)

        except Exception as e:
            print(f"Attempt {attempt} failed: {e}")

            if attempt == MAX_RETRIES:
                raise

            time.sleep(5 * attempt)

def load_existing_data(path):
    """
    Load previously collected results if they exist.
    """

    if path.exists():
        return pd.read_csv(path)

    return pd.DataFrame()


def save_checkpoint(df, path):
    """
    Save to a temporary file, then replace the checkpoint.
    """

    temp_path = path.with_suffix(".tmp")
    df.to_csv(temp_path, index=False)
    os.replace(temp_path, path)


def get_completed_rounds(qualifying_df, race_df):
    """
    Identify race weekends already saved in both datasets.
    """

    if qualifying_df.empty or race_df.empty:
        return set()

    qualifying_rounds = set(
        zip(
            qualifying_df["Year"].astype(int),
            qualifying_df["RoundNumber"].astype(int)
        )
    )

    race_rounds = set(
        zip(
            race_df["Year"].astype(int),
            race_df["RoundNumber"].astype(int)
        )
    )

    return qualifying_rounds & race_rounds


def main():

    qualifying_df = load_existing_data(QUALIFYING_FILE)
    race_df = load_existing_data(RACE_FILE)

    completed_rounds = get_completed_rounds(
        qualifying_df, race_df
    )

    failed_rounds = []

    for year in range(START_YEAR, END_YEAR + 1):

        # Retrieve the season schedule
        try:
            schedule = retry_request(
                fastf1.get_event_schedule, year
            )

        except Exception as e:
            failed_rounds.append({
                "Year": year,
                "RoundNumber": None,
                "Error": str(e)
            })
            continue

        # Remove preseason testing and races after cutoff
        event_dates = pd.to_datetime(
            schedule["EventDate"], errors="coerce"
        )

        schedule = schedule[
            (schedule["RoundNumber"] > 0)
            & (event_dates < CUTOFF_DATE)
        ]

        for round_number in schedule["RoundNumber"]:

            round_number = int(round_number)
            race_key = (year, round_number)

            # Skip races already collected
            if race_key in completed_rounds:
                continue

            try:
                qualifying_results = retry_request(
                    collect_qualifying, year, round_number
                )

                race_results = retry_request(
                    collect_race, year, round_number
                )

                # Validate that driver identifiers are present
                if (
                    qualifying_results["DriverId"].isna().all()
                    or race_results["DriverId"].isna().all()
                ):
                    raise ValueError("Missing driver identifiers")

                # Remove any incomplete prior checkpoint rows
                # for this race before inserting the new results.
                for df_name in ("qualifying_df", "race_df"):
                    df = locals()[df_name]

                    if not df.empty:
                        df = df[
                            ~(
                                (df["Year"].astype(int) == year)
                                & (
                                    df["RoundNumber"].astype(int)
                                    == round_number
                                )
                            )
                        ]

                    if df_name == "qualifying_df":
                        qualifying_df = df
                    else:
                        race_df = df

                qualifying_df = pd.concat(
                    [qualifying_df, qualifying_results],
                    ignore_index=True
                )

                race_df = pd.concat(
                    [race_df, race_results],
                    ignore_index=True
                )

                # Save after every successful race
                save_checkpoint(qualifying_df, QUALIFYING_FILE)
                save_checkpoint(race_df, RACE_FILE)

                completed_rounds.add(race_key)

            except Exception as e:
                failed_rounds.append({
                    "Year": year,
                    "RoundNumber": round_number,
                    "Error": str(e)
                })

    pd.DataFrame(
        failed_rounds,
        columns=["Year", "RoundNumber", "Error"]
    ).to_csv(FAILED_FILE, index=False)

    print("\n========== COLLECTION SUMMARY ==========")
    print("Qualifying:", qualifying_df.shape)
    print("Race:", race_df.shape)
    print("Completed rounds:", len(completed_rounds))
    print("Failed operations:", len(failed_rounds))

    if failed_rounds:
        print("\nFailed rounds:")
        print(pd.DataFrame(failed_rounds))

    print("========================================")


if __name__ == "__main__":
    main()