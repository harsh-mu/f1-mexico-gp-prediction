import pandas as pd


def average_team_finish(df):
    """Calculate team average finishing position over the previous 5 races."""

    df = df.copy()

    team_race_df = (
        df.groupby(
            ["Year", "RoundNumber", "TeamName_race"],
            as_index=False
        )
        .agg(
            team_race_finish=("Position_race", "mean")
        )
    )

    team_race_df = team_race_df.sort_values(
        ["Year", "RoundNumber", "TeamName_race"]
    ).reset_index(drop=True)

    team_race_df["team_avg_finish_5"] = (
        team_race_df.groupby("TeamName_race")["team_race_finish"].transform(
            lambda x: (
                x.shift(1)
                .rolling(window=5, min_periods=1)
                .mean()
            )
        )
    )

    df = pd.merge(
        left=df,
        right=team_race_df[
            ["Year", "RoundNumber", "TeamName_race", "team_avg_finish_5"]
        ],
        on=["Year", "RoundNumber", "TeamName_race"],
        how="left",
        validate="many_to_one"
    )

    return df


def average_team_qualifying(df):
    """Calculate team average qualifying position over the previous 5 races."""

    df = df.copy()

    team_race_df = (
        df.groupby(
            ["Year", "RoundNumber", "TeamName_race"],
            as_index=False
        )
        .agg(
            team_race_qualifying=("Position_quali", "mean")
        )
    )

    team_race_df = team_race_df.sort_values(
        ["Year", "RoundNumber", "TeamName_race"]
    ).reset_index(drop=True)

    team_race_df["team_avg_qualifying_5"] = (
        team_race_df.groupby("TeamName_race")["team_race_qualifying"].transform(
            lambda x: (
                x.shift(1)
                .rolling(window=5, min_periods=1)
                .mean()
            )
        )
    )

    df = pd.merge(
        left=df,
        right=team_race_df[
            ["Year", "RoundNumber", "TeamName_race", "team_avg_qualifying_5"]
        ],
        on=["Year", "RoundNumber", "TeamName_race"],
        how="left",
        validate="many_to_one"
    )

    return df


def engineer_team_features(df):
    """Run all team-performance feature calculations."""

    df = average_team_finish(df)
    df = average_team_qualifying(df)

    return df
