import pandas as pd

def engineer_circuit_features(df):
    """
    Future features:
    - Driver historical performance at this circuit
    - Season progress
    - Potential circuit characteristics
    """
    df = average_circuit_finish(df)

    return df


def average_circuit_finish(df):
    df = df.copy()

    df["driver_circuit_avg_finish"] = (
        df.groupby(["DriverId", "Location"])["Position_race"]
        .transform(
            lambda x: x.shift(1).expanding().mean()
        )
    )

    return df

