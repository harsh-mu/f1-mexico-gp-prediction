from .qualifying import engineer_qualifying_features
from .driver_form import engineer_driver_features
from .team_form import engineer_team_features
from .circuit import engineer_circuit_features


def build_features(df):
    """Apply all feature-engineering modules."""

    df = df.copy()

    df = engineer_qualifying_features(df)
    df = engineer_driver_features(df)
    df = engineer_team_features(df)
    df = engineer_circuit_features(df)

    return df
