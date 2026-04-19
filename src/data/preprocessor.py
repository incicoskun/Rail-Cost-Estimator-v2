"""Feature engineering utilities."""

import numpy as np
import pandas as pd


class FeatureEngineer:
    """Feature creation and transformation for model input."""

    @staticmethod
    def apply_base_features(df: pd.DataFrame) -> pd.DataFrame:
        """Apply base feature engineering transformations.

        Creates derived features:
        - tunnel_pct: Normalized to 0-1 range (handles both % and decimal)
        - station_density: Stations per km
        - log_length: Log-transformed line length
        - mid_year: Mid-point year for inflation adjustment
        - is_regional_rail: Filled NaN with 0

        Args:
            df: DataFrame with raw features

        Returns:
            DataFrame with engineered features added
        """
        df = df.copy()

        # Normalize tunnel percentage (may come as 0-100 or 0-1)
        if "tunnel_pct" in df.columns:
            df["tunnel_pct"] = df["tunnel_pct"].apply(lambda x: x/100 if x > 1 else x)

        # Station density: stations per km (add small offset to avoid division by zero)
        if "num_stations" in df.columns and "length_km" in df.columns:
            df["station_density"] = df["num_stations"] / (df["length_km"] + 0.1)

        # Log-transformed line length
        if "length_km" in df.columns:
            df["log_length"] = np.log(df["length_km"])

        # Mid-year for inflation adjustment
        if "start_year" in df.columns and "end_year" in df.columns:
            df["mid_year"] = (df["start_year"] + df["end_year"]) / 2

        # Fill regional rail indicator
        if "is_regional_rail" in df.columns:
            df["is_regional_rail"] = df["is_regional_rail"].fillna(0.0)

        return df
