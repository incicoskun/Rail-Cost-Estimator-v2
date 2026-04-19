"""Global Rail dataset processing."""

import numpy as np
import pandas as pd

from .processor import DataProcessor

# Global Rail-specific constants
GLOBAL_RAIL_KEEP_COLS = [
    "Country", "City", "Line", "Phase",
    "Start year", "End year", "RR?",
    "Length", "TunnelPer", "Tunnel",
    "Stations", "Anglo?", "PPP rate",
    "Real cost (2023 dollars)", "Cost/km (2023 dollars)",
]

GLOBAL_RAIL_RENAME_MAP = {
    "Country": "country",
    "City": "city",
    "Line": "line",
    "Phase": "phase",
    "Start year": "start_year",
    "End year": "end_year",
    "RR?": "is_regional_rail",
    "Anglo?": "is_anglo",
    "PPP rate": "ppp_rate",
    "Length": "length_km",
    "TunnelPer": "tunnel_pct",
    "Tunnel": "tunnel_km",
    "Stations": "num_stations",
    "Real cost (2023 dollars)": "real_cost_2023_musd",
    "Cost/km (2023 dollars)": "cost_per_km_2023_musd",
}

GLOBAL_RAIL_NUMERIC_PARSE_STRATEGY = {
    "num_stations": "slash",
    "start_year": "year",
    "end_year": "year",
    "length_km": "float",
    "tunnel_pct": "float",
    "tunnel_km": "float",
    "ppp_rate": "float",
}


class GlobalRailProcessor(DataProcessor):
    """Process global rail dataset."""

    @staticmethod
    def _parse_slash(val) -> float:
        """Convert slash-separated strings to float average (e.g., '36/22' -> 29.0)."""
        if pd.isna(val):
            return np.nan
        text = str(val).strip()
        if "/" in text:
            parts = text.split("/")
            try:
                return float(np.mean([float(p.strip()) for p in parts]))
            except ValueError:
                print(f"  _parse_slash: '{text}' - cannot convert, set to NaN")
                return np.nan
        try:
            return float(text)
        except ValueError:
            print(f"  _parse_slash: '{text}' - cannot convert, set to NaN")
            return np.nan

    @staticmethod
    def _parse_year(val) -> float:
        """Clean year column by dropping non-numeric strings."""
        if pd.isna(val):
            return np.nan
        try:
            return float(str(val).strip())
        except ValueError:
            print(f"  _parse_year: '{val}' - not a valid year, set to NaN")
            return np.nan

    @staticmethod
    def _parse_float(val) -> float:
        """Standard float cast; returns NaN on failure."""
        if pd.isna(val):
            return np.nan
        try:
            return float(val)
        except (ValueError, TypeError):
            print(f"  _parse_float: '{val}' - cannot convert, set to NaN")
            return np.nan

    @staticmethod
    def _select_cols(df: pd.DataFrame) -> pd.DataFrame:
        """Select only required columns."""
        keep = [c for c in GLOBAL_RAIL_KEEP_COLS if c in df.columns]
        return df[keep].copy()

    @staticmethod
    def _rename_cols(df: pd.DataFrame) -> pd.DataFrame:
        """Rename columns to snake_case."""
        return df.rename(columns=GLOBAL_RAIL_RENAME_MAP)

    @staticmethod
    def _drop_incomplete_rows(df: pd.DataFrame) -> pd.DataFrame:
        """Drop rows where both country and city are missing."""
        before = len(df)
        df = df.dropna(subset=["country", "city"], how="all")
        dropped = before - len(df)
        if dropped > 0:
            print(f"  Dropped {dropped} rows - country and city both missing")
        return df

    @staticmethod
    def _drop_missing_target(df: pd.DataFrame) -> pd.DataFrame:
        """Drop rows where target variable is missing."""
        before = len(df)
        df = df.dropna(subset=["cost_per_km_2023_musd"])
        dropped = before - len(df)
        if dropped > 0:
            print(f"  Dropped {dropped} rows - cost_per_km_2023_musd missing")
        return df

    @staticmethod
    def _drop_missing_tunnel(df: pd.DataFrame) -> pd.DataFrame:
        """Drop rows where tunnel data is missing."""
        before = len(df)
        df = df.dropna(subset=["tunnel_km", "tunnel_pct"], how="any")
        dropped = before - len(df)
        if dropped > 0:
            print(f"  Dropped {dropped} rows - tunnel_km or tunnel_pct missing")
        return df

    def _clean_numeric(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean dirty values in numeric columns and cast to float."""
        df = df.copy()
        parsers = {
            "slash": self._parse_slash,
            "year": self._parse_year,
            "float": self._parse_float,
        }
        for col, strategy in GLOBAL_RAIL_NUMERIC_PARSE_STRATEGY.items():
            if col not in df.columns:
                continue
            parser = parsers[strategy]
            nan_before = int(df[col].isna().sum())
            df[col] = df[col].apply(parser).astype("float64")
            new_nans = int(df[col].isna().sum()) - nan_before
            if new_nans > 0:
                print(f"  _clean_numeric: '{col}' - {new_nans} dirty values set to NaN")
        return df

    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        """Execute global rail processing pipeline.

        Args:
            df: Raw global rail DataFrame

        Returns:
            Processed DataFrame
        """
        print(f"Raw data: {len(df)} rows")

        df = self._select_cols(df)
        df = self._rename_cols(df)
        df = self._drop_incomplete_rows(df)
        df = self._drop_missing_target(df)
        df = self._drop_missing_tunnel(df)
        df = self._clean_numeric(df)

        return df