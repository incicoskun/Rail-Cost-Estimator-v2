"""FTA-specific data processing."""

import pandas as pd

from .processor import DataProcessor

# FTA-specific constants
MILES_TO_KM = 1.60934

DOLLAR_COLS = [
    "Project Cost", "Cost per Mile", "Hard Costs + ROW", "Hard Costs", "Soft Costs",
    "Avg Cost per Vehicle", "Avg Cost per Station", "Vehicle Costs", "ROW Costs",
    "Systems Costs", "Sitework Costs", "Facilities Costs", "Station Costs",
    "Guideway Costs", "Soft Costs per Mile", "ROW Costs per Mile",
    "System Costs per Mile", "Sitework Costs per Mile", "Facility Costs per Mile",
    "Guideway Costs per Mile",
]

PCT_COLS_FLOAT = [
    "Tunnel %", "Soft/ Hard Percent", "Soft Costs %", "Vehicle Costs %", "ROW Costs %",
    "Systems Costs %", "Sitework Costs %", "Facilities Costs %", "Station Costs %",
    "Guideway Costs %",
]

RENAME_MAP = {
    "Project\xa0(All costs have been adjusted for inflation in 2021 dollars)": "project",
    "Mode": "mode", "Mode.1": "mode_full", "locCity": "city", "Year": "year",
    "Length": "length_miles", "length_km": "length_km", "at grade": "at_grade_miles",
    "above grade": "elevated_miles", "below grade": "below_grade_miles",
    "Tunnel Miles": "tunnel_miles", "Tunnel %": "tunnel_pct",
    "Project Cost": "project_cost", "Cost per Mile": "cost_per_mile",
    "cost_per_km": "cost_per_km", "Hard Costs + ROW": "hard_costs_row",
    "Hard Costs": "hard_costs", "Soft Costs": "soft_costs",
    "Soft/ Hard Percent": "soft_hard_pct", "Vehicles": "vehicles",
    "Avg Cost per Vehicle": "avg_cost_per_vehicle", "Stations": "stations",
    "Avg Cost per Station": "avg_cost_per_station", "Vehicle Costs": "vehicle_costs",
    "ROW Costs": "row_costs", "Systems Costs": "systems_costs",
    "Sitework Costs": "sitework_costs", "Facilities Costs": "facilities_costs",
    "Station Costs": "station_costs", "Guideway Costs": "guideway_costs",
    "Soft Costs %": "soft_costs_pct", "Vehicle Costs %": "vehicle_costs_pct",
    "ROW Costs %": "row_costs_pct", "Systems Costs %": "systems_costs_pct",
    "Sitework Costs %": "sitework_costs_pct", "Facilities Costs %": "facilities_costs_pct",
    "Station Costs %": "station_costs_pct", "Guideway Costs %": "guideway_costs_pct",
    "Soft Costs per Mile": "soft_costs_per_mile", "ROW Costs per Mile": "row_costs_per_mile",
    "System Costs per Mile": "system_costs_per_mile", "Sitework Costs per Mile": "sitework_costs_per_mile",
    "Facility Costs per Mile": "facility_costs_per_mile", "Guideway Costs per Mile": "guideway_costs_per_mile",
    "soft_costs_per_km": "soft_costs_per_km", "row_costs_per_km": "row_costs_per_km",
    "system_costs_per_km": "system_costs_per_km", "sitework_costs_per_km": "sitework_costs_per_km",
    "facility_costs_per_km": "facility_costs_per_km", "guideway_costs_per_km": "guideway_costs_per_km",
}


class FTAProcessor(DataProcessor):
    """FTA cost breakdown data processing."""

    def _clean_dollar_cols(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove $ and commas, convert to float."""
        for col in DOLLAR_COLS:
            if col not in df.columns:
                continue
            cleaned = (
                df[col].astype(str)
                .str.replace(r"[$,]", "", regex=True)
                .str.strip()
            )
            df[col] = pd.to_numeric(cleaned, errors="coerce")
            n_null = df[col].isna().sum()
            if n_null > 0:
                print(f"  '{col}': {n_null} nulls after coercion")
        return df

    def _validate_pct_cols_float(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate float % columns are in 0-1 range."""
        for col in PCT_COLS_FLOAT:
            if col not in df.columns:
                continue
            out_of_range = df[col].dropna()
            out_of_range = out_of_range[(out_of_range < 0) | (out_of_range > 1)]
            if not out_of_range.empty:
                print(f"  '{col}': {len(out_of_range)} values outside 0-1 range")
        return df

    def _convert_miles_to_km(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert length columns from miles to km (factor: 1.60934)."""
        mile_cols = {
            "Length": "length_km",
            "at grade": "at_grade_km",
            "above grade": "elevated_km",
            "below grade": "below_grade_km",
            "Tunnel Miles": "tunnel_km",
        }
        for src, dst in mile_cols.items():
            if src in df.columns:
                df[dst] = df[src] * MILES_TO_KM
        return df

    def _convert_per_mile_costs_to_per_km(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert per-mile costs to per-km (divide by 1.60934)."""
        per_mile_to_per_km = {
            "Cost per Mile": "cost_per_km",
            "Soft Costs per Mile": "soft_costs_per_km",
            "ROW Costs per Mile": "row_costs_per_km",
            "System Costs per Mile": "system_costs_per_km",
            "Sitework Costs per Mile": "sitework_costs_per_km",
            "Facility Costs per Mile": "facility_costs_per_km",
            "Guideway Costs per Mile": "guideway_costs_per_km",
        }
        for src, dst in per_mile_to_per_km.items():
            if src in df.columns:
                df[dst] = df[src] / MILES_TO_KM
        return df

    def _add_subsystem_per_km(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute subsystem cost per km from subsystem totals and length."""
        subsystem_map = {
            "Vehicle Costs": "vehicle_costs_per_km",
            "ROW Costs": "row_costs_per_km_direct",
            "Systems Costs": "systems_costs_per_km",
            "Sitework Costs": "sitework_costs_per_km_direct",
            "Facilities Costs": "facilities_costs_per_km",
            "Station Costs": "station_costs_per_km",
            "Guideway Costs": "guideway_costs_per_km_direct",
            "Soft Costs": "soft_costs_per_km_direct",
            "Hard Costs": "hard_costs_per_km",
        }
        for src, dst in subsystem_map.items():
            if src in df.columns and "length_km" in df.columns:
                df[dst] = df[src] / df["length_km"]
        return df

    def _inflate_to_2023(self, df: pd.DataFrame, multiplier: float) -> pd.DataFrame:
        """Multiply all dollar columns by CPI multiplier (2021→2023)."""
        # Original dollar cols
        for col in DOLLAR_COLS:
            if col in df.columns:
                df[col + "_2023"] = df[col] * multiplier

        # Per-km derived cols
        per_km_cols = [
            "cost_per_km",
            "soft_costs_per_km", "row_costs_per_km", "system_costs_per_km",
            "sitework_costs_per_km", "facility_costs_per_km", "guideway_costs_per_km",
            "vehicle_costs_per_km", "row_costs_per_km_direct", "systems_costs_per_km",
            "sitework_costs_per_km_direct", "facilities_costs_per_km",
            "station_costs_per_km", "guideway_costs_per_km_direct",
            "soft_costs_per_km_direct", "hard_costs_per_km",
        ]
        for col in per_km_cols:
            if col in df.columns:
                df[col + "_2023"] = df[col] * multiplier

        return df

    def _rename_to_snake(self, df: pd.DataFrame) -> pd.DataFrame:
        """Rename all columns to snake_case."""
        df = df.rename(columns=RENAME_MAP)
        df.columns = [
            col.lower().replace(" ", "_").replace("/", "_").replace(".", "_").strip("_")
            for col in df.columns
        ]
        return df

    @staticmethod
    def _get_inflation_multiplier(cpi: pd.DataFrame) -> float:
        """Get multiplier to convert 2021 USD to 2023 USD from CPI."""
        return float(cpi.loc[2021, "Index_2023"])

    def run(self, df: pd.DataFrame, cpi: pd.DataFrame) -> pd.DataFrame:
        """Execute FTA processing pipeline.

        Args:
            df: Raw FTA DataFrame
            cpi: CPI DataFrame indexed by Year with Index_2023 column

        Returns:
            Processed DataFrame with 2023-adjusted costs in km units
        """
        df = df.copy()

        # Pipeline steps
        df = self._clean_dollar_cols(df)
        df = self._validate_pct_cols_float(df)
        df = self._convert_miles_to_km(df)
        df = self._convert_per_mile_costs_to_per_km(df)
        df = self._add_subsystem_per_km(df)

        multiplier = self._get_inflation_multiplier(cpi)
        print(f"  CPI multiplier 2021->2023: {multiplier:.4f}")
        df = self._inflate_to_2023(df, multiplier)

        df = self._rename_to_snake(df)

        return df