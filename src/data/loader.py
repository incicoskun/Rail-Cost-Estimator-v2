"""Unified data loading interface."""

import pandas as pd

from ..config import FileConfig


class DataLoader:
    """Centralized interface for loading raw and processed data."""

    def __init__(self, config: FileConfig):
        """Initialize with file configuration.

        Args:
            config: FileConfig instance with all paths
        """
        self.config = config

    def load_fta_raw(self) -> pd.DataFrame:
        """Load FTA cost summary from Excel.

        Returns:
            Raw FTA cost breakdown data (49 rows)

        Raises:
            FileNotFoundError: If Excel file missing
            ValueError: If sheet not found
        """
        if not self.config.fta_summary_xl.exists():
            raise FileNotFoundError(f"FTA file not found: {self.config.fta_summary_xl}")

        try:
            return pd.read_excel(self.config.fta_summary_xl, sheet_name="Sheet1", header=0)
        except Exception as e:
            raise ValueError(f"Failed to load FTA data: {e}")

    def load_global_rail_raw(self) -> pd.DataFrame:
        """Load global rail projects from Excel.

        Returns:
            Raw project data

        Raises:
            FileNotFoundError: If Excel file missing
            ValueError: If sheet not found
        """
        if not self.config.global_rail_xl.exists():
            raise FileNotFoundError(f"Global Rail file not found: {self.config.global_rail_xl}")

        try:
            return pd.read_excel(
                self.config.global_rail_xl,
                sheet_name=self.config.sheet_latest,
                header=0,
            )
        except Exception as e:
            raise ValueError(f"Failed to load Global Rail data: {e}")

    def load_cpi_raw(self) -> pd.DataFrame:
        """Load CPI inflation data from Excel.

        Returns:
            Raw CPI data (1965-2030) as DataFrame

        Raises:
            FileNotFoundError: If Excel file missing
            ValueError: If sheet not found
        """
        if not self.config.global_rail_xl.exists():
            raise FileNotFoundError(f"Global Rail file not found: {self.config.global_rail_xl}")

        try:
            return pd.read_excel(
                self.config.global_rail_xl,
                sheet_name=self.config.sheet_cpi,
                header=0,
            )
        except Exception as e:
            raise ValueError(f"Failed to load CPI data: {e}")

    def load_processed_rail(self) -> pd.DataFrame:
        """Load pre-processed global rail dataset.

        Returns:
            Cleaned and processed global rail CSV

        Raises:
            FileNotFoundError: If CSV not found
        """
        if not self.config.global_rail_csv.exists():
            raise FileNotFoundError(
                f"Processed Global Rail CSV not found: {self.config.global_rail_csv}"
            )
        return pd.read_csv(self.config.global_rail_csv)

    def load_processed_fta(self) -> pd.DataFrame:
        """Load pre-processed FTA dataset.

        Returns:
            Cleaned and processed FTA CSV

        Raises:
            FileNotFoundError: If CSV not found
        """
        if not self.config.fta_processed_csv.exists():
            raise FileNotFoundError(
                f"Processed FTA CSV not found: {self.config.fta_processed_csv}"
            )
        return pd.read_csv(self.config.fta_processed_csv)
