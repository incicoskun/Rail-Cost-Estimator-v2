"""File path configuration."""

from dataclasses import dataclass
from pathlib import Path


@dataclass
class FileConfig:
    """Centralized file path configuration."""

    base_dir: Path

    # Data paths
    data_processed: Path = None
    data_raw: Path = None

    # Output files
    global_rail_csv: Path = None
    fta_processed_csv: Path = None

    # Input files
    global_rail_xl: Path = None
    fta_summary_xl: Path = None

    # Sheet names
    sheet_latest: str = "1_16_2026"
    sheet_cpi: str = "CPI"

    def __post_init__(self) -> None:
        """Initialize derived path attributes."""
        if self.data_processed is None:
            self.data_processed = self.base_dir / "data" / "processed"
        if self.data_raw is None:
            self.data_raw = self.base_dir / "data" / "raw"
        if self.global_rail_csv is None:
            self.global_rail_csv = self.data_processed / "global_rail_processed.csv"
        if self.fta_processed_csv is None:
            self.fta_processed_csv = self.data_processed / "fta_processed.csv"
        if self.global_rail_xl is None:
            self.global_rail_xl = self.data_raw / "global_rail_costs.xlsx"
        if self.fta_summary_xl is None:
            self.fta_summary_xl = self.data_raw / "fta_summary.xlsx"
