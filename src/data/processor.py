"""Base class for data processing pipelines."""

from abc import ABC, abstractmethod
from pathlib import Path

import pandas as pd


class DataProcessor(ABC):
    """Abstract base class for data processing pipelines."""

    @abstractmethod
    def run(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """Process raw DataFrame and return cleaned output.

        Args:
            df: Raw input DataFrame
            **kwargs: Additional arguments specific to processor

        Returns:
            Processed DataFrame
        """
        pass

    def save(self, df: pd.DataFrame, output_path: Path) -> None:
        """Persist processed DataFrame to CSV.

        Args:
            df: DataFrame to save
            output_path: Path to output CSV file
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"[OK] Saved to {output_path}")