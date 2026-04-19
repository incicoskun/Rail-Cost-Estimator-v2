"""Master ETL and training pipeline orchestrator."""

import pandas as pd

from ..config import AppConfig
from ..data import DataLoader, FTAProcessor, GlobalRailProcessor
from ..model import ModelTrainer


class PipelineOrchestrator:
    """Orchestrates complete ETL + model training pipeline."""

    def __init__(self, config: AppConfig):
        """Initialize orchestrator.

        Args:
            config: AppConfig instance
        """
        self.config = config
        self.loader = DataLoader(config.file_config)
        self.rail_processor = GlobalRailProcessor()
        self.fta_processor = FTAProcessor()
        self.trainer = ModelTrainer(config.model_config, config.file_config)

    def run_full_pipeline(self) -> None:
        """Execute full ETL + training pipeline."""
        print("[PIPELINE] Step 1/3: Global rail data processing")
        rail_raw = self.loader.load_global_rail_raw()
        rail_processed = self.rail_processor.run(rail_raw)
        self.rail_processor.save(rail_processed, self.config.file_config.global_rail_csv)

        print("[PIPELINE] Step 2/3: FTA data processing")
        fta_raw = self.loader.load_fta_raw()
        cpi_raw = self.loader.load_cpi_raw()
        cpi_processed = self._preprocess_cpi(cpi_raw)
        fta_processed = self.fta_processor.run(fta_raw, cpi_processed)
        self.fta_processor.save(fta_processed, self.config.file_config.fta_processed_csv)

        print("[PIPELINE] Step 3/3: Model training")
        self.trainer.run()

        print("[PIPELINE] Completed successfully!")

    @staticmethod
    def _preprocess_cpi(cpi_raw: pd.DataFrame) -> pd.DataFrame:
        """Normalize CPI format for processing.

        Args:
            cpi_raw: Raw CPI DataFrame from Excel

        Returns:
            Processed CPI DataFrame indexed by Year
        """
        cpi_raw.columns = ["Year", "CPI_value", "unused", "Index_2021", "Index_2023"]
        cpi_raw = cpi_raw[pd.to_numeric(cpi_raw["Year"], errors="coerce").notna()].copy()
        cpi_raw["Year"] = cpi_raw["Year"].astype(int)
        return cpi_raw[["Year", "Index_2023"]].set_index("Year")