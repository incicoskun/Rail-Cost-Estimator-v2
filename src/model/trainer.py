"""High-level model training orchestrator."""

from pathlib import Path
from typing import Dict, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor

from ..config import FileConfig, ModelConfig
from ..data import DataLoader, FeatureEngineer
from .encoder import TargetEncoder
from .evaluator import ModelEvaluator
from .pipeline import ModelPipeline


class ModelTrainer:
    """Unified model training orchestrator."""

    def __init__(self, config: ModelConfig, file_config: FileConfig):
        """Initialize trainer.

        Args:
            config: ModelConfig instance
            file_config: FileConfig instance
        """
        self.config = config
        self.file_config = file_config
        self.loader = DataLoader(file_config)
        self.pipeline = ModelPipeline(config)
        self.model: Optional[GradientBoostingRegressor] = None
        self.memory_package: Optional[Dict] = None

    def load_and_prepare(self) -> pd.DataFrame:
        """Load and prepare training data.

        Returns:
            Prepared DataFrame ready for training
        """
        df = self.loader.load_processed_rail()
        return self.pipeline.prepare_data(df)

    def train(self, df: pd.DataFrame) -> Tuple[Dict, pd.DataFrame]:
        """Run full training pipeline.

        Args:
            df: Prepared training DataFrame

        Returns:
            Tuple of (report dict, country_stats DataFrame)
        """
        cv_df, oof_preds = self.pipeline.cross_validate(df)

        report = ModelEvaluator.compute_metrics(
            cv_df["log_cost"].values,
            oof_preds,
        )

        country_stats = ModelEvaluator.country_stats(
            cv_df["log_cost"].values,
            oof_preds,
            cv_df["country"].values,
        )

        print("\nCross-validation summary:")
        print(report)
        print(country_stats.head(10).to_string(index=False))

        return report, country_stats

    def finalize(
        self,
        df: pd.DataFrame,
        report: Dict,
        country_stats: pd.DataFrame,
    ) -> None:
        """Train final model and save artifacts.

        Args:
            df: Full training DataFrame
            report: CV metrics report
            country_stats: Per-country statistics
        """
        # Extract encoding maps from full dataset
        global_mean = df["log_cost"].mean()

        def smooth_encode(col: str, m: int) -> Tuple[Dict, float]:
            encoder = TargetEncoder(m=m)
            encoder.fit(df, "log_cost", col)
            return encoder.get_map()

        country_te_map, _ = smooth_encode(
            "country",
            self.config.encoding_params["country_m"],
        )
        city_te_map, _ = smooth_encode(
            "city",
            self.config.encoding_params["city_m"],
        )

        # Compute frequency maps
        c_freq_map = df["country"].value_counts().to_dict()
        ct_freq_map = df["city"].value_counts().to_dict()

        # Apply encodings to full dataset
        df["country_te"] = df["country"].map(country_te_map).fillna(global_mean)
        df["country_freq"] = df["country"].map(c_freq_map).fillna(1)
        df["city_te"] = df["city"].map(city_te_map).fillna(global_mean)
        df["city_freq"] = df["city"].map(ct_freq_map).fillna(1)

        # Train final model on full dataset
        params = self.config.params_dict
        model_config = self.config.model_config_dict
        self.model = GradientBoostingRegressor(**model_config, **params)
        self.model.fit(df[self.config.feature_all], df["log_cost"])

        # Package memory for inference
        self.memory_package = {
            "country_te_map": country_te_map,
            "city_te_map": city_te_map,
            "global_mean": global_mean,
            "country_freq_map": c_freq_map,
            "city_freq_map": ct_freq_map,
            "train_medians": self.pipeline.train_medians,
            "performance_stats": country_stats.to_dict("records"),
            "report": report,
        }

        # Save artifacts
        self._save_artifacts()

    def _save_artifacts(self) -> None:
        """Save model and memory package to disk."""
        if self.model is None or self.memory_package is None:
            raise RuntimeError("Model or memory package not initialized. Call finalize() first.")

        out_dir = self.file_config.data_processed.parent.parent
        model_path = out_dir / "rail_cost_model.pkl"
        memory_path = out_dir / "memory_package.pkl"
        features_path = out_dir / "feature_names.pkl"

        joblib.dump(self.model, model_path)
        joblib.dump(self.memory_package, memory_path)
        joblib.dump(self.config.feature_all, features_path)

        print("[OK] All artifacts saved:")
        print(f"  - {model_path}")
        print(f"  - {memory_path}")
        print(f"  - {features_path}")

    def run(self) -> None:
        """Execute full training pipeline end-to-end."""
        df = self.load_and_prepare()
        report, country_stats = self.train(df)
        self.finalize(df, report, country_stats)
        print("[OK] Training complete")