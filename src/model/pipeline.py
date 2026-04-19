"""Model training pipeline and cross-validation."""

from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import StratifiedKFold

from ..config import ModelConfig
from ..data.preprocessor import FeatureEngineer
from .encoder import TargetEncoder
from .evaluator import ModelEvaluator


class ModelPipeline:
    """Orchestrates cross-validation training pipeline."""

    def __init__(self, config: ModelConfig):
        """Initialize with model configuration.

        Args:
            config: ModelConfig instance
        """
        self.config = config
        self.train_medians: Optional[Dict] = None
        self.oof_predictions: Optional[np.ndarray] = None
        self.cv_results: list = []

    def prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare data for training (fill medians, feature engineering).

        Args:
            df: Raw training DataFrame

        Returns:
            Prepared DataFrame with features
        """
        df = df.dropna(subset=[self.config.target_col]).reset_index(drop=True)
        df = df[~df["line"].isin(self.config.train_outliers)].reset_index(drop=True)

        # Compute medians for imputation
        self.train_medians = {
            "num_stations": df["num_stations"].median(),
            "mid_year": ((df["start_year"] + df["end_year"]) / 2).median(),
        }

        # Fill missing values
        df["num_stations"] = df["num_stations"].fillna(self.train_medians["num_stations"])

        # Feature engineering
        df = FeatureEngineer.apply_base_features(df)
        df["mid_year"] = df["mid_year"].fillna(self.train_medians["mid_year"])

        # Log-transform target
        df["log_cost"] = np.log(df[self.config.target_col])

        return df

    def cross_validate(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray]:
        """Run stratified K-fold cross-validation.

        Args:
            df: Prepared training DataFrame

        Returns:
            Tuple of (CV DataFrame with encodings, OOF predictions)
        """
        df = df.copy()
        df["cost_quartile"] = pd.qcut(df["log_cost"], q=self.config.oof_qcut, labels=False)

        skf = StratifiedKFold(
            n_splits=self.config.cv_folds,
            shuffle=True,
            random_state=self.config.random_state,
        )

        oof_preds = np.zeros(len(df))

        print("\n" + "=" * 80)
        print(f"STRATIFIED {self.config.cv_folds}-FOLD CROSS-VALIDATION")
        print("=" * 80)

        for fold, (tr_idx, val_idx) in enumerate(skf.split(df, df["cost_quartile"]), 1):
            tr, val = df.iloc[tr_idx].copy(), df.iloc[val_idx].copy()

            # Country target encoding
            country_encoder = TargetEncoder(m=self.config.encoding_params["country_m"])
            country_encoder.fit(tr, "log_cost", "country")
            tr["country_te"] = country_encoder.transform(tr, "country")
            val["country_te"] = country_encoder.transform(val, "country")

            c_freq = tr["country"].value_counts().to_dict()
            tr["country_freq"] = tr["country"].map(c_freq)
            val["country_freq"] = val["country"].map(c_freq).fillna(1)

            # City target encoding
            city_encoder = TargetEncoder(m=self.config.encoding_params["city_m"])
            city_encoder.fit(tr, "log_cost", "city")
            tr["city_te"] = city_encoder.transform(tr, "city")
            val["city_te"] = city_encoder.transform(val, "city")

            ct_freq = tr["city"].value_counts().to_dict()
            tr["city_freq"] = tr["city"].map(ct_freq)
            val["city_freq"] = val["city"].map(ct_freq).fillna(1)

            # Train model
            params = self.config.params_dict
            model_config = self.config.model_config_dict
            model = GradientBoostingRegressor(**model_config, **params)
            model.fit(tr[self.config.feature_all], tr["log_cost"])

            # Generate OOF predictions
            oof_preds[val_idx] = model.predict(val[self.config.feature_all])

            # Show fold metrics
            f_r2 = ModelEvaluator.compute_metrics(
                val["log_cost"].values,
                oof_preds[val_idx]
            )["r2_log"]
            f_mae = mean_absolute_error(val["log_cost"].values, oof_preds[val_idx])
            print(f"Fold {fold}: R2={f_r2:.4f}, MAE={f_mae:.4f}")

        # Store encoded data with OOF predictions
        df["oof_pred_log"] = oof_preds
        self.oof_predictions = oof_preds

        return df, oof_preds

    def get_oof_predictions(self) -> np.ndarray:
        """Get out-of-fold predictions.

        Returns:
            OOF predictions array

        Raises:
            RuntimeError: If cross_validate not yet run
        """
        if self.oof_predictions is None:
            raise RuntimeError("OOF predictions not available. Call cross_validate() first.")
        return self.oof_predictions