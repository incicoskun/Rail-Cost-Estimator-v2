"""SHAP-based cost driver analysis."""

import numpy as np
import pandas as pd
import shap

from ..config import ModelConfig


class SHAPExplainer:
    """Cost driver analysis using SHAP values."""

    def __init__(self, model, config: ModelConfig):
        """Initialize SHAP explainer.

        Args:
            model: Trained GradientBoostingRegressor
            config: ModelConfig with feature names and display names
        """
        self.model = model
        self.config = config
        self.explainer = shap.TreeExplainer(model)

    def explain(self, X: pd.DataFrame) -> pd.DataFrame:
        """Generate SHAP feature importance for prediction.

        Args:
            X: Feature DataFrame (must contain config.feature_all columns)

        Returns:
            DataFrame with columns:
                - Factor: display name of feature
                - Raw_Value: raw SHAP value
                - Impact_Weight: absolute SHAP value
                - Signed_Percent: contribution as % of total absolute impact
                - Direction: "Increases Cost (+)" or "Decreases Cost (-)"
            Filtered to features contributing > 1% of total impact,
            sorted by Signed_Percent ascending (for horizontal bar chart).
        """
        shap_values = self.explainer(X[self.config.feature_all])[0].values
        total_abs = np.sum(np.abs(shap_values))

        impact_df = pd.DataFrame({
            "Factor": X[self.config.feature_all].rename(
                columns=self.config.feature_display_names
            ).columns,
            "Raw_Value": shap_values,
            "Impact_Weight": np.abs(shap_values),
            "Signed_Percent": (shap_values / total_abs) * 100,
        })

        # Filter to features with > 1% contribution
        impact_df = impact_df[
            impact_df["Impact_Weight"] / total_abs > 0.01
        ].copy()

        impact_df["Direction"] = impact_df["Raw_Value"].apply(
            lambda x: "Increases Cost (+)" if x > 0 else "Decreases Cost (-)"
        )

        return impact_df.sort_values("Signed_Percent", ascending=True)