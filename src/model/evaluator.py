"""Model evaluation and metrics."""

from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


class ModelEvaluator:
    """Compute training metrics and performance statistics."""

    @staticmethod
    def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Compute comprehensive metrics on log-scale predictions.

        Args:
            y_true: True log-scaled target values
            y_pred: Predicted log-scaled target values

        Returns:
            Dict with keys:
                - r2_log: R2 score on log scale
                - rmse_log: RMSE on log scale
                - mdape: Median absolute percentage error (%)
                - bias: Median percentage error (%)
                - success_30: % of predictions within 30% error
                - q10_multiplier: Lower uncertainty band (OOF-calibrated)
                - q90_multiplier: Upper uncertainty band (OOF-calibrated)
        """
        # Convert to original scale and compute error rates
        y_true_orig = np.exp(y_true)
        y_pred_orig = np.exp(y_pred)
        errors = (y_pred_orig - y_true_orig) / y_true_orig
        abs_errors = np.abs(errors)

        # Empirical percentile intervals from error distribution
        q10_mult = float(1 + np.percentile(errors, 10))
        q90_mult = float(1 + np.percentile(errors, 90))

        # Safety guards: cost cannot be negative, hi must exceed lo
        q10_safe = max(0.0, q10_mult)
        q90_safe = max(q10_safe + 0.05, q90_mult)

        return {
            "r2_log": r2_score(y_true, y_pred),
            "rmse_log": np.sqrt(mean_squared_error(y_true, y_pred)),
            "mdape": np.median(abs_errors) * 100,
            "bias": np.median(errors) * 100,
            "success_30": np.mean(abs_errors < 0.30) * 100,
            "q10_multiplier": q10_safe,
            "q90_multiplier": q90_safe,
        }

    @staticmethod
    def mode_specific_multipliers(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        modes: np.ndarray,
    ) -> Dict[str, Dict[str, float]]:
        """Compute mode-specific uncertainty bands.

        Args:
            y_true: True log-scaled target values
            y_pred: Predicted log-scaled target values
            modes: Transit mode labels for each sample

        Returns:
            Dict mapping mode → {q10_multiplier, q90_multiplier}
        """
        y_true_orig = np.exp(y_true)
        y_pred_orig = np.exp(y_pred)
        errors = (y_pred_orig - y_true_orig) / y_true_orig

        mode_multipliers = {}
        for mode in np.unique(modes):
            mask = modes == mode
            if np.sum(mask) < 2:
                continue

            mode_errors = errors[mask]
            q10 = float(1 + np.percentile(mode_errors, 10))
            q90 = float(1 + np.percentile(mode_errors, 90))

            # Safety guards
            q10_safe = max(0.0, q10)
            q90_safe = max(q10_safe + 0.05, q90)

            mode_multipliers[str(mode)] = {
                "q10_multiplier": q10_safe,
                "q90_multiplier": q90_safe,
            }

        return mode_multipliers

    @staticmethod
    def country_stats(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        countries: np.ndarray,
    ) -> pd.DataFrame:
        """Per-country performance breakdown.

        Args:
            y_true: True log-scaled target values
            y_pred: Predicted log-scaled target values
            countries: Country codes

        Returns:
            DataFrame with per-country metrics (filtered to n >= 3)
        """
        y_true_orig = np.exp(y_true)
        y_pred_orig = np.exp(y_pred)
        errors = (y_pred_orig - y_true_orig) / y_true_orig
        abs_errors = np.abs(errors)

        stats = pd.DataFrame({
            "Country": countries,
            "True_Log": y_true,
            "Pred_Log": y_pred,
            "True_Cost": y_true_orig,
            "Pred_Cost": y_pred_orig,
            "Pct_Err":  errors * 100,
            "Abs_Pct_Err": abs_errors * 100,
        })

        country_summary = (
            stats.groupby("Country")
            .apply(lambda g: pd.Series({
                "Cnt": len(g),
                "Med_T": g["True_Cost"].median(),
                "Med_P": g["Pred_Cost"].median(),
                "R2": r2_score(g["True_Log"], g["Pred_Log"]) if len(g) > 1 else 0,
                "MdAPE": g["Abs_Pct_Err"].median(),
                "Bias": g["Pct_Err"].median(),
            }))
            .reset_index()
        )

        # Filter to countries with n >= 3
        country_summary = country_summary[country_summary["Cnt"] >= 3].sort_values("Cnt", ascending=False)

        return country_summary
