"""Bayesian target encoding."""

from typing import Optional, Tuple

import pandas as pd


class TargetEncoder:
    """Bayesian smoothed target encoding for categorical features.

    Formula: smooth = (count * mean + m * global_mean) / (count + m)
    where m is the smoothing parameter (larger m = more shrinkage toward global mean)
    """

    def __init__(self, m: float = 10.0):
        """Initialize encoder with smoothing parameter.

        Args:
            m: Smoothing parameter (default 10.0)
        """
        self.m = m
        self.global_mean: Optional[float] = None
        self.mapping: Optional[dict] = None

    def fit(self, train_df: pd.DataFrame, target_col: str, cat_col: str) -> "TargetEncoder":
        """Learn encoding from training data.

        Args:
            train_df: Training DataFrame
            target_col: Name of target column
            cat_col: Name of categorical column

        Returns:
            Self (for chaining)
        """
        self.global_mean = train_df[target_col].mean()
        stats = train_df.groupby(cat_col)[target_col].agg(["mean", "count"])
        stats["smooth"] = (
            (stats["count"] * stats["mean"] + self.m * self.global_mean)
            / (stats["count"] + self.m)
        )
        self.mapping = stats["smooth"].to_dict()
        return self

    def transform(self, val_df: pd.DataFrame, cat_col: str) -> pd.Series:
        """Apply encoding to validation data.

        Args:
            val_df: Validation DataFrame
            cat_col: Name of categorical column

        Returns:
            Encoded Series (unmapped values -> global_mean)

        Raises:
            RuntimeError: If encoder not fitted
        """
        if self.mapping is None or self.global_mean is None:
            raise RuntimeError("TargetEncoder not fitted. Call fit() first.")
        return val_df[cat_col].map(self.mapping).fillna(self.global_mean)

    def get_map(self) -> Tuple[dict, float]:
        """Return encoding map and global mean.

        Returns:
            Tuple of (mapping dict, global mean)

        Raises:
            RuntimeError: If encoder not fitted
        """
        if self.mapping is None or self.global_mean is None:
            raise RuntimeError("TargetEncoder not fitted. Call fit() first.")
        return self.mapping.copy(), self.global_mean
