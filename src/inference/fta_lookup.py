"""
src/inference/fta_lookup.py
─────────────────────────────────────────────────────────────────────────────
Mode-aware subsystem cost breakdown using FTA project data.

Approach:
    For a predicted total cost_per_km (from GBR), this module decomposes
    the total into 8 subsystems using median ratios derived from
    fta_processed.csv, stratified by transit mode.

    This is statistically correct because:
    - Physical decomposition (at-grade / elevated / underground coefficients)
      is NOT feasible: pct_ag + pct_el + pct_ug = 1 creates full
      multicollinearity; LOO-CV shows R² < 0 for all subsystems.
    - Mode is the strongest stratification variable (explains 60%+ of
      station_costs_pct variance between groups).
    - Median is robust to the small-n problem (n=49 total, n=27 LRT,
      n=15 HRT, n=4 BRT, n=2 CRT).

Usage:
    from src.inference import FTALookup

    lookup = FTALookup.from_csv("data/processed/fta_processed.csv")
    ratios = lookup.get_ratios("LRT")
    # {'Guideway': 0.26, 'Stations': 0.06, ...}

    breakdown = lookup.get_breakdown(pred_cost_per_km=45.0, transit_mode="LRT")
    # {'Guideway': 11.7, 'Stations': 2.7, ...}  (M$/km)
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from ..constants import SubsystemLabel, TransitMode


# ── Constants ─────────────────────────────────────────────────────────────────

SUBSYS_COLS: list[str] = [
    "guideway_costs_pct",
    "station_costs_pct",
    "systems_costs_pct",
    "soft_costs_pct",
    "vehicle_costs_pct",
    "row_costs_pct",
    "sitework_costs_pct",
    "facilities_costs_pct",
]

SUBSYS_LABELS: list[str] = [label.value for label in SubsystemLabel]

# Supported modes — inputs outside these fall back to global median
KNOWN_MODES: list[str] = [mode.value for mode in TransitMode]

# Minimum project count — modes below this threshold get a warning
MIN_N_WARN: int = 5


# ── Main Class ─────────────────────────────────────────────────────────────────

class FTALookup:
    """
    FTA-based mode-aware subsystem cost breakdown.

    Attributes:
        _ratios  : dict[mode → dict[subsystem → ratio]]
                   Contains both mode-based and 'global' keys.
        _counts  : dict[mode → int]  — project count per mode
        _fitted  : bool
    """

    def __init__(self) -> None:
        self._ratios: dict[str, dict[str, float]] = {}
        self._counts: dict[str, int] = {}
        self._fitted: bool = False

    # ── Constructors ───────────────────────────────────────────────────────

    @classmethod
    def from_csv(cls, csv_path: str | Path) -> "FTALookup":
        """Create FTALookup instance from fta_processed.csv."""
        path = Path(csv_path)
        if not path.exists():
            raise FileNotFoundError(f"FTA CSV not found: {path}")
        df = pd.read_csv(path)
        return cls.from_dataframe(df)

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame) -> "FTALookup":
        """Create FTALookup instance from DataFrame."""
        obj = cls()
        obj._fit(df)
        return obj

    # ── Fit ────────────────────────────────────────────────────────────────

    def _fit(self, df: pd.DataFrame) -> None:
        """
        Compute median subsystem ratios per mode and normalize.

        Steps:
          1. Per mode, compute median of SUBSYS_COLS
          2. Divide by sum (normalize) — so ratios sum to 1.0
          3. Compute global median as fallback
          4. Warn for modes with n < MIN_N_WARN
        """
        missing = [c for c in SUBSYS_COLS if c not in df.columns]
        if missing:
            raise ValueError(f"Missing columns: {missing}")

        if "mode" not in df.columns:
            raise ValueError("'mode' column not found")

        # Global fallback
        self._ratios["global"] = self._compute_ratios(df)
        self._counts["global"] = len(df)

        # Trolley (n=1) has multiple NaN cols — cannot yield a stable median
        _force_global = {"Trolley"}

        # Per mode — BRT/CRT are small-n but use their own ratios.
        # Falling back to global would erase real differences
        # (e.g. CRT sitework=26% vs global=7%).
        for mode, group in df.groupby("mode"):
            n = len(group)
            self._counts[mode] = n

            if mode in _force_global:
                warnings.warn(
                    f"FTALookup: '{mode}' has n={n} — too few for a stable median. "
                    f"Using global median.",
                    UserWarning,
                    stacklevel=2,
                )
                self._ratios[mode] = self._ratios["global"]
                continue

            if n < MIN_N_WARN:
                warnings.warn(
                    f"FTALookup: '{mode}' has only n={n} projects "
                    f"(recommended minimum: {MIN_N_WARN}). "
                    f"Using mode-specific ratios with caution.",
                    UserWarning,
                    stacklevel=2,
                )

            self._ratios[mode] = self._compute_ratios(group)

        self._fitted = True

    @staticmethod
    def _compute_ratios(df: pd.DataFrame) -> dict[str, float]:
        """
        Compute normalized median ratios for given DataFrame.

        Takes median of each column, divides by sum.
        If sum is zero or NaN, returns equal distribution.
        """
        medians = df[SUBSYS_COLS].median()

        total = medians.sum()
        if total <= 0 or np.isnan(total):
            warnings.warn(
                "FTALookup: Median sum is zero or NaN — using equal distribution.",
                UserWarning,
            )
            n = len(SUBSYS_LABELS)
            return {label: 1.0 / n for label in SUBSYS_LABELS}

        normed = medians / total
        return dict(zip(SUBSYS_LABELS, normed.values))

    # ── Public API ─────────────────────────────────────────────────────────

    def get_ratios(self, transit_mode: str) -> dict[str, float]:
        """
        Get normalized subsystem ratios for given transit mode.

        Args:
            transit_mode: "HRT", "LRT", "BRT", "CRT" or unknown

        Returns:
            {subsystem_label: ratio}  (values sum to ≈ 1.0)

        Note:
            Unknown mode → falls back to global median with warning.
        """
        self._check_fitted()

        if transit_mode not in self._ratios:
            warnings.warn(
                f"FTALookup: '{transit_mode}' not defined. "
                f"Using global median.",
                UserWarning,
            )
            return self._ratios["global"]

        return self._ratios[transit_mode]

    def get_breakdown(
        self,
        pred_cost_per_km: float,
        transit_mode: str,
    ) -> dict[str, float]:
        """
        Decompose predicted total cost into subsystems.

        Args:
            pred_cost_per_km : GBR prediction, in M$/km
            transit_mode     : "HRT", "LRT", "BRT", "CRT"

        Returns:
            {subsystem_label: cost_M_per_km}
        """
        self._check_fitted()
        ratios = self.get_ratios(transit_mode)
        return {
            label: pred_cost_per_km * ratio
            for label, ratio in ratios.items()
        }

    def get_breakdown_interval(
        self,
        pred_cost_per_km: float,
        transit_mode: str,
        q10_multiplier: float,
        q90_multiplier: float,
    ) -> dict[str, dict[str, float]]:
        """
        Return subsystem costs with OOF-calibrated intervals.

        lo/hi come from q10/q90 multipliers derived from the model's
        OOF error distribution — not arbitrary constants.

        Strategy B: total_lo/hi × median_ratio
          Multiplying two uncertainties (total_band × ratio_IQR) creates
          double-counting. Ratio is held constant at median, so the only
          uncertainty source is GBR model error — most defensible approach.

        Args:
            pred_cost_per_km : GBR point estimate (M$/km)
            transit_mode     : "HRT", "LRT", "BRT", "CRT"
            q10_multiplier   : memory_package["report"]["q10_multiplier"]
            q90_multiplier   : memory_package["report"]["q90_multiplier"]

        Returns:
            {subsystem_label: {"point": float, "lo": float, "hi": float}}
        """
        ratios   = self.get_ratios(transit_mode)

        # Robustness guards — never produces negative/inverted intervals
        q10_safe = max(0.0, q10_multiplier)
        q90_safe = max(q10_safe + 0.05, q90_multiplier)

        lo_total = pred_cost_per_km * q10_safe
        hi_total = pred_cost_per_km * q90_safe

        return {
            label: {
                "point": pred_cost_per_km * ratio,
                "lo":    lo_total * ratio,
                "hi":    hi_total * ratio,
            }
            for label, ratio in ratios.items()
        }

    def summary(self) -> pd.DataFrame:
        """
        Return ratios for all modes as a table.
        Useful for reporting and visualization.
        """
        self._check_fitted()
        rows = []
        for mode, ratios in self._ratios.items():
            row = {"mode": mode, "n": self._counts.get(mode, 0)}
            row.update(ratios)
            rows.append(row)
        return pd.DataFrame(rows).set_index("mode")

    def n_projects(self, transit_mode: str) -> int:
        """Get project count for given mode."""
        self._check_fitted()
        return self._counts.get(transit_mode, 0)

    # ── Helpers ────────────────────────────────────────────────────────────

    def _check_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError(
                "FTALookup not fitted yet. "
                "Use from_csv() or from_dataframe()."
            )

    def get_mode_variability(self, transit_mode: str) -> float:
        """
        Estimate cost variability for given mode based on subsystem distribution.

        Returns a scalar in [0.0, 1.0] representing relative uncertainty:
          0.0 = consistent costs (low variance)
          1.0 = highly variable costs (high variance)

        Based on the coefficient of variation across subsystems.
        """
        self._check_fitted()
        ratios = self.get_ratios(transit_mode)
        ratio_values = np.array(list(ratios.values()))

        mean_ratio = np.mean(ratio_values)
        std_ratio = np.std(ratio_values)

        if mean_ratio == 0:
            return 0.5  # Default if no data

        cv = std_ratio / mean_ratio
        # Normalize to [0, 1] using typical CV range [0, 1]
        return float(np.clip(cv, 0, 1))

    def __repr__(self) -> str:
        if not self._fitted:
            return "FTALookup(not fitted)"
        modes = [m for m in self._ratios if m != "global"]
        return (
            f"FTALookup(modes={modes}, "
            f"n_total={self._counts.get('global', 0)})"
        )