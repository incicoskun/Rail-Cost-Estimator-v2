"""Unified inference service combining model, memory, and FTA lookup."""

from typing import Dict

import numpy as np
import pandas as pd

from ..config import ModelConfig, UIConfig
from ..data.preprocessor import FeatureEngineer
from .explainer import SHAPExplainer


class InferenceService:
    """Unified prediction interface (model + memory + FTA lookup)."""

    def __init__(self, model, memory: Dict, fta_lookup, config: ModelConfig, ui_config: UIConfig):
        """Initialize inference service.

        Args:
            model: Trained GradientBoostingRegressor
            memory: Memory package dict with encoding maps
            fta_lookup: FTALookup instance
            config: ModelConfig instance
            ui_config: UIConfig instance (for similarity weights and clamp values)
        """
        self.model = model
        self.memory = memory
        self.fta_lookup = fta_lookup
        self.config = config
        self.ui_config = ui_config
        self._shap_explainer = SHAPExplainer(model, config)

    def predict(self, inputs: Dict) -> Dict:
        """Run inference with user inputs.

        Args:
            inputs: Dict with keys:
                - country, city
                - length_km, tunnel_pct, num_stations
                - start_year, end_year
                - is_regional, transit_mode

        Returns:
            Dict with pred, lo, hi, plus input fields
        """
        gm = self.memory["global_mean"]
        tm = self.memory["train_medians"]

        # Build raw feature dict
        raw_df = pd.DataFrame([{
            "country": inputs["country"],
            "city": inputs["city"],
            "length_km": inputs["length_km"],
            "tunnel_pct": inputs["tunnel_pct"],
            "num_stations": inputs["num_stations"] if inputs["num_stations"] > 0 else tm["num_stations"],
            "start_year": inputs["start_year"],
            "end_year": inputs["end_year"],
            "is_regional_rail": 1.0 if inputs["is_regional"] else 0.0,
        }])

        # Feature engineering
        raw_df = FeatureEngineer.apply_base_features(raw_df)

        # Target encoding — city_te uses 2-level fallback: city → country → global_mean
        raw_df["country_te"] = self.memory["country_te_map"].get(inputs["country"], gm)
        raw_df["country_freq"] = self.memory["country_freq_map"].get(inputs["country"], 1)
        raw_df["city_te"] = self.memory["city_te_map"].get(
            inputs["city"], self.memory["country_te_map"].get(inputs["country"], gm)
        )
        raw_df["city_freq"] = self.memory["city_freq_map"].get(inputs["city"], 1)

        # Prediction
        pred_log = self.model.predict(raw_df[self.config.feature_all])[0]
        pred = np.exp(pred_log)

        # Uncertainty bands
        report = self.memory.get("report", {})
        q10_mult = report.get("q10_multiplier", 0.65)
        q90_mult = report.get("q90_multiplier", 1.45)

        return {
            "pred": pred,
            "lo": pred * q10_mult,
            "hi": pred * q90_mult,
            **inputs,
        }

    def find_similar_projects(self, train_df: pd.DataFrame, inputs: Dict, n: int = 8) -> pd.DataFrame:
        """Find similar projects by similarity scoring.

        Args:
            train_df: Pre-loaded and cached training DataFrame
            inputs: Prediction inputs (country, city, length_km, tunnel_pct)
            n: Number of projects to return

        Returns:
            DataFrame of n most similar projects
        """
        weights = self.ui_config.similarity_weights
        min_clamp = self.ui_config.min_length_clamp

        d = train_df.copy()
        d["loc_match"] = (
            (d["country"] == inputs["country"]) & (d["city"] == inputs["city"])
        ).astype(float)
        d["country_match"] = (d["country"] == inputs["country"]).astype(float)
        d["len_diff"] = np.abs(
            np.log(d["length_km"] + min_clamp) - np.log(inputs["length_km"] + min_clamp)
        )
        d["tun_diff"] = np.abs(d["tunnel_pct"] - inputs["tunnel_pct"])

        d["score"] = (
            d["loc_match"] * weights["loc_match"]
            + d["country_match"] * weights["country_match"]
            + 1.0 / (1.0 + d["len_diff"] * weights["len_diff_scale"])
            + 1.0 / (1.0 + d["tun_diff"] * weights["tun_diff_scale"])
        )

        cols = [
            "score", "country", "city", "line", "start_year", "end_year",
            "length_km", "tunnel_pct", "num_stations", "cost_per_km_2023_musd"
        ]

        return d.nlargest(n, "score")[cols].reset_index(drop=True)

    def get_breakdown(self, inputs: Dict) -> Dict:
        """Compute subsystem cost breakdown with uncertainty intervals.

        Args:
            inputs: Result dict containing pred, transit_mode, and memory report

        Returns:
            Dict with subsystem breakdown intervals from FTALookup,
            plus n_projects for the given mode. Falls back to
            UIConfig.default_subsystem_ratios if FTALookup unavailable.
        """
        pred = inputs["pred"]
        mode = inputs.get("transit_mode", "HRT")

        report = self.memory.get("report", {})
        q10 = report.get("q10_multiplier", 0.65)
        q90 = report.get("q90_multiplier", 1.45)

        if self.fta_lookup is not None:
            bdi = self.fta_lookup.get_breakdown_interval(
                pred_cost_per_km=pred,
                transit_mode=mode,
                q10_multiplier=q10,
                q90_multiplier=q90,
            )
            n_projects = self.fta_lookup.n_projects(mode)
        else:
            # Fallback to default ratios from UIConfig
            bdi = {
                k: {"point": pred * v, "lo": pred * v * q10, "hi": pred * v * q90}
                for k, v in self.ui_config.default_subsystem_ratios.items()
            }
            n_projects = 0

        return {"breakdown": bdi, "n_projects": n_projects}

    def get_market_context(self, inputs: Dict) -> Dict:
        """Return market context signals for display.

        Args:
            inputs: Dict with country and city keys

        Returns:
            Dict with city_baseline, country_baseline, sample_density
        """
        gm = self.memory["global_mean"]
        city_val = self.memory["city_te_map"].get(
            inputs["city"], self.memory["country_te_map"].get(inputs["country"], gm)
        )
        country_val = self.memory["country_te_map"].get(inputs["country"], gm)
        sample_density = self.memory["city_freq_map"].get(inputs["city"], 0)

        return {
            "city_baseline": city_val,
            "country_baseline": country_val,
            "sample_density": sample_density,
        }

    def explain(self, inputs: Dict) -> pd.DataFrame:
        """Generate SHAP cost driver explanation for given inputs.

        Args:
            inputs: Same dict as predict() — country, city, length_km, etc.

        Returns:
            impact_df from SHAPExplainer.explain() — ready for plotting
        """
        gm = self.memory["global_mean"]
        tm = self.memory["train_medians"]

        raw_df = pd.DataFrame([{
            "length_km": inputs["length_km"],
            "tunnel_pct": inputs["tunnel_pct"],
            "num_stations": inputs["num_stations"] if inputs["num_stations"] > 0 else tm["num_stations"],
            "start_year": inputs["start_year"],
            "end_year": inputs["end_year"],
            "is_regional_rail": 1.0 if inputs["is_regional"] else 0.0,
        }])

        raw_df = FeatureEngineer.apply_base_features(raw_df)
        raw_df["country_te"] = self.memory["country_te_map"].get(inputs["country"], gm)
        raw_df["country_freq"] = self.memory["country_freq_map"].get(inputs["country"], 1)
        raw_df["city_te"] = self.memory["city_te_map"].get(
            inputs["city"], self.memory["country_te_map"].get(inputs["country"], gm)
        )
        raw_df["city_freq"] = self.memory["city_freq_map"].get(inputs["city"], 1)

        return self._shap_explainer.explain(raw_df)