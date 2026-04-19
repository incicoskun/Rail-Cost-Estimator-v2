"""Model training and feature engineering configuration."""

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class ModelConfig:
    """Model training and inference configuration."""

    # Cross-validation settings
    cv_folds: int = 5
    random_state: int = 42

    # GBR training parameters
    n_estimators: int = 1200
    learning_rate: float = 0.1
    max_depth: int = 3
    subsample: float = 0.8
    min_samples_leaf: int = 5

    # Loss function config
    loss: str = "quantile"
    alpha: float = 0.5

    # Target variable
    target_col: str = "cost_per_km_2023_musd"

    # Outliers to remove during training
    train_outliers: List[str] = field(default_factory=lambda: [
        "Capital Airport Express",
        "Tozai Line (Sendai)",
    ])

    # Feature engineering
    feature_all: List[str] = field(default_factory=lambda: [
        "country_te", "country_freq", "city_te", "city_freq",
        "tunnel_pct", "station_density", "log_length", "is_regional_rail", "mid_year"
    ])

    # Target encoding smoothing parameters
    encoding_params: Dict[str, int] = field(default_factory=lambda: {
        "country_m": 20,
        "city_m": 30,
    })

    # Feature display names for UI
    feature_display_names: Dict[str, str] = field(default_factory=lambda: {
        "country_te": "National Premium",
        "country_freq": "National Market Size",
        "city_te": "Local Premium",
        "city_freq": "Local Market Size",
        "tunnel_pct": "Tunnel Proportion",
        "station_density": "Station Density",
        "log_length": "Line Length (log)",
        "is_regional_rail": "Is Regional Rail?",
        "mid_year": "Inflation (Mid-Year)",
    })

    # OOF stratification quantiles
    oof_qcut: int = 5

    @property
    def params_dict(self) -> Dict:
        """Return GBR hyperparameters as dict."""
        return {
            "n_estimators": self.n_estimators,
            "learning_rate": self.learning_rate,
            "max_depth": self.max_depth,
            "subsample": self.subsample,
            "min_samples_leaf": self.min_samples_leaf,
            "random_state": self.random_state,
        }

    @property
    def model_config_dict(self) -> Dict:
        """Return model config dict (loss function settings)."""
        return {
            "loss": self.loss,
            "alpha": self.alpha,
        }