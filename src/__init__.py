"""Rail Cost Estimator src module - Clean modular architecture."""

# Config layer
from .config import AppConfig, FileConfig, ModelConfig, UIConfig

# Data layer
from .data import (
    DataLoader,
    DataProcessor,
    FeatureEngineer,
    FTAProcessor,
    GlobalRailProcessor,
)

# Model layer
from .model import (
    ModelEvaluator,
    ModelPipeline,
    ModelTrainer,
    TargetEncoder,
)

# Inference layer
from .inference import (
    FTALookup,
    InferenceService,
    SHAPExplainer,
)

# Pipeline layer
from .pipeline import PipelineOrchestrator

__all__ = [
    # Config
    "AppConfig",
    "FileConfig",
    "ModelConfig",
    "UIConfig",
    # Data
    "DataLoader",
    "DataProcessor",
    "FeatureEngineer",
    "FTAProcessor",
    "GlobalRailProcessor",
    # Model
    "ModelEvaluator",
    "ModelPipeline",
    "ModelTrainer",
    "TargetEncoder",
    # Inference
    "FTALookup",
    "InferenceService",
    "SHAPExplainer",
    # Pipeline
    "PipelineOrchestrator",
]
