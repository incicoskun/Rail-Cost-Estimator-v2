"""
Train the rail cost estimation model from scratch.

This script runs the complete ETL + training pipeline:
  1. Global rail data processing
  2. FTA data processing (with CPI normalization)
  3. Model training with cross-validation

Usage:
    python train.py

Output:
    - rail_cost_model.pkl      (trained GradientBoostingRegressor)
    - memory_package.pkl       (encoders, medians, encoding maps)
    - feature_names.pkl        (feature list for inference)
    - global_rail_processed.csv
    - fta_processed.csv
"""

import sys
from pathlib import Path

from src import AppConfig
from src.pipeline import PipelineOrchestrator


def main():
    """Run complete training pipeline."""
    print("=" * 80)
    print("RAIL COST ESTIMATOR - TRAINING PIPELINE")
    print("=" * 80)
    print()

    try:
        # Initialize configuration
        print("[1/2] Loading configuration...")
        config = AppConfig.from_env()
        print(f"     Base directory: {config.base_dir}")
        print(f"     Config loaded successfully")
        print()

        # Run pipeline
        print("[2/2] Running ETL + Training Pipeline...")
        print()
        orchestrator = PipelineOrchestrator(config)
        orchestrator.run_full_pipeline()

        print()
        print("=" * 80)
        print("TRAINING COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        print()
        print("Output files:")
        print("  [OK] rail_cost_model.pkl       (trained model)")
        print("  [OK] memory_package.pkl        (encoders & maps)")
        print("  [OK] feature_names.pkl         (feature list)")
        print("  [OK] global_rail_processed.csv (processed rail data)")
        print("  [OK] fta_processed.csv         (processed FTA data)")
        print()
        print("Next: Run 'streamlit run app.py' to use the model")
        print()
        return 0

    except FileNotFoundError as e:
        print(f"[ERROR] File not found: {e}")
        print()
        print("Ensure the following raw data files exist:")
        print("  - data/raw/global_rail_costs.xlsx")
        print("  - data/raw/fta_summary.csv")
        return 1

    except Exception as e:
        print(f"[ERROR] Training failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
