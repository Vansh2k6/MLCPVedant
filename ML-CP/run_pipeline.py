"""
run_pipeline.py
===============
Master orchestration script to execute the complete Football Player Market-Value
Prediction project end-to-end:

Step 1: Build the point-in-time analysis table and run data-integrity checks (data_pipeline.py)
Step 2: Exploratory Data Analysis & Sparsity Audit (eda.py)
Step 3, 4, 5: Temporal Split, Baseline Ridge & Main XGBoost Model (models.py)
Step 6, 7: Residual Breakdown & Undervalued Player Rankings (evaluate.py)
Step 8: Refresh generated README.md sections from the outputs (update_readme.py)

Usage:
    python run_pipeline.py             # Corrected upgrades 1–2; outputs in models/upgraded
    python run_pipeline.py --legacy    # Original experiment and README generation
"""

import sys
import time
from src.data_pipeline import build_analysis_dataset, verify_leakage_invariants
from src.eda import run_eda
from src.models import train_and_evaluate
from src.evaluate import run_residual_and_ranking_analysis
from src.update_readme import update_readme


def legacy_main():
    start_all = time.time()
    print("=" * 80)
    print("FOOTBALL PLAYER MARKET-VALUE PREDICTION PIPELINE")
    print("=" * 80)

    # -------------------------------------------------------------
    # Step 1: Build Analysis Table
    # -------------------------------------------------------------
    parquet_path = "data/processed_market_valuations.parquet"
    build_analysis_dataset(
        raw_data_dir=".",
        output_path=parquet_path,
        min_valuation_date="2014-01-01",
    )
    verify_leakage_invariants(parquet_path)

    # -------------------------------------------------------------
    # Step 2: Exploratory Data Analysis
    # -------------------------------------------------------------
    run_eda(
        data_path=parquet_path,
        output_dir="eda_outputs",
    )

    # -------------------------------------------------------------
    # Step 3, 4, 5: Modeling Pipeline (Split, Ridge, XGBoost)
    # -------------------------------------------------------------
    train_and_evaluate(
        data_path=parquet_path,
        models_dir="models",
        eda_dir="eda_outputs",
        tune_hyperparams=True,
    )

    # -------------------------------------------------------------
    # Step 6, 7: Residual Breakdown & Rankings
    # -------------------------------------------------------------
    run_residual_and_ranking_analysis(
        predictions_path="models/test_predictions.parquet",
        models_dir="models",
        eda_dir="eda_outputs",
    )

    # -------------------------------------------------------------
    # Step 8: Refresh README.md Tables From the Outputs Above
    # -------------------------------------------------------------
    update_readme(
        readme_path="README.md",
        models_dir="models",
        data_path=parquet_path,
        raw_data_dir=".",
    )

    elapsed = time.time() - start_all
    print("\n" + "=" * 80)
    print(f"PIPELINE COMPLETED SUCCESSFULLY IN {elapsed:.2f} SECONDS!")
    print("All artifacts generated in data/, eda_outputs/, and models/; README.md refreshed.")
    print("=" * 80)


def main():
    if "--legacy" in sys.argv:
        sys.argv.remove("--legacy")
        legacy_main()
    else:
        from run_upgrade import main as run_upgrade
        run_upgrade()


if __name__ == "__main__":
    main()
