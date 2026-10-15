"""
models.py
=========
Implements Steps 3, 4, and 5 for Football Player Market-Value Prediction.

STEP 3: Temporal Train/Test Split
- Split by valuation date:
    Train set: valuation_date < '2024-01-01'
    Test set:  valuation_date >= '2024-01-01'
- No random train/test split and no standard k-fold. The same player has many valuation dates,
  so a random split would put later valuations of a player in training and earlier ones in test,
  letting the model recall a player's value instead of predicting it.

STEP 4: Baselines
- Regularized Ridge Regression on ln(market_value_in_eur) using age, age^2,
  one-hot position, league tier, and rolling performance features (minutes, goal contributions per 90).
- Naive baseline: training-set median log value by position x league tier, falling back to
  the position median and then the global median for groups with fewer than 30 training rows.

STEP 5: Main Model
- XGBoost Regressor on ln(market_value_in_eur) using demographics, physicals, league tier,
  competition, rolling 365d performance, position-specific interaction terms, and career accumulation.
- Leakage ablation: Ridge and XGBoost are each trained twice, with the full feature set and with
  SNAPSHOT_FEATURES removed. The clean variants are the headline models; the full variants are
  reported only in `models/leakage_ablation.csv`.
- max_depth is chosen with TimeSeriesSplit (n_splits=3) on the training set only, so the test
  period is never used for model selection.
- All log predictions are clipped to [LOG_PRED_FLOOR, LOG_PRED_CEILING] before any metric is
  computed, so log and euro metrics are based on the same predictions.
- Feature importance extraction and visualization saved to `eda_outputs/feature_importances.png`.
- Duan smearing factor from the last TimeSeriesSplit validation fold, saved to
  `models/smearing_factor.json`, gives smear-corrected euro predictions alongside the uncorrected ones.
- Side-by-side metric comparison table saved to `models/model_comparison_metrics.csv`.
- Predictions persisted to `models/test_predictions.parquet` for downstream residual analysis and ranking.
"""

import os
import sys
import argparse
import json
import random
import time
import duckdb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from sklearn.linear_model import Ridge
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import TimeSeriesSplit
import xgboost as xgb

SEED = 42
LOG_PRED_FLOOR = 8.5
LOG_PRED_CEILING = np.log(250_000_000)
NAIVE_MIN_GROUP_ROWS = 30

# Features whose values come from a snapshot taken after the valuation date (Phase 2.5 audit).
# The headline ("clean") models drop them; the "full" variants keep them for the leakage ablation.
SNAPSHOT_FEATURES = [
    # Days from the valuation date to the expiry of the player's *current* contract in players.csv,
    # not the contract in force at the valuation, so old valuations see contracts signed later.
    "contract_days_remaining",
    # Whether players.csv has a current contract expiry date; this partly encodes whether the
    # player is still under contract at the snapshot date.
    "has_contract_info",
    # Squad size from the current row of clubs.csv (one row per club), not the squad at the valuation.
    "club_squad_size",
    # Share of foreign players from the current row of clubs.csv.
    "club_foreigners_percentage",
    # Number of current national-team players from the current row of clubs.csv.
    "club_national_team_players",
]

FEATURE_SET_FULL = "Full (with snapshot features)"
FEATURE_SET_CLEAN = "Clean (snapshot features dropped)"


def clip_log_predictions(preds_log):
    return np.clip(preds_log, LOG_PRED_FLOOR, LOG_PRED_CEILING)


def without_snapshot_features(columns):
    return [c for c in columns if c not in SNAPSHOT_FEATURES]


def regression_metrics(model_name, y_true_log, y_pred_log, y_true_eur, y_pred_eur):
    """Log-space and euro-space RMSE/MAE for one set of test predictions."""
    return {
        "Model": model_name,
        "Log RMSE": float(np.sqrt(np.mean((y_true_log - y_pred_log)**2))),
        "Log MAE": float(np.mean(np.abs(y_true_log - y_pred_log))),
        "Euro RMSE": float(np.sqrt(np.mean((y_true_eur - y_pred_eur)**2))),
        "Euro MAE": float(np.mean(np.abs(y_true_eur - y_pred_eur))),
    }


def relative_improvement(reference: dict, candidate: dict) -> dict:
    """Percentage error reduction of candidate vs reference for each metric."""
    return {
        metric: (reference[metric] - candidate[metric]) / reference[metric] * 100
        for metric in ["Log RMSE", "Log MAE", "Euro RMSE", "Euro MAE"]
    }


def print_metrics(title, metrics):
    print(f"{title}:")
    print(f"  Log RMSE:  {metrics['Log RMSE']:.4f}")
    print(f"  Log MAE:   {metrics['Log MAE']:.4f}")
    print(f"  Euro RMSE: €{metrics['Euro RMSE']:,.2f}")
    print(f"  Euro MAE:  €{metrics['Euro MAE']:,.2f}\n")


def fit_naive_group_median(train_df: pd.DataFrame, y_train_log: np.ndarray, min_rows: int = NAIVE_MIN_GROUP_ROWS):
    """
    Fits medians of training log value by position x league_tier, by position, and globally.
    Groups with fewer than `min_rows` training rows are dropped so prediction falls back
    to the next coarser level.
    """
    train = pd.DataFrame({
        "position": train_df["position"].values,
        "league_tier": train_df["league_tier"].values,
        "y_log": y_train_log,
    })
    pair = train.groupby(["position", "league_tier"])["y_log"].agg(["median", "count"])
    pos = train.groupby("position")["y_log"].agg(["median", "count"])
    return {
        "pair": pair.loc[pair["count"] >= min_rows, "median"],
        "position": pos.loc[pos["count"] >= min_rows, "median"],
        "global": float(np.median(y_train_log)),
    }


def predict_naive_group_median(medians: dict, df: pd.DataFrame) -> np.ndarray:
    pair_index = pd.MultiIndex.from_arrays([df["position"], df["league_tier"]])
    preds = pd.Series(medians["pair"].reindex(pair_index).values, index=df.index)
    preds = preds.fillna(df["position"].map(medians["position"]))
    preds = preds.fillna(medians["global"])
    return preds.values.astype(float)


def setup_plot_style():
    """Configures clean matplotlib aesthetics."""
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Segoe UI", "DejaVu Sans", "Helvetica", "Arial"],
        "axes.edgecolor": "#cccccc",
        "axes.linewidth": 0.8,
        "grid.color": "#e5e5e5",
        "grid.linestyle": "--",
        "grid.linewidth": 0.5,
    })


def prepare_features(df: pd.DataFrame, train_mask: pd.Series):
    """
    Constructs the feature matrix for baseline and gradient boosting models.
    Applies domain-informed regularization (e.g. clipping per-90 rates at realistic max)
    and constructs position-specific interaction terms.
    Imputation statistics (medians) are computed on the rows selected by `train_mask` only
    and applied to every row, so test rows never influence the values used to fill gaps.
    """
    df = df.copy()
    train = df.loc[train_mask]

    # 1. Per-90 rate regularization:
    # Small minutes played can inflate per-90 stats (e.g., 1 goal in 2 mins = 45.0 goals/90).
    # We clip per-90 metrics to realistic domain boundaries (max 3.0 per 90).
    df["goals_per_90_clipped"] = df["goals_per_90_365d"].clip(0, 3.0)
    df["assists_per_90_clipped"] = df["assists_per_90_365d"].clip(0, 3.0)
    df["goal_contributions_per_90_clipped"] = df["goal_contributions_per_90_365d"].clip(0, 3.0)

    # 2. Position Indicators & Interaction Terms (Handling Sparsity Without Blanket Imputation)
    df["is_attack"] = (df["position"] == "Attack").astype(int)
    df["is_midfield"] = (df["position"] == "Midfield").astype(int)
    df["is_defender"] = (df["position"] == "Defender").astype(int)
    df["is_goalkeeper"] = (df["position"] == "Goalkeeper").astype(int)

    # Positional interactions: attacking output is most salient for attackers/midfielders,
    # while minutes played is critical for goalkeepers and defenders.
    df["attack_goals_per_90"] = df["goals_per_90_clipped"] * df["is_attack"]
    df["midfield_assists_per_90"] = df["assists_per_90_clipped"] * df["is_midfield"]
    df["gk_minutes_last_365d"] = df["minutes_last_365d"] * df["is_goalkeeper"]
    df["def_minutes_last_365d"] = df["minutes_last_365d"] * df["is_defender"]

    # 3. Physicals & Contract fields
    train_height_medians = train.groupby("position")["height_in_cm"].median()
    df["height_in_cm"] = df["height_in_cm"].fillna(df["position"].map(train_height_medians)).fillna(182.0)
    df["contract_days_remaining"] = df["contract_days_remaining"].fillna(-1).clip(lower=-1, upper=3650)
    df["has_contract_info"] = df["has_contract_info"].fillna(0).astype(int)

    # 4. Club & League fillna
    df["club_squad_size"] = df["club_squad_size"].fillna(train["club_squad_size"].median())
    df["club_foreigners_percentage"] = df["club_foreigners_percentage"].fillna(train["club_foreigners_percentage"].median())
    df["club_national_team_players"] = df["club_national_team_players"].fillna(0)

    return df


def fit_ridge(df, train_mask, test_mask, num_cols, cat_cols, y_train_log):
    """Fits the Ridge baseline on the given columns; returns clipped test log predictions."""
    preprocessor = ColumnTransformer(transformers=[
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), cat_cols),
    ])
    model = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", Ridge(alpha=100.0, random_state=42)),
    ])
    model.fit(df.loc[train_mask, num_cols + cat_cols], y_train_log)
    return clip_log_predictions(model.predict(df.loc[test_mask, num_cols + cat_cols]))


def select_xgb_depth(X_train, y_train_log, tune_hyperparams):
    """Chooses max_depth by mean TimeSeriesSplit(n_splits=3) log RMSE on the training set."""
    if not tune_hyperparams:
        return 7
    tscv = TimeSeriesSplit(n_splits=3)
    candidate_depths = [6, 7, 8]
    best_depth = 7
    best_cv_rmse = float("inf")

    for depth in candidate_depths:
        cv_scores = []
        for fold, (train_idx, val_idx) in enumerate(tscv.split(X_train)):
            X_tr, y_tr = X_train.iloc[train_idx], y_train_log[train_idx]
            X_val, y_val = X_train.iloc[val_idx], y_train_log[val_idx]

            test_xgb = xgb.XGBRegressor(
                n_estimators=100,
                max_depth=depth,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=SEED,
                n_jobs=-1
            )
            test_xgb.fit(X_tr, y_tr)
            preds_val = test_xgb.predict(X_val)
            fold_rmse = np.sqrt(np.mean((y_val - preds_val)**2))
            cv_scores.append(fold_rmse)

        mean_cv_rmse = np.mean(cv_scores)
        print(f"  - max_depth={depth}: Mean TimeSeriesSplit CV Log RMSE = {mean_cv_rmse:.4f}")
        if mean_cv_rmse < best_cv_rmse:
            best_cv_rmse = mean_cv_rmse
            best_depth = depth

    print(f"Optimal max_depth selected from TimeSeriesSplit: {best_depth}")
    return best_depth


def main_xgb_params(depth):
    return dict(
        n_estimators=300,
        max_depth=depth,
        learning_rate=0.08,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=SEED,
        n_jobs=-1,
    )


def train_and_evaluate(
    data_path: str = "data/processed_market_valuations.parquet",
    models_dir: str = "models",
    eda_dir: str = "eda_outputs",
    tune_hyperparams: bool = True,
):
    """
    Executes Steps 3, 4, and 5:
    - Temporal Split
    - Naive and Ridge baselines (Ridge in full and clean variants)
    - XGBoost Regressor with TimeSeriesSplit (full and clean variants)
    - Leakage ablation table, feature importance plot, side-by-side metrics
    """
    print("=" * 70)
    print("STEPS 3, 4 & 5: TEMPORAL SPLIT, BASELINES & XGBOOST MAIN MODEL")
    print("=" * 70)
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(eda_dir, exist_ok=True)
    setup_plot_style()
    random.seed(SEED)
    np.random.seed(SEED)

    # -------------------------------------------------------------
    # STEP 3: TEMPORAL TRAIN/TEST SPLIT
    # -------------------------------------------------------------
    print("\n--- STEP 3: TEMPORAL TRAIN/TEST SPLIT ---")
    print("Loading analysis table...")
    con = duckdb.connect()
    # player_id breaks ties between same-day valuations; without it DuckDB's multithreaded scan
    # returns them in a different order each run, which shifts CV folds and XGBoost row sampling.
    raw_df = con.execute(
        f"SELECT * FROM '{data_path.replace('\\', '/')}' ORDER BY valuation_date, player_id"
    ).df()

    split_date = "2024-01-01"
    train_mask = raw_df["valuation_date"] < split_date
    test_mask = raw_df["valuation_date"] >= split_date

    df = prepare_features(raw_df, train_mask)

    n_train = train_mask.sum()
    n_test = test_mask.sum()

    print(f"Split Date Boundary: {split_date}")
    print(f"Training observations (valuations < {split_date}):  {n_train:,} records")
    print(f"Testing observations  (valuations >= {split_date}): {n_test:,} records")
    print("Training rows are valuations dated before the split date; test rows are dated on or after it.\n")

    # Define targets
    y_train_eur = df.loc[train_mask, "market_value_in_eur"].values
    y_test_eur = df.loc[test_mask, "market_value_in_eur"].values
    y_train_log = np.log(y_train_eur)
    y_test_log = np.log(y_test_eur)

    def test_metrics(name, preds_log_clipped, eur_scale=1.0):
        return regression_metrics(
            name, y_test_log, preds_log_clipped, y_test_eur, np.exp(preds_log_clipped) * eur_scale
        )

    # -------------------------------------------------------------
    # STEP 4: BASELINE MODEL (RIDGE REGRESSION), FULL AND CLEAN
    # -------------------------------------------------------------
    print("--- STEP 4: BASELINE MODEL (RIDGE REGRESSION) ---")
    base_num_cols = ["age", "age_squared", "minutes_last_365d", "goal_contributions_per_90_clipped"]
    base_cat_cols = ["position", "league_tier"]
    ridge_variants = {
        FEATURE_SET_FULL: (base_num_cols, base_cat_cols),
        FEATURE_SET_CLEAN: (without_snapshot_features(base_num_cols), without_snapshot_features(base_cat_cols)),
    }

    ridge_preds_log = {}
    ridge_metrics = {}
    for feature_set, (num_cols, cat_cols) in ridge_variants.items():
        print(f"Fitting Ridge regression ({feature_set}) on {len(num_cols) + len(cat_cols)} input columns...")
        t_start = time.time()
        ridge_preds_log[feature_set] = fit_ridge(df, train_mask, test_mask, num_cols, cat_cols, y_train_log)
        print(f"Ridge training complete in {time.time() - t_start:.2f} seconds.")
        ridge_metrics[feature_set] = test_metrics("Baseline Ridge Regression", ridge_preds_log[feature_set])
        print_metrics(f"Ridge Test Evaluation ({feature_set})", ridge_metrics[feature_set])

    # -------------------------------------------------------------
    # NAIVE BASELINE (POSITION x LEAGUE TIER MEDIAN)
    # -------------------------------------------------------------
    print("--- NAIVE BASELINE (POSITION x LEAGUE TIER MEDIAN) ---")
    naive_medians = fit_naive_group_median(df.loc[train_mask], y_train_log)
    naive_preds_log = clip_log_predictions(predict_naive_group_median(naive_medians, df.loc[test_mask]))
    print(
        f"Groups with >= {NAIVE_MIN_GROUP_ROWS} training rows: {len(naive_medians['pair'])} position x tier, "
        f"{len(naive_medians['position'])} position-only; global median log value = {naive_medians['global']:.4f}"
    )
    naive_metrics = test_metrics("Naive Baseline (Position x Tier Median)", naive_preds_log)
    print_metrics("Naive Baseline Test Evaluation", naive_metrics)

    # -------------------------------------------------------------
    # STEP 5: MAIN MODEL (XGBOOST REGRESSOR), FULL AND CLEAN
    # -------------------------------------------------------------
    print("--- STEP 5: MAIN MODEL (XGBOOST REGRESSOR) ---")
    full_numeric_cols = [
        "age", "age_squared", "height_in_cm",
        "contract_days_remaining", "has_contract_info",
        "minutes_last_365d", "appearances_last_365d",
        "goals_last_365d", "assists_last_365d",
        "yellow_cards_last_365d", "red_cards_last_365d",
        "goals_per_90_clipped", "assists_per_90_clipped", "goal_contributions_per_90_clipped",
        "minutes_per_appearance_365d",
        "career_appearances_prior", "career_minutes_prior", "career_goals_prior", "career_assists_prior",
        "club_squad_size", "club_foreigners_percentage", "club_national_team_players",
        "is_attack", "is_midfield", "is_defender", "is_goalkeeper",
        "attack_goals_per_90", "midfield_assists_per_90", "gk_minutes_last_365d", "def_minutes_last_365d"
    ]

    full_cat_cols = ["position", "sub_position", "league_tier", "competition_id", "foot"]

    print("One-hot encoding categorical features for XGBoost...")
    df_encoded = pd.get_dummies(df[full_numeric_cols + full_cat_cols], columns=full_cat_cols, drop_first=True)
    xgb_variants = {
        FEATURE_SET_FULL: df_encoded.columns.tolist(),
        FEATURE_SET_CLEAN: without_snapshot_features(df_encoded.columns.tolist()),
    }
    dropped = sorted(set(xgb_variants[FEATURE_SET_FULL]) - set(xgb_variants[FEATURE_SET_CLEAN]))
    print(f"Total features constructed: {len(xgb_variants[FEATURE_SET_FULL])} full, "
          f"{len(xgb_variants[FEATURE_SET_CLEAN])} clean (dropped: {', '.join(dropped)})")

    xgb_results = {}
    for feature_set, features in xgb_variants.items():
        print(f"\n=== XGBoost variant: {feature_set} ({len(features)} features) ===")
        X_train = df_encoded.loc[train_mask, features]
        X_test = df_encoded.loc[test_mask, features]

        print("Validating hyperparameters via TimeSeriesSplit(n_splits=3) on training set...")
        best_depth = select_xgb_depth(X_train, y_train_log, tune_hyperparams)

        print(f"Training XGBoost Regressor (n_estimators=300, max_depth={best_depth}, lr=0.08)...")
        t_start_xgb = time.time()
        model = xgb.XGBRegressor(**main_xgb_params(best_depth))
        model.fit(X_train, y_train_log)
        print(f"XGBoost training complete in {time.time() - t_start_xgb:.2f} seconds.")

        preds_log = clip_log_predictions(model.predict(X_test))
        metrics = test_metrics("Main XGBoost Regressor", preds_log)
        print_metrics(f"XGBoost Test Evaluation ({feature_set})", metrics)
        xgb_results[feature_set] = dict(
            model=model, depth=best_depth, X_train=X_train, preds_log=preds_log, metrics=metrics,
        )

    clean = xgb_results[FEATURE_SET_CLEAN]
    main_xgb, xgb_preds_log, best_depth, X_train = clean["model"], clean["preds_log"], clean["depth"], clean["X_train"]

    # Duan smearing factor for converting log predictions back to euros.
    # exp(log prediction) estimates the conditional median, not the mean, so euro predictions
    # run low; multiplying by mean(exp(residual_log)) corrects this. The residuals must be
    # out-of-sample: in-sample training residuals are shrunk by overfitting and would understate
    # the correction. We refit the headline model configuration on the training portion of the
    # last TimeSeriesSplit fold and score its chronologically later validation portion, which
    # is the closest stand-in for the test period that uses training data only.
    print("\nEstimating Duan smearing factor for the clean model on the last TimeSeriesSplit validation fold...")
    smear_train_idx, smear_val_idx = list(TimeSeriesSplit(n_splits=3).split(X_train))[-1]
    smear_xgb = xgb.XGBRegressor(**main_xgb_params(best_depth))
    smear_xgb.fit(X_train.iloc[smear_train_idx], y_train_log[smear_train_idx])
    smear_val_preds_log = clip_log_predictions(smear_xgb.predict(X_train.iloc[smear_val_idx]))
    smear_residuals_log = y_train_log[smear_val_idx] - smear_val_preds_log
    smearing_factor = float(np.mean(np.exp(smear_residuals_log)))
    smear_val_dates = pd.to_datetime(df.loc[train_mask, "valuation_date"].iloc[smear_val_idx])
    print(f"Smearing factor = {smearing_factor:.6f} "
          f"(validation fold: {len(smear_val_idx):,} rows, "
          f"{smear_val_dates.min():%Y-%m-%d} to {smear_val_dates.max():%Y-%m-%d})")

    smearing_path = os.path.join(models_dir, "smearing_factor.json")
    with open(smearing_path, "w") as f:
        json.dump({
            "smearing_factor": smearing_factor,
            "method": "Duan smearing: mean(exp(residual_log)) on out-of-sample residuals",
            "source": "last TimeSeriesSplit(n_splits=3) validation fold of the training set",
            "feature_set": FEATURE_SET_CLEAN,
            "fold_train_rows": int(len(smear_train_idx)),
            "fold_validation_rows": int(len(smear_val_idx)),
            "fold_validation_start": f"{smear_val_dates.min():%Y-%m-%d}",
            "fold_validation_end": f"{smear_val_dates.max():%Y-%m-%d}",
            "fold_validation_mean_residual_log": float(np.mean(smear_residuals_log)),
            "max_depth": int(best_depth),
        }, f, indent=4)
    print(f"Saved smearing factor to: {smearing_path}")

    # Save headline model artifact
    model_json_path = os.path.join(models_dir, "xgboost_market_value_model.json")
    main_xgb.save_model(model_json_path)
    print(f"Saved trained clean XGBoost model to: {model_json_path}")

    xgb_preds_eur = np.exp(xgb_preds_log)
    xgb_preds_eur_smeared = xgb_preds_eur * smearing_factor

    baseline_metrics = ridge_metrics[FEATURE_SET_CLEAN]
    main_metrics = clean["metrics"]
    # Smearing only rescales euro predictions; log-space metrics equal the uncorrected model's.
    smeared_metrics = test_metrics("Main XGBoost Regressor (Smear-Corrected)", xgb_preds_log, smearing_factor)

    improvement_vs_ridge = relative_improvement(baseline_metrics, main_metrics)
    improvement_vs_naive = relative_improvement(naive_metrics, main_metrics)

    def as_pct_row(label, improvement):
        return {"Model": label, **{k: f"{v:.2f}%" for k, v in improvement.items()}}

    # -------------------------------------------------------------
    # SIDE-BY-SIDE METRICS COMPARISON (HEADLINE = CLEAN MODELS)
    # -------------------------------------------------------------
    metrics_df = pd.DataFrame([
        naive_metrics,
        baseline_metrics,
        main_metrics,
        smeared_metrics,
        as_pct_row("% Improvement (XGBoost vs Ridge Baseline)", improvement_vs_ridge),
        as_pct_row("% Improvement (XGBoost vs Naive Baseline)", improvement_vs_naive),
    ])
    print("\n" + "=" * 70)
    print("SIDE-BY-SIDE MODEL EVALUATION ON TEST SET (VALUATIONS >= 2024-01-01, CLEAN FEATURE SET)")
    print("=" * 70)
    print(metrics_df.to_string(index=False))
    print("=" * 70 + "\n")

    metrics_csv_path = os.path.join(models_dir, "model_comparison_metrics.csv")
    metrics_json_path = os.path.join(models_dir, "model_comparison_metrics.json")
    metrics_df.to_csv(metrics_csv_path, index=False)
    with open(metrics_json_path, "w") as f:
        json.dump({
            "feature_set": FEATURE_SET_CLEAN,
            "models": [naive_metrics, baseline_metrics, main_metrics, smeared_metrics],
            "xgboost_pct_improvement": {
                "vs_ridge_baseline": improvement_vs_ridge,
                "vs_naive_baseline": improvement_vs_naive,
            },
            "smearing_factor": smearing_factor,
        }, f, indent=4)
    print(f"Saved metric tables to: {metrics_csv_path} and {metrics_json_path}")

    # -------------------------------------------------------------
    # LEAKAGE ABLATION (FULL VS CLEAN FEATURE SETS)
    # -------------------------------------------------------------
    metric_names = ["Log RMSE", "Log MAE", "Euro RMSE", "Euro MAE"]
    xgb_full_metrics = xgb_results[FEATURE_SET_FULL]["metrics"]
    ablation_rows = [
        {**naive_metrics, "Feature Set": "Position x league tier medians (no model features)"},
        {**ridge_metrics[FEATURE_SET_FULL], "Feature Set": FEATURE_SET_FULL},
        {**ridge_metrics[FEATURE_SET_CLEAN], "Feature Set": FEATURE_SET_CLEAN},
        {**xgb_full_metrics, "Feature Set": FEATURE_SET_FULL},
        {**main_metrics, "Feature Set": FEATURE_SET_CLEAN},
        {
            "Model": "% Change (XGBoost, clean vs full; positive = worse)",
            "Feature Set": "Clean vs Full",
            **{m: f"{(main_metrics[m] - xgb_full_metrics[m]) / xgb_full_metrics[m] * 100:.2f}%" for m in metric_names},
        },
    ]
    ablation_df = pd.DataFrame(ablation_rows)[["Model", "Feature Set"] + metric_names]
    ablation_path = os.path.join(models_dir, "leakage_ablation.csv")
    ablation_df.to_csv(ablation_path, index=False)
    print("\nLEAKAGE ABLATION (snapshot features: " + ", ".join(SNAPSHOT_FEATURES) + ")")
    print(f"XGBoost max_depth chosen: full={xgb_results[FEATURE_SET_FULL]['depth']}, clean={best_depth}")
    print(ablation_df.to_string(index=False))
    print(f"Saved leakage ablation table to: {ablation_path}")

    # -------------------------------------------------------------
    # FEATURE IMPORTANCES PLOT (CLEAN MODEL)
    # -------------------------------------------------------------
    print("\nExtracting feature importances...")
    booster = main_xgb.get_booster()
    importance_gain = booster.get_score(importance_type="gain")

    imp_df = pd.DataFrame(list(importance_gain.items()), columns=["feature", "gain"])
    imp_df = imp_df.sort_values("gain", ascending=False).reset_index(drop=True)
    imp_csv_path = os.path.join(models_dir, "feature_importances.csv")
    imp_df.to_csv(imp_csv_path, index=False)

    top20_imp = imp_df.head(20).iloc[::-1]

    plt.figure(figsize=(12, 8))
    plt.barh(top20_imp["feature"], top20_imp["gain"], color="#1f77b4", edgecolor="white")
    plt.xlabel("Average Gain per Split (Importance)", fontsize=11, weight="bold")
    plt.title("XGBoost Regressor (Clean Feature Set): Top 20 Most Influential Features", fontsize=13, weight="bold", pad=12)
    plt.gca().xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f"{x:,.0f}"))
    plt.tight_layout()

    feat_imp_plot_path = os.path.join(eda_dir, "feature_importances.png")
    plt.savefig(feat_imp_plot_path, dpi=300)
    plt.close()
    print(f"Saved feature importance plot to: {feat_imp_plot_path}")

    # -------------------------------------------------------------
    # PERSIST PREDICTIONS FOR STEPS 6 & 7 (CLEAN MODELS)
    # -------------------------------------------------------------
    print("\nPersisting test-set predictions for Steps 6 and 7...")
    test_results_df = df.loc[test_mask, [
        "player_id", "player_name", "valuation_date", "age",
        "position", "sub_position", "league_tier", "competition_id",
        "club_name", "minutes_last_365d", "goals_last_365d", "assists_last_365d"
    ]].copy()

    test_results_df["actual_value_eur"] = y_test_eur
    test_results_df["log_actual"] = y_test_log
    test_results_df["baseline_pred_eur"] = np.exp(ridge_preds_log[FEATURE_SET_CLEAN])
    test_results_df["baseline_pred_log"] = ridge_preds_log[FEATURE_SET_CLEAN]
    test_results_df["naive_pred_eur"] = np.exp(naive_preds_log)
    test_results_df["naive_pred_log"] = naive_preds_log
    test_results_df["predicted_value_eur"] = xgb_preds_eur
    test_results_df["predicted_value_eur_smeared"] = xgb_preds_eur_smeared
    test_results_df["log_predicted"] = xgb_preds_log

    # Errors
    test_results_df["residual_eur"] = test_results_df["actual_value_eur"] - test_results_df["predicted_value_eur"]
    test_results_df["abs_residual_eur"] = np.abs(test_results_df["residual_eur"])
    test_results_df["residual_eur_smeared"] = test_results_df["actual_value_eur"] - test_results_df["predicted_value_eur_smeared"]
    test_results_df["abs_residual_eur_smeared"] = np.abs(test_results_df["residual_eur_smeared"])
    test_results_df["residual_log"] = test_results_df["log_actual"] - test_results_df["log_predicted"]
    test_results_df["abs_residual_log"] = np.abs(test_results_df["residual_log"])
    test_results_df["value_delta_eur"] = test_results_df["predicted_value_eur"] - test_results_df["actual_value_eur"]

    test_preds_path = os.path.join(models_dir, "test_predictions.parquet")
    test_results_df.to_parquet(test_preds_path, index=False)
    print(f"Saved test predictions table ({len(test_results_df):,} rows) to: {test_preds_path}")

    print("\n--- Phase 3 Modeling Execution Complete ---")
    return metrics_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Baseline and Main Models for football market value prediction.")
    parser.add_argument("--data", type=str, default="data/processed_market_valuations.parquet", help="Path to processed parquet file.")
    parser.add_argument("--models-dir", type=str, default="models", help="Directory to save models and metrics.")
    parser.add_argument("--eda-dir", type=str, default="eda_outputs", help="Directory to save plots.")
    parser.add_argument("--no-tune", action="store_true", help="Skip TimeSeriesSplit hyperparameter search.")

    args = parser.parse_args()
    train_and_evaluate(
        data_path=args.data,
        models_dir=args.models_dir,
        eda_dir=args.eda_dir,
        tune_hyperparams=not args.no_tune,
    )
