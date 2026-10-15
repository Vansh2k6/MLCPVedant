"""
evaluate.py
===========
Implements Steps 6 and 7 for Football Player Market-Value Prediction.

STEP 6: Residual Analysis
- Stratifies model prediction errors by:
    1. Age Band (<21, 21-24, 25-28, 29-32, 33+)
    2. Position Group (Goalkeeper, Defender, Midfield, Attack)
    3. League Tier (Tier 1 - Big 5, Tier 2 - Top European, Tier 3 - Other First Tier, Tier 4 - Lower/Other)
- Calculates Mean Error (Bias = actual - predicted), MAE, and RMSE in both euros and log-space.
- Produces `eda_outputs/residuals_by_segment.png`.
- Persists tables to `models/residual_analysis_*.csv`.

STEP 7: Undervalued & Overvalued Player Ranking
- Filters the test period to each player's most recent valuation date.
- Calculates delta = (predicted_value - actual_value).
- Generates Top 20 "Undervalued" (predicted >> actual) and Top 20 "Overvalued" (predicted << actual).
- Persists rankings to `models/undervalued_top20.csv` and `models/overvalued_top20.csv`.
- Notes the caveat: This ranking is an exploratory hypothesis-generation tool,
  NOT a proven investment or buy signal (no ground-truth transfer outcomes exist in the data).
"""

import os
import sys
import argparse
import json
import duckdb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


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


def run_residual_and_ranking_analysis(
    predictions_path: str = "models/test_predictions.parquet",
    models_dir: str = "models",
    eda_dir: str = "eda_outputs",
):
    """Executes Steps 6 and 7."""
    print("=" * 70)
    print("STEPS 6 & 7: RESIDUAL ANALYSIS & UNDERVALUED/OVERVALUED RANKINGS")
    print("=" * 70)
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(eda_dir, exist_ok=True)
    setup_plot_style()

    print(f"Loading test predictions from: {predictions_path}")
    con = duckdb.connect()
    df = con.execute(f"SELECT * FROM '{predictions_path.replace('\\', '/')}'").df()
    print(f"Loaded {len(df):,} test valuation predictions.\n")

    # Assign age bands
    def assign_age_band(age):
        if age < 21:
            return "1. <21 (Prospects)"
        elif age < 25:
            return "2. 21-24 (Ascending)"
        elif age < 29:
            return "3. 25-28 (Peak)"
        elif age < 33:
            return "4. 29-32 (Veterans)"
        else:
            return "5. 33+ (Late Career)"

    df["age_band"] = df["age"].apply(assign_age_band)

    # -------------------------------------------------------------
    # STEP 6: RESIDUAL ERROR BREAKDOWN BY SEGMENT
    # -------------------------------------------------------------
    print("--- STEP 6: RESIDUAL BREAKDOWN BY SEGMENT ---")

    def rmse(s):
        return np.sqrt(np.mean(s**2))

    def aggregate_residuals(data: pd.DataFrame, group_col: str) -> pd.DataFrame:
        # Bias = actual - predicted. "_Uncorrected" uses exp(log prediction);
        # "_Smeared" uses the Duan smear-corrected euro prediction. Log metrics are shared.
        return data.groupby(group_col).agg(
            Count=("player_id", "count"),
            Avg_Actual_EUR=("actual_value_eur", "mean"),
            Avg_Pred_EUR_Uncorrected=("predicted_value_eur", "mean"),
            Mean_Bias_EUR_Uncorrected=("residual_eur", "mean"),
            MAE_EUR_Uncorrected=("abs_residual_eur", "mean"),
            RMSE_EUR_Uncorrected=("residual_eur", rmse),
            Avg_Pred_EUR_Smeared=("predicted_value_eur_smeared", "mean"),
            Mean_Bias_EUR_Smeared=("residual_eur_smeared", "mean"),
            MAE_EUR_Smeared=("abs_residual_eur_smeared", "mean"),
            RMSE_EUR_Smeared=("residual_eur_smeared", rmse),
            Mean_Bias_Log=("residual_log", "mean"),
            MAE_Log=("abs_residual_log", "mean"),
            RMSE_Log=("residual_log", rmse),
        ).reset_index()

    # A. Residuals by Age Band
    res_age = aggregate_residuals(df, "age_band")
    res_age_path = os.path.join(models_dir, "residual_analysis_by_age.csv")
    res_age.to_csv(res_age_path, index=False)

    # B. Residuals by Position
    pos_subset = df[df["position"].isin(["Goalkeeper", "Defender", "Midfield", "Attack"])]
    res_pos = aggregate_residuals(pos_subset, "position")
    res_pos_path = os.path.join(models_dir, "residual_analysis_by_position.csv")
    res_pos.to_csv(res_pos_path, index=False)

    # C. Residuals by League Tier
    res_tier = aggregate_residuals(df, "league_tier")
    res_tier_path = os.path.join(models_dir, "residual_analysis_by_tier.csv")
    res_tier.to_csv(res_tier_path, index=False)

    summary_cols = [
        "Count", "Avg_Actual_EUR",
        "Avg_Pred_EUR_Uncorrected", "Mean_Bias_EUR_Uncorrected", "MAE_EUR_Uncorrected",
        "Avg_Pred_EUR_Smeared", "Mean_Bias_EUR_Smeared", "MAE_EUR_Smeared",
        "MAE_Log",
    ]
    print("\nResidual Breakdown by Age Band:")
    print(res_age[["age_band"] + summary_cols].to_string(index=False))

    print("\nResidual Breakdown by Position:")
    print(res_pos[["position"] + summary_cols].to_string(index=False))

    print("\nResidual Breakdown by League Tier:")
    print(res_tier[["league_tier"] + summary_cols].to_string(index=False))

    # Overall bias before and after smearing
    overall_bias = {
        "test_rows": int(len(df)),
        "mean_actual_eur": float(df["actual_value_eur"].mean()),
        "mean_predicted_eur_uncorrected": float(df["predicted_value_eur"].mean()),
        "mean_predicted_eur_smeared": float(df["predicted_value_eur_smeared"].mean()),
        "mean_bias_eur_uncorrected": float(df["residual_eur"].mean()),
        "mean_bias_eur_smeared": float(df["residual_eur_smeared"].mean()),
        "mean_bias_log": float(df["residual_log"].mean()),
    }
    overall_bias_path = os.path.join(models_dir, "overall_bias.json")
    with open(overall_bias_path, "w") as f:
        json.dump(overall_bias, f, indent=4)
    print("\nOverall Mean Bias on Test Set (Actual - Predicted):")
    print(f"  Uncorrected (exp of log prediction): €{overall_bias['mean_bias_eur_uncorrected']:,.0f}")
    print(f"  Smear-corrected:                     €{overall_bias['mean_bias_eur_smeared']:,.0f}")
    print(f"Saved overall bias summary to: {overall_bias_path}")

    # Plotting Residual Distributions
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    # Panel 1: Bias by Age Band, before and after smearing
    ax1 = axes[0]
    age_x = np.arange(len(res_age))
    bar_w = 0.4
    ax1.bar(age_x - bar_w / 2, res_age["Mean_Bias_EUR_Uncorrected"], bar_w, color="#3b82f6", edgecolor="white", label="Uncorrected")
    ax1.bar(age_x + bar_w / 2, res_age["Mean_Bias_EUR_Smeared"], bar_w, color="#93c5fd", edgecolor="white", label="Smear-corrected")
    ax1.set_xticks(age_x)
    ax1.set_xticklabels(res_age["age_band"].str.slice(3))
    ax1.legend(fontsize=9)
    ax1.set_title("Mean Prediction Bias by Age Band\n(Actual - Predicted EUR)", weight="bold")
    ax1.set_ylabel("Mean Bias (€ EUR)")
    ax1.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, p: f"€{y*1e-6:.1f}M"))
    ax1.axhline(0, color="red", linestyle="--", linewidth=1)
    ax1.tick_params(axis="x", rotation=25)

    # Panel 2: MAE by Position
    ax2 = axes[1]
    bars2 = ax2.bar(res_pos["position"], res_pos["MAE_EUR_Uncorrected"], color="#10b981", edgecolor="white")
    ax2.set_title("Mean Absolute Error (MAE) by Position", weight="bold")
    ax2.set_ylabel("MAE (€ EUR)")
    ax2.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, p: f"€{y*1e-6:.1f}M"))

    # Panel 3: MAE by League Tier
    ax3 = axes[2]
    bars3 = ax3.bar(res_tier["league_tier"].str.replace("Tier ", "T"), res_tier["MAE_EUR_Uncorrected"], color="#f59e0b", edgecolor="white")
    ax3.set_title("Mean Absolute Error (MAE) by League Tier", weight="bold")
    ax3.set_ylabel("MAE (€ EUR)")
    ax3.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, p: f"€{y*1e-6:.1f}M"))
    ax3.tick_params(axis="x", rotation=25)

    plt.suptitle("Residual Prediction Error Breakdown Across Player & League Segments", fontsize=14, weight="bold", y=0.98)
    plt.tight_layout()
    res_plot_path = os.path.join(eda_dir, "residuals_by_segment.png")
    plt.savefig(res_plot_path, dpi=300)
    plt.close()
    print(f"\nSaved residual segment plot to: {res_plot_path}")

    # -------------------------------------------------------------
    # STEP 7: UNDERVALUED & OVERVALUED RANKINGS
    # -------------------------------------------------------------
    print("\n--- STEP 7: UNDERVALUED & OVERVALUED PLAYER RANKINGS ---")
    # Take latest test valuation per player
    latest_query = f"""
        WITH latest_val AS (
            SELECT *,
                ROW_NUMBER() OVER (PARTITION BY player_id ORDER BY valuation_date DESC) as rn
            FROM '{predictions_path.replace('\\', '/')}'
        )
        SELECT 
            player_id,
            player_name,
            ROUND(age, 1) as age,
            position,
            sub_position,
            club_name,
            league_tier,
            valuation_date as latest_test_date,
            actual_value_eur,
            ROUND(predicted_value_eur, 0) as predicted_value_eur,
            ROUND(value_delta_eur, 0) as value_delta_eur,
            ROUND((predicted_value_eur - actual_value_eur) / actual_value_eur * 100, 1) as pct_difference
        FROM latest_val
        WHERE rn = 1
    """

    latest_df = con.execute(latest_query).df()
    print(f"Total unique active players evaluated at latest test date: {len(latest_df):,}")

    # Top 20 Undervalued (Predicted >> Actual)
    top20_undervalued = latest_df.sort_values("value_delta_eur", ascending=False).head(20).reset_index(drop=True)
    top20_undervalued.index += 1
    undervalued_csv = os.path.join(models_dir, "undervalued_top20.csv")
    top20_undervalued.to_csv(undervalued_csv, index_label="Rank")

    # Top 20 Overvalued (Predicted << Actual)
    top20_overvalued = latest_df.sort_values("value_delta_eur", ascending=True).head(20).reset_index(drop=True)
    top20_overvalued.index += 1
    overvalued_csv = os.path.join(models_dir, "overvalued_top20.csv")
    top20_overvalued.to_csv(overvalued_csv, index_label="Rank")

    display_cols = [
        "player_name", "age", "position", "league_tier", "club_name",
        "actual_value_eur", "predicted_value_eur", "value_delta_eur", "pct_difference",
    ]
    print("\nTOP 20 PLAYERS WITH THE MODEL ABOVE MARKET VALUE (largest predicted - actual):")
    print("-" * 110)
    print(top20_undervalued[display_cols].to_string())

    print("\nTOP 20 PLAYERS WITH THE MODEL BELOW MARKET VALUE (most negative predicted - actual):")
    print("-" * 110)
    print(top20_overvalued[display_cols].to_string())

    # Write summary text: computed facts only; any interpretation is labelled as a hypothesis.
    rankings_summary_path = os.path.join(models_dir, "rankings_summary.txt")
    with open(rankings_summary_path, "w", encoding="utf-8") as f:
        f.write("=" * 90 + "\n")
        f.write("FOOTBALL PLAYER VALUATION: RESIDUAL BREAKDOWN & RANKINGS REPORT\n")
        f.write("=" * 90 + "\n\n")

        def eur_m(x):
            return f"{'+' if x >= 0 else '-'}€{abs(x) / 1e6:.2f}M"

        def direction(bias):
            # Bias = actual - predicted, so positive bias means the model predicts too low.
            return "under-predicts" if bias > 0 else "over-predicts" if bias < 0 else "is unbiased"

        def segment_line(label, row):
            return (
                f"- {label} ({int(row['Count']):,} rows): mean bias {eur_m(row['Mean_Bias_EUR_Uncorrected'])} uncorrected "
                f"(model {direction(row['Mean_Bias_EUR_Uncorrected'])}), {eur_m(row['Mean_Bias_EUR_Smeared'])} after "
                f"smearing (model {direction(row['Mean_Bias_EUR_Smeared'])}); log bias {row['Mean_Bias_Log']:+.3f} "
                f"(model {direction(row['Mean_Bias_Log'])} in log terms); euro MAE "
                f"€{row['MAE_EUR_Uncorrected'] / 1e6:.2f}M uncorrected; log MAE {row['MAE_Log']:.3f}.\n"
            )

        f.write("1. MODEL BIAS BY SEGMENT (Bias = actual - predicted; clean feature set):\n")
        f.write("-" * 50 + "\n")
        f.write(f"- Overall ({overall_bias['test_rows']:,} test rows): mean bias "
                f"{eur_m(overall_bias['mean_bias_eur_uncorrected'])} uncorrected, "
                f"{eur_m(overall_bias['mean_bias_eur_smeared'])} after smearing "
                f"(log bias {overall_bias['mean_bias_log']:+.3f}).\n")
        for _, row in res_age.iterrows():
            f.write(segment_line(f"Age {row['age_band'][3:]}", row))
        for _, row in res_tier.iterrows():
            f.write(segment_line(row["league_tier"], row))
        worst_log_age = res_age.loc[res_age["MAE_Log"].idxmax(), "age_band"][3:]
        worst_log_tier = res_tier.loc[res_tier["MAE_Log"].idxmax(), "league_tier"]
        worst_eur_tier = res_tier.loc[res_tier["MAE_EUR_Uncorrected"].idxmax(), "league_tier"]
        f.write(f"- Highest log MAE: age band {worst_log_age}; league tier {worst_log_tier}. "
                f"Highest euro MAE: league tier {worst_eur_tier}.\n")
        f.write("- Hypothesis (not tested here): the features contain no signal for potential, hype, or brand\n")
        f.write("  value, which would push young and elite-league players' market values above the model's.\n\n")

        def ranking_facts(table):
            return (
                f"  {len(table)} players; {int((table['league_tier'] == 'Tier 1 - Big 5').sum())} in Tier 1 - Big 5; "
                f"median age {table['age'].median():.1f}; {int((table['actual_value_eur'] >= 100_000_000).sum())} "
                f"with actual value >= €100M; gap range {eur_m(table['value_delta_eur'].min())} to "
                f"{eur_m(table['value_delta_eur'].max())}.\n"
            )

        f.write("2. TOP 20 PLAYERS WITH THE MODEL ABOVE MARKET VALUE (predicted - actual, uncorrected):\n")
        f.write("-" * 50 + "\n")
        f.write(ranking_facts(top20_undervalued))
        f.write(top20_undervalued[display_cols].to_string() + "\n\n")

        f.write("3. TOP 20 PLAYERS WITH THE MODEL BELOW MARKET VALUE (predicted - actual, uncorrected):\n")
        f.write("-" * 50 + "\n")
        f.write(ranking_facts(top20_overvalued))
        f.write(top20_overvalued[display_cols].to_string() + "\n\n")

        f.write("CAVEAT:\n")
        f.write("-" * 50 + "\n")
        f.write("This ranking has no ground-truth validation (no labeled 'good buy' or profitable transfer\n")
        f.write("outcomes exist in this dataset). It lists the largest gaps between the model's prediction\n")
        f.write("and the Transfermarkt value, not proof that either is wrong, and is not investment advice.\n")

    print(f"\nSaved rankings summary report to: {rankings_summary_path}")
    print(f"Saved undervalued ranking to: {undervalued_csv}")
    print(f"Saved overvalued ranking to: {overvalued_csv}")
    print("\n--- Phase 4 Execution Complete ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Residual Analysis and Undervalued Player Rankings.")
    parser.add_argument("--preds", type=str, default="models/test_predictions.parquet", help="Path to predictions parquet.")
    parser.add_argument("--models-dir", type=str, default="models", help="Directory for model tables.")
    parser.add_argument("--eda-dir", type=str, default="eda_outputs", help="Directory for output plots.")

    args = parser.parse_args()
    run_residual_and_ranking_analysis(
        predictions_path=args.preds,
        models_dir=args.models_dir,
        eda_dir=args.eda_dir,
    )
