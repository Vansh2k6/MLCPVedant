"""
eda.py
======
Exploratory Data Analysis for Football Player Market-Value Prediction.

Covers:
1. Distribution of market_value_in_eur:
   - Measures skewness and kurtosis of the raw target and of its natural log.
   - Generates `eda_outputs/market_value_distribution.png`.
2. Market Value vs. Age by Position Group:
   - Median, interquartile range, and mean value by integer age for each position; peak median age.
   - Generates `eda_outputs/value_vs_age_by_position.png`.
3. Missingness and Performance Sparsity Analysis:
   - Missing percentages for key columns.
   - Share of rows with zero minutes, goals, and assists in the last 365 days, by position.
   - Generates `eda_outputs/missingness_and_sparsity_report.png`.
All numbers in the plots and in `eda_outputs/eda_summary.txt` are computed from the analysis table.
"""

import os
import sys
import argparse
import duckdb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


def setup_plot_style():
    """Configures clean, publication-quality matplotlib aesthetics."""
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Segoe UI", "DejaVu Sans", "Helvetica", "Arial"],
        "axes.edgecolor": "#cccccc",
        "axes.linewidth": 0.8,
        "grid.color": "#e5e5e5",
        "grid.linestyle": "--",
        "grid.linewidth": 0.5,
        "figure.titlesize": 14,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
    })


def run_eda(
    data_path: str = "data/processed_market_valuations.parquet",
    output_dir: str = "eda_outputs",
):
    """Executes full EDA pipeline and writes artifacts."""
    print("=" * 70)
    print("STEP 2: EXPLORATORY DATA ANALYSIS (EDA)")
    print("=" * 70)
    os.makedirs(output_dir, exist_ok=True)
    setup_plot_style()

    con = duckdb.connect()
    print(f"Loading data from: {data_path}")

    # Load core columns needed for EDA
    df = con.execute(f"""
        SELECT 
            player_id,
            valuation_date,
            market_value_in_eur,
            ln(market_value_in_eur) as log_market_value,
            age,
            position,
            sub_position,
            league_tier,
            minutes_last_365d,
            goals_last_365d,
            assists_last_365d,
            yellow_cards_last_365d,
            red_cards_last_365d,
            career_appearances_prior,
            career_minutes_prior,
            contract_days_remaining,
            has_contract_info,
            height_in_cm,
            foot,
            club_squad_size
        FROM '{data_path.replace("\\", "/")}'
    """).df()

    print(f"Total dataset records: {len(df):,}")

    # Filter positions to primary 4 categories for clean domain visualization
    primary_positions = ["Goalkeeper", "Defender", "Midfield", "Attack"]
    df_pos = df[df["position"].isin(primary_positions)].copy()

    # -------------------------------------------------------------
    # 1. Target Distribution Plot (Raw vs Log)
    # -------------------------------------------------------------
    print("\n[1/3] Generating market value distribution plots...")
    raw_mv = df["market_value_in_eur"]
    log_mv = df["log_market_value"]

    raw_skew = raw_mv.skew()
    raw_kurt = raw_mv.kurt()
    raw_mean = raw_mv.mean()
    raw_median = raw_mv.median()

    log_skew = log_mv.skew()
    log_kurt = log_mv.kurt()
    log_mean = log_mv.mean()
    log_median = log_mv.median()

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # Left: Raw distribution (clipped at €25M for display clarity with notation)
    ax1 = axes[0]
    clip_val = 25_000_000
    display_raw = np.clip(raw_mv, 0, clip_val)
    ax1.hist(display_raw, bins=60, color="#1f77b4", edgecolor="white", alpha=0.85, density=True)
    ax1.axvline(raw_mean, color="#d62728", linestyle="--", linewidth=2, label=f"Mean: €{raw_mean:,.0f}")
    ax1.axvline(raw_median, color="#2ca02c", linestyle="-", linewidth=2, label=f"Median: €{raw_median:,.0f}")
    ax1.set_title(f"A. Raw Market Value (skewness {raw_skew:.2f})", weight="bold", pad=10)
    ax1.set_xlabel("Market Value in EUR (Display capped at €25M)")
    ax1.set_ylabel("Density")
    ax1.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f"€{x*1e-6:.1f}M"))
    ax1.legend(frameon=True, facecolor="white")
    ax1.text(
        0.50, 0.65,
        f"Skewness: {raw_skew:.2f}\nKurtosis: {raw_kurt:.2f}\nMax Value: €{raw_mv.max():,.0f}",
        transform=ax1.transAxes,
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#f8f9fa", edgecolor="#cccccc"),
    )

    # Right: Log-transformed distribution
    ax2 = axes[1]
    n, bins, patches = ax2.hist(log_mv, bins=50, color="#2b5c8f", edgecolor="white", alpha=0.85, density=True)
    
    # Overlay theoretical normal distribution
    from scipy.stats import norm
    mu, std = norm.fit(log_mv)
    xmin, xmax = ax2.get_xlim()
    x = np.linspace(xmin, xmax, 100)
    p = norm.pdf(x, mu, std)
    ax2.plot(x, p, "r-", linewidth=2, label=f"Normal Fit (μ={mu:.2f}, σ={std:.2f})")

    ax2.axvline(log_mean, color="#d62728", linestyle="--", linewidth=2, label=f"Mean: {log_mean:.2f}")
    ax2.axvline(log_median, color="#2ca02c", linestyle="-", linewidth=2, label=f"Median: {log_median:.2f}")
    ax2.set_title(f"B. Log-Transformed Market Value (skewness {log_skew:.2f})", weight="bold", pad=10)
    ax2.set_xlabel("ln(Market Value in EUR)")
    ax2.set_ylabel("Density")
    ax2.legend(frameon=True, facecolor="white")
    ax2.text(
        0.05, 0.70,
        f"Skewness: {log_skew:.2f} (from {raw_skew:.2f})\nKurtosis: {log_kurt:.2f} (from {raw_kurt:.2f})",
        transform=ax2.transAxes,
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#eef6ff", edgecolor="#99badd"),
    )

    plt.suptitle("Target Variable Distribution & Log-Transformation Assessment", fontsize=15, weight="bold", y=0.98)
    plt.tight_layout()
    dist_plot_path = os.path.join(output_dir, "market_value_distribution.png")
    plt.savefig(dist_plot_path, dpi=300)
    plt.close()
    print(f"Saved: {dist_plot_path}")

    # -------------------------------------------------------------
    # 2. Value vs Age by Position Group
    # -------------------------------------------------------------
    print("\n[2/3] Generating Value vs. Age by Position Group plots...")
    # Bin age into integer bins from 17 to 37
    df_pos["age_bin"] = np.clip(np.floor(df_pos["age"]).astype(int), 17, 37)

    # Compute median and IQR for each position and age bin
    age_pos_stats = df_pos.groupby(["position", "age_bin"])["market_value_in_eur"].agg(
        median="median",
        p25=lambda x: np.percentile(x, 25),
        p75=lambda x: np.percentile(x, 75),
        mean="mean",
        count="count"
    ).reset_index()

    # Create 4-panel figure
    fig, axes = plt.subplots(2, 2, figsize=(16, 12), sharex=True, sharey=True)
    axes = axes.flatten()

    colors = {
        "Attack": "#e41a1c",
        "Midfield": "#377eb8",
        "Defender": "#4daf4a",
        "Goalkeeper": "#984ea3",
    }

    peak_summary = {}

    for idx, pos in enumerate(primary_positions):
        ax = axes[idx]
        subset = age_pos_stats[age_pos_stats["position"] == pos]
        color = colors[pos]

        # Plot median line and IQR band
        ax.plot(subset["age_bin"], subset["median"], color=color, linewidth=2.5, marker="o", markersize=5, label="Median Market Value")
        ax.fill_between(subset["age_bin"], subset["p25"], subset["p75"], color=color, alpha=0.2, label="25th-75th Percentile (IQR)")
        ax.plot(subset["age_bin"], subset["mean"], color="#555555", linestyle=":", linewidth=1.5, label="Mean Market Value")

        # Find peak median age
        peak_row = subset.loc[subset["median"].idxmax()]
        peak_age = int(peak_row["age_bin"])
        peak_val = peak_row["median"]
        peak_summary[pos] = (peak_age, peak_val)

        ax.axvline(peak_age, color=color, linestyle="--", alpha=0.7, label=f"Peak Median Age: {peak_age} yrs")
        ax.scatter([peak_age], [peak_val], color=color, s=100, zorder=5, edgecolor="black")

        ax.set_title(f"{pos} (Peak at Age {peak_age}, Median €{peak_val:,.0f})", weight="bold", color=color, fontsize=12)
        ax.set_xlabel("Age at Valuation Date")
        ax.set_ylabel("Market Value (EUR)")
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, p: f"€{y*1e-6:.1f}M"))
        ax.set_xlim(16.5, 37.5)
        ax.set_ylim(0, 3.5e6)
        ax.legend(loc="upper right", frameon=True, facecolor="white", fontsize=9)

    plt.suptitle("Football Player Market Value Trajectory vs. Age Across Positions", fontsize=15, weight="bold", y=0.98)
    plt.tight_layout()
    age_plot_path = os.path.join(output_dir, "value_vs_age_by_position.png")
    plt.savefig(age_plot_path, dpi=300)
    plt.close()
    print(f"Saved: {age_plot_path}")

    # -------------------------------------------------------------
    # 3. Missingness & Performance Sparsity Audit
    # -------------------------------------------------------------
    print("\n[3/3] Generating Missingness & Sparsity Audit...")

    # A. Feature missingness percentages
    missing_pcts = (df.isnull().sum() / len(df) * 100).sort_values(ascending=False)
    # Filter features that have any missingness or are structurally important
    key_features = [
        "contract_days_remaining", "height_in_cm", "foot", "sub_position",
        "club_squad_size", "league_tier", "minutes_last_365d", "goals_last_365d"
    ]
    missing_subset = missing_pcts[key_features].reset_index()
    missing_subset.columns = ["feature", "missing_pct"]

    # B. Performance sparsity (Zero-rate by position)
    pos_sparsity = df_pos.groupby("position").agg(
        total_observations=("player_id", "count"),
        zero_minutes_pct=("minutes_last_365d", lambda s: (s == 0).mean() * 100),
        zero_goals_pct=("goals_last_365d", lambda s: (s == 0).mean() * 100),
        zero_assists_pct=("assists_last_365d", lambda s: (s == 0).mean() * 100),
        avg_goals_365d=("goals_last_365d", "mean"),
        avg_assists_365d=("assists_last_365d", "mean"),
        avg_minutes_365d=("minutes_last_365d", "mean"),
    ).reset_index()

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Subplot 1: Feature Missingness Bar Chart
    ax_miss = axes[0]
    bars = ax_miss.barh(missing_subset["feature"], missing_subset["missing_pct"], color="#4a7c59", edgecolor="white")
    ax_miss.set_xlabel("Missing Percentage (%)")
    ax_miss.set_title("A. Dataset Feature Missingness", weight="bold")
    ax_miss.set_xlim(0, 100)
    for bar in bars:
        w = bar.get_width()
        if w > 0:
            ax_miss.text(w + 1, bar.get_y() + bar.get_height()/2, f"{w:.1f}%", va="center", fontsize=10, weight="bold")

    # Subplot 2: Positional Performance Zero-Rates
    ax_sparse = axes[1]
    x_indices = np.arange(len(primary_positions))
    bar_width = 0.25

    ordered_sparsity = pos_sparsity.set_index("position").loc[primary_positions].reset_index()

    r1 = ax_sparse.bar(x_indices - bar_width, ordered_sparsity["zero_goals_pct"], width=bar_width, label="Zero Goals (%)", color="#e41a1c")
    r2 = ax_sparse.bar(x_indices, ordered_sparsity["zero_assists_pct"], width=bar_width, label="Zero Assists (%)", color="#377eb8")
    r3 = ax_sparse.bar(x_indices + bar_width, ordered_sparsity["zero_minutes_pct"], width=bar_width, label="Zero Minutes Played (%)", color="#999999")

    ax_sparse.set_xticks(x_indices)
    ax_sparse.set_xticklabels(primary_positions, weight="bold")
    ax_sparse.set_ylabel("Percentage of Records (%)")
    ax_sparse.set_title("B. Rolling 365d Stat Sparsity by Position Group", weight="bold")
    ax_sparse.set_ylim(0, 115)
    ax_sparse.legend(loc="upper right", frameon=True, facecolor="white")

    for rects in [r1, r2]:
        for rect in rects:
            h = rect.get_height()
            ax_sparse.text(rect.get_x() + rect.get_width()/2., h + 2, f"{h:.1f}%", ha="center", va="bottom", fontsize=8.5, rotation=45)

    plt.suptitle("Missingness Audit & Positional Sparsity: Justification Against Blanket Imputation", fontsize=14, weight="bold", y=0.98)
    plt.tight_layout()
    sparsity_plot_path = os.path.join(output_dir, "missingness_and_sparsity_report.png")
    plt.savefig(sparsity_plot_path, dpi=300)
    plt.close()
    print(f"Saved: {sparsity_plot_path}")

    # -------------------------------------------------------------
    # 4. Save Text Summary Report
    # -------------------------------------------------------------
    summary_txt_path = os.path.join(output_dir, "eda_summary.txt")
    with open(summary_txt_path, "w", encoding="utf-8") as f:
        f.write("=" * 70 + "\n")
        f.write("FOOTBALL PLAYER MARKET VALUE: EDA & SPARSITY REPORT\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Total Records Analyzed: {len(df):,}\n")
        f.write(f"Unique Players: {df['player_id'].nunique():,}\n\n")

        f.write("1. TARGET DISTRIBUTION ANALYSIS:\n")
        f.write("-" * 40 + "\n")
        f.write(f"Raw Market Value Mean:      €{raw_mean:,.2f}\n")
        f.write(f"Raw Market Value Median:    €{raw_median:,.2f}\n")
        f.write(f"Raw Market Value Skewness:  {raw_skew:.4f}\n")
        f.write(f"Raw Market Value Kurtosis:  {raw_kurt:.4f}\n")
        f.write(f"Log Market Value Mean:      {log_mean:.4f}\n")
        f.write(f"Log Market Value Median:    {log_median:.4f}\n")
        f.write(f"Log Market Value Skewness:  {log_skew:.4f}\n")
        f.write(f"Log Market Value Kurtosis:  {log_kurt:.4f}\n\n")

        f.write("2. PEAK MARKET VALUE AGE BY POSITION (age with the highest median value, ages binned 17-37):\n")
        f.write("-" * 40 + "\n")
        for pos in primary_positions:
            peak_age, peak_val = peak_summary[pos]
            f.write(f"- {pos:<12}: Peak Age {peak_age} years (Median: €{peak_val:,.0f})\n")
        youngest_peak = min(peak_age for peak_age, _ in peak_summary.values())
        oldest_peak = max(peak_age for peak_age, _ in peak_summary.values())
        youngest = [p for p in primary_positions if peak_summary[p][0] == youngest_peak]
        oldest = [p for p in primary_positions if peak_summary[p][0] == oldest_peak]
        f.write(f"\nEarliest peak: {', '.join(youngest)} (age {youngest_peak}). "
                f"Latest peak: {', '.join(oldest)} (age {oldest_peak}).\n\n")

        sparsity = ordered_sparsity.set_index("position")
        overall_mean_goals = df_pos["goals_last_365d"].mean()
        f.write("3. PERFORMANCE SPARSITY & POSITION-SPECIFIC FEATURE HANDLING:\n")
        f.write("-" * 40 + "\n")
        f.write(ordered_sparsity.to_string(index=False) + "\n\n")
        f.write("Handling of Missingness & Sparsity:\n")
        f.write("a. NO BLANKET IMPUTATION OF PERFORMANCE STATS:\n")
        f.write(f"   - Zero goals in the last 365 days is the norm for some positions "
                f"({sparsity.loc['Goalkeeper', 'zero_goals_pct']:.1f}% of Goalkeeper rows, "
                f"{sparsity.loc['Defender', 'zero_goals_pct']:.1f}% of Defender rows), so zeros are kept as\n")
        f.write(f"     real values rather than replaced by an average (mean goals across the four positions: "
                f"{overall_mean_goals:.2f}).\n")
        f.write("b. POSITION-SPECIFIC INDICATORS & INTERACTIONS (see prepare_features in src/models.py):\n")
        f.write("   - One-hot position flags (`is_goalkeeper`, `is_defender`, `is_midfield`, `is_attack`).\n")
        f.write("   - Interaction terms: `attack_goals_per_90`, `midfield_assists_per_90`, `gk_minutes_last_365d`,\n")
        f.write("     `def_minutes_last_365d`.\n")
        f.write("   - Height: missing values filled with the training-set median for the position (no missing flag).\n")
        f.write("   - Contract and club attributes are current snapshots; the headline model excludes them\n")
        f.write("     (SNAPSHOT_FEATURES in src/models.py).\n\n")

        f.write("4. KEY FINDINGS (computed):\n")
        f.write("-" * 40 + "\n")
        for pos in primary_positions:
            peak_age, peak_val = peak_summary[pos]
            f.write(f"- Peak median value age, {pos}: {peak_age} (median €{peak_val:,.0f}).\n")
        f.write(f"- Rows with zero goals in the last 365 days: Goalkeepers "
                f"{sparsity.loc['Goalkeeper', 'zero_goals_pct']:.1f}%, Defenders "
                f"{sparsity.loc['Defender', 'zero_goals_pct']:.1f}%.\n")
        f.write(f"- Target skewness: {raw_skew:.2f} raw, {log_skew:.2f} after the natural log transform "
                f"(kurtosis {raw_kurt:.2f} raw, {log_kurt:.2f} log).\n")

    print(f"Saved textual report: {summary_txt_path}")
    print("\n--- EDA Pipeline Complete ---")
    return {
        "dist_plot": dist_plot_path,
        "age_plot": age_plot_path,
        "sparsity_plot": sparsity_plot_path,
        "summary_txt": summary_txt_path,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run EDA for football player market value prediction.")
    parser.add_argument("--data", type=str, default="data/processed_market_valuations.parquet", help="Path to processed parquet file.")
    parser.add_argument("--outdir", type=str, default="eda_outputs", help="Directory to save output plots.")
    args = parser.parse_args()
    run_eda(args.data, args.outdir)
