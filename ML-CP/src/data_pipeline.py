"""
data_pipeline.py
================
Constructs the analysis dataset for football player market-value prediction.

What the table does and does not guarantee:
1. One row per player per valuation date.
2. Appearance-based features (rolling 365-day and prior-career minutes, goals, assists, cards,
   per-90 stats) use only appearances dated before the valuation date (a.date < v.valuation_date).
3. Joins player attributes (height, foot, position, contract expiry) from players.csv and club
   attributes from clubs.csv. These are current snapshot values, not values as of the valuation
   date, so they can carry information from after the valuation.
4. Output is persisted to parquet (or CSV) in `data/` for high performance downstream analysis.
"""

import os
import sys
import argparse
import time
import duckdb
import pandas as pd


def build_analysis_dataset(
    raw_data_dir: str = ".",
    output_path: str = "data/processed_market_valuations.parquet",
    min_valuation_date: str = "2014-01-01",
    save_csv: bool = False,
) -> str:
    """
    Builds the analysis table using DuckDB SQL engine. Appearance features use only matches
    before each valuation date; player and club attributes come from the current snapshot.
    """
    start_time = time.time()
    print("=" * 70)
    print("STEP 1: BUILDING ANALYSIS TABLE")
    print("=" * 70)
    print(f"Data source directory: {os.path.abspath(raw_data_dir)}")
    print(f"Minimum valuation date cutoff: {min_valuation_date}")
    print(f"Target output file: {output_path}")

    # Ensure output directories exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Normalize file paths for SQL
    val_file = os.path.join(raw_data_dir, "player_valuations.csv").replace("\\", "/")
    players_file = os.path.join(raw_data_dir, "players.csv").replace("\\", "/")
    appearances_file = os.path.join(raw_data_dir, "appearances.csv").replace("\\", "/")
    clubs_file = os.path.join(raw_data_dir, "clubs.csv").replace("\\", "/")
    competitions_file = os.path.join(raw_data_dir, "competitions.csv").replace("\\", "/")

    for path, name in [
        (val_file, "player_valuations.csv"),
        (players_file, "players.csv"),
        (appearances_file, "appearances.csv"),
        (clubs_file, "clubs.csv"),
        (competitions_file, "competitions.csv"),
    ]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Required data file not found: {path} ({name})")

    con = duckdb.connect()

    # Query description:
    # 1. v_filtered: Filter valuations where market value > 0, DOB is valid, and date >= min_valuation_date.
    # 2. app_agg: Aggregates appearances dated before valuation_date (a.date < v.valuation_date).
    #    Computes rolling 365-day stats as well as prior career stats.
    # 3. Final join brings in competition tiers and club metadata.
    sql_query = f"""
    COPY (
        WITH v_filtered AS (
            SELECT 
                v.player_id,
                v.date as valuation_date,
                v.market_value_in_eur,
                v.current_club_id,
                v.current_club_name,
                v.player_club_domestic_competition_id as competition_id,
                p.name as player_name,
                p.date_of_birth,
                p.position,
                p.sub_position,
                p.height_in_cm,
                p.foot,
                p.country_of_citizenship,
                p.contract_expiration_date,
                -- Age at valuation date
                ROUND(DATEDIFF('day', p.date_of_birth, v.date) / 365.25, 2) as age,
                ROUND(POWER(DATEDIFF('day', p.date_of_birth, v.date) / 365.25, 2), 2) as age_squared,
                -- Contract remaining days at valuation date
                CASE 
                    WHEN p.contract_expiration_date IS NOT NULL 
                    THEN DATEDIFF('day', v.date, TRY_CAST(p.contract_expiration_date AS DATE))
                    ELSE NULL 
                END as contract_days_remaining,
                CASE WHEN p.contract_expiration_date IS NOT NULL THEN 1 ELSE 0 END as has_contract_info
            FROM '{val_file}' v
            JOIN '{players_file}' p ON v.player_id = p.player_id
            WHERE v.market_value_in_eur > 0 
              AND p.date_of_birth IS NOT NULL
              AND v.date >= '{min_valuation_date}'
        ),
        app_agg AS (
            SELECT 
                v.player_id,
                v.valuation_date,
                -- Only appearances dated before valuation_date (join condition a.date < v.valuation_date)
                -- 365-day rolling window
                COUNT(CASE WHEN a.date >= v.valuation_date - INTERVAL 365 DAY THEN a.appearance_id END) as appearances_last_365d,
                COALESCE(SUM(CASE WHEN a.date >= v.valuation_date - INTERVAL 365 DAY THEN a.minutes_played END), 0) as minutes_last_365d,
                COALESCE(SUM(CASE WHEN a.date >= v.valuation_date - INTERVAL 365 DAY THEN a.goals END), 0) as goals_last_365d,
                COALESCE(SUM(CASE WHEN a.date >= v.valuation_date - INTERVAL 365 DAY THEN a.assists END), 0) as assists_last_365d,
                COALESCE(SUM(CASE WHEN a.date >= v.valuation_date - INTERVAL 365 DAY THEN a.yellow_cards END), 0) as yellow_cards_last_365d,
                COALESCE(SUM(CASE WHEN a.date >= v.valuation_date - INTERVAL 365 DAY THEN a.red_cards END), 0) as red_cards_last_365d,
                -- Career cumulative stats prior to valuation date
                COUNT(a.appearance_id) as career_appearances_prior,
                COALESCE(SUM(a.minutes_played), 0) as career_minutes_prior,
                COALESCE(SUM(a.goals), 0) as career_goals_prior,
                COALESCE(SUM(a.assists), 0) as career_assists_prior
            FROM v_filtered v
            LEFT JOIN '{appearances_file}' a 
              ON v.player_id = a.player_id 
             AND a.date < v.valuation_date 
            GROUP BY v.player_id, v.valuation_date
        )
        SELECT 
            v.*,
            -- League Tier classification
            CASE 
                WHEN v.competition_id IN ('GB1', 'ES1', 'IT1', 'L1', 'FR1') THEN 'Tier 1 - Big 5'
                WHEN v.competition_id IN ('PO1', 'NL1', 'BE1', 'TR1', 'RU1', 'SC1', 'GR1', 'UKR1', 'DK1', 'C1', 'NO1', 'PL1', 'RO1', 'SER1', 'TS1') THEN 'Tier 2 - Top European'
                WHEN comp.sub_type = 'first_tier' THEN 'Tier 3 - Other First Tier'
                ELSE 'Tier 4 - Lower / Other'
            END as league_tier,
            comp.name as competition_name,
            comp.confederation as competition_confederation,
            COALESCE(c.name, v.current_club_name, 'Unknown') as club_name,
            c.squad_size as club_squad_size,
            c.foreigners_percentage as club_foreigners_percentage,
            c.national_team_players as club_national_team_players,
            -- Rolling & career stats
            a.appearances_last_365d,
            a.minutes_last_365d,
            a.goals_last_365d,
            a.assists_last_365d,
            a.yellow_cards_last_365d,
            a.red_cards_last_365d,
            ROUND(CASE WHEN a.minutes_last_365d > 0 THEN a.goals_last_365d / (a.minutes_last_365d / 90.0) ELSE 0.0 END, 4) as goals_per_90_365d,
            ROUND(CASE WHEN a.minutes_last_365d > 0 THEN a.assists_last_365d / (a.minutes_last_365d / 90.0) ELSE 0.0 END, 4) as assists_per_90_365d,
            ROUND(CASE WHEN a.minutes_last_365d > 0 THEN (a.goals_last_365d + a.assists_last_365d) / (a.minutes_last_365d / 90.0) ELSE 0.0 END, 4) as goal_contributions_per_90_365d,
            ROUND(CASE WHEN a.appearances_last_365d > 0 THEN a.minutes_last_365d / a.appearances_last_365d ELSE 0.0 END, 2) as minutes_per_appearance_365d,
            a.career_appearances_prior,
            a.career_minutes_prior,
            a.career_goals_prior,
            a.career_assists_prior
        FROM v_filtered v
        JOIN app_agg a ON v.player_id = a.player_id AND v.valuation_date = a.valuation_date
        LEFT JOIN '{competitions_file}' comp ON v.competition_id = comp.competition_id
        LEFT JOIN '{clubs_file}' c ON v.current_club_id = c.club_id
        ORDER BY v.valuation_date, v.player_id
    ) TO '{output_path.replace("\\", "/")}' (FORMAT PARQUET, COMPRESSION ZSTD);
    """

    print("Executing DuckDB rolling aggregation query (appearances before each valuation date)...")
    con.execute(sql_query)

    # Verification summary
    summary_df = con.execute(f"""
        SELECT 
            COUNT(*) as total_rows,
            COUNT(DISTINCT player_id) as unique_players,
            MIN(valuation_date) as min_date,
            MAX(valuation_date) as max_date,
            MIN(market_value_in_eur) as min_market_value,
            MAX(market_value_in_eur) as max_market_value,
            AVG(market_value_in_eur) as mean_market_value,
            AVG(minutes_last_365d) as avg_minutes_365d,
            AVG(career_minutes_prior) as avg_career_minutes
        FROM '{output_path.replace("\\", "/")}'
    """).df()

    elapsed = time.time() - start_time
    print(f"\nSuccessfully generated analysis dataset in {elapsed:.2f} seconds!")
    print("\nDataset Summary Statistics:")
    print(summary_df.to_string(index=False))

    if save_csv:
        csv_path = output_path.replace(".parquet", ".csv")
        print(f"\nExporting to CSV: {csv_path}")
        con.execute(f"COPY (SELECT * FROM '{output_path.replace('\\', '/')}') TO '{csv_path.replace('\\', '/')}' (HEADER, DELIMITER ',');")
        print("CSV export complete.")

    return output_path


def verify_leakage_invariants(data_path: str = "data/processed_market_valuations.parquet") -> bool:
    """
    Data-integrity checks on the analysis table. They verify:
    - 1 row per player per valuation date
    - Target is positive and non-null
    - Ages are between 14 and 50
    - Rolling and career minutes and rolling goals are non-negative
    They do not test for leakage: they do not check appearance dates against valuation dates
    or whether player and club attributes are as of the valuation date.
    """
    print("\n--- Running Data Integrity Checks ---")
    con = duckdb.connect()
    
    # 1. Uniqueness check
    dup_count = con.execute(f"""
        SELECT COUNT(*) - COUNT(DISTINCT (player_id || '_' || CAST(valuation_date AS VARCHAR)))
        FROM '{data_path.replace("\\", "/")}'
    """).fetchone()[0]
    assert dup_count == 0, f"Integrity Failure: Found {dup_count} duplicate player-valuation dates!"
    print("[PASS] Unique constraint: exactly one row per player per valuation date.")

    # 2. Target integrity
    invalid_target = con.execute(f"""
        SELECT COUNT(*)
        FROM '{data_path.replace("\\", "/")}'
        WHERE market_value_in_eur <= 0 OR market_value_in_eur IS NULL
    """).fetchone()[0]
    assert invalid_target == 0, f"Integrity Failure: Found {invalid_target} non-positive market values!"
    print("[PASS] Target integrity: all market values are positive non-null numbers.")

    # 3. Age bounds
    invalid_age = con.execute(f"""
        SELECT COUNT(*)
        FROM '{data_path.replace("\\", "/")}'
        WHERE age < 14 OR age > 50
    """).fetchone()[0]
    assert invalid_age == 0, f"Integrity Failure: Found {invalid_age} anomalous age values (<14 or >50)!"
    print("[PASS] Age integrity: all ages are within valid physiological range (14-50).")

    # 4. Temporal performance sanity check
    negative_stats = con.execute(f"""
        SELECT COUNT(*)
        FROM '{data_path.replace("\\", "/")}'
        WHERE minutes_last_365d < 0 OR career_minutes_prior < 0 OR goals_last_365d < 0
    """).fetchone()[0]
    assert negative_stats == 0, f"Integrity Failure: Found negative performance stats!"
    print("[PASS] Performance stats: rolling minutes, career minutes and rolling goals are non-negative.")

    print("All integrity checks passed successfully!\n")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build the analysis dataset for football market values.")
    parser.add_argument("--data-dir", type=str, default=".", help="Directory containing raw CSVs.")
    parser.add_argument("--output", type=str, default="data/processed_market_valuations.parquet", help="Output parquet file path.")
    parser.add_argument("--min-date", type=str, default="2014-01-01", help="Minimum valuation date cutoff.")
    parser.add_argument("--save-csv", action="store_true", help="Also save a CSV copy.")

    args = parser.parse_args()
    build_analysis_dataset(
        raw_data_dir=args.data_dir,
        output_path=args.output,
        min_valuation_date=args.min_date,
        save_csv=args.save_csv,
    )
    verify_leakage_invariants(args.output)
