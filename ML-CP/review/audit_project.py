"""Read-only review of existing artifacts; writes findings only under review/.

Run from any directory: python review/audit_project.py
No models are fitted and no existing project outputs are changed.
"""
import importlib.metadata
import json
from pathlib import Path

import duckdb
from sklearn.model_selection import TimeSeriesSplit

ROOT = Path(__file__).resolve().parents[1]
con = duckdb.connect()


def path(relative):
    return str(ROOT / relative).replace("\\", "/").replace("'", "''")


def query(sql):
    result = con.execute(sql)
    names = [column[0] for column in result.description]
    return [dict(zip(names, row)) for row in result.fetchall()]


con.execute(f"CREATE VIEW d AS SELECT * FROM '{path('data/processed_market_valuations.parquet')}'")
con.execute(f"CREATE VIEW p AS SELECT * FROM '{path('models/test_predictions.parquet')}'")
report = {"scope": "Existing artifacts and raw data; no retraining", "versions": {
    name: importlib.metadata.version(name)
    for name in ['duckdb', 'numpy', 'pandas', 'scikit-learn', 'xgboost', 'pyarrow']
}}
report['dataset'] = query("""SELECT count(*) AS row_count, count(DISTINCT player_id) players,
    min(valuation_date) first_date, max(valuation_date) last_date,
    count(*)-count(DISTINCT (player_id, valuation_date)) duplicate_keys FROM d""")
report['coverage'] = query("""SELECT league_tier, count(*) AS row_count,
    count(*) FILTER(WHERE minutes_last_365d=0) zero_minutes,
    round(100.0*count(*) FILTER(WHERE minutes_last_365d=0)/count(*),2) zero_minutes_pct
    FROM p GROUP BY 1 ORDER BY 1""")
report['seen_unseen'] = query("""SELECT CASE WHEN player_id IN
    (SELECT player_id FROM d WHERE valuation_date < DATE '2024-01-01')
    THEN 'seen' ELSE 'unseen' END cohort, count(*) AS row_count, count(DISTINCT player_id) players,
    sqrt(avg(pow(residual_log,2))) log_rmse, avg(abs_residual_eur) euro_mae
    FROM p GROUP BY 1""")
dates = con.execute("SELECT valuation_date FROM d WHERE valuation_date < DATE '2024-01-01' ORDER BY valuation_date,player_id").fetchnumpy()['valuation_date']
report['cv_boundaries'] = []
for fold, (tr, va) in enumerate(TimeSeriesSplit(n_splits=3).split(dates), 1):
    report['cv_boundaries'].append({
        'fold': fold, 'train_end': str(dates[tr[-1]]), 'validation_start': str(dates[va[0]]),
        'same_day_overlap': bool(dates[tr[-1]] == dates[va[0]])})
report['recomputed_metrics'] = query("""SELECT sqrt(avg(pow(residual_log,2))) log_rmse,
    avg(abs_residual_log) log_mae, sqrt(avg(pow(residual_eur,2))) euro_rmse,
    avg(abs_residual_eur) euro_mae,
    sqrt(avg(pow(log_actual-ln(predicted_value_eur_smeared),2))) corrected_prediction_log_rmse,
    avg(abs(log_actual-ln(predicted_value_eur_smeared))) corrected_prediction_log_mae,
    count(*) FILTER (WHERE log_predicted <= 8.500001) clipped_floor,
    count(*) FILTER (WHERE log_predicted >= ln(250000000)-0.000001) clipped_ceiling FROM p""")
report['per_year'] = query("""SELECT year(valuation_date) AS valuation_year, count(*) AS row_count,
    sqrt(avg(pow(residual_log,2))) log_rmse, avg(abs_residual_eur) euro_mae FROM p GROUP BY 1 ORDER BY 1""")
report['latest_dates'] = query("""WITH latest AS (SELECT player_id,max(valuation_date) AS latest_date FROM p GROUP BY 1)
    SELECT min(latest_date) oldest, max(latest_date) newest, count(*) players,
    count(*) FILTER(WHERE latest_date < DATE '2026-01-01') before_2026 FROM latest""")
report['persistence_context_baseline'] = query("""WITH history AS (
    SELECT player_id,valuation_date,market_value_in_eur,
    lag(market_value_in_eur) OVER(PARTITION BY player_id ORDER BY valuation_date) previous_value
    FROM d), eligible AS (
    SELECT p.*, greatest(8.5,least(ln(250000000),ln(h.previous_value))) previous_log
    FROM p JOIN history h USING(player_id,valuation_date) WHERE h.previous_value>0)
    SELECT count(*) eligible_rows,
    sqrt(avg(pow(log_actual-previous_log,2))) persistence_log_rmse,
    avg(abs(actual_value_eur-exp(previous_log))) persistence_euro_mae,
    sqrt(avg(pow(residual_log,2))) existing_xgb_log_rmse,
    avg(abs_residual_eur) existing_xgb_euro_mae FROM eligible""")
report['persistence_protocol'] = (
    'Strictly previous observed valuation, updated sequentially during the test period; '
    'same prediction clipping and matched rows. Uses valuation history that the current '
    'performance-only model excludes, so it is a separate information regime, not a fair '
    'algorithm comparison or a fixed-origin future forecast. First-history rows excluded.')
report['appearance_sources'] = query(f"""SELECT min(date) first_date, max(date) last_date,
    count(*) AS row_count, count(DISTINCT appearance_id) unique_appearances,
    count(DISTINCT player_id) players FROM '{path('appearances.csv')}'""")
report['raw_valuation_keys'] = query(f"""SELECT count(*) AS row_count,
    count(*)-count(DISTINCT (player_id,date)) duplicate_keys FROM '{path('player_valuations.csv')}'""")
report['valuation_competition_vs_latest_prior_appearance'] = query(f"""WITH a AS (
    SELECT player_id,date,arg_max(player_club_id,minutes_played) player_club_id
    FROM '{path('appearances.csv')}' GROUP BY player_id,date), matched AS (
    SELECT d.player_id,d.valuation_date,d.current_club_id,a.date appearance_date,a.player_club_id
    FROM d ASOF LEFT JOIN a ON d.player_id=a.player_id AND d.valuation_date>a.date)
    SELECT count(*) recent_matches,
    count(*) FILTER(WHERE current_club_id != player_club_id) different_club,
    round(100.0*count(*) FILTER(WHERE current_club_id != player_club_id)/count(*),2) different_club_pct
    FROM matched WHERE valuation_date-appearance_date<=30""")
report['club_check_caveat'] = 'Diagnostic only: transfers, loans, and national-team appearances can explain club mismatches; this is not proof of leakage.'
out = ROOT / 'review' / 'audit_results.json'
out.write_text(json.dumps(report, indent=2, default=str), encoding='utf-8')
print(json.dumps(report, indent=2, default=str))
