"""
update_readme.py
================
Step 8: Regenerates the data-driven sections of README.md from pipeline outputs.

Each generated section sits between a marker pair:
    <!-- BEGIN:name -->
    ...
    <!-- END:name -->
Only the text between markers is rewritten; everything else in README.md is left as is.
A missing or duplicated marker stops the script with an error naming the marker.

Inputs (all written by earlier pipeline steps):
- models/model_comparison_metrics.csv
- models/residual_analysis_by_age.csv, residual_analysis_by_position.csv, residual_analysis_by_tier.csv
- models/undervalued_top20.csv, models/overvalued_top20.csv
- models/smearing_factor.json, models/overall_bias.json, models/feature_importances.csv
- models/test_predictions.parquet, data/processed_market_valuations.parquet
- games.csv in the raw-data directory (optional; only used for the match-data date in the snapshot text)
"""

import argparse
import json
import os
import re
import sys

import duckdb
import numpy as np
import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models import (
    FEATURE_SET_CLEAN, FEATURE_SET_FULL, LOG_PRED_CEILING, LOG_PRED_FLOOR, NAIVE_MIN_GROUP_ROWS, SNAPSHOT_FEATURES,
)

# Must match split_date in src/models.py and the position filter in src/evaluate.py.
SPLIT_DATE = "2024-01-01"
PRIMARY_POSITIONS = ["Goalkeeper", "Defender", "Midfield", "Attack"]

NAIVE_MODEL = "Naive Baseline (Position x Tier Median)"
RIDGE_MODEL = "Baseline Ridge Regression"
XGB_MODEL = "Main XGBoost Regressor"
XGB_SMEARED_MODEL = "Main XGBoost Regressor (Smear-Corrected)"
MODEL_LABELS = {
    NAIVE_MODEL: "Naive Baseline (position × league-tier median)",
    RIDGE_MODEL: "Ridge",
    XGB_MODEL: "XGBoost",
    XGB_SMEARED_MODEL: "XGBoost (Smear-Corrected)",
}
NAIVE_IMPROVEMENT_ROW = "% Improvement (XGBoost vs Naive Baseline)"
METRICS = ["Log RMSE", "Log MAE", "Euro RMSE", "Euro MAE"]

PROSPECT_BAND = "<21"
LATE_CAREER_BAND = "33+"
TIER_1 = "Tier 1 - Big 5"
SUPERSTAR_VALUE_EUR = 100_000_000
YOUNG_AGE = 25  # upper edge of the "21-24" age band in evaluate.py

SNAPSHOT_REASONS = {
    "contract_days_remaining": "days to the expiry of the player's current contract in `players.csv`, not the "
                               "contract in force at the valuation",
    "has_contract_info": "whether `players.csv` has a current contract expiry date, which partly encodes whether "
                         "the player is still under contract at the snapshot date",
    "club_squad_size": "squad size from the current row of `clubs.csv`",
    "club_foreigners_percentage": "share of foreign players from the current row of `clubs.csv`",
    "club_national_team_players": "number of current national-team players from the current row of `clubs.csv`",
}
if set(SNAPSHOT_REASONS) != set(SNAPSHOT_FEATURES):
    raise RuntimeError("SNAPSHOT_REASONS in update_readme.py must list exactly SNAPSHOT_FEATURES from src/models.py.")

SECTIONS = [
    "tldr", "snapshot", "split_counts", "sparsity", "benchmark", "leakage_ablation", "smearing", "feature_importance",
    "residuals_age", "findings_age", "residuals_position", "findings_position",
    "residuals_tier", "findings_tier", "ranking_scope", "top20_model_above",
    "top20_model_below", "limitations", "zero_minutes", "zero_minutes_errors",
]


class ReadmeMarkerError(RuntimeError):
    pass


# -------------------------------------------------------------
# Formatting
# -------------------------------------------------------------
def eur(x):
    v = int(round(x))
    return f"-€{abs(v):,}" if v < 0 else f"€{v:,}"


def eur_signed(x):
    v = int(round(x))
    return f"{'+' if v >= 0 else '-'}€{abs(v):,}"


def pct(x):
    return f"{x:.1f}%"


def pct_signed(x):
    return f"{x:+.1f}%"


def change_word(reduction_pct):
    """`reduction_pct` > 0 means the second value is lower than the first."""
    return f"{pct(abs(reduction_pct))} {'lower' if reduction_pct > 0 else 'higher'}"


def bias_direction(bias):
    # Bias = actual - predicted, so positive bias means predictions are too low.
    if bias > 0:
        return "under-predicts"
    if bias < 0:
        return "over-predicts"
    return "has zero mean bias for"


def md_table(headers, aligns, rows):
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(aligns) + " |"]
    lines += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return "\n".join(lines)


def band_label(age_band):
    return re.sub(r"^\d+\.\s*", "", age_band)


# -------------------------------------------------------------
# Loading pipeline outputs
# -------------------------------------------------------------
def zero_minutes_errors(predictions_path):
    """Headline XGBoost Euro MAE and Log MAE for test rows with zero vs some previous-365-day minutes."""
    rows = duckdb.sql(
        f"""SELECT minutes_last_365d = 0, count(*), avg(abs_residual_eur), avg(abs_residual_log), avg(actual_value_eur)
            FROM '{predictions_path.replace(chr(92), '/')}' GROUP BY 1 ORDER BY 1 DESC"""
    ).fetchall()
    return [{"zero_minutes": z, "rows": n, "euro_mae": e, "log_mae": l, "avg_actual": a} for z, n, e, l, a in rows]


def snapshot_dates_text(ctx):
    text = f"valuations through {ctx['last_valuation']}"
    if ctx["last_match"] is not None:
        text += f"; match data through {ctx['last_match']}"
    return text


def load_context(models_dir, data_path, raw_data_dir):
    def path(name):
        p = os.path.join(models_dir, name)
        if not os.path.exists(p):
            raise FileNotFoundError(f"Pipeline output not found: {p}. Run `python run_pipeline.py` first.")
        return p

    ctx = {}

    metrics_raw = pd.read_csv(path("model_comparison_metrics.csv"), dtype=str)
    ctx["metrics_raw"] = metrics_raw
    ctx["metrics"] = {}
    ctx["improvements"] = {}
    for _, row in metrics_raw.iterrows():
        if row["Model"].startswith("%"):
            ctx["improvements"][row["Model"]] = {m: float(row[m].rstrip("%")) for m in METRICS}
        else:
            if row["Model"] not in MODEL_LABELS:
                raise KeyError(f"Unknown model '{row['Model']}' in model_comparison_metrics.csv; add it to MODEL_LABELS.")
            ctx["metrics"][row["Model"]] = {m: float(row[m]) for m in METRICS}

    ctx["ablation_raw"] = pd.read_csv(path("leakage_ablation.csv"), dtype=str)
    ctx["ablation"] = {
        (r["Model"], r["Feature Set"]): {m: float(r[m]) for m in METRICS}
        for _, r in ctx["ablation_raw"].iterrows() if not r["Model"].startswith("%")
    }
    ctx["ablation_change"] = next(
        {m: float(r[m].rstrip("%")) for m in METRICS}
        for _, r in ctx["ablation_raw"].iterrows() if r["Model"].startswith("%")
    )

    ctx["age"] = pd.read_csv(path("residual_analysis_by_age.csv"))
    ctx["age"]["label"] = ctx["age"]["age_band"].map(band_label)
    ctx["position"] = pd.read_csv(path("residual_analysis_by_position.csv"))
    ctx["tier"] = pd.read_csv(path("residual_analysis_by_tier.csv"))
    ctx["above"] = pd.read_csv(path("undervalued_top20.csv"))
    ctx["below"] = pd.read_csv(path("overvalued_top20.csv"))
    ctx["importances"] = pd.read_csv(path("feature_importances.csv"))
    with open(path("smearing_factor.json")) as f:
        ctx["smear"] = json.load(f)
    with open(path("overall_bias.json")) as f:
        ctx["bias"] = json.load(f)

    con = duckdb.connect()
    preds = path("test_predictions.parquet").replace("\\", "/")
    n_test, test_start, test_end, n_players = con.execute(
        f"SELECT count(*), min(valuation_date), max(valuation_date), count(DISTINCT player_id) FROM '{preds}'"
    ).fetchone()
    position_list = ", ".join(f"'{p}'" for p in PRIMARY_POSITIONS)
    ctx["excluded_positions"] = con.execute(
        f"""SELECT coalesce(position, 'NULL'), count(*) FROM '{preds}'
            WHERE position IS NULL OR position NOT IN ({position_list})
            GROUP BY 1 ORDER BY 1"""
    ).fetchall()

    data_file = data_path.replace("\\", "/")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Analysis table not found: {data_path}. Run `python run_pipeline.py` first.")
    n_rows, n_train, n_test_table, last_valuation = con.execute(
        f"""SELECT count(*),
                   count(*) FILTER (WHERE valuation_date < DATE '{SPLIT_DATE}'),
                   count(*) FILTER (WHERE valuation_date >= DATE '{SPLIT_DATE}'),
                   max(valuation_date)
            FROM '{data_file}'"""
    ).fetchone()
    ctx["zero_goals_pct"] = dict(con.execute(
        f"""SELECT position, avg(CASE WHEN goals_last_365d = 0 THEN 1.0 ELSE 0.0 END) * 100
            FROM '{data_file}' GROUP BY 1"""
    ).fetchall())
    ctx["contract_days_by_year"] = con.execute(
        f"""SELECT year(valuation_date), median(contract_days_remaining)
            FROM '{data_file}' GROUP BY 1 ORDER BY 1"""
    ).fetchall()

    n_test_zero, n_test_no_minutes = con.execute(
        f"""SELECT count(*) FILTER (WHERE minutes_last_365d = 0), count(*) FILTER (WHERE minutes_last_365d IS NULL)
            FROM '{data_file}' WHERE valuation_date >= DATE '{SPLIT_DATE}'"""
    ).fetchone()
    if n_test_no_minutes:
        raise ValueError(f"{n_test_no_minutes:,} test rows have a missing minutes_last_365d in {data_path}.")
    ctx["zero_minutes_by_tier"] = con.execute(
        f"""SELECT league_tier, count(*), avg(CASE WHEN minutes_last_365d = 0 THEN 1.0 ELSE 0.0 END) * 100
            FROM '{data_file}' WHERE valuation_date >= DATE '{SPLIT_DATE}' GROUP BY 1 ORDER BY 1"""
    ).fetchall()
    ctx["zero_minutes_errors"] = zero_minutes_errors(preds)
    if sum(r["rows"] for r in ctx["zero_minutes_errors"] if r["zero_minutes"]) != n_test_zero:
        raise ValueError("Zero-minute test rows differ between the analysis table and test_predictions.parquet.")

    # games.csv is optional: it only supplies the match-data date in the snapshot text.
    games_file = os.path.join(raw_data_dir, "games.csv")
    if os.path.exists(games_file):
        last_match = con.execute(
            f"SELECT max(date) FROM read_csv_auto('{games_file.replace(chr(92), '/')}')"
        ).fetchone()[0]
    else:
        last_match = None
        print(f"Note: {games_file} not found; the README will state the valuation date range only.")

    if n_test != n_test_table or n_test != ctx["bias"]["test_rows"]:
        raise ValueError(
            f"Test row counts disagree: test_predictions.parquet={n_test:,}, "
            f"analysis table >= {SPLIT_DATE}: {n_test_table:,}, overall_bias.json={ctx['bias']['test_rows']:,}."
        )
    if int(ctx["position"]["Count"].sum()) + sum(c for _, c in ctx["excluded_positions"]) != n_test:
        raise ValueError("Position table rows plus excluded rows do not add up to the test set size.")

    ctx.update(
        n_rows=n_rows, n_train=n_train, n_test=n_test, n_players=n_players,
        test_start=f"{test_start:%Y-%m-%d}", test_end=f"{test_end:%Y-%m-%d}",
        last_valuation=f"{last_valuation:%Y-%m-%d}",
        last_match=f"{last_match:%Y-%m-%d}" if last_match is not None else None,
        n_test_zero_minutes=n_test_zero,
        prediction_cap_eur=float(np.exp(LOG_PRED_CEILING)),
        prediction_floor_eur=float(np.exp(LOG_PRED_FLOOR)),
    )
    return ctx


# -------------------------------------------------------------
# Section builders
# -------------------------------------------------------------
def section_tldr(ctx):
    m = ctx["metrics"]
    best = min(m, key=lambda k: m[k]["Log RMSE"])
    imp = ctx["improvements"][NAIVE_IMPROVEMENT_ROW]
    naive, xgb = m[NAIVE_MODEL], m[XGB_MODEL]
    return "\n".join([
        "> **TL;DR**",
        "> - **Question:** can on-pitch and league features (playing time, goals and assists, age, position, "
        "competition and league tier) predict a player's Transfermarkt market value?",
        f"> - **Best model:** {MODEL_LABELS[best]}, with the lowest Log RMSE of the {len(m)} models "
        f"({m[best]['Log RMSE']:.4f}) on {ctx['n_test']:,} test valuations from {ctx['test_start']} to {ctx['test_end']}. "
        f"Headline models use the clean feature set: {len(SNAPSHOT_FEATURES)} contract and club features taken from "
        "a later snapshot are dropped (see Leakage ablation).",
        f"> - **XGBoost vs the naive baseline** (median value for the player's position and league tier): "
        f"Log RMSE is {change_word(imp['Log RMSE'])} ({naive['Log RMSE']:.4f} → {xgb['Log RMSE']:.4f}) and "
        f"Euro MAE is {change_word(imp['Euro MAE'])} ({eur(naive['Euro MAE'])} → {eur(xgb['Euro MAE'])}).",
        "> - **Main limitation:** the features carry no signal for potential or hype, so young stars and "
        "superstars are predicted far below their market value (see Limitations).",
    ])


def section_snapshot(ctx):
    return "\n".join([
        "> [!IMPORTANT]",
        f"> **Static snapshot: {snapshot_dates_text(ctx)}.**",
        "> The data used in this project is a fixed historical snapshot and is **not being refreshed**. ",
        "> It is **not live data** and should not be treated as current transfer market information. Player "
        "ages are computed at each valuation date; positions, heights, contracts, and club attributes reflect "
        "the snapshot (see the point-in-time audit in Section 4).",
    ])


def section_split_counts(ctx):
    return "\n".join([
        f"   - **Training Set**: All valuations recorded before `{SPLIT_DATE}` ({ctx['n_train']:,} observations).",
        f"   - **Testing Set**: All valuations recorded from `{SPLIT_DATE}` onward ({ctx['n_test']:,} observations, "
        f"up to {ctx['test_end']}).",
    ])


def section_sparsity(ctx):
    z = ctx["zero_goals_pct"]
    return (
        f"- Performance metrics for Goalkeepers and Defenders exhibit structural zeros "
        f"({pct(z['Goalkeeper'])} of Goalkeeper and {pct(z['Defender'])} of Defender valuations had 0 goals "
        f"in the preceding 365 days, across all {ctx['n_rows']:,} rows of the analysis table)."
    )


def section_benchmark(ctx):
    rows = []
    for _, r in ctx["metrics_raw"].iterrows():
        if r["Model"].startswith("%"):
            rows.append([r["Model"]] + [r[m] for m in METRICS])
        else:
            v = ctx["metrics"][r["Model"]]
            rows.append([
                MODEL_LABELS[r["Model"]], f"{v['Log RMSE']:.4f}", f"{v['Log MAE']:.4f}",
                eur(v["Euro RMSE"]), eur(v["Euro MAE"]),
            ])
    intro = (
        f"All models predict ln(market value) and are scored on the held-out test set: {ctx['n_test']:,} valuations "
        f"from {ctx['test_start']} to {ctx['test_end']}. Ridge and XGBoost use the clean feature set (snapshot "
        "features dropped). Every model's log predictions are clipped to the same range before scoring, so log and "
        "euro metrics use identical predictions. Euro metrics use exp(log prediction); the smear-corrected "
        f"row multiplies that by the smearing factor. The naive baseline predicts the training-set median log value "
        f"for the player's position × league tier, falling back to the position median, then the overall median, "
        f"for groups with fewer than {NAIVE_MIN_GROUP_ROWS} training rows."
    )
    note = (
        "The two improvement rows are copied from `models/model_comparison_metrics.csv` and compare the "
        "uncorrected XGBoost row with each baseline (positive = XGBoost error is lower)."
    )
    table = md_table(["Model"] + METRICS, [":---", "---:", "---:", "---:", "---:"], rows)
    return f"{intro}\n\n{table}\n\n{note}"


def section_leakage_ablation(ctx):
    ab, change = ctx["ablation"], ctx["ablation_change"]
    feature_lines = "\n".join(f"- `{f}`: {SNAPSHOT_REASONS[f]}." for f in SNAPSHOT_FEATURES)

    rows = []
    for _, r in ctx["ablation_raw"].iterrows():
        if r["Model"].startswith("%"):
            rows.append([r["Model"], r["Feature Set"]] + [r[m] for m in METRICS])
        else:
            v = ab[(r["Model"], r["Feature Set"])]
            rows.append([MODEL_LABELS[r["Model"]], r["Feature Set"], f"{v['Log RMSE']:.4f}", f"{v['Log MAE']:.4f}",
                         eur(v["Euro RMSE"]), eur(v["Euro MAE"])])
    table = md_table(["Model", "Feature Set"] + METRICS, [":---", ":---", "---:", "---:", "---:", "---:"], rows)

    def worse_better(p):
        return f"{abs(p):.1f}% {'worse' if p > 0 else 'better' if p < 0 else 'unchanged'}"

    xgb_full, xgb_clean = ab[(XGB_MODEL, FEATURE_SET_FULL)], ab[(XGB_MODEL, FEATURE_SET_CLEAN)]
    ridge_same = ab[(RIDGE_MODEL, FEATURE_SET_FULL)] == ab[(RIDGE_MODEL, FEATURE_SET_CLEAN)]
    ridge_note = (
        " Ridge results are identical for both feature sets because the Ridge baseline uses none of the snapshot "
        "features (it uses age, age², minutes, goal contributions per 90, position, and league tier)."
        if ridge_same else ""
    )
    effect = (
        f"Dropping them makes XGBoost {worse_better(change['Log RMSE'])} on Log RMSE "
        f"({xgb_full['Log RMSE']:.4f} → {xgb_clean['Log RMSE']:.4f}) and {worse_better(change['Euro MAE'])} on "
        f"Euro MAE ({eur(xgb_full['Euro MAE'])} → {eur(xgb_clean['Euro MAE'])}).{ridge_note} Every headline number "
        "in this README (benchmark, smearing, residuals, rankings, feature importances) comes from the clean models; "
        "the full-feature results appear only in this table and in `models/leakage_ablation.csv`."
    )
    known = "\n".join([
        "Known remaining issues, not fixed by the ablation:",
        "- **Imputation inside the cross-validation folds.** Imputation medians are computed on the whole training "
        "set and then reused inside the `TimeSeriesSplit` folds that choose `max_depth` and fit the smearing factor, "
        "so each validation fold's fill values include its own rows. This is a small, known leak within the training "
        "set; the test set is not affected.",
        "- **Club at valuation time is assumed.** The club and competition of each valuation come from "
        "`player_valuations.current_club_id` and `player_club_domestic_competition_id`. These change over a "
        "player's history, so they are treated as the club at the valuation date, but the dataset does not "
        "document this.",
        "- **Other snapshot attributes are kept.** Position, sub-position, foot, and height are current values from "
        "`players.csv` (usually stable over a career), and the Tier 3/4 league classification uses the current "
        "`competitions.csv`.",
    ])
    return (
        "These features come from a snapshot taken after the valuation date (see the point-in-time audit in "
        "Section 4), so the headline models drop them:\n"
        f"{feature_lines}\n\n"
        "Each model is trained with the full feature set and with these features dropped, using the same split, "
        "seeds, and `max_depth` search (from `models/leakage_ablation.csv`; the last row is the percent change of "
        f"clean vs full, where positive means the clean model's error is higher):\n\n{table}\n\n{effect}\n\n{known}"
    )


def section_smearing(ctx):
    m, s, b = ctx["metrics"], ctx["smear"], ctx["bias"]
    raw, smeared = m[XGB_MODEL], m[XGB_SMEARED_MODEL]
    age = ctx["age"].set_index("label")
    tier = ctx["tier"].set_index("league_tier")

    def metric_line(name):
        before, after = raw[name], smeared[name]
        reduction = (before - after) / before * 100
        verdict = "improved" if after < before else "got worse"
        return f"- {name}: {eur(before)} → {eur(after)} ({change_word(reduction)}; {verdict})."

    def segment_line(label, row):
        before, after = row["Mean_Bias_EUR_Uncorrected"], row["Mean_Bias_EUR_Smeared"]
        return (f"- {label}: {eur_signed(before)} → {eur_signed(after)} "
                f"(model {bias_direction(before)} before, {bias_direction(after)} after).")

    paragraphs = [
        "The models predict the log of market value. Converting a log prediction back with exp() gives a "
        "median-type estimate (roughly the typical value for players with those features), not the average. "
        "Market values are heavily right-skewed, so the average sits well above the median and plain exp() "
        f"predictions run low in euros: on the test set the uncorrected XGBoost predictions average "
        f"{eur(b['mean_predicted_eur_uncorrected'])} against an actual average of {eur(b['mean_actual_eur'])}.",

        f"To correct this, euro predictions are multiplied by a Duan smearing factor of **{s['smearing_factor']:.4f}** "
        "(the mean of exp(residual) over out-of-sample log residuals). The residuals come from the last "
        f"`TimeSeriesSplit(n_splits=3)` fold of the training set: the clean XGBoost settings refit on the first "
        f"{s['fold_train_rows']:,} training rows and scored on the next {s['fold_validation_rows']:,} rows "
        f"({s['fold_validation_start']} to {s['fold_validation_end']}). No test data is used to fit the factor.",

        "Effect on the test set (from `models/model_comparison_metrics.csv` and `models/overall_bias.json`):\n"
        + "\n".join([metric_line("Euro RMSE"), metric_line("Euro MAE")])
        + f"\n- Log RMSE and Log MAE are identical in both rows ({raw['Log RMSE']:.4f} and {raw['Log MAE']:.4f}) "
          "because they are computed on the uncorrected log predictions; smearing only changes euro predictions."
        + f"\n- Overall mean bias (actual − predicted): {eur_signed(b['mean_bias_eur_uncorrected'])} → "
          f"{eur_signed(b['mean_bias_eur_smeared'])}."
        + ("\n\nSmearing targets the average, so it reduces mean bias and squared error, but absolute error is "
           "smallest for median-type predictions, which is why Euro MAE can rise after the correction, as it "
           "does here."
           if smeared["Euro MAE"] > raw["Euro MAE"] else ""),

        "One global factor scales every prediction by the same amount, so it removes most of the overall bias "
        "but not the bias inside segments. It shifts each segment's bias rather than removing it "
        "(from `models/residual_analysis_by_tier.csv` and `models/residual_analysis_by_age.csv`):\n"
        + "\n".join([
            segment_line(TIER_1, tier.loc[TIER_1]),
            segment_line(f"Age {age.index[0]}", age.iloc[0]),
            segment_line(f"Age {age.index[-1]}", age.iloc[-1]),
        ])
        + "\n\nThe segment-level bias is therefore not solved; correcting it would need segment-specific "
          "factors or better features.",
    ]
    return "\n\n".join(paragraphs)


def section_feature_importance(ctx):
    top = ctx["importances"].head(10)
    return "\n".join(
        f"{i}. `{r.feature}` (gain {r.gain:,.1f})" for i, r in enumerate(top.itertuples(), start=1)
    )


def residual_table(df, label_col, header):
    rows = [[
        r[label_col], f"{int(r['Count']):,}", eur(r["Avg_Actual_EUR"]),
        eur_signed(r["Mean_Bias_EUR_Uncorrected"]), eur_signed(r["Mean_Bias_EUR_Smeared"]),
        eur(r["MAE_EUR_Uncorrected"]), eur(r["MAE_EUR_Smeared"]),
        f"{r['Mean_Bias_Log']:+.3f}", f"{r['MAE_Log']:.3f}",
    ] for _, r in df.iterrows()]
    headers = [
        header, "Rows", "Avg Actual", "Mean Bias (uncorrected)", "Mean Bias (smear-corrected)",
        "Euro MAE (uncorrected)", "Euro MAE (smear-corrected)", "Log Bias", "Log MAE",
    ]
    return md_table(headers, [":---"] + ["---:"] * 8, rows)


def section_residuals_age(ctx):
    return residual_table(ctx["age"], "label", "Age Band")


def section_residuals_position(ctx):
    excluded = ctx["excluded_positions"]
    n_excluded = sum(c for _, c in excluded)
    names = ", ".join(f'"{p}" ({c:,} rows)' for p, c in excluded)
    footnote = (
        f"*Excludes test rows whose position is {names}, so this table covers "
        f"{int(ctx['position']['Count'].sum()):,} rows instead of the {ctx['n_test']:,} in the age and tier tables.*"
        if n_excluded else "*No test rows are excluded from this table.*"
    )
    return residual_table(ctx["position"], "position", "Position") + "\n\n" + footnote


def section_residuals_tier(ctx):
    return residual_table(ctx["tier"], "league_tier", "League Tier")


def section_findings_age(ctx):
    age = ctx["age"].set_index("label")
    prospect = next(l for l in age.index if l.startswith(PROSPECT_BAND))
    late = next(l for l in age.index if l.startswith(LATE_CAREER_BAND))
    worst_log = age["MAE_Log"].idxmax()
    lowest_eur = age["MAE_EUR_Uncorrected"].idxmin()
    p, v = age.loc[prospect], age.loc[late]

    lines = [
        f"- **Largest relative error:** {worst_log} has the highest log MAE ({age.loc[worst_log, 'MAE_Log']:.3f}).",
        f"- **{prospect}:** the model {bias_direction(p['Mean_Bias_EUR_Uncorrected'])} on average: mean bias "
        f"{eur_signed(p['Mean_Bias_EUR_Uncorrected'])} uncorrected, {eur_signed(p['Mean_Bias_EUR_Smeared'])} after "
        f"smearing, log bias {p['Mean_Bias_Log']:+.3f}. Hypothesis (not tested here): Transfermarkt values young "
        "players partly on expected development, which features built from past minutes and output cannot see.",
        f"- **{late}:** in log terms the model {bias_direction(v['Mean_Bias_Log'])} the typical player (log bias "
        f"{v['Mean_Bias_Log']:+.3f}). In euros the uncorrected mean bias is {eur_signed(v['Mean_Bias_EUR_Uncorrected'])} "
        f"(model {bias_direction(v['Mean_Bias_EUR_Uncorrected'])}) and the smear-corrected one is "
        f"{eur_signed(v['Mean_Bias_EUR_Smeared'])} (model {bias_direction(v['Mean_Bias_EUR_Smeared'])})."
        + (" Log bias weights every player equally, while euro bias is dominated by the few high-value players, so "
           "the two can differ in sign, as they do here."
           if np.sign(v["Mean_Bias_Log"]) != np.sign(v["Mean_Bias_EUR_Uncorrected"]) else "")
        + " Hypothesis (not tested here): career totals keep growing with age while market value falls.",
    ]
    smallest = (f"- **Smallest euro error:** {lowest_eur} has the lowest uncorrected euro MAE "
                f"({eur(age.loc[lowest_eur, 'MAE_EUR_Uncorrected'])})")
    if age["Avg_Actual_EUR"].idxmin() == lowest_eur:
        smallest += (f", largely because it also has the lowest values (average actual "
                     f"{eur(age.loc[lowest_eur, 'Avg_Actual_EUR'])}); its log MAE is "
                     f"{age.loc[lowest_eur, 'MAE_Log']:.3f}.")
    else:
        smallest += "."
    lines.append(smallest)
    return "\n".join(lines)


def section_findings_position(ctx):
    pos = ctx["position"].set_index("position")
    lowest_eur = pos["MAE_EUR_Uncorrected"].idxmin()
    worst_log = pos["MAE_Log"].idxmax()
    lowest_value = pos["Avg_Actual_EUR"].idxmin()
    top2 = pos["MAE_EUR_Uncorrected"].sort_values(ascending=False).index[:2]

    lines = []
    if lowest_eur == worst_log:
        line = (f"- **{lowest_eur}s** have the lowest uncorrected euro MAE "
                f"({eur(pos.loc[lowest_eur, 'MAE_EUR_Uncorrected'])}) but the highest log MAE "
                f"({pos.loc[worst_log, 'MAE_Log']:.3f}).")
        if lowest_value == lowest_eur:
            line += (f" Their euro errors look small because their values are the lowest of the four positions "
                     f"(average actual {eur(pos.loc[lowest_value, 'Avg_Actual_EUR'])}); relative to value, they "
                     "are the hardest position to predict.")
        else:
            line += " Relative to value, they are the hardest position to predict."
        lines.append(line)
    else:
        lines.append(f"- **Lowest euro MAE:** {lowest_eur} ({eur(pos.loc[lowest_eur, 'MAE_EUR_Uncorrected'])}).")
        lines.append(f"- **Highest log MAE:** {worst_log} ({pos.loc[worst_log, 'MAE_Log']:.3f}).")
    if worst_log == "Goalkeeper":
        lines.append("- Hypothesis (not tested here): goals and assists carry almost no information about "
                     "goalkeepers, which may explain their higher relative error.")
    lines.append(
        f"- **Highest euro MAE:** {top2[0]} ({eur(pos.loc[top2[0], 'MAE_EUR_Uncorrected'])}) and "
        f"{top2[1]} ({eur(pos.loc[top2[1], 'MAE_EUR_Uncorrected'])}). Hypothesis (not tested here): the most "
        "expensive players, whose premiums the features do not capture, are concentrated in these positions."
    )
    if (pos["Mean_Bias_EUR_Uncorrected"] > 0).all():
        lines.append("- Uncorrected mean bias is positive for every position, so the model under-predicts all of them "
                     "on average before smearing.")
    return "\n".join(lines)


def section_findings_tier(ctx):
    tier = ctx["tier"].set_index("league_tier")
    abs_error = tier["Count"] * tier["MAE_EUR_Uncorrected"]
    top_eur = tier["MAE_EUR_Uncorrected"].idxmax()
    worst_log = tier["MAE_Log"].idxmax()
    t1 = tier.loc[TIER_1]
    return "\n".join([
        f"- **Largest euro error:** {top_eur} has the highest uncorrected euro MAE "
        f"({eur(tier.loc[top_eur, 'MAE_EUR_Uncorrected'])}) and accounts for "
        f"{pct(abs_error[top_eur] / abs_error.sum() * 100)} of the total absolute euro error on the test set while "
        f"making up {pct(tier.loc[top_eur, 'Count'] / tier['Count'].sum() * 100)} of test rows "
        f"(average actual value {eur(tier.loc[top_eur, 'Avg_Actual_EUR'])}).",
        f"- **Largest relative error:** {worst_log} has the highest log MAE ({tier.loc[worst_log, 'MAE_Log']:.3f}); "
        f"{TIER_1} has a log MAE of {t1['MAE_Log']:.3f}.",
        f"- **{TIER_1} bias:** the model {bias_direction(t1['Mean_Bias_EUR_Uncorrected'])} by "
        f"{eur(abs(t1['Mean_Bias_EUR_Uncorrected']))} on average before smearing (mean bias "
        f"{eur_signed(t1['Mean_Bias_EUR_Uncorrected'])}) and {bias_direction(t1['Mean_Bias_EUR_Smeared'])} by "
        f"{eur(abs(t1['Mean_Bias_EUR_Smeared']))} after (mean bias {eur_signed(t1['Mean_Bias_EUR_Smeared'])}). "
        "Hypothesis (not tested here): elite-league values include premiums such as brand and commercial value "
        "that the features do not capture.",
    ])


def section_ranking_scope(ctx):
    above_t1 = int((ctx["above"]["league_tier"] == TIER_1).sum())
    below_t1 = int((ctx["below"]["league_tier"] == TIER_1).sum())
    return (
        f"Rankings use the test period, taking each player's most recent test valuation ({ctx['n_players']:,} "
        "players), and compare the model's prediction with the Transfermarkt value: Difference = Predicted − Actual. "
        f"Because the ranking uses euro differences, it is dominated by high-value players: {above_t1} of the "
        f"{len(ctx['above'])} players in the first table and {below_t1} of the {len(ctx['below'])} in the second "
        f"play in {TIER_1}."
    )


def ranking_table(df, cap_eur):
    rows = [[
        int(r["Rank"]), r["player_name"], f"{r['age']:.1f}", r["position"], r["club_name"],
        eur(r["actual_value_eur"]), eur(r["predicted_value_eur"]) + ("*" if r["predicted_value_eur"] >= cap_eur - 1 else ""),
        eur_signed(r["value_delta_eur"]), pct_signed(r["pct_difference"]),
    ] for _, r in df.iterrows()]
    table = md_table(
        ["Rank", "Player", "Age", "Position", "Club", "Actual", "Predicted", "Difference", "% Difference"],
        [":---:", ":---", "---:", ":---", ":---", "---:", "---:", "---:", "---:"],
        rows,
    )
    capped = df.loc[df["predicted_value_eur"] >= cap_eur - 1, "player_name"].tolist()
    if capped:
        table += (f"\n\n\\* Prediction hit the {eur(cap_eur)} cap applied in `src/models.py` "
                  f"({', '.join(capped)}); the model's raw prediction was at or above the cap.")
    return table


def section_top20_model_above(ctx):
    return ranking_table(ctx["above"], ctx["prediction_cap_eur"])


def section_top20_model_below(ctx):
    return ranking_table(ctx["below"], ctx["prediction_cap_eur"])


def section_limitations(ctx):
    below = ctx["below"]
    superstars = int((below["actual_value_eur"] >= SUPERSTAR_VALUE_EUR).sum())
    young = int((below["age"] < YOUNG_AGE).sum())
    (first_year, first_days), (last_year, last_days) = ctx["contract_days_by_year"][0], ctx["contract_days_by_year"][-1]
    snapshot_list = ", ".join(f"`{f}`" for f in SNAPSHOT_FEATURES)
    return "\n".join([
        f"- **Snapshot features are excluded from the headline model.** {snapshot_list} come from the current rows of "
        "`players.csv` and `clubs.csv`, not values as of the valuation date. For example, the median "
        f"`contract_days_remaining` is {first_days:,.0f} days for {first_year} valuations versus {last_days:,.0f} days "
        f"for {last_year}, which is what contracts signed after the valuation would produce. The headline models drop "
        "these features; the Leakage ablation section shows their effect.",
        "- **Some information from after the valuation date can still reach the model.** Position, sub-position, "
        "foot, and height are current values from `players.csv`; imputation medians inside the `TimeSeriesSplit` "
        "folds are computed on the whole training set; and the club at each valuation is assumed, not documented, "
        "to be the club at that date. Test metrics may therefore still be slightly optimistic.",
        f"- **Predictions are capped.** Ridge and XGBoost log predictions are clipped to [{LOG_PRED_FLOOR}, "
        f"ln({ctx['prediction_cap_eur']:,.0f})] in `src/models.py`, so their euro predictions lie between "
        f"{eur(ctx['prediction_floor_eur'])} and {eur(ctx['prediction_cap_eur'])}.",
        "- **Features describe on-pitch output, the player's profile, and league.** The headline model uses playing "
        "time, goals, assists, cards, career totals, age, position, sub-position, foot, height, competition, and "
        "league tier. There is no signal for potential, hype, brand value, contract situation, or injuries.",
        f"- **Large disagreements reflect model limits, not market errors.** Of the {len(below)} players the model "
        f"places furthest below market value, {superstars} are valued at {eur(SUPERSTAR_VALUE_EUR)} or more and "
        f"{young} are under {YOUNG_AGE}. These gaps show what the features cannot see; they are not proof that "
        "Transfermarkt is wrong.",
        "- **Market value is not a transfer fee.** The target is Transfermarkt's crowdsourced estimate, not the "
        "price clubs actually pay.",
        f"- **Static snapshot.** {snapshot_dates_text(ctx).capitalize()}. The data is not refreshed.",
    ])


def section_zero_minutes(ctx):
    n_zero, n_test = ctx["n_test_zero_minutes"], ctx["n_test"]
    by_tier = "; ".join(f"{tier} {share:.1f}% of {n:,}" for tier, n, share in ctx["zero_minutes_by_tier"])
    return (
        f"- **Most test valuations have no recent playing time.** {n_zero:,} of the {n_test:,} test valuations "
        f"({n_zero / n_test * 100:.1f}%) have `minutes_last_365d` = 0, i.e. no recorded minutes in the 365 days "
        f"before the valuation. By league tier: {by_tier}. For these rows the model has no recent playing-time "
        "signal. Hypothesis (not verified "
        "here): the appearance data covers fewer of the competitions these players play in, so some zeros mean "
        "\"not recorded\" rather than \"did not play\"."
    )


def section_zero_minutes_errors(ctx):
    rows = [[
        "0 minutes" if r["zero_minutes"] else "More than 0 minutes", f"{r['rows']:,}",
        f"{r['rows'] / ctx['n_test'] * 100:.1f}%", eur(r["euro_mae"]), f"{r['log_mae']:.4f}",
    ] for r in ctx["zero_minutes_errors"]]
    table = md_table(["Minutes in previous 365 days", "Test rows", "Share", "Euro MAE", "Log MAE"],
                     [":---", "---:", "---:", "---:", "---:"], rows)
    zero = next(r for r in ctx["zero_minutes_errors"] if r["zero_minutes"])
    some = next(r for r in ctx["zero_minutes_errors"] if not r["zero_minutes"])
    reading = ""
    if zero["euro_mae"] < some["euro_mae"] and zero["log_mae"] > some["log_mae"] and zero["avg_actual"] < some["avg_actual"]:
        reading = (
            f"\n\nEuro MAE is lower for zero-minute rows ({eur(zero['euro_mae'])} vs {eur(some['euro_mae'])}) only "
            f"because those players have lower market values (average actual {eur(zero['avg_actual'])} vs "
            f"{eur(some['avg_actual'])}); relative error is higher (Log MAE {zero['log_mae']:.4f} vs "
            f"{some['log_mae']:.4f}), so the model is less accurate in relative terms where it has no recent playing time."
        )
    return ("Headline XGBoost errors (uncorrected) for test rows with and without previous-year minutes, from "
            f"`models/test_predictions.parquet`:\n\n{table}{reading}")


BUILDERS = {name: globals()[f"section_{name}"] for name in SECTIONS}


# -------------------------------------------------------------
# Marker replacement
# -------------------------------------------------------------
def replace_marker_sections(text, sections, source="README.md"):
    spans, errors = [], []
    for name in sections:
        begins = list(re.finditer(rf"^[ \t]*<!-- BEGIN:{re.escape(name)} -->[ \t]*$", text, re.M))
        ends = list(re.finditer(rf"^[ \t]*<!-- END:{re.escape(name)} -->[ \t]*$", text, re.M))
        if len(begins) != 1 or len(ends) != 1:
            errors.append(f"'{name}': found {len(begins)} BEGIN and {len(ends)} END markers (expected exactly 1 each)")
        elif ends[0].start() < begins[0].end():
            errors.append(f"'{name}': END marker appears before BEGIN marker")
        else:
            spans.append((begins[0].end(), ends[0].start(), name))
    if errors:
        raise ReadmeMarkerError(
            f"{source} marker problem(s):\n  - " + "\n  - ".join(errors)
            + "\nEach generated section needs '<!-- BEGIN:name -->' and '<!-- END:name -->' on their own lines."
        )
    for start, end, name in sorted(spans, reverse=True):
        text = text[:start] + "\n" + sections[name] + "\n" + text[end:]
    return text


def update_readme(
    readme_path: str = "README.md",
    models_dir: str = "models",
    data_path: str = "data/processed_market_valuations.parquet",
    raw_data_dir: str = ".",
):
    print("=" * 70)
    print("STEP 8: REFRESHING README.md FROM PIPELINE OUTPUTS")
    print("=" * 70)
    ctx = load_context(models_dir, data_path, raw_data_dir)
    sections = {name: build(ctx) for name, build in BUILDERS.items()}

    with open(readme_path, encoding="utf-8") as f:
        original = f.read()
    updated = replace_marker_sections(original, sections, source=readme_path)
    if updated != original:
        with open(readme_path, "w", encoding="utf-8") as f:
            f.write(updated)
        print(f"Updated {len(sections)} generated sections in {readme_path}.")
    else:
        print(f"{readme_path} already up to date ({len(sections)} generated sections).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Regenerate README.md sections from pipeline outputs.")
    parser.add_argument("--readme", type=str, default="README.md", help="Path to README.md.")
    parser.add_argument("--models-dir", type=str, default="models", help="Directory with model outputs.")
    parser.add_argument("--data", type=str, default="data/processed_market_valuations.parquet", help="Analysis table.")
    parser.add_argument("--raw-data-dir", type=str, default=".", help="Directory containing the raw CSVs.")
    args = parser.parse_args()
    update_readme(
        readme_path=args.readme, models_dir=args.models_dir, data_path=args.data, raw_data_dir=args.raw_data_dir
    )
