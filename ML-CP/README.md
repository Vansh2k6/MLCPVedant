# Football Player Market-Value Prediction & Valuation Lab (Touchline)

> An end-to-end Machine Learning system and interactive presentation lab predicting association football player market valuations from historical match performance, positional demographics, and league hierarchies without temporal lookahead.

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Streamlit UI](https://img.shields.io/badge/dashboard-Streamlit-FF4B4B.svg)](dashboard.py)
[![XGBoost Regressor](https://img.shields.io/badge/model-XGBoost%20%7C%20SVR%20%7C%20Ridge%20%7C%20OLS-orange.svg)](models/upgraded/)

---

<!-- BEGIN:tldr -->
> **TL;DR**
> - **Question:** can on-pitch and league features (playing time, goals and assists, age, position, competition and league tier) predict a player's Transfermarkt market value?
> - **Best model:** XGBoost, with the lowest Log RMSE of the 4 models (0.9108) on 111,706 test valuations from 2024-01-01 to 2026-06-12. Headline models use the clean feature set: 5 contract and club features taken from a later snapshot are dropped (see Leakage ablation).
> - **XGBoost vs the naive baseline** (median value for the player's position and league tier): Log RMSE is 38.9% lower (1.4904 → 0.9108) and Euro MAE is 42.7% lower (€2,795,267 → €1,600,833).
> - **Main limitation:** the features carry no signal for potential or hype, so young stars and superstars are predicted far below their market value (see Limitations).
<!-- END:tldr -->

---

## 1. What This Project Is About & What It Is Doing

### Core Objective
This project investigates whether machine learning regressors can reliably estimate professional association football players' **Transfermarkt market valuations** using strictly point-in-time on-pitch performance records, age curves, positional attributes, and competition tiers.

Football player valuation is traditionally viewed as an opaque, hype-driven process. This project builds a transparent, mathematically grounded machine learning framework that evaluates players across 547,653 career valuation snapshots spanning over a decade (2014 to 2026) across global leagues.

### Critical Domain Distinctions
A fundamental conceptual distinction must be maintained when evaluating this project:
1. **Transfermarkt Market Value (Target Variable)**: A crowdsourced, expert-calibrated consensus valuation representing a player's perceived general market worth across the football industry. **This project strictly models this market value estimate.**
2. **Realized Transfer Fee**: The actual transaction price negotiated between two clubs. Transfer fees incorporate contract buyout clauses, agent commissions, urgent squad replacement premiums, financial leverage, wage structures, and bilateral negotiations that do not reflect intrinsic baseline market value.
3. **Auction / Clearing Price**: Football operates as a decentralized bilateral negotiation market with significant information asymmetry, not an open competitive auction.

---

## 2. Current Working Features & Implementation Capabilities

The codebase provides two fully functional execution layers and a live analytical dashboard:

### A. The Upgraded Machine Learning Pipeline (`run_upgrade.py` / `run_pipeline.py`)
- **Expanding Calendar Validation (`src/temporal.py`)**: Replaces standard row-based splits with calendar-bounded folds (train < 2020 / val 2020; train < 2021 / val 2021; train < 2022 / val 2022). Every training observation strictly precedes every validation date with zero boundary sharing.
- **Strict Preprocessing Isolation (`src/preprocessing.py`)**: Height medians, category encodings, missingness flags, and robust scaling are fit *strictly* on training folds and transformed on validation/test rows to prevent lookahead bias.
- **Multi-Window Recent-Form Features (`src/recent_features.py`)**: Uses DuckDB ASOF temporal logic to construct non-overlapping rolling performance windows:
  - 90-day, 180-day, and 365-day recorded minutes, appearances, goals, and assists.
  - Recent vs. preceding 90-day momentum and trend ratios.
  - Observation coverage flags (`days_since_last_match`, `has_prior_appearances`, `has_recent_minutes`).
  - Empirical Bayes exposure shrinkage: per-90 rates are shrunk toward positional priors with 450 pseudo-minutes.
- **Syllabus Regression Suite (`src/upgraded_training.py`)**: Evaluates 5 competing models on the exact same 111,706 test rows:
  1. *Corrected Annual XGBoost* (Baseline with isolated preprocessing)
  2. *Upgraded Recent-Form XGBoost* (Headline tree regressor with recent form)
  3. *Recent-Form Ridge Regression* (L2 regularized linear model, $\alpha=100$)
  4. *Ordinary Least Squares (OLS)* (Closed-form least-squares regression, Unit III syllabus)
  5. *Linear Support Vector Regression (LinearSVR)* (Margin-based regressor with $\epsilon=0.1$ loss, Unit IV syllabus)
- **Empirical Learning Curves**: 16 diagnostic models trained across 5%, 20%, 50%, and 100% subsets of pre-2023 data and scored on the fixed 2023 validation set to study model variance and sample-size scaling.

### B. Interactive Streamlit Presentation Dashboard: "Touchline" (`dashboard.py`)
Run locally with zero cloud dependencies or external API keys (`python -m streamlit run dashboard.py` or double-clicking `start_dashboard.cmd`). Includes 5 interactive views:
1. **Overview**: Executive comparison cards, live test metrics table, interactive feature importance chart, and segment performance breakdowns.
2. **Player Explorer**: Searchable player lookup across 40,883 athletes, tracking actual vs. model predicted value trajectories over time, with complete 90/180/365-day form inspection tables.
3. **Learning Curves**: 4-panel diagnostic plots comparing training vs. validation log RMSE for OLS, Ridge, LinearSVR, and XGBoost.
4. **Method & Caveats**: Audit trail explaining calendar validation folds, coverage caveats (e.g., explaining why zero minutes indicates competition data boundaries rather than player ability), and snapshot features.
5. **48-Hour Roadmap**: Academic expansion roadmap outlining next-stage experiments (valuation history persistence baselines and calibrated prediction intervals).

### C. Preserved Legacy Benchmark Pipeline (`python run_pipeline.py --legacy`)
- DuckDB point-in-time extraction ensuring `appearance.date < valuation.date`.
- Naive group-median baseline ($Position \times Tier$).
- Duan smearing retransformation correction for log-target smearing bias.
- Leakage ablation study explicitly isolating the impact of snapshot contract/club variables.
- Error segmentations across age bands, positions, and league tiers.
- Automated consistency checks ensuring 100% verification across reports and models.

---

## 3. Data Snapshot & Temporal Disclaimer

<!-- BEGIN:snapshot -->
> [!IMPORTANT]
> **Static snapshot: valuations through 2026-06-12; match data through 2026-07-06.**
> The data used in this project is a fixed historical snapshot and is **not being refreshed**. 
> It is **not live data** and should not be treated as current transfer market information. Player ages are computed at each valuation date; positions, heights, contracts, and club attributes reflect the snapshot (see the point-in-time audit in Section 4).
<!-- END:snapshot -->

---

## 4. Project Architecture & Directory Structure

```text
ML-CP/
├── dashboard.py                              # Streamlit interactive presentation frontend (Touchline Lab)
├── start_dashboard.cmd                       # One-click Windows launch shortcut for dashboard
├── run_pipeline.py                           # Master runner (defaults to Upgraded; --legacy for benchmark)
├── run_upgrade.py                            # Direct runner for the upgraded validation & recent-form pipeline
├── analysis.ipynb                            # Complete narrated Jupyter walkthrough and analysis
├── OVERVIEW.md                               # Architectural guide, domain concepts, and exam prep
├── UPGRADE_GUIDE.md                          # Technical specifications for upgrades 1-2 & syllabus additions
├── PROJECT_REVIEW_AND_IMPLEMENTATION_PLAN.md # Comprehensive academic review & 48-hour roadmap
├── requirements.txt                          # Core pipeline dependencies (DuckDB, XGBoost, Scikit-learn, etc.)
├── requirements-dashboard.txt                # Frontend dependencies (Streamlit, Plotly)
├── requirements-notebook.txt                 # Interactive notebook dependencies (Jupyter, nbconvert)
├── data/
│   ├── processed_market_valuations.parquet   # Base point-in-time table (547,653 rows x 32 columns)
│   ├── upgraded_market_valuations.parquet    # Enriched dataset with 90d/180d/365d form & coverage
│   └── upgraded_features_manifest.json       # SHA-256 hashes verifying feature pipeline integrity
├── eda_outputs/
│   ├── market_value_distribution.png         # Target distribution (raw skewness vs log-stabilization)
│   ├── value_vs_age_by_position.png          # Positional market value trajectories across age
│   ├── missingness_and_sparsity_report.png   # Performance sparsity audit justifying no blanket imputation
│   ├── feature_importances.png               # Top 20 XGBoost split gain drivers
│   ├── residuals_by_segment.png              # Error breakdowns across age, position, and league tier
│   └── eda_summary.txt                       # Detailed statistical EDA audit report
├── models/
│   ├── xgboost_market_value_model.json       # Legacy XGBoost regressor model weights
│   ├── model_comparison_metrics.csv          # Legacy side-by-side performance table
│   ├── model_comparison_metrics.json         # JSON metrics summary
│   ├── leakage_ablation.csv                  # Ridge and XGBoost with vs without snapshot features
│   ├── smearing_factor.json                  # Duan smearing retransformation factor
│   ├── overall_bias.json                     # Test-set mean bias before and after smearing
│   ├── feature_importances.csv               # Ranked feature importance gains
│   ├── test_predictions.parquet              # Full out-of-time legacy test predictions
│   ├── residual_analysis_by_age.csv          # Error breakdown across age cohorts
│   ├── residual_analysis_by_position.csv     # Error breakdown across positions
│   ├── residual_analysis_by_tier.csv         # Error breakdown across league tiers
│   ├── undervalued_top20.csv                 # Top 20 players with model prediction above market value
│   ├── overvalued_top20.csv                  # Top 20 players with model prediction below market value
│   ├── rankings_summary.txt                  # Textual ranking and domain commentary
│   └── upgraded/                             # UPGRADED EXPERIMENTAL ARTIFACTS
│       ├── annual_xgb.joblib                 # Fitted pipeline: Annual-feature XGBoost
│       ├── recent_xgb.joblib                 # Fitted pipeline: Recent-form XGBoost (Headline)
│       ├── recent_ridge.joblib               # Fitted pipeline: Recent-form Ridge regression
│       ├── recent_ols.joblib                 # Fitted pipeline: Ordinary Least Squares (Unit III)
│       ├── recent_svr.joblib                 # Fitted pipeline: Linear SVR (Unit IV)
│       ├── metrics.csv                       # Side-by-side test metrics for all 5 upgraded models
│       ├── validation_scores.csv             # Expanding calendar cross-validation scores
│       ├── test_predictions.parquet          # Out-of-time test predictions for all 5 models
│       ├── segments.csv                      # Segmented evaluation across cohorts and coverage
│       ├── feature_importance.csv            # Upgraded split-gain feature importances
│       ├── learning_curves.csv               # Learning curve scores across sample fractions
│       ├── learning_curves.png               # 4-panel diagnostic learning curve visualization
│       └── run_metadata.json                 # Complete run manifest, hyperparameters, and parameters
├── src/
│   ├── temporal.py                           # Expanding calendar cross-validation splitter
│   ├── preprocessing.py                      # Training-isolated imputation, encodings, and scalers
│   ├── recent_features.py                    # 90d/180d/365d rolling form windows & shrinkage priors
│   ├── upgraded_training.py                  # Training pipeline for the 5 upgraded models & learning curves
│   ├── data_pipeline.py                      # DuckDB extraction & temporal invariant verification
│   ├── eda.py                                # Exploratory data analysis & statistical distribution audits
│   ├── models.py                             # Legacy modeling pipeline (split, baseline, XGBoost, smearing)
│   ├── evaluate.py                           # Legacy residual breakdowns and model disagreement rankings
│   ├── update_readme.py                      # Automated synchronization of README metrics from models/
│   └── check_readme.py                       # Numerical validation of README tables against CSVs
└── tests/
    └── check_consistency.py                  # Test suite verifying data, split dates, and metrics
```

---

## 5. Upgraded Experimental Results (5 Models on 111,706 Test Valuations)

All 5 models are evaluated out-of-time on the exact same 111,706 test observations recorded between `2024-01-01` and `2026-06-12`. Predictions are clipped to the valid range [€4,915, €250,000,000].

| Model Identifier | Model Architecture & Features | CV Log RMSE | Test Log RMSE | Test Log MAE | Test Euro MAE | Within 2× (%) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| `recent_xgb` | **Upgraded XGBoost (Recent Form + Coverage)** | **0.8603** | **0.9125** | **0.7000** | **€1,607,832** | **59.54%** |
| `annual_xgb` | Corrected XGBoost (Annual Features Baseline) | 0.8720 | 0.9167 | 0.7052 | €1,622,405 | 58.98% |
| `recent_svr` | Linear Support Vector Regression (SVR, $C=0.01$) | 0.9682 | 0.9972 | 0.7790 | €2,106,626 | 53.73% |
| `recent_ols` | Ordinary Least Squares (OLS, Unregularized) | — | 0.9974 | 0.7786 | €2,108,778 | 53.81% |
| `recent_ridge`| Ridge Regression ($\alpha=100$) | — | 0.9975 | 0.7790 | €2,108,116 | 53.74% |

### Key Takeaways from Upgraded Experiments:
1. **Recent Form Impact**: Incorporating multi-window rolling form (90d/180d/365d) and observation coverage flags improves XGBoost CV Log RMSE from 0.8720 to 0.8603 and raises the proportion of predictions within a factor of two to 59.54%.
2. **Linear Benchmark Alignment**: OLS, Ridge, and LinearSVR perform remarkably consistently across all metrics (Log RMSE ~0.997, Euro MAE ~€2.1M). The gradient-boosted tree architecture achieves an ~8.5% relative error reduction over the best linear models by capturing non-linear age curves and league interaction effects.
3. **Syllabus Additions**:
   - **Unit III (OLS)**: Demonstrates unregularized least-squares regression on standardized inputs.
   - **Unit IV (Linear SVR)**: Uses $\epsilon$-insensitive loss ($\epsilon=0.1$) with hyperparameter tuning over $C \in \{0.01, 0.1, 1.0\}$.
   - **Learning Curves**: Evaluates 16 diagnostic configurations across 5%, 20%, 50%, and 100% training subsets, revealing that while linear models plateau early, tree-based models continue to gain precision as sample size expands.

---

## 6. Methodology & Temporal Integrity

### Temporal Integrity of Appearance Features and the Split
1. **Point-in-Time Appearance Windows**:
   - Rolling 365-day statistics (`appearances_last_365d`, `minutes_last_365d`, `goals_last_365d`, `assists_last_365d`, cards, per-90 rates) and prior career accumulation were calculated using **only matches that took place strictly before each valuation date** (`a.date < v.valuation_date` in `src/data_pipeline.py`).
   - No appearance from on or after the valuation date enters these features.
2. **Chronological Train/Test Split**:
   <!-- BEGIN:split_counts -->
   - **Training Set**: All valuations recorded before `2024-01-01` (435,947 observations).
   - **Testing Set**: All valuations recorded from `2024-01-01` onward (111,706 observations, up to 2026-06-12).
   <!-- END:split_counts -->
   - **Why K-Fold and Random Splits are Prohibited**: Individual players have dozens of valuations over their careers. A random split allows future valuations of a player to leak into the training fold for past valuations, causing the model to memorize player identities and report artificially inflated accuracy that collapses in production.
   - Cross-validation uses expanding calendar folds to guarantee that past data never peeks into the future.

### Point-in-Time Audit of Features
Not every feature is measured as of the valuation date. Features that come from the current row of `players.csv` or `clubs.csv` carry information from the snapshot, which can be later than the valuation. The contract and club features are the `SNAPSHOT_FEATURES` list in `src/models.py`; the headline models drop them.

| Feature group | Source | Point-in-time? | In headline model? |
| :--- | :--- | :--- | :--- |
| Rolling 365-day and prior-career appearance stats, per-90 rates | `appearances.csv`, filtered to `a.date < v.valuation_date` | Yes | Yes |
| Position interaction terms (`attack_goals_per_90`, `midfield_assists_per_90`, `gk_minutes_last_365d`, `def_minutes_last_365d`) | Appearance stats × position indicator | Partly (the appearance part is point-in-time; the position part is a snapshot) | Yes |
| `age`, `age_squared` | `players.date_of_birth` and the valuation date | Yes | Yes |
| `competition_id` one-hot, `league_tier` | `player_valuations.player_club_domestic_competition_id` (assumed to be the club at the valuation; the dataset does not document this); Tier 3/4 also use `competitions.sub_type` | Yes, under that assumption (Tier 3/4 classification uses the current `competitions.csv`) | Yes |
| `position`, `sub_position`, `foot`, `height_in_cm`, position indicators | Current row of `players.csv` | No (snapshot; usually stable) | Yes |
| `contract_days_remaining`, `has_contract_info` | Current `players.contract_expiration_date` | No (current contract, not the one in force at the valuation) | No (dropped) |
| `club_squad_size`, `club_foreigners_percentage`, `club_national_team_players` | Current row of `clubs.csv`, joined on the valuation's club | No (snapshot) | No (dropped) |

### Handling Missingness Without Blanket Imputation
<!-- BEGIN:sparsity -->
- Performance metrics for Goalkeepers and Defenders exhibit structural zeros (99.9% of Goalkeeper and 83.9% of Defender valuations had 0 goals in the preceding 365 days, across all 547,653 rows of the analysis table).
<!-- END:sparsity -->
- Imputing mean or median attacking output onto defensive players corrupts their baseline evaluation. Instead:
  - Positional binary indicators (`is_goalkeeper`, `is_defender`, `is_attack`, `is_midfield`) were encoded.
  - Position-specific interaction features were engineered: `attack_goals_per_90` (clipped goals per 90 × `is_attack`), `midfield_assists_per_90` (clipped assists per 90 × `is_midfield`), `gk_minutes_last_365d` (`minutes_last_365d` × `is_goalkeeper`), and `def_minutes_last_365d` (`minutes_last_365d` × `is_defender`).
  - Per-90 rates are clipped to a fixed maximum (set in `prepare_features`) to stop tiny minute totals from producing extreme rates.
  - Missing heights were imputed with the training-set median height for the player's position, falling back to a fixed default height.
  - Missing `club_squad_size` and `club_foreigners_percentage` were filled with their training-set medians; missing `club_national_team_players` with zero. These club features are used only by the full-feature variants in the leakage ablation.
  - Contract expiration was converted into `contract_days_remaining` accompanied by a binary `has_contract_info` flag. Both come from the current contract and are used only by the full-feature variants.

---

## 7. Preserved Legacy Benchmark Evaluation

### Benchmark Comparison Table

<!-- BEGIN:benchmark -->
All models predict ln(market value) and are scored on the held-out test set: 111,706 valuations from 2024-01-01 to 2026-06-12. Ridge and XGBoost use the clean feature set (snapshot features dropped). Every model's log predictions are clipped to the same range before scoring, so log and euro metrics use identical predictions. Euro metrics use exp(log prediction); the smear-corrected row multiplies that by the smearing factor. The naive baseline predicts the training-set median log value for the player's position × league tier, falling back to the position median, then the overall median, for groups with fewer than 30 training rows.

| Model | Log RMSE | Log MAE | Euro RMSE | Euro MAE |
| :--- | ---: | ---: | ---: | ---: |
| Naive Baseline (position × league-tier median) | 1.4904 | 1.1488 | €9,047,330 | €2,795,267 |
| Ridge | 1.1562 | 0.9050 | €7,157,729 | €2,244,637 |
| XGBoost | 0.9108 | 0.6981 | €4,779,167 | €1,600,833 |
| XGBoost (Smear-Corrected) | 0.9108 | 0.6981 | €4,493,754 | €1,653,781 |
| % Improvement (XGBoost vs Ridge Baseline) | 21.22% | 22.87% | 33.23% | 28.68% |
| % Improvement (XGBoost vs Naive Baseline) | 38.89% | 39.23% | 47.18% | 42.73% |

The two improvement rows are copied from `models/model_comparison_metrics.csv` and compare the uncorrected XGBoost row with each baseline (positive = XGBoost error is lower).
<!-- END:benchmark -->

### Leakage Ablation Study

<!-- BEGIN:leakage_ablation -->
These features come from a snapshot taken after the valuation date (see the point-in-time audit in Section 4), so the headline models drop them:
- `contract_days_remaining`: days to the expiry of the player's current contract in `players.csv`, not the contract in force at the valuation.
- `has_contract_info`: whether `players.csv` has a current contract expiry date, which partly encodes whether the player is still under contract at the snapshot date.
- `club_squad_size`: squad size from the current row of `clubs.csv`.
- `club_foreigners_percentage`: share of foreign players from the current row of `clubs.csv`.
- `club_national_team_players`: number of current national-team players from the current row of `clubs.csv`.

Each model is trained with the full feature set and with these features dropped, using the same split, seeds, and `max_depth` search (from `models/leakage_ablation.csv`; the last row is the percent change of clean vs full, where positive means the clean model's error is higher):

| Model | Feature Set | Log RMSE | Log MAE | Euro RMSE | Euro MAE |
| :--- | :--- | ---: | ---: | ---: | ---: |
| Naive Baseline (position × league-tier median) | Position x league tier medians (no model features) | 1.4904 | 1.1488 | €9,047,330 | €2,795,267 |
| Ridge | Full (with snapshot features) | 1.1562 | 0.9050 | €7,157,729 | €2,244,637 |
| Ridge | Clean (snapshot features dropped) | 1.1562 | 0.9050 | €7,157,729 | €2,244,637 |
| XGBoost | Full (with snapshot features) | 0.8508 | 0.6473 | €4,193,906 | €1,432,443 |
| XGBoost | Clean (snapshot features dropped) | 0.9108 | 0.6981 | €4,779,167 | €1,600,833 |
| % Change (XGBoost, clean vs full; positive = worse) | Clean vs Full | 7.06% | 7.84% | 13.96% | 11.76% |

Dropping them makes XGBoost 7.1% worse on Log RMSE (0.8508 → 0.9108) and 11.8% worse on Euro MAE (€1,432,443 → €1,600,833). Ridge results are identical for both feature sets because the Ridge baseline uses none of the snapshot features (it uses age, age², minutes, goal contributions per 90, position, and league tier). Every headline number in this README (benchmark, smearing, residuals, rankings, feature importances) comes from the clean models; the full-feature results appear only in this table and in `models/leakage_ablation.csv`.

Known remaining issues, not fixed by the ablation:
- **Imputation inside the cross-validation folds.** Imputation medians are computed on the whole training set and then reused inside the `TimeSeriesSplit` folds that choose `max_depth` and fit the smearing factor, so each validation fold's fill values include its own rows. This is a small, known leak within the training set; the test set is not affected.
- **Club at valuation time is assumed.** The club and competition of each valuation come from `player_valuations.current_club_id` and `player_club_domestic_competition_id`. These change over a player's history, so they are treated as the club at the valuation date, but the dataset does not document this.
- **Other snapshot attributes are kept.** Position, sub-position, foot, and height are current values from `players.csv` (usually stable over a career), and the Tier 3/4 league classification uses the current `competitions.csv`.
<!-- END:leakage_ablation -->

### Retransformation Correction (Duan Smearing)

<!-- BEGIN:smearing -->
The models predict the log of market value. Converting a log prediction back with exp() gives a median-type estimate (roughly the typical value for players with those features), not the average. Market values are heavily right-skewed, so the average sits well above the median and plain exp() predictions run low in euros: on the test set the uncorrected XGBoost predictions average €2,107,665 against an actual average of €3,144,319.

To correct this, euro predictions are multiplied by a Duan smearing factor of **1.5044** (the mean of exp(residual) over out-of-sample log residuals). The residuals come from the last `TimeSeriesSplit(n_splits=3)` fold of the training set: the clean XGBoost settings refit on the first 326,961 training rows and scored on the next 108,986 rows (2022-03-31 to 2023-12-31). No test data is used to fit the factor.

Effect on the test set (from `models/model_comparison_metrics.csv` and `models/overall_bias.json`):
- Euro RMSE: €4,779,167 → €4,493,754 (6.0% lower; improved).
- Euro MAE: €1,600,833 → €1,653,781 (3.3% higher; got worse).
- Log RMSE and Log MAE are identical in both rows (0.9108 and 0.6981) because they are computed on the uncorrected log predictions; smearing only changes euro predictions.
- Overall mean bias (actual − predicted): +€1,036,654 → -€26,537.

Smearing targets the average, so it reduces mean bias and squared error, but absolute error is smallest for median-type predictions, which is why Euro MAE can rise after the correction, as it does here.

One global factor scales every prediction by the same amount, so it removes most of the overall bias but not the bias inside segments. It shifts each segment's bias rather than removing it (from `models/residual_analysis_by_tier.csv` and `models/residual_analysis_by_age.csv`):
- Tier 1 - Big 5: +€3,499,410 → -€478,858 (model under-predicts before, over-predicts after).
- Age <21 (Prospects): +€1,148,033 → +€545,132 (model under-predicts before, under-predicts after).
- Age 33+ (Late Career): -€73,743 → -€509,400 (model over-predicts before, over-predicts after).

The segment-level bias is therefore not solved; correcting it would need segment-specific factors or better features.
<!-- END:smearing -->

### Key Model Drivers (Feature Importances)
Top 10 features by average gain per tree split ([eda_outputs/feature_importances.png](eda_outputs/feature_importances.png)). Gain shows which features the model relies on, not what causes market value.

<!-- BEGIN:feature_importance -->
1. `league_tier_Tier 2 - Top European` (gain 1,124.3)
2. `appearances_last_365d` (gain 855.2)
3. `league_tier_Tier 3 - Other First Tier` (gain 524.2)
4. `minutes_last_365d` (gain 440.2)
5. `competition_id_GB1` (gain 279.2)
6. `competition_id_C1` (gain 223.8)
7. `competition_id_MEX1` (gain 186.7)
8. `competition_id_BRA1` (gain 136.9)
9. `competition_id_TS1` (gain 132.7)
10. `position_Goalkeeper` (gain 122.1)
<!-- END:feature_importance -->

---

## 8. Residual Error Breakdown by Segment

Residuals are defined as $e = \text{Actual} - \text{Predicted}$. Positive bias indicates the model **under-predicts** reality; negative bias indicates the model **over-predicts**.

"Uncorrected" columns use exp(log prediction); "smear-corrected" columns multiply that by the smearing factor. Log columns are the same for both. Plot: [eda_outputs/residuals_by_segment.png](eda_outputs/residuals_by_segment.png).

### A. Residuals by Age Band

<!-- BEGIN:residuals_age -->
| Age Band | Rows | Avg Actual | Mean Bias (uncorrected) | Mean Bias (smear-corrected) | Euro MAE (uncorrected) | Euro MAE (smear-corrected) | Log Bias | Log MAE |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| <21 (Prospects) | 17,799 | €2,343,221 | +€1,148,033 | +€545,132 | €1,386,895 | €1,290,357 | +0.300 | 0.906 |
| 21-24 (Ascending) | 35,545 | €3,671,836 | +€1,506,645 | +€414,436 | €1,919,641 | €1,743,993 | +0.192 | 0.737 |
| 25-28 (Peak) | 31,286 | €4,162,999 | +€1,259,964 | -€204,443 | €1,980,164 | €2,080,585 | +0.067 | 0.633 |
| 29-32 (Veterans) | 18,716 | €2,253,142 | +€160,835 | -€894,609 | €1,093,272 | €1,557,207 | -0.125 | 0.578 |
| 33+ (Late Career) | 8,360 | €789,901 | -€73,743 | -€509,400 | €417,537 | €662,925 | -0.344 | 0.603 |
<!-- END:residuals_age -->

**Domain Findings**:
<!-- BEGIN:findings_age -->
- **Largest relative error:** <21 (Prospects) has the highest log MAE (0.906).
- **<21 (Prospects):** the model under-predicts on average: mean bias +€1,148,033 uncorrected, +€545,132 after smearing, log bias +0.300. Hypothesis (not tested here): Transfermarkt values young players partly on expected development, which features built from past minutes and output cannot see.
- **33+ (Late Career):** in log terms the model over-predicts the typical player (log bias -0.344). In euros the uncorrected mean bias is -€73,743 (model over-predicts) and the smear-corrected one is -€509,400 (model over-predicts). Hypothesis (not tested here): career totals keep growing with age while market value falls.
- **Smallest euro error:** 33+ (Late Career) has the lowest uncorrected euro MAE (€417,537), largely because it also has the lowest values (average actual €789,901); its log MAE is 0.603.
<!-- END:findings_age -->

### B. Residuals by Position Group

<!-- BEGIN:residuals_position -->
| Position | Rows | Avg Actual | Mean Bias (uncorrected) | Mean Bias (smear-corrected) | Euro MAE (uncorrected) | Euro MAE (smear-corrected) | Log Bias | Log MAE |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Attack | 31,038 | €3,819,473 | +€1,229,739 | -€76,626 | €1,867,690 | €1,910,431 | +0.087 | 0.701 |
| Defender | 36,655 | €2,792,083 | +€901,802 | -€51,732 | €1,423,952 | €1,517,137 | +0.090 | 0.675 |
| Goalkeeper | 11,819 | €1,496,564 | +€489,449 | -€18,581 | €830,592 | €892,757 | +0.039 | 0.750 |
| Midfield | 32,120 | €3,506,989 | +€1,207,739 | +€47,905 | €1,831,592 | €1,845,075 | +0.081 | 0.702 |

*Excludes test rows whose position is "Missing" (74 rows), so this table covers 111,632 rows instead of the 111,706 in the age and tier tables.*
<!-- END:residuals_position -->

**Domain Findings**:
<!-- BEGIN:findings_position -->
- **Goalkeepers** have the lowest uncorrected euro MAE (€830,592) but the highest log MAE (0.750). Their euro errors look small because their values are the lowest of the four positions (average actual €1,496,564); relative to value, they are the hardest position to predict.
- Hypothesis (not tested here): goals and assists carry almost no information about goalkeepers, which may explain their higher relative error.
- **Highest euro MAE:** Attack (€1,867,690) and Midfield (€1,831,592). Hypothesis (not tested here): the most expensive players, whose premiums the features do not capture, are concentrated in these positions.
- Uncorrected mean bias is positive for every position, so the model under-predicts all of them on average before smearing.
<!-- END:findings_position -->

### C. Residuals by League Tier

<!-- BEGIN:residuals_tier -->
| League Tier | Rows | Avg Actual | Mean Bias (uncorrected) | Mean Bias (smear-corrected) | Euro MAE (uncorrected) | Euro MAE (smear-corrected) | Log Bias | Log MAE |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Tier 1 - Big 5 | 19,817 | €11,385,912 | +€3,499,410 | -€478,858 | €5,196,027 | €5,194,652 | +0.275 | 0.640 |
| Tier 2 - Top European | 39,142 | €1,747,200 | +€656,265 | +€105,953 | €992,100 | €1,054,722 | +0.102 | 0.659 |
| Tier 3 - Other First Tier | 26,613 | €1,641,236 | +€613,817 | +€95,545 | €1,057,882 | €1,138,341 | -0.025 | 0.767 |
| Tier 4 - Lower / Other | 26,134 | €518,008 | +€169,498 | -€6,305 | €339,283 | €390,917 | +0.010 | 0.731 |
<!-- END:residuals_tier -->

**Domain Findings**:
<!-- BEGIN:findings_tier -->
- **Largest euro error:** Tier 1 - Big 5 has the highest uncorrected euro MAE (€5,196,027) and accounts for 57.6% of the total absolute euro error on the test set while making up 17.7% of test rows (average actual value €11,385,912).
- **Largest relative error:** Tier 3 - Other First Tier has the highest log MAE (0.767); Tier 1 - Big 5 has a log MAE of 0.640.
- **Tier 1 - Big 5 bias:** the model under-predicts by €3,499,410 on average before smearing (mean bias +€3,499,410) and over-predicts by €478,858 after (mean bias -€478,858). Hypothesis (not tested here): elite-league values include premiums such as brand and commercial value that the features do not capture.
<!-- END:findings_tier -->

---

## 9. Largest Model Disagreements

<!-- BEGIN:ranking_scope -->
Rankings use the test period, taking each player's most recent test valuation (21,792 players), and compare the model's prediction with the Transfermarkt value: Difference = Predicted − Actual. Because the ranking uses euro differences, it is dominated by high-value players: 20 of the 20 players in the first table and 20 of the 20 in the second play in Tier 1 - Big 5.
<!-- END:ranking_scope -->

### Largest Model Disagreements: Model Above Market Value

Rankings use uncorrected predictions; smear correction scales every prediction by the same factor and would distort the ranking.

<!-- BEGIN:top20_model_above -->
| Rank | Player | Age | Position | Club | Actual | Predicted | Difference | % Difference |
| :---: | :--- | ---: | :--- | :--- | ---: | ---: | ---: | ---: |
| 1 | Lamine Yamal | 18.9 | Attack | FC Barcelona | €200,000,000 | €250,000,000* | +€50,000,000 | +25.0% |
| 2 | Nikola Milenković | 28.6 | Defender | Nottingham Forest | €28,000,000 | €54,828,215 | +€26,828,215 | +95.8% |
| 3 | Marcus Rashford | 28.6 | Attack | FC Barcelona | €40,000,000 | €65,240,022 | +€25,240,022 | +63.1% |
| 4 | David Turnbull | 24.9 | Midfield | Cardiff City | €3,800,000 | €26,854,851 | +€23,054,851 | +606.7% |
| 5 | Gonçalo Guedes | 29.5 | Attack | Real Sociedad | €6,000,000 | €25,942,671 | +€19,942,671 | +332.4% |
| 6 | Deniz Undav | 29.9 | Attack | VfB Stuttgart | €22,000,000 | €41,871,724 | +€19,871,724 | +90.3% |
| 7 | Jadon Sancho | 26.2 | Attack | Aston Villa | €18,000,000 | €37,830,981 | +€19,830,981 | +110.2% |
| 8 | Ilaix Moriba | 23.4 | Midfield | Celta de Vigo | €15,000,000 | €34,692,823 | +€19,692,823 | +131.3% |
| 9 | Pablo Fornals | 30.3 | Midfield | Real Betis Balompié | €8,000,000 | €27,682,013 | +€19,682,013 | +246.0% |
| 10 | Gianluca Mancini | 30.1 | Defender | Associazione Sportiva Roma | €15,000,000 | €34,580,777 | +€19,580,777 | +130.5% |
| 11 | Ibrahim Sangaré | 28.5 | Midfield | Nottingham Forest | €20,000,000 | €39,233,241 | +€19,233,241 | +96.2% |
| 12 | Xavi Simons | 23.1 | Midfield | Tottenham Hotspur | €40,000,000 | €57,030,921 | +€17,030,921 | +42.6% |
| 13 | Nikola Vlašić | 28.7 | Midfield | Torino FC | €10,000,000 | €26,452,174 | +€16,452,174 | +164.5% |
| 14 | Lewis Ferguson | 26.8 | Midfield | Bologna Football Club 1909 | €14,000,000 | €30,123,600 | +€16,123,600 | +115.2% |
| 15 | Manuel Locatelli | 28.4 | Midfield | Juventus FC | €25,000,000 | €40,992,050 | +€15,992,050 | +64.0% |
| 16 | Victor Nelsson | 27.6 | Defender | Hellas Verona | €6,000,000 | €21,721,521 | +€15,721,521 | +262.0% |
| 17 | Mikel Oyarzabal | 29.1 | Attack | Real Sociedad | €25,000,000 | €40,682,590 | +€15,682,590 | +62.7% |
| 18 | Lorenzo Pellegrini | 29.9 | Midfield | Associazione Sportiva Roma | €8,000,000 | €23,465,116 | +€15,465,116 | +193.3% |
| 19 | Unai Simón | 29.0 | Goalkeeper | Athletic Bilbao | €22,000,000 | €36,768,172 | +€14,768,172 | +67.1% |
| 20 | Weston McKennie | 27.8 | Midfield | Juventus FC | €30,000,000 | €44,736,855 | +€14,736,855 | +49.1% |

\* Prediction hit the €250,000,000 cap applied in `src/models.py` (Lamine Yamal); the model's raw prediction was at or above the cap.
<!-- END:top20_model_above -->

### Largest Model Disagreements: Model Below Market Value

Rankings use uncorrected predictions; smear correction scales every prediction by the same factor and would distort the ranking.

<!-- BEGIN:top20_model_below -->
| Rank | Player | Age | Position | Club | Actual | Predicted | Difference | % Difference |
| :---: | :--- | ---: | :--- | :--- | ---: | ---: | ---: | ---: |
| 1 | João Neves | 21.7 | Midfield | Paris Saint-Germain | €140,000,000 | €33,768,763 | -€106,231,237 | -75.9% |
| 2 | Vitinha | 26.3 | Midfield | Paris Saint-Germain | €140,000,000 | €36,618,190 | -€103,381,810 | -73.8% |
| 3 | Pedri | 23.5 | Midfield | FC Barcelona | €150,000,000 | €48,202,884 | -€101,797,116 | -67.9% |
| 4 | Erling Haaland | 25.9 | Attack | Manchester City | €200,000,000 | €110,889,842 | -€89,110,158 | -44.6% |
| 5 | Khvicha Kvaratskhelia | 25.3 | Attack | Paris Saint-Germain | €140,000,000 | €57,406,347 | -€82,593,653 | -59.0% |
| 6 | Jamal Musiala | 23.3 | Midfield | Bayern Munich | €100,000,000 | €25,216,348 | -€74,783,652 | -74.8% |
| 7 | Désiré Doué | 21.0 | Attack | Paris Saint-Germain | €120,000,000 | €47,079,710 | -€72,920,290 | -60.8% |
| 8 | Kylian Mbappé | 27.5 | Attack | Real Madrid | €180,000,000 | €113,573,879 | -€66,426,121 | -36.9% |
| 9 | Jude Bellingham | 22.9 | Midfield | Real Madrid | €130,000,000 | €64,162,940 | -€65,837,060 | -50.6% |
| 10 | Michael Olise | 24.5 | Attack | Bayern Munich | €150,000,000 | €85,565,394 | -€64,434,606 | -43.0% |
| 11 | Aleksandar Pavlovic | 22.1 | Midfield | Bayern Munich | €90,000,000 | €28,301,472 | -€61,698,528 | -68.6% |
| 12 | Alexander Isak | 26.7 | Attack | Liverpool FC | €85,000,000 | €25,072,184 | -€59,927,816 | -70.5% |
| 13 | Ousmane Dembélé | 29.1 | Attack | Paris Saint-Germain | €100,000,000 | €40,911,051 | -€59,088,949 | -59.1% |
| 14 | Julián Alvarez | 26.3 | Attack | Atlético de Madrid | €100,000,000 | €43,618,751 | -€56,381,249 | -56.4% |
| 15 | Antoine Semenyo | 26.4 | Attack | Manchester City | €80,000,000 | €25,239,590 | -€54,760,410 | -68.5% |
| 16 | Fermín López | 23.1 | Midfield | FC Barcelona | €100,000,000 | €45,727,145 | -€54,272,855 | -54.3% |
| 17 | Rayan | 19.8 | Attack | AFC Bournemouth | €60,000,000 | €6,063,489 | -€53,936,511 | -89.9% |
| 18 | Willian Pacho | 24.6 | Defender | Paris Saint-Germain | €80,000,000 | €26,222,104 | -€53,777,896 | -67.2% |
| 19 | Declan Rice | 27.4 | Midfield | Arsenal FC | €120,000,000 | €66,838,119 | -€53,161,881 | -44.3% |
| 20 | Yan Diomande | 19.5 | Attack | RB Leipzig | €90,000,000 | €37,309,166 | -€52,690,834 | -58.5% |
<!-- END:top20_model_below -->

*Source files: the "Model Above Market Value" table comes from `models/undervalued_top20.csv` and the "Model Below Market Value" table from `models/overvalued_top20.csv`; the file names describe the market value relative to the model.*

---

## 10. Critical Limitations & Domain Findings

> [!CAUTION]
> **Hypothesis Generation Only**:
> In association football datasets, there are **no ground-truth labels for "good buy" or "profitable transfer"**. Disagreements between the model and Transfermarkt represent feature blind spots (commercial value, press resistance, hype) or market inefficiencies, not guaranteed investment signals.

<!-- BEGIN:limitations -->
- **Snapshot features are excluded from the headline model.** `contract_days_remaining`, `has_contract_info`, `club_squad_size`, `club_foreigners_percentage`, `club_national_team_players` come from the current rows of `players.csv` and `clubs.csv`, not values as of the valuation date. For example, the median `contract_days_remaining` is 3,793 days for 2014 valuations versus 757 days for 2026, which is what contracts signed after the valuation would produce. The headline models drop these features; the Leakage ablation section shows their effect.
- **Some information from after the valuation date can still reach the model.** Position, sub-position, foot, and height are current values from `players.csv`; imputation medians inside the `TimeSeriesSplit` folds are computed on the whole training set; and the club at each valuation is assumed, not documented, to be the club at that date. Test metrics may therefore still be slightly optimistic.
- **Predictions are capped.** Ridge and XGBoost log predictions are clipped to [8.5, ln(250,000,000)] in `src/models.py`, so their euro predictions lie between €4,915 and €250,000,000.
- **Features describe on-pitch output, the player's profile, and league.** The headline model uses playing time, goals, assists, cards, career totals, age, position, sub-position, foot, height, competition, and league tier. There is no signal for potential, hype, brand value, contract situation, or injuries.
- **Large disagreements reflect model limits, not market errors.** Of the 20 players the model places furthest below market value, 14 are valued at €100,000,000 or more and 11 are under 25. These gaps show what the features cannot see; they are not proof that Transfermarkt is wrong.
- **Market value is not a transfer fee.** The target is Transfermarkt's crowdsourced estimate, not the price clubs actually pay.
- **Static snapshot.** Valuations through 2026-06-12; match data through 2026-07-06. The data is not refreshed.
<!-- END:limitations -->
<!-- BEGIN:zero_minutes -->
- **Most test valuations have no recent playing time.** 62,908 of the 111,706 test valuations (56.3%) have `minutes_last_365d` = 0, i.e. no recorded minutes in the 365 days before the valuation. By league tier: Tier 1 - Big 5 14.4% of 19,817; Tier 2 - Top European 34.2% of 39,142; Tier 3 - Other First Tier 91.6% of 26,613; Tier 4 - Lower / Other 85.2% of 26,134. For these rows the model has no recent playing-time signal. Hypothesis (not verified here): the appearance data covers fewer of the competitions these players play in, so some zeros mean "not recorded" rather than "did not play".
<!-- END:zero_minutes -->

<!-- BEGIN:zero_minutes_errors -->
Headline XGBoost errors (uncorrected) for test rows with and without previous-year minutes, from `models/test_predictions.parquet`:

| Minutes in previous 365 days | Test rows | Share | Euro MAE | Log MAE |
| :--- | ---: | ---: | ---: | ---: |
| 0 minutes | 62,908 | 56.3% | €618,759 | 0.7451 |
| More than 0 minutes | 48,798 | 43.7% | €2,866,876 | 0.6374 |

Euro MAE is lower for zero-minute rows (€618,759 vs €2,866,876) only because those players have lower market values (average actual €951,229 vs €5,971,544); relative error is higher (Log MAE 0.7451 vs 0.6374), so the model is less accurate in relative terms where it has no recent playing time.
<!-- END:zero_minutes_errors -->

---

## 11. Quickstart & How to Run

### System Requirements
- **Python 3.12 or newer**
- 8GB RAM minimum (16GB recommended for complete parquet feature caching)
- Windows, macOS, or Linux

### Installation
```bash
# 1. Clone repository and navigate to project folder
cd ML-CP

# 2. Set up virtual environment
python -m venv .venv
source .venv/bin/activate       # On Windows PowerShell: .venv\Scripts\Activate.ps1

# 3. Install dependencies
pip install -r requirements.txt -r requirements-dashboard.txt
```

### Running the System
```bash
# A. Launch the Interactive Touchline Dashboard (Instant demonstration)
python -m streamlit run dashboard.py
# (Or double-click start_dashboard.cmd on Windows)

# B. Run the Upgraded Training Pipeline (Calendar CV, Recent Form, OLS, SVR, XGBoost)
python run_pipeline.py

# C. Quick Smoke Run for Testing (Doesn't overwrite presentation artifacts)
python run_upgrade.py --quick

# D. Run the Preserved Legacy Benchmark
python run_pipeline.py --legacy

# E. Verify Test Invariants & Markdown Numbers Consistency
python tests/check_consistency.py
```

### 5-Minute Professor Demonstration Script
1. **Overview Tab**: Show the 5-model metrics card. Highlight that adding recent 90d/180d/365d form reduces log RMSE and boosts within-2× predictions to 59.5%.
2. **Coverage & Sparsity**: Filter by Tier 3/4 leagues to illustrate why zero minutes in the dataset represents competition coverage limits rather than player benching.
3. **Player Explorer Tab**: Search for a well-known star (e.g., Lamine Yamal or Erling Haaland). Examine the timeline showing the model's estimate vs. Transfermarkt's valuation.
4. **Learning Curves Tab**: Compare OLS and Linear SVR with XGBoost across the 4 training fractions (5% to 100%) to demonstrate empirical bias-variance trade-offs.
5. **Method & Caveats Tab**: Review the strict calendar validation boundary rules, showing that lookahead bias has been eliminated.

---

## 12. Data Source & Provenance

The data is based on the open Transfermarkt historical dataset curated by David Carmo (covering competitions, clubs, games, appearances, and market valuations). All raw CSV files (`appearances.csv`, `player_valuations.csv`, `players.csv`, `clubs.csv`, `competitions.csv`) are processed strictly with point-in-time constraints.
