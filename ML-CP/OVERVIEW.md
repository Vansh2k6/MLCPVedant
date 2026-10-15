# Project Overview and Concept Guide

> This guide describes the original benchmark. The corrected experiment and presentation dashboard
> are documented in [UPGRADE_GUIDE.md](UPGRADE_GUIDE.md). The default runner now executes that upgrade;
> use `python run_pipeline.py --legacy` for the workflow described below.

A study guide for this project: what it does, every concept it relies on, and the questions you should be ready to answer. Settings quoted here (split date, seed, search space, caps) are copied from the code. **Results (metrics, biases, counts) are deliberately not repeated here**, because they are regenerated on every run; read them from the README TL;DR, the README tables, or `analysis.ipynb`.

---

## 1. The project in one paragraph

We predict a football player's **Transfermarkt market value** on a given valuation date from information available on that date: recent and career playing record, age, position, physical profile, and the league the player's club plays in. Three predictors are compared on valuations from a later period than the training data: a **naive group-median baseline**, a **Ridge linear regression**, and **XGBoost** (gradient-boosted trees). The project's main emphasis is honest evaluation: a chronological split, point-in-time features, a **leakage audit and ablation**, a **retransformation (smearing) correction** for the log target, segment-level **residual analysis**, and a README whose numbers are generated and checked automatically.

---

## 2. Pipeline map

`python run_pipeline.py` runs every step in order.

| Step | File | What it does | Main outputs |
| :--- | :--- | :--- | :--- |
| 1 | `src/data_pipeline.py` | Joins the raw CSVs with DuckDB SQL into one row per player per valuation date; builds point-in-time features; runs data-integrity checks | `data/processed_market_valuations.parquet` |
| 2 | `src/eda.py` | Target distribution, value-vs-age curves by position, sparsity audit | `eda_outputs/*.png`, `eda_outputs/eda_summary.txt` |
| 3–5 | `src/models.py` | Chronological split, naive baseline, Ridge, XGBoost (full and clean feature sets), depth tuning, smearing factor, metrics | `models/model_comparison_metrics.csv`, `models/leakage_ablation.csv`, `models/smearing_factor.json`, `models/test_predictions.parquet`, `models/xgboost_market_value_model.json`, `models/feature_importances.csv` |
| 6–7 | `src/evaluate.py` | Residuals by age band, position, league tier; top-20 model disagreements | `models/residual_analysis_by_*.csv`, `models/undervalued_top20.csv`, `models/overvalued_top20.csv`, `models/rankings_summary.txt` |
| 8 | `src/update_readme.py` | Rewrites every README section between `<!-- BEGIN:name -->` / `<!-- END:name -->` markers from the outputs | `README.md` |
| check | `src/check_readme.py`, `tests/check_consistency.py` | Verifies README tables against the CSVs and checks the split, features, notebook, and wording | exit code 0 or a list of problems |
| walkthrough | `analysis.ipynb` | Narrated tour built only from `data/` and `models/` outputs | rendered charts and computed text |

---

## 3. Data

- **Source tables (raw CSVs in the project root):** `player_valuations.csv` (the target and the club/competition at each valuation), `players.csv` (date of birth, position, foot, height, contract expiry), `appearances.csv` (per-match minutes, goals, assists, cards), `clubs.csv`, `competitions.csv`. `games.csv` is optional and only supplies the match-data date in the README.
- **Grain (unit of analysis):** one row = one player on one valuation date. A player appears many times across a career, which matters for how we split the data (Section 6).
- **Target:** `market_value_in_eur`, Transfermarkt's crowdsourced estimate. It is **not** a transfer fee (the price actually paid between clubs, which includes negotiation, contract clauses, and urgency).
- **Filters:** positive market value, known date of birth; the integrity checks also assert unique player-date rows, plausible ages, and non-negative stats. These checks confirm data quality; they do not test for leakage.
- **Static snapshot:** the data is a fixed historical download, not live.

---

## 4. Feature engineering concepts

| Concept | What it means here | Why it matters |
| :--- | :--- | :--- |
| **Point-in-time feature** | A value computed only from information that existed on the valuation date | Using later information makes test scores look better than real-world performance |
| **Rolling 365-day window** | Sums of minutes, appearances, goals, assists, cards from matches in the 365 days before the valuation (`a.date < valuation_date` in the SQL join) | Captures current form without looking ahead |
| **Prior-career totals** | Cumulative appearances, minutes, goals, assists before the valuation date | Experience and track record |
| **Per-90 rates** | Goals (or assists) per 90 minutes played; 0 when no minutes | Normalises output by playing time |
| **Clipping per-90 rates at 3.0** | `goals_per_90_clipped` etc. | One goal in 2 minutes would be 45 goals per 90; clipping stops tiny samples creating absurd values |
| **Age and age²** | Age at the valuation date and its square | Value rises then falls with age; the squared term lets a linear model bend |
| **One-hot encoding** | A 0/1 column per category (position, sub-position, competition, league tier, foot) | Models need numbers; one-hot avoids implying an order between categories |
| **League tier** | Tier 1 = Big 5 leagues; Tier 2 = a fixed list of other top European leagues; Tier 3 = other first divisions; Tier 4 = everything else | Market prices differ hugely by league |
| **Interaction terms** | e.g. `attack_goals_per_90` = goals per 90 × is_attack; `gk_minutes_last_365d` = minutes × is_goalkeeper | Lets the model treat the same statistic differently by position (goals matter for attackers, minutes for keepers) |
| **Structural zeros** | Most goalkeepers and many defenders score 0 goals in a year | These zeros are real, so they are kept rather than "imputed" |
| **Imputation** | Missing heights are filled with the **training-set** median for the player's position | Using test rows to compute fill values would leak test information |

---

## 5. Data leakage (the central theme)

**Leakage** = the model sees information during training or prediction that would not be available at prediction time in the real world. It inflates scores and the model then disappoints in practice.

Types relevant here:
1. **Temporal leakage:** training on the future and testing on the past. Prevented by the chronological split.
2. **Feature (snapshot) leakage:** a feature measured *after* the valuation date. `players.csv` and `clubs.csv` hold one *current* row per player or club, so contract expiry and club attributes describe the snapshot date, not the valuation date. Example: a 2015 valuation sees the contract the player signed in 2024.
3. **Preprocessing leakage:** computing statistics (medians, scalers) on data that includes the evaluation rows.

What the project does about it:
- **Point-in-time audit:** a table in the README (Section 4) classifies every feature as point-in-time or snapshot.
- **`SNAPSHOT_FEATURES`** (top of `src/models.py`): `contract_days_remaining`, `has_contract_info`, `club_squad_size`, `club_foreigners_percentage`, `club_national_team_players`. The **headline ("clean") models drop them.**
- **Leakage ablation:** every model is trained twice, *full* (with snapshot features) and *clean* (without), with identical split, seeds and search. `models/leakage_ablation.csv` shows the difference. Correct reading: "the model is X% worse without these features", **not** "leakage inflated the score by X%", because part of what they carry may be genuine signal.
- **Known remaining issues (stated openly):** position, sub-position, foot and height are also snapshot values (kept because they rarely change); the club at each valuation is assumed to be correct; imputation medians are computed on the whole training set and reused inside the cross-validation folds (a small leak within training only; the test set is unaffected); the Tier 3/4 classification uses the current competitions table.

---

## 6. Train/test split and validation

- **Chronological split:** train = valuations before **2024-01-01**; test = valuations on or after it. No random split anywhere.
- **Why not a random split?** Valuations of the same player a few months apart are nearly identical. A random split puts some of them in training and their neighbours in testing, so the model can "look up" the answer. It would also learn later market price levels and be tested on earlier dates.
- **Players can appear in both sets:** the split separates *time*, not players, which mirrors real use (valuing known players in the future).
- **`TimeSeriesSplit(n_splits=3)`:** cross-validation inside the training set where each validation fold is later than its training fold. Used to choose XGBoost's `max_depth` from **[6, 7, 8]** and to fit the smearing factor on out-of-sample residuals.
- **Hold-out discipline:** the test set is used only once, for final scoring.

---

## 7. Target transformation

- **Skewness:** a measure of asymmetry. Market values are heavily right-skewed: a few superstars are worth hundreds of times a typical player. **Kurtosis** measures how heavy the tails are.
- **Log transform:** every model predicts `ln(market value)`. This makes the target nearly symmetric, stops the few superstars dominating the loss, and turns errors into *relative* errors (being off by half is equally bad for cheap and expensive players).
- **Back-transform:** euro predictions = `exp(log prediction)`. This gives a **median-type** estimate, not the mean (see Section 10).

---

## 8. The models

| Model | How it works | Role |
| :--- | :--- | :--- |
| **Naive baseline** | Predicts the training median log value for the player's position × league tier; falls back to position median, then global median, for groups with fewer than 30 training rows | The minimum bar: "what kind of player in which league" with no model |
| **Ridge regression** | Linear regression with an L2 penalty (`alpha=100`) that shrinks coefficients to reduce overfitting; numeric features standardised (`StandardScaler`), categoricals one-hot encoded. Uses age, age², minutes, goal contributions per 90, position, league tier | A simple, interpretable linear benchmark |
| **XGBoost** | Gradient boosting: builds many shallow decision trees in sequence, each fitting the errors of the trees before it. Settings: 300 trees, learning rate 0.08, `max_depth` tuned from [6, 7, 8], row and column subsampling 0.85, L1/L2 regularisation (`reg_alpha=0.1`, `reg_lambda=1.0`), seed 42 | The main model; captures non-linear effects and interactions automatically |

Key terms:
- **Decision tree:** splits the data with yes/no questions ("minutes > 1,500?") and predicts the average in each leaf.
- **Boosting vs bagging:** boosting adds trees sequentially to correct errors (XGBoost); bagging averages independent trees (random forest).
- **`max_depth`:** how many questions a tree can ask; deeper = more complex interactions, more risk of overfitting.
- **Learning rate:** how much each new tree is allowed to change the prediction; smaller = slower, more stable learning.
- **Subsampling:** each tree sees a random share of rows and columns, which reduces overfitting (hence the fixed seed).
- **Regularisation:** penalties that discourage overly complex models (L1 = `reg_alpha`, L2 = `reg_lambda`; Ridge's `alpha`).
- **Overfitting:** fitting noise in the training data so test performance is worse.

---

## 9. Evaluation metrics

| Metric | Formula idea | Interpretation |
| :--- | :--- | :--- |
| **RMSE** | √(mean of squared errors) | Penalises large errors heavily; dominated by the biggest misses |
| **MAE** | mean of absolute errors | "Typical" error size; less sensitive to outliers |
| **Log RMSE / Log MAE** | computed on ln(value) | Relative error: a Log MAE of *m* means a typical prediction is off by a factor of about e^*m* |
| **Euro RMSE / Euro MAE** | computed on euro values | Absolute error in money; dominated by expensive players |
| **% improvement** | (baseline error − model error) / baseline error | How much error the model removes relative to a baseline |

- **Why both log and euro metrics?** Log metrics say how good the model is *relatively* for every player; euro metrics say how much money it is off by, which is driven by superstars.
- **Prediction clipping:** every model's log predictions are clipped to **[8.5, ln(250,000,000)]**, i.e. about **€4,915 to €250,000,000**, *before* any metric, so log and euro metrics use the same predictions. Predictions at the ceiling are flagged in the README ranking tables.

---

## 10. Retransformation bias and Duan smearing

- **Problem:** for a skewed variable, exp(mean of logs) = the geometric mean, which is below the arithmetic mean. So exp(log prediction) systematically **under-predicts the average euro value**.
- **Duan smearing:** multiply every euro prediction by a constant factor = mean(exp(residual)), where residuals are log-scale errors measured **out of sample** (the last `TimeSeriesSplit` validation fold of the training set). No test data is used. Stored in `models/smearing_factor.json`.
- **Effect:** removes most of the *overall* mean bias and usually reduces Euro RMSE. It can **increase Euro MAE**, because absolute error is minimised by median-type predictions, and smearing moves predictions toward the mean.
- **Limitation:** one global factor scales every player the same way, so it cannot fix **segment bias**: segments under-predicted before smearing may stay under-predicted, and others may flip to over-prediction (see the Tier 1 discussion in the README and notebook).
- **Rankings use uncorrected predictions**, because scaling every prediction by the same factor would distort the euro gaps used for ranking.

---

## 11. Residual analysis

- **Residual / bias sign convention:** bias = actual − predicted. **Positive = model under-predicts; negative = model over-predicts.**
- **Mean bias vs MAE:** mean bias shows *systematic* direction (errors that don't cancel); MAE shows *size*. A segment can have small bias but large MAE (errors cancel out) or vice versa.
- **Segments:** age bands (<21, 21–24, 25–28, 29–32, 33+), positions, league tiers. Each is reported uncorrected and smear-corrected, in euros and in logs.
- **Why log and euro bias can disagree in sign:** log bias weights every player equally; euro bias is dominated by the few very valuable players.
- **Feature importance (gain):** how much each feature's splits reduced training error in total. It shows what the model *uses*, not what *causes* value. Correlated features share importance.

---

## 12. Model-disagreement rankings

- Two top-20 tables per player's latest test valuation: the model **above** market value (`models/undervalued_top20.csv`) and the model **below** market value (`models/overvalued_top20.csv`; the file name describes the market value relative to the model).
- These are **hypothesis-generating only**. There is no ground truth for "good buys", and large gaps mostly show what the features cannot see (potential, hype, brand, injuries), not that Transfermarkt is wrong.

---

## 13. Reproducibility and verification

- **Fixed seed 42** for Ridge, XGBoost and sampling; deterministic row order (`ORDER BY valuation_date, player_id`) so repeated runs give byte-identical metrics.
- **Generated README:** every number between markers is written by `src/update_readme.py`; nothing is typed by hand. Interpretations are labelled "Hypothesis (not tested here)".
- **`src/check_readme.py`:** every euro/percentage figure in the README tables must match its source CSV within display rounding.
- **`tests/check_consistency.py`:** also fails on absolute paths, banned wording, train + test ≠ total rows, test rows before the split date, snapshot features in the saved model, notebook errors, or raw CSV references in the notebook.
- **Pinned environment:** `requirements.txt` and `requirements-notebook.txt`; Python 3.12+.

---

## 14. Limitations to state up front

1. Snapshot features are dropped, so contract situation and club strength are not modelled.
2. Position, height and foot are assumed stable; the club at each valuation is assumed correct.
3. Small imputation leak inside the cross-validation folds (training only).
4. Predictions are capped at both ends.
5. The target is a crowdsourced opinion, with its own biases; it is not a transfer fee.
6. No signal for potential, hype, brand value, or injuries, so young stars and superstars are under-predicted.
7. Most test valuations have zero recorded minutes in the previous year, concentrated in lower tiers; for those rows the model has no recent playing-time signal and its relative error is higher.
8. Static, non-refreshed data.

---

## 15. Questions a professor is likely to ask

**Why a log target?** The raw target is extremely right-skewed; log makes it near-symmetric, stabilises variance, and makes errors relative.

**Why not a random train/test split or K-fold?** Repeated valuations of the same player and market-wide price trends would leak future information; a chronological split and `TimeSeriesSplit` mimic real forecasting.

**How do you know there is no leakage?** We don't claim there is none. Appearance features are point-in-time by construction (strict date filter); snapshot features are listed, dropped from the headline model, and their effect measured by ablation; remaining small sources are documented.

**Why is the full model better than the clean one, and why not use it?** It relies on features measured after the valuation date, so its score would not be achievable in real use. The gap shows how much the model leaned on them, not a pure "leakage amount".

**Why compare against a naive baseline?** Any model must beat "median value for this position in this league"; otherwise the features add nothing.

**Why XGBoost over Ridge?** Market value depends on non-linear interactions (age × position × league × minutes) that trees capture automatically; Ridge would need these engineered by hand.

**What is the smearing factor and why can it make MAE worse?** It corrects the downward bias of exp(log prediction) for the mean. MAE is minimised by median-type predictions, so shifting toward the mean can raise MAE while lowering mean bias and RMSE.

**Why does one smearing factor not fix segment bias?** It scales every prediction equally; segments biased in different directions need segment-specific corrections or better features.

**What does feature importance tell you?** Which features the trees split on most to reduce error. It is not causal, and correlated features split credit.

**Why is Euro MAE lower for zero-minute rows if the model is worse there?** Those players have much lower values, so absolute errors are smaller; Log MAE (relative error) is higher.

**Are the "undervalued" players good buys?** No claim is made. There is no ground truth; the lists show where the features and the market disagree.

**How would you improve it?** Dated contract and squad records (so contract and club features can be used without leakage), fold-internal imputation, signals for potential and reputation, segment-specific smearing, and a probabilistic or quantile model to express uncertainty.

---

## 16. Glossary

- **Ablation:** removing a component (here, a feature group) to measure its contribution.
- **Bias (residual):** actual − predicted, averaged over a group.
- **Categorical feature:** a label without numeric order (position, league).
- **Cross-validation:** repeatedly splitting training data into fit/validate parts to choose settings without touching the test set.
- **DuckDB:** an in-process SQL database used to join large CSVs efficiently.
- **Geometric mean:** exp(mean of logs); below the arithmetic mean for skewed data.
- **Hyperparameter:** a setting chosen before training (e.g. `max_depth`), as opposed to learned parameters.
- **Kurtosis:** heaviness of a distribution's tails.
- **L1 / L2 regularisation:** penalties on the absolute / squared size of model weights.
- **Parquet:** a compressed, columnar file format for tables.
- **Point-in-time:** known as of a specific date.
- **Residual:** the error of one prediction.
- **Skewness:** asymmetry of a distribution; right skew = long tail of large values.
- **Smearing (Duan):** multiplying back-transformed predictions by mean(exp(residual)) to correct retransformation bias.
- **Snapshot feature:** a value taken from a single later download rather than as of each valuation date.
- **Structural zero:** a zero that is a genuine value (a keeper scoring no goals), not missing data.
