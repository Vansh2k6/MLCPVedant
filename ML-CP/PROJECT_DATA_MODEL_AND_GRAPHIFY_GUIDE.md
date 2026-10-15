# Project Data, Model Comparison, and Graphify Guide

## Current project status

This project is now a time-aware football player market-value regression system with:

- point-in-time 90, 180, and 365-day performance features;
- expanding calendar validation using only earlier years;
- a final 2024-2026 test period;
- Ordinary Least Squares (OLS), Ridge regression, Linear Support Vector Regression, and XGBoost comparisons;
- empirical learning curves;
- an interactive Streamlit dashboard; and
- a local Graphify knowledge graph for navigating the code and documentation.

The original raw files remain unchanged. The model-ready table is stored as compressed Parquet, which is better suited than CSV for repeated analytical reads.

Current verification status: all 11 upgrade/dashboard tests and all 6 preserved legacy consistency checks pass. All five saved model pipelines reproduce their stored test predictions within numerical tolerance.

## Graphify initialization

Graphify 0.9.91 was installed from the official `graphifyy` Python package. Project-scoped Codex support was registered with:

```powershell
graphify install --project --platform codex
```

The installation created:

- `AGENTS.md` with graph-first repository instructions;
- `.codex/skills/graphify/` with the local Graphify skill and references;
- `.codex/hooks.json` with the project hook configuration; and
- `graphify-out/` with the generated knowledge graph.

The first graph build detected 44 supported files: 25 code files, 13 documents, and 6 images. No files were skipped as sensitive. Its main outputs are:

- `graphify-out/graph.html`: interactive, searchable visualization;
- `graphify-out/graph.json`: machine-readable graph;
- `graphify-out/GRAPH_REPORT.md`: graph audit and architecture report.

The first clean post-build graph contained 443 nodes and 804 edges. Indexing this guide adds documentation nodes, so exact totals can change after an incremental refresh; use `graphify-out/GRAPH_REPORT.md` for the current count. Communities cover Core Model Training, Pipeline Orchestration and EDA, Learning Curves, Residual Diagnostics, Multisport Architecture, and Dashboard Tests.

### Graph health and cost record

The final integrity diagnostic reported zero dangling endpoints, missing endpoints, self-loops, or collapsed same-endpoint edges. The graph is suitable for project navigation, scoped code queries, and relationship discovery. It is still a generated representation rather than a mathematical proof that every real-world dependency was modeled. Semantic extraction used the active Codex session; no external Gemini API was configured. Graphify therefore recorded zero external API tokens, while host-agent token use was not exposed by the collaboration interface.

Useful commands:

```powershell
graphify query "how does temporal validation work?"
graphify path "train_upgraded" "calendar_folds"
graphify explain "build_learning_curves"
graphify update .
```

## Why all CSV files should not be flattened into one CSV

The files describe different kinds of records. This is called the **grain** of a table.

| File | Rows | Natural grain |
|---|---:|---|
| `players.csv` | 50,149 | one row per player |
| `player_valuations.csv` | 656,301 | one row per player and valuation date |
| `appearances.csv` | 1,894,350 | one row per player appearance in a match |
| `games.csv` | 88,958 | one row per match |
| `club_games.csv` | 177,916 | one row per club in a match; normally two rows per match |
| `game_events.csv` | 1,274,469 | one row per event within a match |
| `game_lineups.csv` | 3,179,016 | one row per player lineup record within a match |
| `transfers.csv` | 175,165 | one row per transfer event |
| `clubs.csv` | 796 | one row per club |
| `competitions.csv` | 65 | one row per competition |

Joining tables at different grains without aggregation creates a many-to-many multiplication. For example, joining every valuation for a player to every appearance for that player produces **50,426,443 rows** before events or lineups are added. The valid model table has only **547,653 player-valuation rows** because appearances are aggregated inside historical windows before they are attached to each valuation.

A single flattened CSV would introduce four major problems:

1. **Duplicate targets.** One market valuation would appear once for every matching appearance, event, or lineup row. The model would effectively give players with more recorded activity much more weight.
2. **Incorrect statistics.** Counts, averages, and totals would be inflated by repeated rows.
3. **Temporal leakage.** A simple player join can attach matches, events, transfers, or contract data that happened after the valuation date.
4. **Storage and maintenance cost.** Player and match attributes would be copied millions of times. Correcting one source record would require rebuilding a very large flat file.

### A safe way to reduce the visible file count

It is possible to have fewer physical files without destroying the relational structure:

1. Keep the downloaded CSV files as immutable raw-source evidence.
2. Optionally import each CSV as a separate table inside one `football_raw.duckdb` database. This gives one physical database file while preserving table grains and keys.
3. Build one model-ready table at the player-and-valuation-date grain.
4. Store that table as `data/upgraded_market_valuations.parquet`.
5. Keep experiment outputs under one versioned directory such as `models/upgraded/`.

For the course demonstration, the clean explanation is: **the raw layer is normalized, and the feature layer is intentionally denormalized after time-aware aggregation**.

## Should multiple regressors be compared?

Yes. A model comparison is academically stronger than presenting only one high-performing model. It shows whether improvement comes from the data and feature engineering or merely from choosing a more flexible algorithm.

The current project already compares four useful model families on the same task:

| Model | What it demonstrates | 2024-2026 test Log RMSE | Test MAE |
|---|---|---:|---:|
| OLS | unregularized linear baseline from Unit III | 0.9974 | EUR 2.109M |
| Ridge | regularized linear regression | 0.9975 | EUR 2.108M |
| Linear SVR | margin-based regression from Unit IV | 0.9972 | EUR 2.107M |
| Recent-form XGBoost | nonlinear interactions and boosted trees | **0.9125** | **EUR 1.608M** |

An annual-feature XGBoost ablation is also retained, with test Log RMSE 0.9167 and MAE EUR 1.622M. Comparing it with recent-form XGBoost isolates the effect of the new time-window features.

OLS, Ridge, and Linear SVR perform similarly and plateau at roughly the same validation error. That suggests the main remaining limitation of these models is linear functional form rather than insufficient training data. XGBoost performs better because market value depends on nonlinear relationships and interactions among age, position, competition, exposure, and recent performance.

The project uses `LinearSVR` rather than kernel RBF `SVR` because the training set contains 435,947 rows. Kernel SVR does not scale well to that sample size. Linear SVR still demonstrates the support-vector regression objective at a practical college-project scale.

### Fair comparison protocol

Every regressor comparison should use:

- the same eligible observations;
- the same target, `log1p(market_value_in_eur)`;
- the same feature definitions;
- preprocessing fitted only on the training portion of each fold;
- the same expanding calendar validation years;
- hyperparameter selection using validation years only; and
- one untouched final test period for the final table.

Report at least Log RMSE, Log MAE, euro MAE, and percentage of predictions within a factor of two. Do not select a model using the final test set.

Adding many nearly identical regressors will not make the project look more advanced. The current set is enough for a strong comparison because it includes an unregularized linear model, a regularized linear model, a support-vector model, and a nonlinear ensemble. A Random Forest can be added as an optional syllabus model, but it is lower priority than explaining the controlled comparison and learning curves well.

## Learning-curve result

The learning curves use nested samples of 18,919, 75,676, 189,190, and 378,379 pre-2023 rows. Every model is evaluated on the same fixed 57,568-row 2023 validation set; no 2024-2026 test rows are used.

- OLS, Ridge, and Linear SVR converge near 0.986 validation Log RMSE at large sample sizes.
- XGBoost improves from about 0.916 to 0.890 validation Log RMSE as training size grows.
- The XGBoost train-validation gap shrinks from roughly 0.211 to 0.089.

This supports two college-level conclusions:

1. The linear models show higher bias: more rows do not materially improve their final validation error.
2. XGBoost benefits from additional data and learns useful nonlinear structure, although the remaining gap also indicates some variance, temporal drift, or unobserved variables.

These are descriptive curves from one seeded sampling experiment, not confidence intervals or a formal proof of bias and variance.

## Connection to Units I-IV

The upgraded project now uses a substantial portion of the requested syllabus:

- **Unit I:** well-posed learning problem, training/testing, temporal cross-validation, bias, variance, underfitting, overfitting, error analysis, and learning curves.
- **Unit II:** feature engineering, missing-value handling, standardization, scaling for linear/SVR models, and regression metrics.
- **Unit III:** descriptive statistics, OLS linear regression, regularized linear regression, and model comparison.
- **Unit IV:** support-vector regression with validation-based selection of `C`.

XGBoost is technically an ensemble method related to Unit V, but it can be presented as the advanced benchmark while the syllabus discussion focuses on Units I-IV.

## Recommended presentation order

1. Define the prediction point: estimate a player's market value on a given historical date.
2. Show why all features must occur before that date.
3. Explain the different raw-table grains and the aggregation into one valuation-level feature table.
4. Compare OLS, Ridge, Linear SVR, and XGBoost under the same temporal protocol.
5. Show the learning curves and explain the linear-model plateau.
6. Open the player explorer and discuss one prediction and its recent-form history.
7. End with limitations: Transfermarkt value is an estimate, some source attributes are static snapshots, and cross-sport targets need sport-specific definitions.

## Key commands

```powershell
# Rebuild and train the upgraded experiment
python run_pipeline.py

# Run the automated checks
python -m pytest tests/test_upgrades.py tests/test_dashboard.py -q
python tests/check_consistency.py

# Open the presentation dashboard
start_dashboard.cmd

# Explore the project graph
graphify query "what connects recent features to model evaluation?"
```

## Final recommendation

Keep the raw CSVs separate or package them as separate tables in one DuckDB database. Use the Parquet feature table as the single modeling input. Continue comparing OLS, Ridge, Linear SVR, and XGBoost with the existing time-aware protocol. This gives a clear syllabus story, a fair experimental design, and a defensible reason why the nonlinear model performs better.
