# Graph Report - ML-CP  (2026-10-15)

## Corpus Check
- 1 files · ~107,978 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 455 nodes · 814 edges · 29 communities (25 shown, 4 thin omitted)
- Extraction: 94% EXTRACTED · 6% INFERRED · 0% AMBIGUOUS · INFERRED: 50 edges (avg confidence: 0.92)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- Core Model Training
- Report Generation
- Pipeline Orchestration and EDA
- Model Evaluation Utilities
- Feature Importance
- Consistency Checks
- Residual Diagnostics
- Graphify Data Architecture
- Regressor Comparison
- Project Dependencies
- Learning Curves
- Missingness Analysis
- Age Value Analysis
- Temporal Validation Design
- Upgrade Experiment
- Validation Evidence
- EDA Transformations
- Multisport Architecture
- Rankings and Bias
- Dashboard Tests
- Core Project Design
- Target Distribution
- Multisport Data Sources
- Documentation and Provenance
- Streamlit Valuation Lab
- Multimodal Pilot
- Graphify Workflow
- Test Dependencies
- Competition Features

## God Nodes (most connected - your core abstractions)
1. `train_and_evaluate()` - 27 edges
2. `XGBoost Regressor Top 20 Feature Importances Chart` - 23 edges
3. `Residual Prediction Error Breakdown Across Player and League Segments` - 20 edges
4. `run_residual_and_ranking_analysis()` - 18 edges
5. `train_upgraded()` - 17 edges
6. `eur()` - 13 edges
7. `build_learning_curves()` - 12 edges
8. `Core Pipeline Requirements` - 11 edges
9. `Football Market Value Prediction` - 11 edges
10. `main()` - 10 edges

## Surprising Connections (you probably didn't know these)
- `Segment-Level Residual Analysis` --semantically_similar_to--> `Age and League Segment Bias`  [INFERRED] [semantically similar]
  OVERVIEW.md → models/rankings_summary.txt
- `Log Target Transformation` --semantically_similar_to--> `Log-Scale Distribution Stabilization`  [INFERRED] [semantically similar]
  OVERVIEW.md → eda_outputs/eda_summary.txt
- `Model Disagreement Rankings` --semantically_similar_to--> `No Good-Buy Ground Truth`  [INFERRED] [semantically similar]
  OVERVIEW.md → models/rankings_summary.txt
- `main()` --calls--> `file_hash()`  [EXTRACTED]
  run_upgrade.py → src/upgraded_training.py
- `main()` --calls--> `train_upgraded()`  [EXTRACTED]
  run_upgrade.py → src/upgraded_training.py

## Import Cycles
- None detected.

## Hyperedges (group relationships)
- **Four-Model Learning-Curve Experiment** — models_upgraded_learning_curves_ordinary_least_squares, models_upgraded_learning_curves_ridge, models_upgraded_learning_curves_linear_svr, models_upgraded_learning_curves_xgboost [EXTRACTED 1.00]
- **Market Value Distribution Assessment** — ml_cp_eda_outputs_market_value_distribution_raw_market_value, ml_cp_eda_outputs_market_value_distribution_log_transformed_market_value, ml_cp_eda_outputs_market_value_distribution_normal_fit [EXTRACTED 1.00]
- **Market Value Summary Statistics** — ml_cp_eda_outputs_value_vs_age_by_position_median_market_value, ml_cp_eda_outputs_value_vs_age_by_position_mean_market_value, ml_cp_eda_outputs_value_vs_age_by_position_interquartile_range [EXTRACTED 1.00]
- **Position Market Value Trajectory Comparison** — ml_cp_eda_outputs_value_vs_age_by_position_goalkeeper_trajectory, ml_cp_eda_outputs_value_vs_age_by_position_defender_trajectory, ml_cp_eda_outputs_value_vs_age_by_position_midfield_trajectory, ml_cp_eda_outputs_value_vs_age_by_position_attack_trajectory [EXTRACTED 1.00]
- **Age-Band Prediction Bias Comparison** — ml_cp_eda_outputs_residuals_by_segment_age_segment_error_analysis, ml_cp_eda_outputs_residuals_by_segment_mean_prediction_bias, ml_cp_eda_outputs_residuals_by_segment_uncorrected_predictions, ml_cp_eda_outputs_residuals_by_segment_smear_corrected_predictions, ml_cp_eda_outputs_residuals_by_segment_zero_bias_reference, ml_cp_eda_outputs_residuals_by_segment_under_21_prospects_bias, ml_cp_eda_outputs_residuals_by_segment_age_21_24_ascending_bias, ml_cp_eda_outputs_residuals_by_segment_age_25_28_peak_bias, ml_cp_eda_outputs_residuals_by_segment_age_29_32_veterans_bias, ml_cp_eda_outputs_residuals_by_segment_age_33_plus_late_career_bias [EXTRACTED 1.00]
- **League-Tier MAE Comparison** — ml_cp_eda_outputs_residuals_by_segment_league_tier_error_analysis, ml_cp_eda_outputs_residuals_by_segment_mean_absolute_error, ml_cp_eda_outputs_residuals_by_segment_t1_big_5_mae, ml_cp_eda_outputs_residuals_by_segment_t2_top_european_mae, ml_cp_eda_outputs_residuals_by_segment_t3_other_first_tier_mae, ml_cp_eda_outputs_residuals_by_segment_t4_lower_other_mae [EXTRACTED 1.00]
- **Position-Level MAE Comparison** — ml_cp_eda_outputs_residuals_by_segment_position_segment_error_analysis, ml_cp_eda_outputs_residuals_by_segment_mean_absolute_error, ml_cp_eda_outputs_residuals_by_segment_attack_mae, ml_cp_eda_outputs_residuals_by_segment_defender_mae, ml_cp_eda_outputs_residuals_by_segment_goalkeeper_mae, ml_cp_eda_outputs_residuals_by_segment_midfield_mae [EXTRACTED 1.00]
- **Sport-Specific Extension Architecture** — multisport_feasibility_sport_specific_models, multisport_feasibility_shared_temporal_evaluation, multisport_feasibility_common_record_envelope [EXTRACTED 1.00]
- **Controlled Model Comparison Framework** — overview_ridge_regression, overview_xgboost, project_review_and_implementation_plan_matched_feature_model_comparison, readme_five_model_suite, upgrade_guide_five_model_experiment [INFERRED 0.85]
- **Competition Identity Signals** — ml_cp_eda_outputs_feature_importances_competition_identity_features, ml_cp_eda_outputs_feature_importances_competition_id_gb1, ml_cp_eda_outputs_feature_importances_competition_id_c1, ml_cp_eda_outputs_feature_importances_competition_id_mex1, ml_cp_eda_outputs_feature_importances_competition_id_bra1, ml_cp_eda_outputs_feature_importances_competition_id_ts1, ml_cp_eda_outputs_feature_importances_competition_id_ser1, ml_cp_eda_outputs_feature_importances_competition_id_pl1, ml_cp_eda_outputs_feature_importances_competition_id_jap1, ml_cp_eda_outputs_feature_importances_competition_id_ukr1, ml_cp_eda_outputs_feature_importances_competition_id_aus1, ml_cp_eda_outputs_feature_importances_competition_id_rsk1, ml_cp_eda_outputs_feature_importances_competition_id_dk1, ml_cp_eda_outputs_feature_importances_competition_id_col1 [INFERRED 0.95]
- **Goalkeeper Role Signals** — ml_cp_eda_outputs_feature_importances_goalkeeper_role_features, ml_cp_eda_outputs_feature_importances_position_goalkeeper, ml_cp_eda_outputs_feature_importances_sub_position_goalkeeper, ml_cp_eda_outputs_feature_importances_is_goalkeeper [INFERRED 0.95]
- **League Tier and Recent Activity Signals** — ml_cp_eda_outputs_feature_importances_league_tier_features, ml_cp_eda_outputs_feature_importances_league_tier_tier_2_top_european, ml_cp_eda_outputs_feature_importances_league_tier_tier_3_other_first_tier, ml_cp_eda_outputs_feature_importances_recent_playing_time_features, ml_cp_eda_outputs_feature_importances_appearances_last_365d, ml_cp_eda_outputs_feature_importances_minutes_last_365d [INFERRED 0.95]
- **Temporal Integrity Across Project Documentation** — overview_point_in_time_features, project_review_and_implementation_plan_calendar_fold_preprocessing_repair, readme_expanding_calendar_validation, upgrade_guide_validation_and_inference_upgrade [INFERRED 0.95]
- **Graphify Artifact and Integrity System** — project_data_model_and_graphify_guide_graphify_initialization, project_data_model_and_graphify_guide_graph_outputs, project_data_model_and_graphify_guide_first_clean_graph_statistics, project_data_model_and_graphify_guide_clean_graph_integrity [EXTRACTED 1.00]
- **Raw-to-Feature Data Architecture** — project_data_model_and_graphify_guide_normalized_raw_layer, project_data_model_and_graphify_guide_duckdb_raw_packaging, project_data_model_and_graphify_guide_time_aware_denormalized_feature_layer, project_data_model_and_graphify_guide_parquet_model_table [EXTRACTED 1.00]
- **Four-Family Regressor Comparison** — project_data_model_and_graphify_guide_ordinary_least_squares, project_data_model_and_graphify_guide_ridge_regression, project_data_model_and_graphify_guide_linear_svr, project_data_model_and_graphify_guide_recent_form_xgboost [EXTRACTED 1.00]

## Communities (29 total, 4 thin omitted)

### Community 0 - "Core Model Training"
Cohesion: 0.07
Nodes (24): load_outputs(), metrics_table(), FootballFeatures, model_pipeline(), build_recent_features(), sql_path(), calendar_folds(), build_learning_curves() (+16 more)

### Community 1 - "Report Generation"
Cohesion: 0.09
Nodes (33): band_label(), bias_direction(), change_word(), eur(), eur_signed(), load_context(), md_table(), pct() (+25 more)

### Community 2 - "Pipeline Orchestration and EDA"
Cohesion: 0.12
Nodes (16): legacy_main(), main(), main(), build_analysis_dataset(), verify_leakage_invariants(), run_eda(), setup_plot_style(), run_residual_and_ranking_analysis() (+8 more)

### Community 3 - "Model Evaluation Utilities"
Cohesion: 0.09
Nodes (14): clip_log_predictions(), fit_naive_group_median(), fit_ridge(), main_xgb_params(), predict_naive_group_median(), prepare_features(), print_metrics(), regression_metrics() (+6 more)

### Community 4 - "Feature Importance"
Cohesion: 0.09
Nodes (27): appearances_last_365d, Average Gain per Split (Importance), XGBoost Regressor Top 20 Feature Importances Chart, Clean Feature Set, competition_id_AUS1, competition_id_BRA1, competition_id_C1, competition_id_COL1 (+19 more)

### Community 5 - "Consistency Checks"
Cohesion: 0.18
Nodes (14): check(), csv_rows(), figures(), marker_block(), table_rows(), check_absolute_paths(), check_banned_strings(), check_headline_features() (+6 more)

### Community 6 - "Residual Diagnostics"
Cohesion: 0.10
Nodes (24): Actual Minus Predicted Error (EUR), Age 21-24 Ascending Bias (Uncorrected ~+€1.5M; Smear-Corrected ~+€0.4M), Age 25-28 Peak Bias (Uncorrected ~+€1.25M; Smear-Corrected ~-€0.2M), Age 29-32 Veterans Bias (Uncorrected ~+€0.15M; Smear-Corrected ~-€0.9M), Age 33+ Late-Career Bias (Uncorrected ~-€0.1M; Smear-Corrected ~-€0.5M), Age-Segment Error Analysis, Attack MAE (~€1.9M), Residual Prediction Error Breakdown Across Player and League Segments (+16 more)

### Community 7 - "Graphify Data Architecture"
Cohesion: 0.12
Nodes (16): Final 2024-2026 Test Period, Zero-Issue Graph Integrity Diagnostic, Project Data, Model Comparison, and Graphify Guide, Fair Model Comparison Protocol, First Clean Post-Build Graph: 443 Nodes and 804 Edges, Graphify Architecture Communities, Graphify HTML, JSON, and Architecture Report Outputs, Project-Scoped Graphify Initialization (+8 more)

### Community 8 - "Regressor Comparison"
Cohesion: 0.15
Nodes (12): Annual-Feature XGBoost Ablation, Separate Raw Tables in One DuckDB Database, Nested-Sample Learning-Curve Experiment, Linear Support Vector Regression, Many-to-Many Row Multiplication, Normalized Immutable Raw Layer, Ordinary Least Squares, Recent-Form XGBoost (+4 more)

### Community 9 - "Project Dependencies"
Cohesion: 0.12
Nodes (18): Dashboard Requirements, Plotly 7.1.0, Streamlit 1.64.0, Core Pipeline Requirements, DuckDB 1.5.6, Joblib 1.5.3, Matplotlib 3.11.2, Notebook Requirements (+10 more)

### Community 10 - "Learning Curves"
Cohesion: 0.26
Nodes (9): Fixed 2023 Validation Set, Linear SVR Learning Curve, Learning-Curve Model Comparison, Nested Pre-2023 Training Samples, Ordinary Least Squares Learning Curve, Ridge Learning Curve, Four-Model Learning Curves, XGBoost Learning Curve (+1 more)

### Community 11 - "Missingness Analysis"
Cohesion: 0.22
Nodes (9): Club Squad Size Missingness, Complete Rolling Performance and League Fields, Contract Days Remaining Missingness, Dataset Feature Missingness, Goalkeeper Performance Sparsity, Low-Missingness Player Attributes, Outfield Position Sparsity Gradient, Rolling 365-Day Stat Sparsity by Position (+1 more)

### Community 12 - "Age Value Analysis"
Cohesion: 0.36
Nodes (10): Age-Market Value Trajectory, Attack Market Value Trajectory, Defender Market Value Trajectory, Goalkeeper Market Value Trajectory, 25th-75th Percentile Market Value Range, Mean Market Value, Median Market Value, Midfield Market Value Trajectory (+2 more)

### Community 13 - "Temporal Validation Design"
Cohesion: 0.20
Nodes (8): Calibrated Prediction Intervals, History-Informed Information Regime, Performance-Only Information Regime, Previous-Valuation Persistence Baseline, Player-Cluster Bootstrap Comparison, Six-Month Value-Change Forecasting, Time-Aware Valuation with Explanations and Calibrated Uncertainty, Validation and Inference Upgrade

### Community 14 - "Upgrade Experiment"
Cohesion: 0.24
Nodes (7): Five-Model Regression Suite, Empirical Learning Curves, Multi-Window Recent-Form Features, Upgraded Machine Learning Pipeline, Five-Model Controlled Experiment, Five-Minute Professor Demonstration, Recent Form and Observation Coverage Upgrade

### Community 15 - "Validation Evidence"
Cohesion: 0.20
Nodes (9): Dashboard and Learning-Curve Navigation Inspection, Upgrade Validation Record, Five-Model Test Metric Comparison, Full Feature Generation Validation, Sixteen-Point Learning-Curve Validation, Calendar Model Selection Validation, Five-Pipeline Serialization Replay Validation, Upgrade, Learning-Curve, and Dashboard Test Validation (+1 more)

### Community 16 - "EDA Transformations"
Cohesion: 0.22
Nodes (5): EDA and Sparsity Report, Performance Sparsity by Position, Position-Specific Feature Interactions, Position-Specific Peak Market Value Age, Market Value Target Distribution

### Community 17 - "Multisport Architecture"
Cohesion: 0.22
Nodes (6): Common Cross-Sport Record Envelope, Multi-Sport Feasibility Study, Multi-Sport Player Value Platform, NBA Stats, Shared Temporal Evaluation Framework, Sport and Target Specific Models

### Community 18 - "Rankings and Bias"
Cohesion: 0.32
Nodes (5): Residual Breakdown and Rankings Report, Largest Model-Above-Market Gaps, Largest Model-Below-Market Gaps, Age and League Segment Bias, Segment-Level Residual Analysis

### Community 19 - "Dashboard Tests"
Cohesion: 0.39
Nodes (5): app(), test_learning_curves_model_selection(), test_method_and_roadmap(), test_overview_filters(), test_player_search_and_history()

### Community 20 - "Core Project Design"
Cohesion: 0.29
Nodes (6): Football Market Value Prediction, Leakage Audit and Snapshot-Feature Ablation, Position-by-Tier Median Baseline, Generated-Report Reproducibility and Consistency Checks, Ridge Regression Benchmark, XGBoost Market Value Model

### Community 21 - "Target Distribution"
Cohesion: 0.53
Nodes (5): Log-Transformed Market Value Distribution, Normal Distribution Fit, Extreme-Value Heavy Tail, Raw Market Value Distribution, Target Variable Distribution and Log-Transformation Assessment

### Community 22 - "Multisport Data Sources"
Cohesion: 0.33
Nodes (5): Historical Baseball Salary Pilot, IPL Auction Price Pilot, Cricsheet Match Data, Official IPL Auction Reporting, SABR Lahman Database

### Community 23 - "Documentation and Provenance"
Cohesion: 0.47
Nodes (6): Project Overview and Concept Guide, Project Review and Implementation Plan, Football Player Market-Value Prediction and Valuation Lab README, Transfermarkt Dataset Provenance, College Upgrade Guide, Scikit-Learn Preprocessing Leakage Guidance

### Community 24 - "Streamlit Valuation Lab"
Cohesion: 0.33
Nodes (4): Preserved Legacy Benchmark Pipeline, Interactive Touchline Streamlit Dashboard, Touchline Valuation Lab, Transfermarkt Market Value Target

## Ambiguous Edges - Review These
- `Age and League Segment Bias` → `Missing Potential, Hype, and Brand Signal Hypothesis`  [AMBIGUOUS]
  models/rankings_summary.txt · relation: rationale_for

## Knowledge Gaps
- **87 isolated node(s):** `Transfermarkt Dataset Provenance`, `Dashboard and Learning-Curve Navigation Inspection`, `Five-Model Test Metric Comparison`, `Full Feature Generation Validation`, `Sixteen-Point Learning-Curve Validation` (+82 more)
  These have ≤1 connection - possible missing edges or undocumented components. (Counts symbols only; 177 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **4 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `Expanding Calendar Validation` connect `Graphify Data Architecture` to `Multisport Architecture`, `Upgrade Experiment`?**
  _High betweenness centrality (0.030) - this node is a cross-community bridge._
- **What connects `Transfermarkt Dataset Provenance`, `Dashboard and Learning-Curve Navigation Inspection`, `Five-Model Test Metric Comparison` to the rest of the system?**
  _87 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `Core Model Training` be split into smaller, more focused modules?**
  _Cohesion score 0.06829488919041157 - nodes in this community are weakly interconnected._
- **Why does `Football Market Value Prediction` connect `Core Project Design` to `Temporal Validation Design`, `EDA Transformations`, `Multisport Architecture`, `Rankings and Bias`, `Documentation and Provenance`?**
  _High betweenness centrality (0.027) - this node is a cross-community bridge._
- **Should `Report Generation` be split into smaller, more focused modules?**
  _Cohesion score 0.08985200845665962 - nodes in this community are weakly interconnected._
- **Why does `run_residual_and_ranking_analysis()` connect `Pipeline Orchestration and EDA` to `Core Model Training`, `Consistency Checks`?**
  _High betweenness centrality (0.019) - this node is a cross-community bridge._

### Low-confidence Hints
_AMBIGUOUS edges — the extractor was unsure. Verify before acting on these._

- **What is the exact relationship between `Age and League Segment Bias` and `Missing Potential, Hype, and Brand Signal Hypothesis`?**
  _Edge tagged AMBIGUOUS (relation: rationale_for) - confidence is low._