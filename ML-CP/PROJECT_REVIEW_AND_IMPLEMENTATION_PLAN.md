# Project review and implementation plan

> Implementation update: upgrades 1–2 (validation/preprocessing and recent-form features) plus a local
> presentation dashboard are now available. See [UPGRADE_GUIDE.md](UPGRADE_GUIDE.md) for current commands,
> scope and the separate 48-hour roadmap. Findings below refer to the preserved original benchmark.

Reviewed 14 October 2026. Scope: source code, notebook narrative, raw CSV schemas and selected data checks, saved predictions, metrics, and existing consistency checks. The training pipeline was not rerun. Existing source code, models, and generated reports were not changed. Supporting measurements are in `review/audit_results.json`; rerun them with `python review/audit_project.py` from this project directory.

**Recommendation**

Develop this into **Time-aware Football Player Valuation with Explainable Predictions and Calibrated Uncertainty**. Keep the current performance-based valuation task, add a separate history-informed model, and investigate whether recent performance trends add information beyond the player's previous valuation. Add prediction intervals, explanations for individual players, and evaluation across time and player cohorts.

This is appropriate for a college ML project: regression, feature engineering, temporal validation, quantile regression, calibration, and explainability. XGBoost is already a capable model. A neural network is not necessary to make the project academically stronger.

The current work is a solid end-to-end regression application, but its main experiment is still a small-feature Ridge model versus a larger-feature boosted model. The stronger contribution would be a set of clearly defined hypotheses, fair comparisons, and evidence about when predictions can be trusted. This is my assessment of the work, not a claim about your professor's exact reasoning.

**1. What already works**

- The appearance join requires `a.date < valuation_date`, which is the right direction for historical features.
- The final evaluation is chronological, and the five identified contract/club snapshot features are excluded from the headline model.
- Log targets, a naive baseline, segment errors, leakage ablation, and retransformation analysis are already present.
- The README distinguishes market valuations from transfer fees and acknowledges several limitations.
- All six existing consistency checks passed. Independently recomputing the saved headline predictions reproduces log RMSE **0.91084** and euro MAE **EUR 1,600,833**. These checks establish consistency of existing outputs, not a fresh reproduction of model training.

**2. Issues to address, in priority order**

| Priority | Finding and evidence | Why it matters / action |
|---|---|---|
| High | Preprocessing occurs outside the validation folds. `src/models.py:308` applies `prepare_features` using the entire pre-2024 training set; `:384` calls `get_dummies` on both training and test rows. | Validation rows influence imputation statistics, and future/test categories influence the encoded schema. The category issue is not target leakage and its effect is unmeasured; nevertheless fit all learned transformations separately within each training fold. |
| High | All three `TimeSeriesSplit` boundaries share a date between training and validation: 2017-06-26, 2020-03-11, and 2022-03-31. See `src/models.py:221` and the audit output. | Row-based folds are not strictly earlier-date folds. Group by calendar date and use fixed calendar validation windows. This does not imply that a duplicated player-date exists; the processed table has zero duplicate keys. |
| High | Missing coverage is represented as zero performance through `COALESCE` in `src/data_pipeline.py:110`. **62,908/111,706 test rows (56.3%)** have zero previous-year minutes; the rates are **91.61%** in Tier 3 and **85.24%** in Tier 4. | A player with no captured matches is not necessarily inactive. Add observation/coverage features, distinguish unavailable rates from observed zeros, and report performance separately by coverage. Never label every zero-minute row as an injury or lack of ability. |
| High | The model comparison confounds algorithms and inputs. Ridge receives four numeric fields plus position/tier (`src/models.py:333`); XGBoost receives substantially more (`:367`). | Keep the simple Ridge as a baseline, but also compare Ridge and XGBoost on the same available features. The current difference cannot be attributed entirely to boosting. |
| High | There is no valuation-history baseline. On **108,409 eligible test rows**, carrying forward the strictly preceding observed value produces **0.38347 log RMSE / EUR 525,039 MAE**, versus **0.90162 / EUR 1,641,756** for the existing XGBoost predictions on exactly those rows. | This is a different information regime: the existing model deliberately excludes valuation history. It does not invalidate the performance-only task. It does show that a history-informed extension must beat persistence, and that adding previous values alone is not an adequate research contribution. |
| Medium | The tuning model uses 100 trees, learning rate 0.1, and sampling 0.8 (`src/models.py:232`); the final model uses 300 trees, learning rate 0.08, sampling 0.85, and additional regularization (`:256`). | The chosen depth is not selected for the configuration actually evaluated. Use one estimator factory for tuning and fitting; record the complete search results. |
| Medium | The smearing row combines corrected euro predictions with uncorrected log metrics (`src/models.py:464`). The README explains this, but the table mixes two predictors. | Recomputing log metrics from the actual corrected euro predictions gives **0.96448 log RMSE** and **0.75939 log MAE**, rather than 0.91084 and 0.69806. Report coherent metrics for each predictor or explicitly separate base-log scores from corrected-euro scores. |
| Medium | Smearing calibration reuses the final fold already involved in choosing depth (`src/models.py:426`). | This is not final-test leakage, but calibration is not independent of model selection. Reserve a later calibration period after tuning. |
| Medium | The project does not separately evaluate previously unseen players or time drift. The audit finds log RMSE **0.87665** for previously seen players versus **1.15790** for unseen players; annual scores are **0.88396 (2024)**, **0.91944 (2025)**, and **0.94883 (partial 2026)**. | Aggregate scores conceal different use cases and deterioration. These differences do not establish causes; cohort composition and partial-year coverage also matter. Publish cohort counts and calendar-period results. |
| Medium | Rankings take each player's latest test observation (`src/evaluate.py:203`), spanning **2024-01-14 to 2026-06-12**. Of 21,792 players, 5,081 have their latest observation before 2026. | This is not a common-date shortlist or proof of active status. Specify an as-of date and maximum valuation age, display staleness, and name outputs “model disagreements.” Large disagreement is not evidence of a good transfer. |
| Medium | Only XGBoost weights are serialized (`src/models.py:455`), without fitted imputation and encoding state. | A separate prediction process cannot reliably reproduce the training transformation. Save the preprocessing bundle, feature schema, model, configuration, and calibration parameters together. |
| Medium | Existing integrity checks explicitly do not test temporal feature correctness (`src/data_pipeline.py:196`). | Add small, hand-checkable examples that test date boundaries, future-event exclusion, joins, and history lags. README consistency is valuable but cannot detect these modeling bugs. |
| Medium | Historical club/competition attribution is assumed, and player physical/position fields are current snapshots. The README's source/license section is still a TODO (`README.md:472`). | Verify provenance before claiming full point-in-time integrity. A diagnostic finds a different club from the latest appearance within 30 days in 14.95% of matched rows; transfers, loans, and national teams can explain this, so it is not proof of leakage. Audit examples and run a conservative feature ablation. |

Additional reporting corrections: “career” totals begin with recorded appearances in July 2012, not necessarily a player's actual debut; call these observed-history totals. The appearance table ends on 2026-06-28, whereas the README's broader match-date limit comes from a different table; report per-source date coverage. `exp(E[log(Y)|X])` is a geometric-mean-type estimate and is only a conditional median under additional assumptions. Also, the notebook describes `gain` as total error reduction, while XGBoost's `gain` is average gain per split. Age and historical performance can contain proxies for development potential, so “no signal for potential” is stronger than the evidence supports.

**3. Research questions and experimental boundaries**

Use three questions in the report:

1. Do recent performance trends improve valuation estimates beyond demographics and annual totals?
2. When previous valuations are available, do performance features improve on carrying forward the last value?
3. How do error and uncertainty change for unseen players, young players, and poorly covered competitions?

Maintain two clearly labeled information regimes:

- **Performance-only:** estimate the valuation at date t using information strictly before t, excluding previous market values. This preserves the original question.
- **History-informed:** estimate the valuation at t using the same information plus valuations strictly before t. Compare against persistence on matched eligible rows; use the performance-only model as a separately reported fallback for first-history players.

Both are valuation-at-date tasks, not six-month forecasts. Model parameters can remain fixed during a test period while historical inputs update as new observations become available. Declare this sequential evaluation protocol explicitly; never use the current row's target or later valuations as an input. Avoid player-ID features in the core experiment.

**4. Implementation schedule — approximately four part-time weeks**

Budget roughly **45–65 focused hours** for one student already familiar with the code. Hardware, debugging, and course deadlines can change this. Stages depend on the earlier stages; the demo comes last.

| Stage | Effort | Changes and deliverables | Completion criterion |
|---|---|---|---|
| 1. Repair evaluation | 10–14 h | Add `src/splits.py`, `src/preprocessing.py`, `configs/experiment.json`; refactor `src/models.py`; add temporal tests. | Every fold has `max(train_date) < min(validation_date)`, transformations learn only from its training rows, and inference survives save/reload. |
| 2. Improve information | 10–14 h | Extend `src/data_pipeline.py`; add coverage flags, multiple windows, shrinkage, and strict previous-valuation features. Produce a feature dictionary. | Hand-calculated fixtures match feature outputs; a future appended event cannot change earlier features. |
| 3. Run controlled experiments | 10–14 h | Add experiment runner, matched-feature baselines, ablation table, cohort metrics, and player-cluster bootstrap comparisons. | All models use declared information regimes, common folds, identical evaluation rows where applicable, and logged configurations. |
| 4. Uncertainty and explanation | 10–15 h | Add `src/uncertainty.py`, `src/explain.py`; train quantile models and calibrate; create global and individual explanations. | Interval coverage/width are reported overall and by cohort; explanations correspond to the actual saved model and target scale. |
| 5. Presentation | 5–8 h | Update notebook/report and optionally add a small local dashboard or player report. | A reader can trace a displayed prediction to its date, feature history, saved model, uncertainty, and coverage status. |

**Stage 1 details: evaluation protocol**

Use expanding training windows with calendar validation blocks. A feasible development schedule is train through 2019 / validate 2020, train through 2020 / validate 2021, and train through 2021 / validate 2022. Keep 2023 for calibration after selecting the configuration and fitting the final base model on data through 2022. Evaluate 2024 onward using that frozen model and calibration. This sacrifices some recent training data for a simple, defensible protocol; rerun the reference models under the same protocol.

The existing 2024–2026 outputs have already been inspected extensively, including in this review. Call this an existing historical evaluation set, not a newly untouched test. Freeze decisions before rerunning it, report that limitation, and use genuinely unseen later data if the course permits obtaining another snapshot. Do not present a relabeled slice of already examined data as a fresh blind holdout.

Implement a training-only position-height imputer, missingness flags, scaler for Ridge, and `OneHotEncoder(handle_unknown='ignore')`. Keep full one-hot columns for trees. Make snapshot-sensitive attributes removable by configuration. Fit preprocessing afresh within every fold. Add calendar trend features only as an explicit ablation; a year feature alone does not guarantee extrapolation.

Use one common XGBoost parameter factory. Start with approximately 12–20 randomized configurations over depth, learning rate, minimum child weight, regularization, and row/column sampling. Keep the search budget equal across feature ablations. If using early stopping, choose the tree count with development folds, not the calibration or final evaluation data. A quick reduced-data run is for debugging only; publish full-data results.

**Stage 2 details: features feasible with these files**

| Feature family | Implementation | Guardrail |
|---|---|---|
| Recent form | Minutes, appearances, goals, assists over 90, 180, and 365 days; compare recent 90 days with preceding 90 days using rates and exposure. | All windows end strictly before the prediction date. Do not compare raw counts from unequal window lengths as a trend. |
| Stable per-90 output | `90 * (goals + k * prior_rate / 90) / (minutes + k)`, with analogous assists. | Estimate the position-level prior using training data only; tune pseudo-minutes k in development. Preserve exposure indicators. For no observations, this is a prior estimate, not an observed performance statistic. |
| Observation coverage | Observed appearance count, days since last recorded match, length of observed history, and recent captured games for the relevant club/competition. | No recorded match does not prove no match occurred. Keep ambiguous coverage explicit and require historical club attribution for club features. |
| Role and usage | Prior starting share and recorded captaincy share from `game_lineups.csv`; international appearances from historical games and appearances where identifiable. | Deduplicate player-game rows and restrict to prior dates. Do not use current total international caps from `players.csv` as historical counts. |
| Prior valuation | Last known log value, days since it was recorded, previous change, and a history-available flag. | Strictly lag within player. Never use `players.market_value_in_eur`, `highest_market_value_in_eur`, or the target-row value as substitutes. |

Build the first three feature families before lineups. Cache a daily player aggregate to reuse across windows; avoid repeatedly joining every historical appearance to every valuation if runtime becomes a problem. Existing data do not include rich individual defensive metrics such as tackles, interceptions, saves, or xG. Adding those requires a separately sourced, historically aligned dataset and should be optional.

**Stage 3 details: experiment matrix and evaluation**

| Experiment | Model / feature set | Question answered |
|---|---|---|
| A | Position × tier median | How much does a coarse grouping explain? |
| B | Ridge, current simple inputs | Preserve the original reference. |
| C | Ridge and XGBoost on identical clean inputs | What does nonlinear modeling contribute? |
| D | C + coverage and recent-form features | Does better historical information help? |
| E | Strict previous-value persistence, eligible rows only | How strong is valuation inertia? |
| F | XGBoost with valuation history only | Does learning history patterns beat persistence? |
| G | F + performance/coverage features | Does football performance add information beyond history? |

Run feature-group removals from D and G with the same tuning budget. An optional CatBoost comparison can follow, but it is lower priority than these experiments. A stacking ensemble or an LSTM would add implementation burden before resolving the main scientific questions.

Report log RMSE and log MAE, euro MAE and RMSE, and the fraction within a factor of two of the true value. Include row-weighted results and player-macro MAE so frequently valued players do not silently dominate every conclusion. Compare paired errors with player-cluster bootstrap intervals, keeping each player's observations together; describe these as conditional-on-fitted-model comparisons, not uncertainty over the full training process. Report year/quarter variation separately because player clustering does not capture every shared time shock.

Publish results by seen/unseen player, age, position, league, observed minutes, and target-value band, with counts. Same players appearing in earlier training and later testing is legitimate for returning-player valuation; it is not automatically leakage. Evaluate first-history players separately. If doing an additional player-disjoint stress test, preserve chronological boundaries too.

A useful success criterion is a reproducible improvement over the appropriate baseline across multiple development periods, supported by paired uncertainty estimates and no unexplained large cohort regressions. Do not promise a particular accuracy gain. A negative result such as “history dominates, and performance adds little” can still support a strong report.

**Stage 4 details: intervals and explanations**

Train 5th, 50th, and 95th conditional quantile models on log value; exponentiate endpoints to obtain a positive, asymmetric euro interval. XGBoost supports quantile regression, including the possibility of quantile crossing, so check ordering and apply any chosen rearrangement consistently before calibration and evaluation. See the official [prediction interval example](https://xgboost.readthedocs.io/en/latest/python/examples/prediction_intervals.html).

For a manageable calibration extension, use the separate 2023 block and nonconformity scores `max(lower - y, y - upper, 0)` in log space. For nominal 90% coverage, take the finite-sample corrected order statistic at `ceil((n+1)*0.90)` and expand future log intervals by that amount. Handle too-small calibration groups explicitly. Keep the model fixed after calibration; refitting it changes the residual distribution and requires a new valid calibration strategy.

Report empirical coverage, median width, pinball loss, and interval score. Break coverage down by year and cohorts and examine whether intervals become unhelpfully wide. Ordinary split-conformal guarantees rely on exchangeability, which repeated players and temporal drift challenge here; describe this as an empirical calibration experiment, not guaranteed future 90% coverage. This limitation is explained in [Conformal prediction beyond exchangeability](https://arxiv.org/abs/2202.13415). Adaptive methods can remain future work.

Use [Tree SHAP](https://shap.readthedocs.io/en/latest/generated/shap.TreeExplainer.html) for a global summary and 4–6 preselected contrasting player cases: a well-covered attacker, defender, goalkeeper, young player, and a low-coverage case. Use training-only background data where required. Explain contributions in log-value units; do not label them euro amounts or causal effects. Group related encoded categories for display and state the limitations from correlated features. Explain the point model separately from interval calibration.

**Stage 5 details: final demonstration and submission**

A compact notebook or local dashboard should let the examiner select a player and historical date and see the estimate, reference valuation, prediction interval, recent performance trend, top explanatory factors, and coverage/staleness status. Keep retrospective reference values separate from inference inputs. Use this as a presentation layer after the experiments, not as the main claim of technical advancement.

For any disagreement list, use a common reporting date with a declared freshness threshold. Optionally flag reference valuations outside the model interval, but call these unusual disagreements rather than validated bargains. A common-date selection still needs a clearly described historical or current prediction protocol.

Submit a feature dictionary with source timestamps, data-source/download/license notes, a split diagram, baseline and ablation tables, coverage/error plots, individual explanations, reproducible configuration, and a concise limitations discussion. Preserve the original run as a versioned benchmark instead of overwriting it during experiments.

**5. Required tests and reproducibility checks**

- A fixture with matches before, on, and after a valuation verifies that only earlier matches enter; test the exact 365-day boundary.
- Appending a future match or valuation leaves all older feature rows unchanged.
- Every validation date follows every training date; for a future-horizon extension, training labels must also be available before validation begins.
- Changing validation/test values cannot change fitted imputation statistics or category vocabulary.
- Duplicate appearance/player-game input is detected, and zero observation coverage remains distinguishable from observed zero goals.
- Persistence uses the immediately previous known valuation and handles first-history players with the declared fallback.
- A saved/reloaded preprocessing-plus-model bundle reproduces predictions within numeric tolerance and handles unseen categories.
- Reported log metrics equal metrics recomputed from the corresponding euro predictions; interval order and scoring are consistent.

Save data hashes, package versions, seeds, feature schema, split boundaries, full parameters, metrics, and run time under a distinct run directory. The audit used locally installed versions listed in `review/audit_results.json`, which differ from several requirement pins; a clean pinned-environment training reproduction remains to be done. Consult scikit-learn's [preprocessing leakage guidance](https://scikit-learn.org/stable/common_pitfalls) and [TimeSeriesSplit documentation](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html) when implementing the fold-local transformations and calendar splits.

**6. Optional larger advancement: six-month value-change forecasting**

If another 1–2 weeks are available, this is a more substantial extension than adding several algorithms. At an anchor valuation date t, predict `log(V_future / V_t)`. Define the future label as the first observed valuation in `[t+180 days, t+240 days]`; record its actual horizon and exclude anchors with no eligible label, reporting how many were excluded and the resulting selection bias. Never use that future row's performance or metadata as features.

Train only on anchors whose future labels were already observable by the training cutoff. This requires purging overlapping label horizons, not merely splitting anchor dates. Use zero log-change / last value as the primary baseline. Evaluate log-value error and value-change direction, including treatment of unchanged values. A ranking of predicted risers can be evaluated using future value changes, but that does not establish profitable transfers. Keep this task separate from valuation-at-date results.

**7. Scope choices**

- **Two-week minimum:** repair validation, coverage flags and recent-form features, matched-feature baselines, persistence/history comparison, cohort analysis, and a clear ablation report. Defer the dashboard and complex calibration.
- **Recommended four-week version:** complete Stages 1–5, prioritizing prediction intervals and explanations over another model family.
- **Extended version:** add six-month forecasting after the main evaluation is stable.

A suitable proposal to the professor: “I will extend the project from a single valuation regression into a controlled study of performance-based and history-informed valuation. I will test whether recent performance adds predictive information beyond the last known valuation, estimate uncertainty, explain individual predictions, and evaluate reliability across time, unseen players, and data-coverage levels.”
