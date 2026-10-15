# Upgrade validation record

Validated on 15 October 2026 using the locally installed versions captured in `models/upgraded/run_metadata.json`.

- Full feature generation completed on 547,653 valuation rows. Independently rebuilt 365-day minutes/goals/assists matched the original base table.
- Full training and model selection completed: 435,947 training rows, 111,706 historical test rows; calendar validation in 2020, 2021 and 2022; two depths per XGBoost variant; 250 trees for both validation and final fitting; and three candidate `C` values for Linear SVR.
- All 11 tests in `tests/test_upgrades.py` and `tests/test_dashboard.py` passed. They cover historical boundaries, future exclusion, duplicates, training-only preprocessing, unknown categories, OLS/SVR prediction and serialization, nested learning-curve samples, dashboard filters, player/date selection, empty search, learning-curve selection and method/roadmap views.
- All 6 original consistency checks passed against the preserved legacy artifacts.
- All five saved pipelines replayed the first 128 test predictions within numerical tolerance. Parameter metadata was read back from the saved fitted estimators and normalized to strict JSON, without changing predictions.
- The dashboard was opened and visually inspected in the browser at `http://127.0.0.1:8501`; overview, player navigation, and learning-curve navigation were verified. A clipped top label was corrected before the final preview.
- Sixteen learning-curve measurements were generated from four nested pre-2023 training sizes and a fixed 2023 validation set. The CSV and four-panel PNG were saved under `models/upgraded/`.
- Streamlit's Windows runtime requires local socket access. The local server and interaction tests ran outside the tool sandbox; the server is bound only to 127.0.0.1. No external hosting was configured.

| Model | Log RMSE | Euro MAE | Euro RMSE |
|---|---:|---:|---:|
| Corrected annual-feature XGBoost | 0.91667 | 1,622,405 | 4,857,420 |
| Recent-form XGBoost | 0.91248 | 1,607,832 | 4,884,332 |
| Ridge with the same recent features | 0.99753 | 2,108,116 | 7,321,074 |
| Ordinary least squares with the same recent features | 0.99744 | 2,108,778 | 7,350,095 |
| Linear SVR with the same recent features | 0.99719 | 2,106,626 | 7,327,365 |

Recent-form features modestly improved log RMSE and euro MAE, while euro RMSE worsened. The linear models plateaued near 0.986 validation Log RMSE at the largest learning-curve sample, whereas recent-form XGBoost reached 0.890. These results do not establish statistical significance. The original inspected test period is a historical benchmark, not a new blind holdout. A clean installation using all pinned requirements was not performed; no multimodal model, prediction intervals or SHAP explanations are claimed as implemented.
