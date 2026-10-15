# Extending the project to other sports

**Yes: a multi-sport player-value estimation platform is feasible.** A sensible college project would reuse the validation, model-comparison and dashboard framework while fitting a separate model for each sport and economic target. A model trained on football alone cannot reliably price a cricket or baseball player without corresponding training labels.

This document is a proposed extension. No other-sport dataset has been imported and the current predictor supports football only. “Estimator” or “predictor” describes this regression task more accurately than “detector.”

## Decide what value means

| Target | Meaning | Can it be mixed directly with the current labels? |
|---|---|---|
| Football market valuation | An external assessment of a player's market worth | This is the current target |
| Annual salary | Compensation paid to the player over a specified period | No: a different economic quantity |
| Transfer fee | A transaction payment negotiated between clubs | No: depends on transaction/contract conditions |
| Auction bid | A winning price under a particular auction's rules | No: depends on budgets, eligibility and bidding context |
| Performance score | A sporting-output measure defined by the researcher | No: it is not a financial valuation |

Changing rupees to euros does not solve this mismatch. Nor does scaling every target to 0–1 make the meaning equivalent. If the goal is one monetary model, first obtain labels for the **same kind of target**, define the time period, adjust currency/price levels consistently, and test whether pooling helps relative to separate models. Even then, league rules and sport-specific incentives can differ.

## Candidate sources

**Baseball: the cleanest structured proof of concept.** [SABR's Lahman database](https://sabr.org/lahman-database/) is a candidate source for historical baseball records. The package's [Salaries documentation](https://cdalzell.github.io/Lahman/reference/Salaries.html) describes player/year/team salary records and shows coverage from **1985 to 2016**. Do not assume that the salary table extends through the latest year of the wider database. Use the actual downloaded table to verify dates, missingness and identifiers before implementation.

A manageable task is estimating season-t salary from performance observed through season t−1, age and known experience. Restrict the first pilot to batters to avoid mixing pitching and batting statistics prematurely. Aggregate team stints carefully so joins do not multiply observations. A salary proxy measures compensation patterns, not intrinsic talent or football-style market worth.

**Cricket: attractive for an IPL-focused demonstration, with more data preparation.** [Cricsheet](https://cricsheet.org/downloads/) provides historical match data. Its [player register](https://cricsheet.org/article/introducing-the-cricsheet-register-and-a-new-data-format-json/) helps match identities across sources. Monetary labels must come from separate, verified auction records; match statistics alone do not supply player prices. Official IPL reporting documents winning bids—for example, [this IPL article](https://www.iplt20.com/news/article/rishabh-pant-and-kuldeep-yadav-complete-high-profile-trade-between-dc-and-lsg)—but a single article is not a complete reusable dataset. Build and verify a multi-year auction table with provenance before modeling.

Use performance strictly before the actual auction date. “2025 auction” does not necessarily mean an event during calendar 2025: the [official 2025 player-list announcement](https://www.iplt20.com/news/article/tata-ipl-2025-player-auction-list-announced) places that auction in November 2024. Mixing in 2025-season matches would leak future information. Record sold/unsold status; unsold players do not have a zero intrinsic value. A useful later extension is classification for probability of being sold plus price regression conditional on sale, with the selection limitation explained.

**Basketball:** [NBA Stats](https://www.nba.com/stats/players/traditional) supplies performance statistics; salary targets require a separately verified source and historical alignment.

Download and reuse rights, source timestamps and label completeness must be checked for whichever dataset is chosen. We have verified documentation, not downloaded and audited complete cross-sport training tables in this change.

## Recommended design

```text
Sport + target selection
          |
Source-specific data adapter
          |
Historical feature builder
          |
Sport/target-specific preprocessing and model
          |
Shared temporal evaluation and presentation components
```

Create a common record envelope:

```text
sport, league, player_id, as_of_date, season, target_kind,
target_amount, currency, amount_period, feature_cutoff,
source_reference, observation_coverage
```

Names and identifiers are metadata, not automatic predictor features. Each adapter then supplies the appropriate measurements:

- Football: recorded minutes, goals, assists, age, role and recent-form changes.
- Cricket: batting/bowling role, runs, strike rate, wickets, economy, exposure and prior form.
- Baseball: age, position, plate appearances, on-base output and other prior-season batting statistics.

Do not place cricket runs into a field named football goals. Separate preprocessing avoids that false equivalence. Where common concepts are useful, define them explicitly: age, experience, participation, role-relative production and trend. Fit normalizations on training data; do not compute target percentiles from a future test season before predicting it.

Select the model by `(sport, target_kind)` and reject unsupported combinations. The interface should say “Football market value,” “Baseball annual salary,” or “Cricket auction price” next to the currency amount. Maintain separate metrics and model versions. A “sport” dropdown with no trained second-sport model is only a UI demonstration and should be labeled as such.

## Evaluation that would demonstrate genuine generalization

1. Start with one additional sport and evaluate it on later seasons using training-only preprocessing.
2. Compare OLS, Ridge and linear SVR using the same eligible rows and a sport-appropriate baseline such as previous salary or a role/season group estimate.
3. Report error by sport, role, season and observation coverage. Keep raw-currency metrics separate; use log error or baseline-relative skill for a cautiously interpreted summary across tasks.
4. If you subsequently build one pooled model, compare it with separate models on exactly the same per-sport test cases. An average score dominated by the largest dataset is insufficient.
5. Only claim transfer to an unseen sport after a dedicated held-out-sport or limited-label adaptation experiment. Success on football and baseball after training a separate model on each demonstrates framework reuse, not zero-shot generalization.

## College-level scope recommendation

For the current deadline, retain the finished football experiment and explain its linear regression, SVR and learning curves well. A **football + historical baseball salary** pilot is the most straightforward next structured-data extension based on the sources reviewed. An **IPL auction-price** pilot may be easier to present to your audience, but price-label collection and identity matching will likely take more time.

With clean, aligned second-sport labels already available, a small adapter, three baseline models and a dashboard view could fit into a few focused days. Collecting, cleaning and validating that data is additional work and cannot be promised inside 48 hours. Training a universal cross-sport monetary model is a substantially larger research project.

Suggested title for the eventual extension: **Multi-Sport Player Value Estimation with Sport-Specific Models and Temporal Validation**.
