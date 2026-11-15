# Phase 4 notes: drift monitoring, traffic simulator, dashboard

## What was built

- `monitoring.statistics`: the drift maths in numpy and scipy. `observed_fractions`
  bins a sample with the reference decile edges (outer bins open to infinity),
  `population_stability_index` compares two fraction vectors, `reference_cdf`
  rebuilds the training distribution's empirical CDF from the stored quantiles, and
  `ks_against_reference` runs a one-sample Kolmogorov-Smirnov test against it.
- `monitoring.drift`: `DriftThresholds` (PSI >= 0.2, or KS statistic >= 0.1 with
  p < 0.01, and at least two drifted features for the report to count as drifted),
  `compute_drift` producing one `FeatureDrift` per feature and a `DriftResult`.
- `monitoring.job`: `run_drift_job` reads the recent requests for the served model
  version from the `predictions` table, computes drift against the metadata's
  reference statistics and stores a row in `drift_reports` with the full per-feature
  detail as JSON. Too few requests means no report, not a misleading one.
- `monitoring.simulator`: `TrafficPlan` samples real test rows, sends a normal batch,
  then a batch whose drift intensity ramps from 0 to 1 (income x2, age x0.4,
  occupancy x1.8, latitude -2), always clipped to the request bounds.
- `api.db`: two new tables, `drift_reports` and `model_versions`. The API registers
  the model it serves at startup (`register_model_version`), so version history
  exists from the first run and phase 5 only has to call the same function on
  promotion.
- `scripts/simulate_traffic.py` and `scripts/drift_check.py` (exit code 2 with
  `--fail-on-drift`, which phase 5 uses as the trigger).
- `dashboard.app`: prediction volume per minute or hour, PSI per feature for the
  latest report with the threshold line and drifted features highlighted, PSI
  history per feature across reports, model version history with test RMSE.
- Tests: 31 unit tests for the maths including 6 Hypothesis property tests, 7 for
  drift decisions, 5 for the job on SQLite, 13 for the simulator, 2 for model version
  registration, 2 headless dashboard tests with Streamlit's `AppTest`, and 1
  PostgreSQL integration test for the job.

## Demo on this machine (Compose stack, real model)

```
300 normal requests          report 1: 0 drifted features, drifted=False
400 drifted requests         report 2: 700 requests in window, 4 drifted features
  MedInc    PSI 0.186  KS 0.177  p 1e-19   mean 3.87 -> 4.88   DRIFT
  HouseAge  PSI 0.145  KS 0.158  p 1e-15   mean 28.6 -> 24.3   DRIFT
  AveOccup  PSI 0.524  KS 0.306  p 1e-58   mean 3.11 -> 3.64   DRIFT
  Latitude  PSI 0.514  KS 0.238  p 3e-35   mean 35.6 -> 35.1   DRIFT
  AveRooms, AveBedrms, Population, Longitude: PSI < 0.04, not flagged
drift_check --fail-on-drift   exit code 2
```

Report 1 is instructive: `HouseAge` had p = 0.010 but a KS statistic of 0.093, below
the 0.1 floor, so it was not flagged. With hundreds of requests, p-values alone get
small for differences that do not matter; the statistic floor keeps the rule honest.

## Design choices and why

**Reference statistics from the model, not from the database.** The drift job never
reads the training parquet. Everything it needs (decile edges, bin fractions, 101
quantiles, means) is in `metadata.json`, so the baseline moves with the model when a
new version is promoted.

**PSI and KS together, with both a statistic floor and a p-value.** PSI is the
industry habit (0.1 moderate, 0.2 significant) and is insensitive to sample size. KS
is a proper test but its p-value collapses with large samples. Flagging on PSI >= 0.2
*or* (KS >= 0.1 *and* p < 0.01) means a feature needs either a big distribution
change or a statistically solid moderate one.

**Open outer bins.** Requests outside the training range would otherwise be dropped
from the PSI histogram, which is exactly the traffic a drift monitor must count.

**A step-aware reference CDF.** Discrete-ish features (`HouseAge`) produce repeated
quantiles. Linear interpolation across a tie overstates the CDF just below the tied
value; the first version did that and a test caught it. The CDF now jumps at ties:
just below a tie it uses the first grid probability, at the tie the last.

**Two drifted features before the report counts as drifted.** One feature can move
for benign reasons (a marketing campaign in one region). Retraining is expensive and
should respond to a pattern, not a blip. The number is a threshold parameter, not a
constant.

**Time window, not "last N rows".** A window is what an operator means by "recent",
it is what a cron job needs, and it makes the report self-describing
(`window_start`, `window_end`). The demo's second report therefore contains both the
normal and the drifted batch (700 rows) and still flags four features.

**Registering model versions from the API, not from the training script.** The thing
that matters operationally is which version served traffic and when. The API knows
that; the training script does not.

**Hypothesis for the maths.** Properties such as "PSI of a distribution with itself is
zero", "PSI is symmetric and non-negative", "observed fractions always sum to one",
"the reference CDF is monotonic and bounded for any sorted quantiles" hold for all
inputs, so they are stated once and Hypothesis searches for counterexamples. The
example-based tests then pin down specific numbers.

**Headless dashboard tests.** Streamlit's `AppTest` runs the page without a browser and
exposes widgets and exceptions. The page is tested against an empty database and a
populated one with two drift reports. `GET /_stcore/health` only proves the server is
up; `AppTest` proves the page renders.

**Dashboard colours by job.** One blue for single-series magnitude (volume, PSI, RMSE),
a status orange for "drifted" with a text label (never colour alone), and a fixed
per-feature colour assignment in `FEATURE_COLUMNS` order for the PSI history so a
feature keeps its colour whatever subset is selected. One axis per chart.

## Alternatives rejected

- **Evidently or another drift library.** They do more, but the brief asks for PSI and
  KS implemented with numpy and scipy, and the implementation is sixty lines.
- **Two-sample KS against a stored training sample.** Simpler maths but leaks training
  rows into the metadata and grows the file with the sample. The quantile CDF keeps
  the metadata small and still gives a faithful reference distribution.
- **Chi-square on the same bins as PSI.** Largely redundant with PSI on ten bins.
- **Alembic migrations for the two new tables.** Still one service owning the schema
  and `create_all` adds new tables without touching existing ones. The moment a
  column changes on an existing table, migrations become necessary.
- **Running the drift job inside the API process on a timer.** It couples the API's
  availability to the job and hides failures. A separate script that cron, Compose or
  GitHub Actions can call is simpler to reason about and to test.

## Places where a code comment would help

1. `statistics.py`, `observed_fractions`: why the outer edges are replaced with
   infinity.
2. `statistics.py`, `reference_cdf`: the tie handling (first versus last grid value).
3. `drift.py`, `DriftThresholds.feature_drifted`: why KS needs both a statistic floor
   and a p-value.
4. `job.py`, `run_drift_job`: why too few requests returns `None` instead of a report.
5. `simulator.py`, `clip_to_bounds`: the tiny margin above the lower bound and the
   bedrooms-versus-rooms fix-up.
6. `dashboard/app.py`, `FEATURE_COLORS`: colour follows the entity, never its rank.
7. `api/db.py`, `register_model_version`: the single-current invariant.

## Five questions to test your understanding

1. PSI uses an epsilon clip of 1e-4 on both fraction vectors. What would happen
   without it when a bin is empty in the recent sample, and why does the clip make
   the demo's `AveOccup` PSI of 0.52 slightly different from the "true" value?
2. Report 1 had `HouseAge` with p = 0.01 and KS = 0.093. Which single threshold
   change would have flagged it, and why would that be a bad change?
3. Why does `recent_requests` filter by `model_version` as well as by time? Describe a
   sequence of events after a promotion where omitting that filter produces a wrong
   report.
4. The simulator ramps intensity linearly from 0 to 1 over the drifted batch. If the
   drift job ran halfway through the batch, roughly what would the `MedInc` mean be
   and would it be flagged?
5. `register_model_version` runs an `UPDATE ... SET is_current = false` for all other
   rows and then sets the new row current, inside one commit. What goes wrong if the
   API restarts twice with two different model directories in quick succession, and
   does the function protect against it?
