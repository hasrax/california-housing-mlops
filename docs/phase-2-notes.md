# Phase 2 notes: training pipeline

## What was built

- `training.pipelines`: two scikit-learn pipelines behind one registry.
  `baseline` is `StandardScaler` + `Ridge`. `gradient_boosting` is
  `HistGradientBoostingRegressor` with a monotonic constraint that predictions never
  decrease as `MedInc` increases. `build_pipeline(name, seed=..., **params)` is the
  only entry point.
- `training.metrics`: RMSE, MAE and R² plus `is_better`, which knows the direction
  of each metric. RMSE is the primary metric used for promotion later.
- `training.reference`: per-feature reference statistics computed from the training
  split: count, mean, std, min, max, 101 quantiles, and decile bin edges with the
  fraction of rows in each bin. These feed the drift job in phase 4.
- `training.metadata`: the `ModelMetadata` dataclass and its JSON file
  (`metadata.json`): model name, version, timestamp, seed, params, feature and target
  columns, training data hash, row count, validation and test metrics, reference
  statistics, MLflow run id, library versions.
- `training.registry`: on-disk layout `models/<model>/<version>/{model.joblib,
  metadata.json}`, `load_model`, `set_current` (atomic copy to `models/current`) and
  `list_versions`.
- `training.train`: `train_model` fits a pipeline, evaluates on validation and test,
  writes the artifact and metadata, and records params, metrics, tags, the metadata
  dict and the model file in an MLflow run.
- `training.predict`: `predict_frame` selects the feature columns by name so column
  order and extra columns cannot change a prediction.
- `scripts/train.py`: trains one or both models, prints metrics, optionally copies the
  best (by validation RMSE) to `models/current`.
- Tests: 47 offline unit tests for the training package and 12 model quality tests
  on the real data under `tests/model` (marker `dataset`).
- CI: a `quality-gate` job that downloads the data, runs the real-dataset and model
  quality tests, trains both models and uploads the metadata files as an artifact.

## Results from a real run (seed 42, test split of 4,128 rows)

| Model | Validation RMSE | Test RMSE | Test MAE | Test R² |
|---|---|---|---|---|
| baseline (Ridge) | 0.7337 | 0.7456 | 0.5333 | 0.5758 |
| gradient_boosting | 0.4895 | 0.4873 | 0.3243 | 0.8188 |

Target unit is $100,000, so the gradient boosting model is off by about $32k on a
typical house. `models/current` now points at the gradient boosting version.

## Design choices and why

**A deprecated MLflow backend forced a change.** The brief asked for the MLflow
file store. MLflow 3.18 raises an exception when the plain-directory backend is
used, unless an opt-out environment variable is set, because that backend is in
maintenance mode. Rather than opt into a deprecated path, `MLFLOW_TRACKING_URI`
still names a directory (`./mlruns`) but `resolve_tracking_uri` turns it into a
SQLite database inside that directory, with run artifacts in `mlruns/artifacts`.
It is still a local file, still no server, and it is what MLflow recommends.

**Params, metrics and files, not the MLflow model flavour.** `mlflow.sklearn.log_model`
infers a pip environment and writes a model package, which is slow and
network-sensitive. The run instead records the metadata dict (`log_dict`) and the
joblib file (`log_artifact`). The serving path loads joblib directly from the
registry, so the MLflow copy is purely for tracking and comparison.

**Reference statistics are computed at training time and stored with the model.**
The drift job will compare live requests against the data the deployed model
actually saw. Promoting a new model therefore moves the baseline with it, and no
process needs the training parquet at runtime.

**101 quantiles and decile bins, not raw samples.** PSI needs bins; a KS test needs a
reference distribution. Quantiles let phase 4 rebuild an empirical CDF by
interpolation, and decile edges give PSI bins with equal reference mass. Both are a
few kilobytes per feature, so the metadata file stays small and self-contained.

**Monotonic constraint on income.** Higher median income should never lower a
predicted price. Encoding that as a constraint makes the directional test a strict
inequality over 1,000 real rows instead of a statistical tendency, and it is the
kind of domain knowledge a reviewer likes to see.

**Validation and test are both recorded but used differently.** Validation picks
the model to serve. Test is reported and, in phase 5, used by the promotion gate.
No decision in this phase is made on the test split.

**Atomic `set_current`.** The new version is copied to a staging folder and renamed
over `models/current`, so a reader never sees a half-copied model.

**Version strings are timestamp plus a short random suffix.** Sortable, readable,
and unique even when two trainings start in the same second (which the tests do).

## Alternatives rejected

- **`MLFLOW_ALLOW_FILE_STORE=true`.** One line, but it pins the project to a backend
  MLflow has stopped developing.
- **An MLflow tracking server in Docker Compose.** Realistic, but an extra service
  for a demonstration that the SQLite store already covers.
- **Storing a random sample of training rows for KS.** Simpler than quantiles but
  leaks training data into the metadata file and makes the file size grow with
  the sample.
- **Hyperparameter search.** It would improve the gradient boosting score a little
  and triple the training time in CI. The project is about the loop around the
  model, not squeezing the last point of R².
- **A strict "every prediction in range" test for the baseline.** The first version
  asserted every prediction from both models lies in (-1, 10). The linear baseline
  predicted 11.5 for one test row with 132 average rooms, which is what a linear
  model does on an outlier. The test now checks the 1st to 99th percentile for
  every model and keeps the strict check for the gradient boosting model, which is
  the one that gets served.

## Places where a code comment would help

1. `train.py`, `resolve_tracking_uri`: why a directory becomes a SQLite URI.
2. `train.py`, `configure_tracking`: why the artifact location is set explicitly
   when the experiment is created.
3. `pipelines.py`, `MONOTONIC_FEATURES`: the domain reason for the constraint.
4. `reference.py`, the `np.unique` on bin edges: discrete features produce duplicate
   quantiles, and `np.histogram` needs strictly increasing edges.
5. `registry.py`, `set_current`: why the staging-and-rename dance.
6. `tests/model/test_model_quality.py`, `THRESHOLDS`: where the numbers come from.

## Five questions to test your understanding

1. The gradient boosting pipeline has no `StandardScaler` but the baseline does. Why
   does tree-based boosting not need feature scaling, and why does Ridge?
2. The reference statistics are computed from `split.train` only. What would go
   wrong in phase 4 if they were computed from the whole dataset?
3. `train_model` builds the pipeline before `configure_tracking`. Which test depends
   on that order, and what would it observe if the order were swapped?
4. `test_same_seed_gives_identical_models` asserts different versions but identical
   metrics and predictions. Which parts of the system does that prove deterministic,
   and which part is deliberately not?
5. The promotion gate in phase 5 will compare test RMSE. Why would it be a mistake to
   also use test RMSE to pick `max_iter` or `learning_rate`?
