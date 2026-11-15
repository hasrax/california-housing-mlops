# Phase 5 notes: retraining with a promotion gate, full CI

## What was built

- `retraining.promotion`: `gate` scores the candidate and the current champion on
  the same held-out test split (it never trusts metrics stored in metadata), computes
  the relative improvement and promotes only if the candidate is strictly better and
  clears `min_improvement`. `promote` copies the candidate to `models/current` and
  registers it as the current version. `record_decision` stores every verdict,
  promoted or not, in the new `promotion_decisions` table.
- `retraining.retrain`: `retrain_if_drifted` decides whether to train at all (forced,
  no champion yet, or the latest drift report for the served version is drifted),
  trains a candidate with stronger defaults (`CANDIDATE_PARAMS`), runs the gate,
  records the decision and promotes when allowed. It returns a `RetrainingOutcome`
  that serialises to JSON for CI.
- `scripts/retrain.py` (`--force`, `--model`, `--min-improvement`, `--json`) and
  `scripts/retrain_summary.py`, which renders the outcome as Markdown for the GitHub
  Actions job summary.
- `.github/workflows/retrain.yml`: a manually triggered workflow with inputs
  (`force`, `model`, `min_improvement`). It trains a champion, runs the gated
  retraining against a PostgreSQL service, writes the summary and uploads the
  outcome and metadata as artifacts.
- `.github/workflows/ci.yml`: all actions bumped to current majors (checkout v7,
  setup-python v7, upload-artifact v7), coverage XML uploaded, and the Docker job
  now also starts `db` and `api` with Compose and smoke tests `/health` and
  `/predict`.
- Dashboard: a "Promotion decisions" table.
- Tests: 28 new unit tests (gate rules, decision recording, promotion, the trigger
  logic, forced and drifted paths, the summary renderer), one PostgreSQL integration
  test, and a regression test for an MLflow environment leak found along the way.

## The loop, end to end, on this machine

```
champion served           20261114-181555-319faf   test RMSE 0.4873
traffic                   300 normal + 400 drifted requests
drift report 1            4 drifted features, drifted=True
retrain.py                triggered: "drift report 1 flagged 4 features"
candidate                 20261115-113208-9bb8af   test RMSE 0.4784
gate                      +1.82% better than champion, required >= 0%: promoted
retrain.py (again)        not triggered: no drift report for the new champion
docker compose restart api
health                    model_version 20261115-113208-9bb8af
model_versions            old: startup, not current | new: promotion, current
promotion_decisions       report 1, 0.4784 vs 0.4873, promoted=true
```

## Design choices and why

**Score both models on the test split, now.** The champion's metadata already holds
a test RMSE, but trusting it would compare numbers computed at different times,
possibly on different splits. Scoring both models on the same frame in the same
process is the only comparison that is unambiguous. The gate also warns when the two
models have different training data hashes, because then the test split may not be
held out for both.

**Strictly better, with an optional margin.** The default `min_improvement` is 0, so
an identical candidate (same data, same seed, same params) is rejected because it is
not better. A margin such as 1% is available for teams who do not want to churn
models for noise.

**Candidates use stronger hyperparameters, not new labels.** The logged requests have
no ground truth, so the candidate cannot learn from them. In this project the
candidate differs by configuration (lower learning rate, more iterations, larger
trees, seed unchanged). In production the same code path would append freshly
labelled rows to the training split. The gate does not care where the improvement
comes from.

**Every decision is recorded, including rejections.** A rejected candidate is
evidence that the current model is still the best available, which an operator wants
to see on the dashboard next to the drift report that triggered the attempt.

**Trigger on the latest report for the served version.** A report for an older
version is irrelevant once a new model is serving, which is why the second
`retrain.py` run correctly did nothing until new traffic produces a new report.

**The API does not hot-reload.** Promotion changes `models/current` on disk; the API
reads it at startup. A restart is explicit and visible, and Compose makes it one
command. Hot reloading would hide the moment the serving model changed.

**A separate manual workflow.** Retraining is a decision with cost, so it is not run
on every push. `workflow_dispatch` gives a button with inputs, a job summary with the
gate's verdict, and artifacts for inspection. The workflow builds its own champion in
the runner because CI has no access to a deployed model store; the code path is the
same one the operator runs locally.

**The MLflow environment leak.** `mlflow.set_tracking_uri` writes
`MLFLOW_TRACKING_URI` into the process environment. In the test suite that leaked a
temporary SQLite URI into `Settings` and broke an unrelated test once the retraining
tests ran before it. `configure_tracking` now restores the variable to its previous
state, with a test that pins the behaviour.

## Alternatives rejected

- **MLflow Model Registry stages.** Attractive, but the registry needs the tracking
  backend to be shared by API and training, which the SQLite file in a bind mount is
  not. A directory and a database row are simpler and sufficient here.
- **Promotion on validation RMSE.** Validation already chose the hyperparameters, so
  it is biased in the candidate's favour. The test split is the only untouched data.
- **Hot-swapping the model in the API process.** See above; it hides the change and
  adds thread-safety concerns for little benefit.
- **Automatic retraining on a schedule.** Drift is the trigger the brief asks for. A
  schedule could call the same script; it would be one cron line.

## Places where a code comment would help

1. `promotion.py`, `gate`: why both models are scored on the fly and what the hash
   warning means.
2. `promotion.py`, `relative_improvement`: the direction handling for lower-is-better
   versus higher-is-better metrics.
3. `retrain.py`, `CANDIDATE_PARAMS`: why the candidate config differs from the
   training defaults.
4. `retrain.py`, `decide_trigger`: the precedence of force, missing champion, missing
   report and non-drifted report.
5. `train.py`, `configure_tracking`: the environment restore.
6. `retrain.yml`, the `${{ inputs.force && '--force' || '' }}` expression.

## Five questions to test your understanding

1. The candidate in the demo was promoted with a 1.82% RMSE improvement. What
   argument would you give `scripts/retrain.py` to make that same candidate be
   rejected, and what would the dashboard then show?
2. Why does `gate` refuse an identical candidate even though it is "not worse"? What
   would happen over many retraining runs if ties were promoted?
3. After promotion, `scripts/retrain.py` reported "no drift report for champion
   <new version>". Walk through what has to happen before retraining can trigger
   again, and why the old report must not be reused.
4. The retraining workflow trains its own champion inside the runner. What would you
   have to add to make it operate on the model that is actually serving traffic?
5. `configure_tracking` restores `MLFLOW_TRACKING_URI`. Which test caught the leak,
   why did it only start failing in phase 5, and what does that tell you about test
   ordering as a tool?
