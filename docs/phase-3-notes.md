# Phase 3 notes: API, PostgreSQL logging, Docker

## What was built

- `api.schemas`: `HousingFeatures` (the request body) with per-field bounds taken from
  the same `FEATURE_BOUNDS` dict the Pandera schema uses, `extra="forbid"`, and the
  bedrooms-versus-rooms rule as a model validator. Response models for the three
  endpoints.
- `api.db`: SQLAlchemy 2 declarative `PredictionLog` table (`predictions`) with one
  column per feature, the prediction, model name and version, a unique `request_id`,
  a timezone-aware `created_at` and the latency in milliseconds. Engine and session
  helpers, `init_db` (create tables) and `check_connection`.
- `api.service`: `ModelService` loads `models/current` at startup and predicts
  through `predict_frame`; `log_prediction` writes one row per request.
- `api.main`: `create_app(settings)` builds the FastAPI app with a lifespan that opens
  the engine, creates tables and loads the model. Routes: `GET /health`,
  `POST /predict`, `GET /model-info`. The module-level `app` is what uvicorn serves.
- `dashboard.app`: a first Streamlit page (prediction count and predictions per
  model version) so the `dashboard` Compose service is real. Phase 4 adds the drift
  and version panels. `scripts/run_dashboard.py` launches it from the installed
  package.
- `Dockerfile`: two stages. The builder installs the project into a virtualenv; the
  runtime image copies only that virtualenv and `scripts/`, creates a system user
  `app`, and runs uvicorn as that user. One image serves both API and dashboard.
- `docker-compose.yml`: `db` (PostgreSQL 17 with a `pg_isready` health check and a
  named volume), `api` (built image, `./models` mounted read-only, waits for a
  healthy db) and `dashboard` (same image, Streamlit command, its own health check).
- Tests: 46 offline API unit tests on SQLite (schemas, routes, logging, 503 paths)
  and 4 integration tests against real PostgreSQL (`integration` marker).
- CI: the test job now runs unit and integration tests against a PostgreSQL service
  container; a new `docker` job validates the Compose file, builds the image, imports
  the app inside it and asserts the container user is not root.

## Smoke test on this machine

```
docker compose up -d --wait            db healthy, api healthy, dashboard healthy
id (inside api)                        uid=999(app) gid=999(app)
GET  /health                           200 {"status":"ok","model_loaded":true,...}
POST /predict (valid)                  200 prediction=4.067, model gradient_boosting
POST /predict (MedInc=-1)              422
GET  /model-info                       200 training_rows=14448, test rmse 0.4873
psql select count(*) from predictions  1
GET  :8501/_stcore/health              200
```

Image size: 1.57 GB. Most of it is MLflow, Streamlit and scikit-learn.

## Design choices and why

**One bounds dictionary, two validators.** Pandera validates the training data and
Pydantic validates requests, but both read `FEATURE_BOUNDS`. A row that would fail
training validation is rejected at the API with a 422 that names the field and the
rule. Drift (unusual but valid values) is not the API's job.

**One column per feature in the predictions table.** A JSON blob would be shorter to
write, but the drift job in phase 4 needs per-feature arrays, and the dashboard needs
simple aggregates. Typed columns make both plain SQL.

**Synchronous logging inside the request.** The row is written before the response
is returned, so a 200 means the prediction was recorded. A background task or a
queue would be faster under heavy load but makes "was it logged?" a harder question
in tests and in the drift job.

**`create_app(settings)` instead of a global app configured by import.** Tests build
an app per test with a temporary SQLite file and a temporary model directory, so the
same route code runs against SQLite in unit tests and PostgreSQL in integration
tests. The module-level `app` exists only for uvicorn.

**Missing model is a degraded service, not a crash.** The API starts without a model
so that `GET /health` can say why it is unhealthy. `/predict` and `/model-info`
return 503 until a model is present. Compose mounts `./models`, so `python
scripts/train.py --set-current` on the host fixes it without a rebuild.

**`create_all` instead of Alembic migrations.** There is one table today and the
schema is owned by this service. Migrations would be the right next step once a
second service or a schema change arrives; the notes for phase 4 revisit this when
drift reports are added.

**SQLite in unit tests.** The ORM model avoids PostgreSQL-only types, so the same
table definition runs on SQLite. Unit tests therefore need no network and no
container, and the integration tests check the things SQLite cannot: that
`created_at` is `timestamp with time zone` and `request_id` has a unique index.

**httpx2 for the test client.** Starlette 1.10 deprecates the old `httpx` transport
in its `TestClient` in favour of `httpx2`. It is a dev dependency only; the
traffic simulator in phase 4 can use either.

**Non-root user and a read-only model mount.** The container cannot write to its own
code or to the model directory. A compromised request handler cannot swap the model.

**Two health checks.** The image's `HEALTHCHECK` probes the API port. The dashboard
service overrides it in Compose to probe Streamlit's `_stcore/health`, otherwise
Compose would report the dashboard unhealthy even when it works (which is exactly
what happened on the first run).

## Alternatives rejected

- **Separate images for API and dashboard.** Smaller images and fewer dependencies
  per service, but two Dockerfiles to maintain. Splitting the optional dependencies
  (`api`, `dashboard`, `training`) is the first thing to do if image size becomes a
  concern.
- **Baking the model into the image.** Simple for deployment but it couples a model
  version to an image build. Mounting lets retraining in phase 5 promote a model
  without rebuilding or restarting the API image.
- **Async SQLAlchemy and asyncpg.** More throughput, but the prediction itself is
  CPU-bound and synchronous, and the extra machinery buys nothing at demo scale.
- **Skipping integration tests when `DATABASE_URL` is unset.** They fail with a
  message that tells you how to start the database instead.

## Places where a code comment would help

1. `api/db.py`, `make_engine`: why `check_same_thread=False` is needed for SQLite
   under the TestClient and harmless for PostgreSQL.
2. `api/main.py`, the lifespan: order of engine, tables, model, and why the engine is
   disposed on shutdown.
3. `api/service.py`, `ModelService.from_model_dir`: why a missing model is a warning
   and not an exception.
4. `Dockerfile`, the `HEALTHCHECK`: why it uses `urllib` instead of `curl`.
5. `docker-compose.yml`, the `api` volume: why `:ro`.
6. `tests/integration/test_api_postgres.py`, `clean_database`: why tables are dropped
   rather than truncated.

## Five questions to test your understanding

1. The unit tests run the API against SQLite and pass. Name two properties the
   integration tests check that SQLite could never prove.
2. If `extra="forbid"` were removed from `HousingFeatures`, which test fails first,
   and what kind of production bug would that setting have caught?
3. `GET /health` returns 200 even when the status is `degraded`. Why might a load
   balancer want a non-200 in that case, and what would you change?
4. The builder stage installs the project with `pip install .` (not `-e .`). Why
   does the runtime stage then not need the `src/` directory?
5. The `dashboard` service depends on `api` with `condition: service_started`, not
   `service_healthy`. What does the dashboard actually need to work, and is that
   dependency even necessary?
