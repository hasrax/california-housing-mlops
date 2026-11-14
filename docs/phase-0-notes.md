# Phase 0 notes: setup

## What was built

- Environment check: Windows 11, PowerShell 5.1, Python 3.13.1, git 2.43,
  Docker 29.7 with Compose 5.3 (daemon not needed until phase 3).
- A virtual environment at `.venv` with the full dependency stack installed once,
  so that every version in `pyproject.toml` is a version that actually resolved and
  installed together on this machine.
- `pyproject.toml` with the package metadata, pinned runtime and dev dependencies,
  and configuration for ruff, pytest and coverage.
- An installable `src/housing_mlops` package with a `config.py` settings module and
  empty sub-packages for each later phase.
- First unit tests (`tests/unit/test_config.py`) and a shared `conftest.py`.
- `docs/architecture.md` with the component table, a system diagram and a request
  sequence diagram in Mermaid.
- A minimal GitHub Actions workflow with a lint job and a unit test job.
- `.gitignore`, `.env.example`, a README stub.

## Design choices and why

**src layout.** The package lives in `src/housing_mlops`, not at the repo root.
With a root layout, `import housing_mlops` works from the repo folder even when the
package is not installed, which hides packaging mistakes until CI or Docker. The
src layout forces `pip install -e .` and makes tests run against the installed
package, exactly as Docker will.

**Settings via pydantic-settings.** One `Settings` class reads defaults, `.env`
and real environment variables in that order. The API, the drift job and the
dashboard all import the same object, so there is one place that knows about
`DATABASE_URL`. The test fixture builds `Settings(_env_file=None, ...)` so unit
tests never read a developer's `.env` by accident.

**Pin by installing, not by guessing.** The exact versions in `pyproject.toml`
came from `pip freeze` after a real install. Pinning versions from memory risks
combinations that never co-existed on PyPI.

**Python 3.13.** It is the default interpreter on this machine, and every library
in the stack publishes 3.13 wheels. The CI and Docker images use the same minor
version so behaviour is identical everywhere.

**Ruff for lint and format.** One tool replaces flake8, isort and black, and it is
fast enough to run on every commit. Formatting is checked in CI with
`ruff format --check` so style is never a review topic.

**Coverage configured now, enforced later.** `fail_under = 85` is set in the
coverage config from the start. In phase 0 the only code is `config.py`, which is
fully covered, so the gate passes trivially. Setting it now means later phases
cannot forget it.

**Initial commit on `main`, everything else on branches.** The first commit holds
only `.gitignore` and the README stub so that `main` exists as a base. The rest of
phase 0 is on `feat/phase-0-setup` and arrives through a pull request, like every
later phase.

## Alternatives rejected

- **Poetry or uv for dependency management.** Both are good, but neither was
  installed, and plain `pip` with a pinned `pyproject.toml` is enough for a project
  this size. Fewer tools for a reader to learn.
- **A Makefile.** Make is not available in PowerShell by default. Commands are
  documented in the README instead; a `justfile` or `tasks.py` can be added later
  if the command list grows.
- **A `requirements.txt` alongside `pyproject.toml`.** Two sources of truth drift
  apart. `pyproject.toml` with `[project.optional-dependencies]` is the single
  source.
- **MLflow tracking server in Compose.** More realistic, but it adds a service,
  a volume and a port for a demonstration that the file store handles fine.
- **Putting tests next to the code.** Keeps the package importable without pytest
  installed in the Docker image and makes the unit/integration split obvious.

## Places where a code comment would help

1. `src/housing_mlops/config.py`, the `model_config` line: why `extra="ignore"`
   (Docker Compose and CI inject unrelated variables).
2. `tests/conftest.py`, the `settings` fixture: why `_env_file=None` is passed.
3. `pyproject.toml`, the `[tool.coverage.report]` section: why `fail_under` is 85
   and which files are omitted.
4. `.github/workflows/ci.yml`: why the `permissions` block is `contents: read`.

## Five questions to test your understanding

1. If you deleted `src/` and moved `housing_mlops/` to the repo root, which test
   would still pass locally but could fail inside Docker, and why?
2. The `Settings` class has `database_url` as a plain string default. What happens
   in CI if `DATABASE_URL` is not set, and is that the behaviour we want for unit
   tests versus integration tests?
3. Why does `.gitignore` list `data/raw/*` and then `!data/raw/.gitkeep` instead of
   just `data/`?
4. The CI workflow installs with `pip install -e ".[dev]"`. What would break if a
   runtime dependency were accidentally listed only under `dev`?
5. `pytest` is configured with `--import-mode=importlib`. What problem does that
   avoid when two test folders both contain a file called `test_config.py`?
