from pathlib import Path

import housing_mlops
from housing_mlops.config import Settings, get_settings


def test_package_exposes_version():
    assert housing_mlops.__version__ == "0.1.0"


def test_defaults_are_deterministic(settings):
    assert settings.random_seed == 42
    assert settings.mlflow_tracking_uri == "./mlruns"
    assert settings.database_url.startswith("postgresql+psycopg://")


def test_environment_overrides_defaults(monkeypatch):
    monkeypatch.setenv("RANDOM_SEED", "7")
    monkeypatch.setenv("MODEL_DIR", "/tmp/somewhere")
    s = Settings(_env_file=None)
    assert s.random_seed == 7
    assert s.model_dir == Path("/tmp/somewhere")


def test_get_settings_returns_settings_instance():
    assert isinstance(get_settings(), Settings)
