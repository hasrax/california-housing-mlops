import pytest

from housing_mlops.config import Settings


@pytest.fixture
def settings(tmp_path, monkeypatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.delenv("RANDOM_SEED", raising=False)
    return Settings(_env_file=None, data_dir=tmp_path / "data", model_dir=tmp_path / "models")
