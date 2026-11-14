from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    database_url: str = "postgresql+psycopg://housing:housing@localhost:5432/housing"
    mlflow_tracking_uri: str = "./mlruns"
    model_dir: Path = Path("./models")
    data_dir: Path = Path("./data")
    random_seed: int = 42
    log_level: str = "INFO"


def get_settings() -> Settings:
    return Settings()
