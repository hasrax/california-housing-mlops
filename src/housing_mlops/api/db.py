from collections.abc import Iterator
from datetime import datetime
from typing import Any

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Float,
    Integer,
    String,
    create_engine,
    func,
    select,
    text,
    update,
)
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker


class Base(DeclarativeBase):
    pass


class PredictionLog(Base):
    __tablename__ = "predictions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    request_id: Mapped[str] = mapped_column(String(36), unique=True, index=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), index=True
    )
    model_name: Mapped[str] = mapped_column(String(64))
    model_version: Mapped[str] = mapped_column(String(64), index=True)
    MedInc: Mapped[float] = mapped_column(Float)
    HouseAge: Mapped[float] = mapped_column(Float)
    AveRooms: Mapped[float] = mapped_column(Float)
    AveBedrms: Mapped[float] = mapped_column(Float)
    Population: Mapped[float] = mapped_column(Float)
    AveOccup: Mapped[float] = mapped_column(Float)
    Latitude: Mapped[float] = mapped_column(Float)
    Longitude: Mapped[float] = mapped_column(Float)
    prediction: Mapped[float] = mapped_column(Float)
    latency_ms: Mapped[float] = mapped_column(Float)


class DriftReport(Base):
    __tablename__ = "drift_reports"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), index=True
    )
    model_version: Mapped[str] = mapped_column(String(64), index=True)
    window_start: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    window_end: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    recent_count: Mapped[int] = mapped_column(Integer)
    n_drifted: Mapped[int] = mapped_column(Integer)
    drifted: Mapped[bool] = mapped_column(Boolean, index=True)
    details: Mapped[dict[str, Any]] = mapped_column(JSON)


class ModelVersion(Base):
    __tablename__ = "model_versions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    model_name: Mapped[str] = mapped_column(String(64))
    version: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    created_at: Mapped[str] = mapped_column(String(40))
    registered_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    training_rows: Mapped[int] = mapped_column(Integer)
    training_data_hash: Mapped[str] = mapped_column(String(64))
    validation_rmse: Mapped[float] = mapped_column(Float)
    test_rmse: Mapped[float] = mapped_column(Float)
    test_r2: Mapped[float] = mapped_column(Float)
    is_current: Mapped[bool] = mapped_column(Boolean, default=False)
    source: Mapped[str] = mapped_column(String(32), default="startup")


class PromotionDecision(Base):
    __tablename__ = "promotion_decisions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), index=True
    )
    drift_report_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    candidate_version: Mapped[str] = mapped_column(String(64), index=True)
    champion_version: Mapped[str | None] = mapped_column(String(64), nullable=True)
    metric: Mapped[str] = mapped_column(String(16))
    candidate_value: Mapped[float] = mapped_column(Float)
    champion_value: Mapped[float | None] = mapped_column(Float, nullable=True)
    promoted: Mapped[bool] = mapped_column(Boolean, index=True)
    reason: Mapped[str] = mapped_column(String(255))


def make_engine(database_url: str) -> Engine:
    connect_args = {"check_same_thread": False} if database_url.startswith("sqlite") else {}
    return create_engine(database_url, pool_pre_ping=True, connect_args=connect_args)


def make_session_factory(engine: Engine) -> sessionmaker[Session]:
    return sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)


def init_db(engine: Engine) -> None:
    Base.metadata.create_all(engine)


def check_connection(engine: Engine) -> bool:
    try:
        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))
        return True
    except Exception:
        return False


def session_scope(factory: sessionmaker[Session]) -> Iterator[Session]:
    session = factory()
    try:
        yield session
    finally:
        session.close()


def register_model_version(session: Session, metadata, *, source: str = "startup") -> ModelVersion:
    row = session.scalar(select(ModelVersion).where(ModelVersion.version == metadata.version))
    if row is None:
        row = ModelVersion(
            model_name=metadata.model_name,
            version=metadata.version,
            created_at=metadata.created_at,
            training_rows=metadata.training_rows,
            training_data_hash=metadata.training_data_hash,
            validation_rmse=metadata.metrics["validation"]["rmse"],
            test_rmse=metadata.metrics["test"]["rmse"],
            test_r2=metadata.metrics["test"]["r2"],
            source=source,
        )
        session.add(row)
    session.execute(
        update(ModelVersion)
        .where(ModelVersion.version != metadata.version)
        .values(is_current=False)
    )
    row.is_current = True
    session.commit()
    return row
