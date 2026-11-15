from collections.abc import Iterator
from datetime import datetime

from sqlalchemy import DateTime, Float, Integer, String, create_engine, func, text
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
