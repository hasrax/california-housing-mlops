import logging
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import APIRouter, Depends, FastAPI, HTTPException, Request, status
from sqlalchemy.orm import Session

from housing_mlops import __version__
from housing_mlops.api.db import (
    check_connection,
    init_db,
    make_engine,
    make_session_factory,
    register_model_version,
    session_scope,
)
from housing_mlops.api.schemas import (
    HealthResponse,
    HousingFeatures,
    ModelInfoResponse,
    PredictionResponse,
)
from housing_mlops.api.service import ModelService, log_prediction
from housing_mlops.config import Settings, get_settings

log = logging.getLogger(__name__)

router = APIRouter()


def get_session(request: Request) -> Iterator[Session]:
    yield from session_scope(request.app.state.session_factory)


def get_model_service(request: Request) -> ModelService:
    return request.app.state.model_service


SessionDep = Annotated[Session, Depends(get_session)]
ServiceDep = Annotated[ModelService, Depends(get_model_service)]


@router.get("/health", response_model=HealthResponse)
def health(request: Request, service: ServiceDep) -> HealthResponse:
    database_ok = check_connection(request.app.state.engine)
    healthy = service.loaded and database_ok
    return HealthResponse(
        status="ok" if healthy else "degraded",
        model_loaded=service.loaded,
        model_version=service.model.metadata.version if service.model else None,
        database="ok" if database_ok else "unavailable",
    )


@router.post("/predict", response_model=PredictionResponse)
def predict(features: HousingFeatures, service: ServiceDep, session: SessionDep):
    if not service.loaded:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "no model loaded")
    model = service.require()
    prediction = service.predict(features)
    log_prediction(session, model, features, prediction)
    return PredictionResponse(
        request_id=prediction.request_id,
        prediction=prediction.value,
        model_name=model.metadata.model_name,
        model_version=model.metadata.version,
    )


@router.get("/model-info", response_model=ModelInfoResponse)
def model_info(service: ServiceDep) -> ModelInfoResponse:
    if not service.loaded:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "no model loaded")
    metadata = service.require().metadata
    return ModelInfoResponse(
        model_name=metadata.model_name,
        version=metadata.version,
        created_at=metadata.created_at,
        seed=metadata.seed,
        params=metadata.params,
        feature_columns=metadata.feature_columns,
        target_column=metadata.target_column,
        training_rows=metadata.training_rows,
        training_data_hash=metadata.training_data_hash,
        metrics=metadata.metrics,
        mlflow_run_id=metadata.mlflow_run_id,
    )


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        logging.basicConfig(level=settings.log_level)
        engine = make_engine(settings.database_url)
        init_db(engine)
        app.state.engine = engine
        app.state.session_factory = make_session_factory(engine)
        service = ModelService.from_model_dir(settings.model_dir)
        app.state.model_service = service
        if service.model is not None:
            with app.state.session_factory() as session:
                register_model_version(session, service.model.metadata, source="startup")
        yield
        engine.dispose()

    app = FastAPI(title="California Housing Price API", version=__version__, lifespan=lifespan)
    app.include_router(router)
    return app


app = create_app()
