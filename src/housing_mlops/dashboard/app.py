import pandas as pd
import streamlit as st
from sqlalchemy import func, select

from housing_mlops.api.db import PredictionLog, check_connection, make_engine, make_session_factory
from housing_mlops.config import get_settings


def main() -> None:
    settings = get_settings()
    st.set_page_config(page_title="California Housing MLOps", layout="wide")
    st.title("California Housing MLOps")

    engine = make_engine(settings.database_url)
    if not check_connection(engine):
        st.error("Database is not reachable. Start the stack with `docker compose up`.")
        return

    factory = make_session_factory(engine)
    with factory() as session:
        total = session.scalar(select(func.count(PredictionLog.id))) or 0
        by_version = session.execute(
            select(
                PredictionLog.model_version,
                func.count(PredictionLog.id).label("predictions"),
                func.min(PredictionLog.created_at).label("first_seen"),
                func.max(PredictionLog.created_at).label("last_seen"),
            )
            .group_by(PredictionLog.model_version)
            .order_by(func.min(PredictionLog.created_at))
        ).all()

    st.metric("Predictions logged", total)
    st.subheader("Predictions per model version")
    if by_version:
        st.dataframe(
            pd.DataFrame(
                by_version, columns=["model_version", "predictions", "first_seen", "last_seen"]
            ),
            width="stretch",
        )
    else:
        st.info("No predictions yet. Send a request to POST /predict.")
    st.caption("Drift and version history panels arrive in phase 4.")


if __name__ == "__main__":
    main()
