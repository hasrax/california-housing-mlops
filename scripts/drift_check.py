import argparse
import logging
import sys
from datetime import timedelta

from housing_mlops.api.db import init_db, make_engine, make_session_factory
from housing_mlops.config import get_settings
from housing_mlops.monitoring import DriftThresholds, run_drift_job
from housing_mlops.training import load_current


def parse_args() -> argparse.Namespace:
    settings = get_settings()
    parser = argparse.ArgumentParser(description="Compare recent requests with the training data.")
    parser.add_argument("--database-url", default=settings.database_url)
    parser.add_argument("--model-dir", default=settings.model_dir)
    parser.add_argument("--window-hours", type=float, default=24.0)
    parser.add_argument("--min-requests", type=int, default=100)
    parser.add_argument("--psi-threshold", type=float, default=DriftThresholds.psi)
    parser.add_argument("--min-drifted-features", type=int, default=2)
    parser.add_argument(
        "--fail-on-drift", action="store_true", help="exit with status 2 when drift is detected"
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=get_settings().log_level, format="%(levelname)s %(name)s: %(message)s"
    )
    model = load_current(args.model_dir)
    engine = make_engine(args.database_url)
    init_db(engine)
    thresholds = DriftThresholds(
        psi=args.psi_threshold, min_drifted_features=args.min_drifted_features
    )
    with make_session_factory(engine)() as session:
        report = run_drift_job(
            session,
            model.metadata,
            window=timedelta(hours=args.window_hours),
            min_requests=args.min_requests,
            thresholds=thresholds,
        )
    if report is None:
        print("not enough recent requests; no report stored")
        return 0

    print(f"drift report {report.id} for {report.model_version}: {report.recent_count} requests")
    print(
        f"{'feature':12s} {'psi':>8s} {'ks':>8s} {'p':>10s} {'ref mean':>10s} {'recent':>10s}  flag"
    )
    for item in report.details["features"]:
        flag = "DRIFT" if item["drifted"] else ""
        print(
            f"{item['feature']:12s} {item['psi']:8.4f} {item['ks_statistic']:8.4f} "
            f"{item['ks_p_value']:10.2e} {item['reference_mean']:10.3f} "
            f"{item['recent_mean']:10.3f}  {flag}"
        )
    print(f"drifted={report.drifted} ({report.n_drifted} features)")
    return 2 if (report.drifted and args.fail_on_drift) else 0


if __name__ == "__main__":
    sys.exit(main())
