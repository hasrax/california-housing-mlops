import argparse
import json
import logging
import sys
from pathlib import Path

from housing_mlops.api.db import init_db, make_engine, make_session_factory
from housing_mlops.config import get_settings
from housing_mlops.data import prepare_dataset, processed_dir, read_splits
from housing_mlops.retraining import retrain_if_drifted
from housing_mlops.training import GRADIENT_BOOSTING, MODEL_NAMES


def parse_args() -> argparse.Namespace:
    settings = get_settings()
    parser = argparse.ArgumentParser(
        description="Train a candidate when drift was detected and promote it if it beats "
        "the current model on the held-out test set."
    )
    parser.add_argument("--force", action="store_true", help="retrain even without drift")
    parser.add_argument("--model", choices=MODEL_NAMES, default=GRADIENT_BOOSTING)
    parser.add_argument("--min-improvement", type=float, default=0.0)
    parser.add_argument("--data-dir", type=Path, default=settings.data_dir)
    parser.add_argument("--model-dir", type=Path, default=settings.model_dir)
    parser.add_argument("--seed", type=int, default=settings.random_seed)
    parser.add_argument("--tracking-uri", default=settings.mlflow_tracking_uri)
    parser.add_argument("--experiment", default=settings.mlflow_experiment)
    parser.add_argument("--database-url", default=settings.database_url)
    parser.add_argument("--json", type=Path, help="write the outcome to this JSON file")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=get_settings().log_level, format="%(levelname)s %(name)s: %(message)s"
    )
    processed = processed_dir(args.data_dir)
    if not processed.exists():
        prepare_dataset(args.data_dir, seed=args.seed)
    split = read_splits(processed)

    engine = make_engine(args.database_url)
    init_db(engine)
    with make_session_factory(engine)() as session:
        outcome = retrain_if_drifted(
            session,
            split,
            model_dir=args.model_dir,
            tracking_uri=args.tracking_uri,
            experiment_name=args.experiment,
            seed=args.seed,
            model_name=args.model,
            force=args.force,
            min_improvement=args.min_improvement,
        )
    engine.dispose()

    payload = outcome.to_dict()
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"triggered={outcome.triggered}: {outcome.reason}")
    if outcome.gate:
        print(outcome.gate.reason)
        print(
            f"promoted={outcome.promoted}"
            + (f" -> {outcome.promoted_dir}" if outcome.promoted else "")
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
