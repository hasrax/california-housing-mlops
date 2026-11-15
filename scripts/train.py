import argparse
import logging
from pathlib import Path

from housing_mlops.config import get_settings
from housing_mlops.data import prepare_dataset, processed_dir, read_splits
from housing_mlops.training import MODEL_NAMES, PRIMARY_METRIC, set_current, train_model


def parse_args() -> argparse.Namespace:
    settings = get_settings()
    parser = argparse.ArgumentParser(description="Train a model and track it with MLflow.")
    parser.add_argument("--model", choices=[*MODEL_NAMES, "all"], default="all")
    parser.add_argument("--data-dir", type=Path, default=settings.data_dir)
    parser.add_argument("--model-dir", type=Path, default=settings.model_dir)
    parser.add_argument("--seed", type=int, default=settings.random_seed)
    parser.add_argument("--tracking-uri", default=settings.mlflow_tracking_uri)
    parser.add_argument("--experiment", default=settings.mlflow_experiment)
    parser.add_argument(
        "--set-current",
        action="store_true",
        help="copy the best model (by validation RMSE) to models/current",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=get_settings().log_level, format="%(levelname)s %(name)s: %(message)s"
    )
    processed = processed_dir(args.data_dir)
    if not processed.exists():
        prepare_dataset(args.data_dir, seed=args.seed)
    split = read_splits(processed)

    names = list(MODEL_NAMES) if args.model == "all" else [args.model]
    results = [
        train_model(
            split,
            model_name=name,
            seed=args.seed,
            model_dir=args.model_dir,
            tracking_uri=args.tracking_uri,
            experiment_name=args.experiment,
        )
        for name in names
    ]

    for result in results:
        m = result.metadata
        print(f"{m.model_name:18s} {m.version}")
        for split_name, values in m.metrics.items():
            line = "  ".join(f"{k}={v:.4f}" for k, v in values.items())
            print(f"  {split_name:10s} {line}")

    if args.set_current:
        best = min(results, key=lambda r: r.metadata.metrics["validation"][PRIMARY_METRIC])
        target = set_current(best.artifact_dir, args.model_dir)
        print(f"current -> {best.metadata.model_name} {best.metadata.version} at {target}")


if __name__ == "__main__":
    main()
