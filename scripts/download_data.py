import argparse
import logging
from pathlib import Path

from housing_mlops.config import get_settings
from housing_mlops.data import prepare_dataset


def parse_args() -> argparse.Namespace:
    settings = get_settings()
    parser = argparse.ArgumentParser(
        description="Download, validate and split the California Housing dataset."
    )
    parser.add_argument("--data-dir", type=Path, default=settings.data_dir)
    parser.add_argument("--seed", type=int, default=settings.random_seed)
    parser.add_argument("--force", action="store_true", help="re-download even if cached")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=get_settings().log_level, format="%(levelname)s %(name)s: %(message)s"
    )
    split = prepare_dataset(args.data_dir, seed=args.seed, force=args.force)
    print(f"Prepared dataset in {args.data_dir}: {split.sizes()}")


if __name__ == "__main__":
    main()
