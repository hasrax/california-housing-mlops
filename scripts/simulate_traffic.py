import argparse
import logging
from pathlib import Path

import httpx

from housing_mlops.config import get_settings
from housing_mlops.data import processed_dir, read_splits
from housing_mlops.monitoring import TrafficPlan, run_traffic


def parse_args() -> argparse.Namespace:
    settings = get_settings()
    parser = argparse.ArgumentParser(
        description="Send normal and then gradually drifted requests to the API."
    )
    parser.add_argument("--api-url", default="http://localhost:8000")
    parser.add_argument("--data-dir", type=Path, default=settings.data_dir)
    parser.add_argument("--normal", type=int, default=300)
    parser.add_argument("--drifted", type=int, default=300)
    parser.add_argument("--seed", type=int, default=settings.random_seed)
    parser.add_argument("--pause", type=float, default=0.0, help="seconds between requests")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=get_settings().log_level, format="%(levelname)s %(name)s: %(message)s"
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    split = read_splits(processed_dir(args.data_dir))
    plan = TrafficPlan(
        source=split.test,
        normal_requests=args.normal,
        drifted_requests=args.drifted,
        seed=args.seed,
    )
    with httpx.Client(base_url=args.api_url, timeout=10.0) as client:

        def send(payload: dict[str, float]) -> int:
            return client.post("/predict", json=payload).status_code

        summary = run_traffic(plan, send, pause_seconds=args.pause)
    print(
        f"sent={summary.sent} succeeded={summary.succeeded} failed={summary.failed} "
        f"by_phase={summary.by_phase}"
    )


if __name__ == "__main__":
    main()
