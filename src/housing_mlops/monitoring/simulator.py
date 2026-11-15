import logging
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from housing_mlops.data.schema import FEATURE_BOUNDS, FEATURE_COLUMNS

log = logging.getLogger(__name__)

Payload = dict[str, float]
Sender = Callable[[Payload], int]


@dataclass(frozen=True)
class DriftScenario:
    income_multiplier: float = 2.0
    age_multiplier: float = 0.4
    occupancy_multiplier: float = 1.8
    latitude_shift: float = -2.0

    def apply(self, frame: pd.DataFrame, intensity: float) -> pd.DataFrame:
        intensity = float(np.clip(intensity, 0.0, 1.0))
        drifted = frame.copy()
        drifted["MedInc"] *= 1.0 + (self.income_multiplier - 1.0) * intensity
        drifted["HouseAge"] *= 1.0 + (self.age_multiplier - 1.0) * intensity
        drifted["AveOccup"] *= 1.0 + (self.occupancy_multiplier - 1.0) * intensity
        drifted["Latitude"] += self.latitude_shift * intensity
        return clip_to_bounds(drifted)


def clip_to_bounds(frame: pd.DataFrame) -> pd.DataFrame:
    clipped = frame.copy()
    for name in FEATURE_COLUMNS:
        lower, upper = FEATURE_BOUNDS[name]
        margin = (upper - lower) * 1e-6
        clipped[name] = clipped[name].clip(lower + margin, upper)
    clipped["AveBedrms"] = np.minimum(clipped["AveBedrms"], clipped["AveRooms"])
    return clipped


def to_payloads(frame: pd.DataFrame) -> list[Payload]:
    return [
        {name: float(value) for name, value in row.items()}
        for row in frame[FEATURE_COLUMNS].to_dict(orient="records")
    ]


@dataclass
class TrafficPlan:
    source: pd.DataFrame
    normal_requests: int
    drifted_requests: int
    seed: int = 0
    scenario: DriftScenario = field(default_factory=DriftScenario)

    def batches(self) -> Iterator[tuple[str, float, list[Payload]]]:
        rng = np.random.default_rng(self.seed)
        features = self.source[FEATURE_COLUMNS]
        if self.normal_requests < 0 or self.drifted_requests < 0:
            raise ValueError("request counts must not be negative")
        if self.normal_requests:
            normal = features.sample(
                n=self.normal_requests, replace=True, random_state=rng.integers(2**31)
            )
            yield "normal", 0.0, to_payloads(normal)
        if not self.drifted_requests:
            return

        drifted = features.sample(
            n=self.drifted_requests, replace=True, random_state=rng.integers(2**31)
        )
        intensities = np.linspace(0.0, 1.0, self.drifted_requests, endpoint=True)
        for (_, row), intensity in zip(drifted.iterrows(), intensities, strict=True):
            payload = to_payloads(self.scenario.apply(row.to_frame().T, intensity))
            yield "drifted", float(intensity), payload


@dataclass
class TrafficSummary:
    sent: int = 0
    succeeded: int = 0
    failed: int = 0
    by_phase: dict[str, int] = field(default_factory=dict)

    def record(self, phase: str, status_code: int) -> None:
        self.sent += 1
        self.by_phase[phase] = self.by_phase.get(phase, 0) + 1
        if 200 <= status_code < 300:
            self.succeeded += 1
        else:
            self.failed += 1


def run_traffic(plan: TrafficPlan, send: Sender, *, pause_seconds: float = 0.0) -> TrafficSummary:
    summary = TrafficSummary()
    for phase, intensity, payloads in plan.batches():
        for payload in payloads:
            status = send(payload)
            summary.record(phase, status)
            if status >= 400:
                log.warning("%s request (intensity %.2f) failed with %s", phase, intensity, status)
            if pause_seconds:
                time.sleep(pause_seconds)
    return summary
