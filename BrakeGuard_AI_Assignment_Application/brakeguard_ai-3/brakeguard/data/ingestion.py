"""Data-ingestion layer.

Every data source implements :class:`TelemetryDataSource` and returns a
:class:`pandas.DataFrame` with the columns defined in
:mod:`brakeguard.schemas`. Training code depends only on the abstract
interface, so a CSV export, a database query or the synthetic generator used
for the prototype are interchangeable (Dependency-Inversion principle).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import pandas as pd

from ..exceptions import DataIngestionError
from ..logging_config import get_logger
from ..schemas import ID_COLUMN, RAW_FEATURES, TARGET_COLUMN

logger = get_logger(__name__)


class TelemetryDataSource(ABC):
    """Abstract source of labelled brake telemetry."""

    @abstractmethod
    def load(self) -> pd.DataFrame:
        """Return a data frame of telemetry records."""


class SyntheticTelemetrySource(TelemetryDataSource):
    """Generates synthetic, labelled brake telemetry.

    The generative process is the same one used in the Assignment I
    prototype: a hidden linear risk score over the six signals, squashed by a
    sigmoid, and sampled as a Bernoulli label. ``shift`` lets tests and the
    drift monitor simulate a fleet whose operating conditions have changed.
    """

    def __init__(self, n_samples: int = 3000, seed: int = 42, shift: float = 0.0) -> None:
        if n_samples <= 0:
            raise DataIngestionError(f"n_samples must be positive, got {n_samples}")
        self.n_samples = n_samples
        self.seed = seed
        self.shift = shift

    def load(self) -> pd.DataFrame:
        rng = np.random.default_rng(self.seed)
        n = self.n_samples
        temp = rng.normal(155 + 40 * self.shift, 35, n).clip(70, 280)
        pressure = rng.normal(65, 15, n).clip(20, 110)
        wear = (rng.beta(2.0, 2.5, n) * 100 + 15 * self.shift).clip(0, 100)
        vibration = rng.gamma(2.0, 0.8, n).clip(0.1, 8)
        fluid_age = rng.gamma(2.2, 120, n).clip(5, 900)
        odo = rng.uniform(5000, 180000, n)
        score = (
            0.022 * (temp - 120)
            + 0.028 * np.abs(pressure - 65)
            + 0.065 * (wear - 35)
            + 0.55 * (vibration - 1.2)
            + 0.0022 * (fluid_age - 180)
            + 0.000008 * (odo - 60000)
        )
        prob = 1 / (1 + np.exp(-(score - 2.2)))
        label = (rng.random(n) < prob).astype(int)
        frame = pd.DataFrame(
            np.column_stack([temp, pressure, wear, vibration, fluid_age, odo]),
            columns=list(RAW_FEATURES),
        )
        frame.insert(0, ID_COLUMN, [f"SYN-{i:05d}" for i in range(n)])
        frame[TARGET_COLUMN] = label
        logger.info(
            "Generated %d synthetic telemetry rows (seed=%d, shift=%.2f, positive rate=%.3f)",
            n,
            self.seed,
            self.shift,
            label.mean(),
        )
        return frame


class CSVTelemetrySource(TelemetryDataSource):
    """Loads telemetry from a CSV export of the fleet data lake."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def load(self) -> pd.DataFrame:
        if not self.path.exists():
            logger.error("Telemetry file not found: %s", self.path)
            raise DataIngestionError(f"Telemetry file not found: {self.path}")
        try:
            frame = pd.read_csv(self.path)
        except (pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeDecodeError) as exc:
            logger.error("Could not parse telemetry file %s: %s", self.path, exc)
            raise DataIngestionError(f"Could not parse {self.path}: {exc}") from exc
        if frame.empty:
            logger.warning("Telemetry file %s contains no rows", self.path)
        logger.info("Loaded %d rows from %s", len(frame), self.path)
        return frame
