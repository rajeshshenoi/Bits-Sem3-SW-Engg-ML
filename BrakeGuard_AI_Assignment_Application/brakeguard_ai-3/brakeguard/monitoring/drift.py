"""Data-drift detection: Population Stability Index (PSI) + KS test.

PSI rule of thumb: < 0.10 stable, 0.10-0.25 moderate shift (warning),
> 0.25 significant shift (alert - investigate / retrain).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from ..logging_config import get_logger
from ..schemas import RAW_FEATURES

logger = get_logger(__name__)


def population_stability_index(reference, current, n_bins: int = 10) -> float:
    """PSI between two 1-D samples using reference quantile bins."""
    reference = np.asarray(reference, dtype=float)
    current = np.asarray(current, dtype=float)
    reference = reference[~np.isnan(reference)]
    current = current[~np.isnan(current)]
    if reference.size == 0 or current.size == 0:
        raise ValueError("PSI needs non-empty reference and current samples")
    edges = np.unique(np.quantile(reference, np.linspace(0, 1, n_bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    ref_pct = np.histogram(reference, edges)[0] / reference.size
    cur_pct = np.histogram(current, edges)[0] / current.size
    eps = 1e-6
    ref_pct = np.clip(ref_pct, eps, None)
    cur_pct = np.clip(cur_pct, eps, None)
    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


@dataclass
class DriftReport:
    """Per-feature drift statistics."""

    n_reference: int
    n_current: int
    psi: dict[str, float] = field(default_factory=dict)
    ks_pvalue: dict[str, float] = field(default_factory=dict)
    status: dict[str, str] = field(default_factory=dict)

    @property
    def max_psi(self) -> float:
        """Largest PSI over all features."""
        return max(self.psi.values()) if self.psi else 0.0

    @property
    def drifted_features(self) -> list[str]:
        """Features whose status is ``alert``."""
        return [k for k, v in self.status.items() if v == "alert"]

    def to_dict(self) -> dict:
        """Serialisable representation."""
        data = asdict(self)
        data["max_psi"] = round(self.max_psi, 4)
        data["drifted_features"] = self.drifted_features
        return data


class DriftDetector:
    """Compares live telemetry with the training reference distribution."""

    def __init__(self, reference: pd.DataFrame, psi_warning: float = 0.10, psi_alert: float = 0.25) -> None:
        self.reference = reference[list(RAW_FEATURES)].reset_index(drop=True)
        self.psi_warning = psi_warning
        self.psi_alert = psi_alert

    def evaluate(self, current: pd.DataFrame) -> DriftReport:
        """Compute PSI and KS p-values for every raw feature."""
        report = DriftReport(n_reference=len(self.reference), n_current=len(current))
        for name in RAW_FEATURES:
            psi = population_stability_index(self.reference[name], current[name])
            pvalue = float(ks_2samp(self.reference[name], current[name].dropna()).pvalue)
            report.psi[name] = round(psi, 4)
            report.ks_pvalue[name] = round(pvalue, 6)
            if psi >= self.psi_alert:
                report.status[name] = "alert"
                logger.warning("Significant drift on %s (PSI=%.3f)", name, psi)
            elif psi >= self.psi_warning:
                report.status[name] = "warning"
                logger.warning("Moderate drift on %s (PSI=%.3f)", name, psi)
            else:
                report.status[name] = "stable"
        logger.info("Drift check: max PSI=%.3f, drifted=%s", report.max_psi, report.drifted_features)
        return report
