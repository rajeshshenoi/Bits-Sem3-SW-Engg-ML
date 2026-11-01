"""Data-quality validation (schema, completeness, validity, uniqueness).

The validator produces a :class:`DataQualityReport` with measurable data
quality metrics. The training pipeline refuses to train on data that fails
the report (a *quality gate*), and the same checks are exercised by the
``tests/data_quality`` test-suite.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import pandas as pd
from pandas.api.types import is_numeric_dtype

from ..exceptions import DataValidationError
from ..logging_config import get_logger
from ..schemas import TARGET_COLUMN, TELEMETRY_SCHEMA

logger = get_logger(__name__)


@dataclass
class DataQualityReport:
    """Measured data-quality metrics for one data set."""

    n_rows: int
    schema_valid: bool
    missing_columns: list[str] = field(default_factory=list)
    non_numeric_columns: list[str] = field(default_factory=list)
    missing_ratio: dict[str, float] = field(default_factory=dict)
    out_of_range_ratio: dict[str, float] = field(default_factory=dict)
    duplicate_ratio: float = 0.0
    label_values_valid: bool = True
    issues: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        """``True`` when no blocking issue was found."""
        return not self.issues

    @property
    def overall_missing_ratio(self) -> float:
        """Mean share of missing cells across all schema columns."""
        if not self.missing_ratio:
            return 0.0
        return sum(self.missing_ratio.values()) / len(self.missing_ratio)

    @property
    def overall_out_of_range_ratio(self) -> float:
        """Mean share of physically implausible values across columns."""
        if not self.out_of_range_ratio:
            return 0.0
        return sum(self.out_of_range_ratio.values()) / len(self.out_of_range_ratio)

    def to_dict(self) -> dict:
        """Serialisable representation (used by the API and the report)."""
        data = asdict(self)
        data["passed"] = self.passed
        data["overall_missing_ratio"] = round(self.overall_missing_ratio, 4)
        data["overall_out_of_range_ratio"] = round(self.overall_out_of_range_ratio, 4)
        return data


class DataValidator:
    """Validates telemetry frames against :data:`TELEMETRY_SCHEMA`."""

    def __init__(self, max_missing_ratio: float = 0.05, max_out_of_range_ratio: float = 0.01) -> None:
        self.max_missing_ratio = max_missing_ratio
        self.max_out_of_range_ratio = max_out_of_range_ratio

    def validate(self, frame: pd.DataFrame, require_target: bool = True) -> DataQualityReport:
        """Measure data quality and collect blocking issues."""
        report = DataQualityReport(n_rows=len(frame), schema_valid=True)
        if frame.empty:
            report.issues.append("data set is empty")

        self._check_schema(frame, report, require_target)
        present = [s for s in TELEMETRY_SCHEMA if s.name in frame.columns]
        for spec in present:
            column = frame[spec.name]
            if not is_numeric_dtype(column):
                report.non_numeric_columns.append(spec.name)
                report.issues.append(f"column '{spec.name}' is not numeric")
                continue
            self._check_completeness(spec.name, column, report)
            self._check_range(spec, column, report)

        if require_target and TARGET_COLUMN in frame.columns:
            labels = set(frame[TARGET_COLUMN].dropna().unique().tolist())
            report.label_values_valid = labels.issubset({0, 1})
            if not report.label_values_valid:
                report.issues.append(f"target has values outside {{0, 1}}: {sorted(labels)}")

        report.duplicate_ratio = float(frame.duplicated().mean()) if len(frame) else 0.0
        if report.duplicate_ratio > 0.0:
            logger.warning("%.2f%% duplicate rows detected", 100 * report.duplicate_ratio)

        if report.passed:
            logger.info("Data validation passed for %d rows", report.n_rows)
        else:
            logger.warning("Data validation found %d issue(s): %s", len(report.issues), report.issues)
        return report

    def validate_or_raise(self, frame: pd.DataFrame, require_target: bool = True) -> DataQualityReport:
        """Validate and raise :class:`DataValidationError` on failure."""
        report = self.validate(frame, require_target=require_target)
        if not report.passed:
            logger.error("Rejecting data set: %s", "; ".join(report.issues))
            raise DataValidationError("Data quality gate failed", report.issues)
        return report

    # -- individual checks ---------------------------------------------------
    @staticmethod
    def _check_schema(frame: pd.DataFrame, report: DataQualityReport, require_target: bool) -> None:
        expected = [s.name for s in TELEMETRY_SCHEMA]
        if require_target:
            expected.append(TARGET_COLUMN)
        report.missing_columns = [c for c in expected if c not in frame.columns]
        if report.missing_columns:
            report.schema_valid = False
            report.issues.append(f"missing columns: {report.missing_columns}")

    def _check_completeness(self, name: str, column: pd.Series, report: DataQualityReport) -> None:
        ratio = float(column.isna().mean()) if len(column) else 0.0
        report.missing_ratio[name] = round(ratio, 4)
        if ratio > self.max_missing_ratio:
            report.issues.append(
                f"column '{name}' missing ratio {ratio:.2%} exceeds {self.max_missing_ratio:.2%}"
            )
        elif ratio > 0:
            logger.warning("Column '%s' has %.2f%% missing values", name, 100 * ratio)

    def _check_range(self, spec, column: pd.Series, report: DataQualityReport) -> None:
        valid = column.dropna()
        if valid.empty:
            report.out_of_range_ratio[spec.name] = 0.0
            return
        bad = (valid < spec.min_value) | (valid > spec.max_value)
        ratio = float(bad.mean())
        report.out_of_range_ratio[spec.name] = round(ratio, 4)
        if ratio > self.max_out_of_range_ratio:
            report.issues.append(
                f"column '{spec.name}' has {ratio:.2%} values outside "
                f"[{spec.min_value}, {spec.max_value}] {spec.unit}"
            )
