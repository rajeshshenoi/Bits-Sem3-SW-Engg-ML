"""Command-line entry point for the training pipeline.

Usage::

    python -m scripts.train                       # synthetic data
    python -m scripts.train --csv fleet.csv       # CSV export
    python -m scripts.train --metrics-out reports/metrics/metrics.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from brakeguard.config import get_settings
from brakeguard.data import CSVTelemetrySource
from brakeguard.exceptions import BrakeGuardError
from brakeguard.logging_config import configure_logging, get_logger
from brakeguard.pipeline import TrainingPipeline

logger = get_logger("brakeguard.cli")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Train and register the BrakeGuard model")
    parser.add_argument("--csv", type=Path, help="train from a CSV file instead of synthetic data")
    parser.add_argument("--metrics-out", type=Path, help="write metrics + data quality as JSON")
    parser.add_argument("--no-register", action="store_true", help="do not save the model")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run the pipeline; return a process exit code."""
    args = parse_args(argv)
    settings = get_settings()
    configure_logging(settings.log_level)
    source = CSVTelemetrySource(args.csv) if args.csv else None
    try:
        bundle = TrainingPipeline(settings=settings, source=source).run(register=not args.no_register)
    except BrakeGuardError as exc:
        logger.error("Training failed: %s", exc)
        return 1
    if args.metrics_out:
        args.metrics_out.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "model_version": bundle.version,
            "model_quality": bundle.metrics,
            "data_quality": bundle.data_quality,
        }
        args.metrics_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        logger.info("Metrics written to %s", args.metrics_out)
    print(json.dumps(bundle.metrics, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
