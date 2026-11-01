"""Logging set-up shared by the CLI, the API and the notebooks.

Log levels are used with a consistent meaning across the code base:

* ``DEBUG``   - developer detail (feature vectors, intermediate values)
* ``INFO``    - normal lifecycle events (model loaded, prediction served)
* ``WARNING`` - degraded but recoverable situations (drift, missing values
  imputed, model retrained because the artefact was missing)
* ``ERROR``   - an operation failed and the caller receives an error
"""

from __future__ import annotations

import logging
import sys

LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
_CONFIGURED = False


def configure_logging(level: str = "INFO") -> None:
    """Configure the root ``brakeguard`` logger once (idempotent)."""
    global _CONFIGURED  # pylint: disable=global-statement
    logger = logging.getLogger("brakeguard")
    logger.setLevel(level.upper())
    if not _CONFIGURED:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(LOG_FORMAT))
        logger.addHandler(handler)
        logger.propagate = True  # lets pytest's caplog capture records
        _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Return a child logger of the ``brakeguard`` namespace."""
    if not name.startswith("brakeguard"):
        name = f"brakeguard.{name}"
    return logging.getLogger(name)
