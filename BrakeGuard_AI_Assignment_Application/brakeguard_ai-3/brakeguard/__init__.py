"""BrakeGuard AI - predictive maintenance for automotive brake systems.

Production-style package for AIML ZG546 Assignment II. The package is split
into small, single-responsibility modules:

* ``brakeguard.data``        - data ingestion and data-quality validation
* ``brakeguard.features``    - feature engineering and the online feature store
* ``brakeguard.models``      - training, model registry (persistence), inference
* ``brakeguard.evaluation``  - model-quality metrics (discrimination + calibration)
* ``brakeguard.monitoring``  - audit log, event bus, drift monitoring
* ``brakeguard.api``         - FastAPI REST layer
"""

__version__ = "2.0.0"
