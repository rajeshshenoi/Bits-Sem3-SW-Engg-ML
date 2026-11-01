"""End-to-end training pipeline: ingest -> validate -> train -> gate -> register."""

from __future__ import annotations

from .config import Settings, get_settings
from .data import DataValidator, SyntheticTelemetrySource, TelemetryDataSource
from .exceptions import ModelRegistryError, ModelTrainingError
from .features import ENGINEERED_FEATURES
from .logging_config import get_logger
from .models import ModelBundle, ModelRegistry, ModelTrainer

logger = get_logger(__name__)


class TrainingPipeline:
    """Orchestrates the training workflow with explicit quality gates."""

    def __init__(
        self,
        settings: Settings | None = None,
        source: TelemetryDataSource | None = None,
        validator: DataValidator | None = None,
        trainer: ModelTrainer | None = None,
        registry: ModelRegistry | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.source = source or SyntheticTelemetrySource(
            n_samples=self.settings.training_samples, seed=self.settings.random_seed
        )
        self.validator = validator or DataValidator(max_missing_ratio=self.settings.max_missing_ratio)
        self.trainer = trainer or ModelTrainer(self.settings.hyper_params, self.settings.test_size)
        self.registry = registry or ModelRegistry(self.settings.model_path)

    def run(self, register: bool = True) -> ModelBundle:
        """Execute the pipeline and (optionally) register the model."""
        logger.info("Training pipeline started (model version %s)", self.settings.model_version)
        frame = self.source.load()
        quality = self.validator.validate_or_raise(frame, require_target=True)
        result = self.trainer.train(frame)
        self._check_quality_gates(result.test_metrics.to_dict())
        bundle = ModelBundle(
            pipeline=result.pipeline,
            version=self.settings.model_version,
            metrics=result.test_metrics.to_dict(),
            feature_names=list(ENGINEERED_FEATURES),
            reference_frame=result.reference_frame.sample(
                n=min(1000, len(result.reference_frame)), random_state=self.settings.random_seed
            ).reset_index(drop=True),
            data_quality=quality.to_dict(),
        )
        if register:
            self.registry.save(bundle)
        logger.info("Training pipeline finished")
        return bundle

    def _check_quality_gates(self, metrics: dict) -> None:
        failures = []
        if metrics["recall"] < self.settings.min_recall:
            failures.append(f"recall {metrics['recall']:.3f} < {self.settings.min_recall}")
        if metrics["pr_auc"] < self.settings.min_pr_auc:
            failures.append(f"pr_auc {metrics['pr_auc']:.3f} < {self.settings.min_pr_auc}")
        if failures:
            logger.error("Model rejected by quality gate: %s", failures)
            raise ModelTrainingError("Model failed quality gate: " + "; ".join(failures))
        logger.info("Model passed quality gates")


def load_or_train(settings: Settings | None = None) -> ModelBundle:
    """Load the registered model; fall back to training if it is missing/corrupt."""
    settings = settings or get_settings()
    registry = ModelRegistry(settings.model_path)
    try:
        return registry.load()
    except ModelRegistryError as exc:
        logger.warning("Model unavailable (%s) - training a new one", exc)
        return TrainingPipeline(settings=settings, registry=registry).run(register=True)
