"""Measure model-quality and data-quality metrics and write the evidence.

Outputs (under ``reports/metrics``):

* ``quality_report.json`` - model quality (discrimination + calibration),
  data quality (schema, missing, range, duplicates) for a clean and a
  corrupted batch, and data drift (PSI / KS) for a stable and a shifted fleet.
* ``calibration.png``, ``precision_recall.png``, ``loss_curve.png``,
  ``drift_psi.png`` - figures used in the report.

Usage::

    python -m scripts.quality_report
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import precision_recall_curve

from brakeguard.config import PROJECT_ROOT, get_settings
from brakeguard.data import DataValidator, SyntheticTelemetrySource
from brakeguard.logging_config import configure_logging, get_logger
from brakeguard.models import ModelTrainer
from brakeguard.monitoring import DriftDetector
from brakeguard.schemas import RAW_FEATURES

logger = get_logger("brakeguard.quality_report")
OUT = PROJECT_ROOT / "reports" / "metrics"

# Reference palette (dataviz skill): categorical slots 1-2 + recessive ink.
BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def _style(ax, title: str, xlabel: str, ylabel: str) -> None:
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", color=INK, fontsize=12, fontweight="bold")
    ax.set_xlabel(xlabel, color=MUTED)
    ax.set_ylabel(ylabel, color=MUTED)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED)


def _save(fig, name: str) -> Path:
    path = OUT / name
    fig.patch.set_facecolor(SURFACE)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def corrupted_batch(seed: int = 3):
    """A realistic bad batch: sensor dropouts, a unit bug and duplicates."""
    frame = SyntheticTelemetrySource(500, seed=seed).load()
    rng = np.random.default_rng(seed)
    frame.loc[rng.choice(len(frame), 40, replace=False), "brake_temperature"] = np.nan
    frame.loc[rng.choice(len(frame), 25, replace=False), "pad_wear_pct"] *= 100  # fraction -> % bug
    return pd.concat([frame, frame.iloc[:15]], ignore_index=True)


def main() -> int:
    """Compute the metrics, write JSON and figures."""
    settings = get_settings()
    plt.switch_backend("Agg")
    configure_logging("WARNING")
    OUT.mkdir(parents=True, exist_ok=True)

    trainer = ModelTrainer(settings.hyper_params, settings.test_size)
    data = SyntheticTelemetrySource(settings.training_samples, seed=settings.random_seed).load()
    result = trainer.train(data)
    proba = result.pipeline.predict_proba(result.X_test)[:, 1]

    validator = DataValidator(max_missing_ratio=settings.max_missing_ratio)
    clean_quality = validator.validate(SyntheticTelemetrySource(500, seed=3).load())
    bad_quality = validator.validate(corrupted_batch())

    detector = DriftDetector(result.reference_frame, settings.psi_warning, settings.psi_alert)
    stable = detector.evaluate(SyntheticTelemetrySource(500, seed=8).load())
    shifted = detector.evaluate(SyntheticTelemetrySource(500, seed=8, shift=1.0).load())

    steps = (1, 2, 5, 10, 25, 50, 100, 160)
    X, y = trainer.split_xy(data)
    losses = trainer.loss_curve(X, y, steps=steps)

    report = {
        "model_version": settings.model_version,
        "model_quality": {"test": result.test_metrics.to_dict(), "train": result.train_metrics.to_dict()},
        "data_quality": {"clean_batch": clean_quality.to_dict(), "corrupted_batch": bad_quality.to_dict()},
        "drift": {"stable_fleet": stable.to_dict(), "shifted_fleet": shifted.to_dict()},
        "training_loss_curve": dict(zip(map(str, steps), [round(v, 4) for v in losses])),
    }
    (OUT / "quality_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    plot_calibration(result, proba)
    plot_precision_recall(result, proba)
    plot_loss_curve(steps, losses)
    plot_drift(stable, shifted, settings)

    print(json.dumps({"model_quality": report["model_quality"]["test"]}, indent=2))
    print("clean batch passed:", clean_quality.passed, "| corrupted batch issues:", bad_quality.issues)
    print("drifted features (shifted fleet):", shifted.drifted_features)
    return 0


def plot_calibration(result, proba) -> None:
    """Reliability diagram."""
    frac_pos, mean_pred = calibration_curve(result.y_test, proba, n_bins=10)
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.plot([0, 1], [0, 1], color=MUTED, linewidth=1, linestyle="--", label="Perfect calibration")
    ax.plot(mean_pred, frac_pos, color=BLUE, linewidth=2, marker="o", markersize=6, label="BrakeGuard RF")
    _style(
        ax,
        f"Reliability diagram (ECE {result.test_metrics.expected_calibration_error:.3f}, "
        f"Brier {result.test_metrics.brier_score:.3f})",
        "Mean predicted fault probability",
        "Observed fault rate",
    )
    ax.legend(frameon=False, labelcolor=INK)
    _save(fig, "calibration.png")


def plot_precision_recall(result, proba) -> None:
    """Precision-recall curve with the operating point."""
    precision, recall, _ = precision_recall_curve(result.y_test, proba)
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.plot(recall, precision, color=BLUE, linewidth=2)
    ax.axhline(result.y_test.mean(), color=MUTED, linewidth=1, linestyle="--")
    ax.text(0.02, result.y_test.mean() + 0.015, "Random baseline", color=MUTED, fontsize=9)
    m = result.test_metrics
    ax.scatter([m.recall], [m.precision], s=60, color=BLUE, edgecolor=SURFACE, linewidth=2, zorder=3)
    ax.annotate(
        f"threshold 0.5\nrecall {m.recall:.2f}, precision {m.precision:.2f}",
        (m.recall, m.precision),
        xytext=(-150, -45),
        textcoords="offset points",
        color=INK,
        fontsize=9,
    )
    _style(ax, f"Precision-recall curve (PR-AUC {m.pr_auc:.3f})", "Recall", "Precision")
    ax.set_ylim(0.4, 1.02)
    _save(fig, "precision_recall.png")


def plot_loss_curve(steps, losses) -> None:
    """Training log-loss as trees are added."""
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.plot(steps, losses, color=BLUE, linewidth=2, marker="o", markersize=6)
    ax.set_xscale("log")
    ax.set_xticks(steps)
    ax.set_xticklabels([str(s) for s in steps])
    _style(ax, "Training log-loss vs number of trees", "Trees in the forest (log scale)", "Training log-loss")
    _save(fig, "loss_curve.png")


def plot_drift(stable, shifted, settings) -> None:
    """PSI per feature for a stable and a shifted fleet."""
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    idx = np.arange(len(RAW_FEATURES))
    width = 0.38
    ax.bar(
        idx - width / 2 - 0.01, [stable.psi[f] for f in RAW_FEATURES], width, color=BLUE, label="Stable fleet"
    )
    ax.bar(
        idx + width / 2 + 0.01,
        [shifted.psi[f] for f in RAW_FEATURES],
        width,
        color=ORANGE,
        label="Shifted fleet",
    )
    for level, label in ((settings.psi_warning, "warning 0.10"), (settings.psi_alert, "alert 0.25")):
        ax.axhline(level, color=MUTED, linewidth=1, linestyle="--")
        ax.text(len(RAW_FEATURES) - 0.45, level + 0.01, label, color=MUTED, fontsize=9, ha="right")
    ax.set_xticks(idx)
    ax.set_xticklabels([f.replace("_", "\n") for f in RAW_FEATURES], fontsize=9)
    _style(ax, "Data drift: Population Stability Index per feature", "", "PSI")
    ax.legend(frameon=False, labelcolor=INK, loc="center right")
    _save(fig, "drift_psi.png")


if __name__ == "__main__":
    raise SystemExit(main())
