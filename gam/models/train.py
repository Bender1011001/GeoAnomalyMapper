"""Model training pipeline for GeoAnomalyMapper."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import joblib
import lightgbm as lgb
import mlflow
import numpy as np
import pandas as pd
import shap
from matplotlib import pyplot as plt
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ..features.schema import FeatureSchema
from ..utils.hashing import sha256_path
from ..utils.logging import get_logger

LOGGER = get_logger(__name__)


@dataclass
class FoldMetrics:
    auprc: float
    auroc: float
    recall_at_1fpr: float
    ece: float


def recall_at_fpr(y_true: np.ndarray, y_score: np.ndarray, target_fpr: float = 0.01) -> float:
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    mask = fpr <= target_fpr
    if not np.any(mask):
        return 0.0
    return float(np.max(tpr[mask]))


def expected_calibration_error(y_true: np.ndarray, y_score: np.ndarray, n_bins: int = 15) -> float:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.digitize(y_score, bins) - 1
    ece = 0.0
    total = len(y_true)
    for i in range(n_bins):
        mask = bin_ids == i
        if not np.any(mask):
            continue
        acc = np.mean(y_true[mask])
        conf = np.mean(y_score[mask])
        ece += np.abs(acc - conf) * np.sum(mask) / total
    return float(ece)


def compute_metrics(y_true: np.ndarray, y_score: np.ndarray) -> FoldMetrics:
    auprc = average_precision_score(y_true, y_score)
    try:
        auroc = roc_auc_score(y_true, y_score)
    except ValueError:
        auroc = float("nan")
    return FoldMetrics(
        auprc=float(auprc),
        auroc=float(auroc),
        recall_at_1fpr=recall_at_fpr(y_true, y_score),
        ece=expected_calibration_error(y_true, y_score),
    )


def summarize_metrics(metrics: Sequence[FoldMetrics]) -> Dict[str, float]:
    return {
        "auprc": float(np.mean([m.auprc for m in metrics])),
        "auroc": float(np.nanmean([m.auroc for m in metrics])),
        "recall_at_1fpr": float(np.mean([m.recall_at_1fpr for m in metrics])),
        "ece": float(np.mean([m.ece for m in metrics])),
    }


def load_training_data(path: Path, feature_schema: FeatureSchema) -> Tuple[pd.DataFrame, np.ndarray, pd.Series]:
    df = pd.read_csv(path)
    features = df[feature_schema.bands].astype(float)
    labels = df["label"].astype(int)
    groups = None
    if {"zone", "tile"}.issubset(df.columns):
        groups = df["zone"].astype(str) + "_" + df["tile"].astype(str)
    elif "zone" in df.columns:
        groups = df["zone"].astype(str)
    else:
        groups = pd.Series(np.arange(len(df)) // 10, name="group")
    return features, labels.to_numpy(), groups


def build_logistic_pipeline() -> Pipeline:
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    penalty="l2",
                    C=1.0,
                    class_weight="balanced",
                    solver="lbfgs",
                    max_iter=200,
                ),
            ),
        ]
    )


def build_lgbm_classifier() -> lgb.LGBMClassifier:
    return lgb.LGBMClassifier(
        objective="binary",
        n_estimators=300,
        learning_rate=0.05,
        max_depth=-1,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        class_weight="balanced",
        n_jobs=-1,
    )


def git_sha() -> str:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
            .decode()
            .strip()
        )
    except Exception:
        return "unknown"


def dvc_hash() -> str:
    try:
        status = subprocess.check_output(["dvc", "status", "-q", "--json"], stderr=subprocess.DEVNULL).decode()
        return hashlib.sha256(status.encode()).hexdigest()
    except Exception:
        return "unavailable"


def log_to_mlflow(
    schema: FeatureSchema,
    schema_path: Path,
    dataset_path: Path,
    model_path: Path,
    shap_path: Path,
    params: Dict[str, object],
    metrics: Dict[str, float],
) -> None:
    with mlflow.start_run():
        mlflow.log_param("git_sha", git_sha())
        mlflow.log_param("dvc_hash", dvc_hash())
        mlflow.log_param("feature_schema_hash", schema.hash)
        mlflow.log_param("dataset_hash", sha256_path(dataset_path))
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        mlflow.log_artifact(str(model_path))
        if schema_path.exists():
            mlflow.log_artifact(str(schema_path))
        mlflow.log_artifact(str(shap_path))


def compute_shap_summary(model, X: np.ndarray, feature_names: Sequence[str], out_path: Path) -> None:
    if X.shape[0] > 5000:
        rng = np.random.default_rng(42)
        idx = rng.choice(X.shape[0], size=5000, replace=False)
        X = X[idx]
    if isinstance(model, CalibratedClassifierCV):
        estimator = model.calibrated_classifiers_[0].estimator
    else:
        estimator = model
    if isinstance(estimator, Pipeline):
        steps = estimator.named_steps
        scaler = steps.get("scaler")
        clf = steps.get("clf")
        transformed = scaler.transform(X) if scaler is not None else X
        base_estimator = clf
    else:
        transformed = X
        base_estimator = estimator
    if hasattr(base_estimator, "booster_"):
        explainer = shap.TreeExplainer(base_estimator.booster_)
        shap_values = explainer.shap_values(X)
        values = shap_values[1] if isinstance(shap_values, list) else shap_values
    elif isinstance(base_estimator, LogisticRegression):
        explainer = shap.LinearExplainer(base_estimator, transformed)
        values = explainer.shap_values(transformed)
    else:
        explainer = shap.KernelExplainer(base_estimator.predict_proba, transformed[:200])
        values = explainer.shap_values(transformed, nsamples=200)[1]
    plt.figure(figsize=(10, 6))
    shap.summary_plot(values, transformed, feature_names=feature_names, show=False)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close()


def train_models(
    dataset_path: Path,
    schema_path: Path,
    output_dir: Path,
    cv_folds: int = 5,
) -> None:
    schema = FeatureSchema.from_file(schema_path)
    X_df, y, groups = load_training_data(dataset_path, schema)
    X = X_df.to_numpy()

    if isinstance(groups, pd.Series):
        groups_values = groups.to_numpy()
        n_groups = len(np.unique(groups_values))
    else:
        groups_values = groups
        n_groups = len(np.unique(groups_values))

    folds = min(cv_folds, n_groups)
    if folds < 2:
        splitter = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        splits = splitter.split(X, y)
    else:
        splitter = GroupKFold(n_splits=folds)
        splits = splitter.split(X, y, groups_values)

    lr_metrics: List[FoldMetrics] = []
    gbm_metrics: List[FoldMetrics] = []

    for train_idx, val_idx in splits:
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        lr_pipeline = build_logistic_pipeline()
        lr_pipeline.fit(X_train, y_train)
        lr_cal = CalibratedClassifierCV(lr_pipeline, method="sigmoid", cv="prefit")
        lr_cal.fit(X_val, y_val)
        lr_scores = lr_cal.predict_proba(X_val)[:, 1]
        lr_metrics.append(compute_metrics(y_val, lr_scores))

        gbm = build_lgbm_classifier()
        gbm.fit(X_train, y_train)
        gbm_cal = CalibratedClassifierCV(gbm, method="isotonic", cv="prefit")
        gbm_cal.fit(X_val, y_val)
        gbm_scores = gbm_cal.predict_proba(X_val)[:, 1]
        gbm_metrics.append(compute_metrics(y_val, gbm_scores))

    lr_summary = summarize_metrics(lr_metrics)
    gbm_summary = summarize_metrics(gbm_metrics)

    LOGGER.info("LR metrics: %s", lr_summary)
    LOGGER.info("GBM metrics: %s", gbm_summary)

    if gbm_summary["auprc"] >= lr_summary["auprc"] + 0.05:
        selected_model_name = "lightgbm"
        base_estimator = build_lgbm_classifier()
        calibration_method = "isotonic"
        summary = gbm_summary
    else:
        selected_model_name = "logistic_regression"
        base_estimator = build_logistic_pipeline()
        calibration_method = "sigmoid"
        summary = lr_summary

    LOGGER.info("Selected model: %s", selected_model_name)

    calibrator = CalibratedClassifierCV(base_estimator, method=calibration_method, cv=min(5, cv_folds))
    calibrator.fit(X, y)

    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "selected_model.pkl"
    joblib.dump(calibrator, model_path)

    shap_path = output_dir / "shap_summary.png"
    compute_shap_summary(calibrator, X, schema.bands, shap_path)

    probabilities = calibrator.predict_proba(X)[:, 1]
    fpr, tpr, thresholds = roc_curve(y, probabilities)
    mask = fpr <= 0.01
    if np.any(mask):
        operating_threshold = float(thresholds[mask][-1])
    else:
        operating_threshold = 0.5

    params = {
        "selected_model": selected_model_name,
        "schema_path": str(schema_path),
        "dataset_path": str(dataset_path),
        "operating_threshold": operating_threshold,
    }
    params.update({f"lr_{k}": v for k, v in lr_summary.items()})
    params.update({f"gbm_{k}": v for k, v in gbm_summary.items()})

    log_to_mlflow(schema, schema_path, dataset_path, model_path, shap_path, params, summary)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train LR and GBM models")
    sub = parser.add_subparsers(dest="command", required=True)
    run_parser = sub.add_parser("run", help="Execute training")
    run_parser.add_argument("--config", type=Path, required=False)
    run_parser.add_argument("--dataset", type=Path, default=Path("data/labels/training_points.csv"))
    run_parser.add_argument("--schema", type=Path, default=Path("data/feature_schema.json"))
    run_parser.add_argument("--output", type=Path, default=Path("artifacts"))
    run_parser.add_argument("--folds", type=int, default=5)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    if args.command == "run":
        train_models(args.dataset, args.schema, args.output, args.folds)


if __name__ == "__main__":
    main()
