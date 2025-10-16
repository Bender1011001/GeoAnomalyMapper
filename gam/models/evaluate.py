"""Model evaluation utilities."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Sequence

import pandas as pd

from .train import compute_metrics
from ..utils.logging import get_logger

LOGGER = get_logger(__name__)


def evaluate_predictions(truth_path: Path, predictions_path: Path) -> None:
    truth = pd.read_csv(truth_path)
    preds = pd.read_csv(predictions_path)
    if "label" not in truth.columns:
        raise ValueError("Truth file must contain 'label' column")
    if "p" not in preds.columns:
        raise ValueError("Predictions file must contain 'p' column")
    merged = truth.merge(preds, on=["lon", "lat"], suffixes=("", "_pred"))
    metrics = compute_metrics(merged["label"].to_numpy(), merged["p"].to_numpy())
    LOGGER.info("Evaluation metrics: %s", metrics)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate predictions against ground truth")
    parser.add_argument("--truth", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    evaluate_predictions(args.truth, args.predictions)


if __name__ == "__main__":
    main()
