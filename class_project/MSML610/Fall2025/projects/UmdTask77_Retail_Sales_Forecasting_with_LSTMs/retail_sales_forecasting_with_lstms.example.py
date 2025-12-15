"""
Executable entry point illustrating how the forecasting API is applied.

The script mirrors the steps in `retail_sales_forecasting_with_lstms.example.ipynb`
so that automated tests (to be added) can validate the notebooks' logic without
requiring a GUI runtime. It now exposes CLI flags so SLURM jobs (e.g., on UMD
Nexus) can launch configurable LSTM or GRU runs.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Sequence

from retail_sales_forecasting_utils import (
    DataSourceConfig,
    ModelConfig,
    TemporalFeatureConfig,
    build_feature_pipeline,
    ensure_data_root,
    evaluate_model,
    prepare_dataloader,
    train_model,
)

_LOG = logging.getLogger(__name__)


def _build_data_config(args: argparse.Namespace) -> DataSourceConfig:
    """Create a configuration pointing at the Kaggle Store Sales dataset layout."""
    return DataSourceConfig(
        root_dir=args.data_root,
        sales_file=args.sales_file,
        calendar_file=args.calendar_file,
        oil_file=args.oil_file,
        transactions_file=args.transactions_file,
        id_columns=("store_nbr", "family"),
        date_column="date",
        target_column="sales",
        frequency="D",
        horizon_days=args.horizon_days,
        allow_synthetic=args.allow_synthetic,
    )


def _build_feature_config(args: argparse.Namespace) -> TemporalFeatureConfig:
    """Generate a temporal feature configuration using CLI overrides."""
    return TemporalFeatureConfig(
        include_holidays=not args.skip_holidays,
        include_promotions=not args.skip_promotions,
        include_external_regressors=args.include_external_regressors,
        lookback_days=args.lookback_days,
        train_ratio=args.train_ratio,
    )


def _build_model_config(args: argparse.Namespace) -> ModelConfig:
    """Assemble the recurrent model configuration from CLI overrides."""
    return ModelConfig(
        cell_type=args.cell_type,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout_rate=args.dropout_rate,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        gradient_clip=args.gradient_clip,
        batch_size=args.batch_size,
        epochs=args.epochs,
        seed=args.seed,
    )


def run_training(args: argparse.Namespace) -> None:
    """Orchestrate the end-to-end workflow using configuration defaults."""
    data_cfg = _build_data_config(args)
    feature_cfg = _build_feature_config(args)
    model_cfg = _build_model_config(args)
    metrics: Sequence[str] = tuple(args.metrics)

    _LOG.info("Starting %s run with data root %s", model_cfg.cell_type.upper(), args.data_root)
    ensure_data_root(data_cfg)
    feature_generators = build_feature_pipeline(data_cfg, feature_cfg)
    train_dataset, val_dataset, metadata = prepare_dataloader(
        data_cfg, feature_generators, feature_cfg=feature_cfg
    )
    trained_state = train_model(train_dataset, val_dataset, model_cfg, metadata)
    artifacts = evaluate_model(trained_state, val_dataset, metadata, metrics)

    _LOG.info("Metrics: %s", artifacts.metrics)
    _LOG.info("Predictions preview:\n%s", artifacts.predictions.head())
    _LOG.debug("Model params summary: %s", trained_state["state"].params)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train the Retail Sales Forecasting model (LSTM/GRU) with configurable settings.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("/data/store_sales"),
        help="Directory containing the Kaggle Store Sales dataset files.",
    )
    parser.add_argument("--sales-file", default="train.parquet")
    parser.add_argument("--calendar-file", default="holidays_events.csv")
    parser.add_argument("--oil-file", default="oil.csv")
    parser.add_argument("--transactions-file", default="transactions.csv")
    parser.add_argument(
        "--allow-synthetic",
        action="store_true",
        help="Enable synthetic data fallback if Kaggle assets are missing.",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=("mae", "rmse", "mape"),
        help="Evaluation metrics to compute after training.",
    )

    # Feature config overrides.
    parser.add_argument("--lookback-days", type=int, default=120)
    parser.add_argument("--horizon-days", type=int, default=28)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument(
        "--include-external-regressors",
        action="store_true",
        help="Include regressors such as oil price and transactions.",
    )
    parser.add_argument("--skip-holidays", action="store_true")
    parser.add_argument("--skip-promotions", action="store_true")

    # Model config overrides.
    parser.add_argument("--cell-type", choices=("lstm", "gru"), default="lstm")
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout-rate", type=float, default=0.0)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    args.data_root = args.data_root.expanduser()

    run_training(args)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
