<!-- toc -->

- [Retail Sales Forecasting API — Learn in 60 Minutes](#retail-sales-forecasting-api--learn-in-60-minutes)
  * [Audience & Prerequisites](#audience--prerequisites)
  * [60-Minute Roadmap](#60-minute-roadmap)
  * [Core Concepts](#core-concepts)
  * [API Surface](#api-surface)
  * [Hands-On Workflow](#hands-on-workflow)
  * [Resilience, Dependencies, and Tests](#resilience-dependencies-and-tests)
  * [Open Items](#open-items)

<!-- tocstop -->

# Retail Sales Forecasting API — Learn in 60 Minutes

Structure follows the guidance in `tutorials/docs/all.learn_X_in_60_minutes.how_to_guide.md`
and the class instructions. The API describes the contracts implemented in
`retail_sales_forecasting_utils.py` and consumed by the notebooks.

## Audience & Prerequisites

- Practitioners comfortable with pandas/JAX who need a repeatable demand
  forecasting workflow.
- Basic familiarity with Kaggle Store Sales data schema.
- Local setup from the README (`docker_simple`) or equivalent Python 3.10 env.

## 60-Minute Roadmap

| Minute | Topic                                 | Talking Points |
|--------|---------------------------------------|----------------|
| 0–5    | Why sequence models for demand        | Business problem, dataset scope |
| 5–15   | Config dataclasses & feature toggles  | `DataSourceConfig`, `TemporalFeatureConfig` |
| 15–30  | Model/config objects                  | `ModelConfig`, cell types, optimizer choices |
| 30–40  | Data pipeline helpers                 | `build_feature_pipeline`, `prepare_dataloader` |
| 40–50  | Training/evaluation contracts         | `train_model`, `evaluate_model`, `ForecastArtifacts` |
| 50–60  | Testing + next steps                  | Synthetic fixtures, Kaggle integration plan |

## Core Concepts

- Provide a typed JAX pipeline for multi-store retail sales: data loading,
  feature creation, sliding windows, RNN training, and evaluation.
- Support both LSTM and GRU recurrences using a shared configuration contract.
- Keep notebooks declarative; all reusable logic lives in the utils module so
  the API tutorial can focus on intent, not cell-by-cell code.

## API Surface

| Component | Responsibility | Status |
|-----------|----------------|--------|
| `DataSourceConfig` | Dataset locations, schema metadata | ✅ Implemented |
| `TemporalFeatureConfig` | Seasonalities + feature toggles | ✅ Implemented |
| `ModelConfig` | RNN hyperparameters and optimizer knobs | ✅ Implemented |
| `FeatureGenerator` | Protocol for feature callables | ✅ Implemented |
| `WindowedDataset` | Sliding-window container | ✅ Implemented |
| `ForecastArtifacts` | Metrics, predictions, params snapshot | ✅ Implemented |
| `load_sales_data()` | Loads Kaggle files or synthetic fallback | ✅ Implemented |
| `build_feature_pipeline()` | Ordered callables to add temporal/event features | ✅ Implemented |
| `prepare_dataloader()` | Generates scaled sliding windows for train/val splits | ✅ Implemented |
| `create_rnn_model()` | Builds a Flax LSTM/GRU backbone with dense head | ✅ Implemented |
| `train_model()` | Optax-powered training loop with mini-batching | ✅ Implemented |
| `evaluate_model()` | Computes MAE/RMSE/MAPE and tidy prediction frame | ✅ Implemented |

## Hands-On Workflow

1. **Configure Data Sources**

   ```python
   from retail_sales_forecasting_with_lstms.API import DataSourceConfig

   data_cfg = DataSourceConfig(
       root_dir="/data/store_sales",
       sales_file="train.parquet",
       calendar_file="holidays_events.csv",
       oil_file="oil.csv",
       transactions_file="transactions.csv",
       id_columns=("store_nbr", "family"),
       date_column="date",
       target_column="sales",
       frequency="D",
       horizon_days=28,
   )
   ```

2. **Specify Temporal Features and Model Hyperparameters**

   ```python
   feature_cfg = TemporalFeatureConfig(
       include_holidays=True,
       include_promotions=True,
       include_external_regressors=True,
       seasonalities=(7, 28, 365),
   )

   model_cfg = ModelConfig(
       cell_type="lstm",
       hidden_size=128,
       num_layers=2,
       dropout_rate=0.1,
       learning_rate=3e-4,
       weight_decay=1e-4,
       gradient_clip=1.0,
       epochs=5,
       batch_size=64,
   )
   ```

3. **Prepare Features, Train, and Evaluate**

   ```python
   pipeline = build_feature_pipeline(data_cfg, feature_cfg)
   train_ds, val_ds, metadata = prepare_dataloader(
       data_cfg,
       pipeline,
       feature_cfg=feature_cfg,
   )
   training_state = train_model(train_ds, val_ds, model_cfg, metadata)
   artifacts = evaluate_model(training_state, val_ds, metadata, metrics=("mae", "rmse", "mape"))
   ```

4. **Consume Outputs**

   - `artifacts.metrics` contains metric dictionaries keyed by scope (currently `overall`).
   - `artifacts.predictions` is a tidy DataFrame with `(store_nbr, family, horizon_step)` rows.
   - `artifacts.model_params` exposes the trained Flax parameter PyTree for serialization.

## Resilience, Dependencies, and Tests

- `ensure_data_root()` warns when Kaggle files are missing and falls back to a
  synthetic dataset so notebooks run offline.
- Sliding-window creation raises informative `ValueError`s when the dataset is
  too short to satisfy the requested context window or horizon.
- Training leverages gradient clipping to avoid exploding gradients and logs the
  training/validation losses every epoch for traceability.
- Dependencies: `jax`, `flax`, `optax`, `pandas`, `numpy`, `scikit-learn`. No
  GPU-specific wheels are required for the demo notebooks.
- Tests: unit tests validate shapes/metadata, and smoke tests execute the
  training loop to guarantee compilation on CPU-only environments.

## Open Items

- Incorporate real holiday calendars (e.g., Ecuador-specific events) once raw
  files are available.
- Extend the evaluation module with hierarchical roll-ups (store vs family) and
  add forecast visualizations for major event periods.
