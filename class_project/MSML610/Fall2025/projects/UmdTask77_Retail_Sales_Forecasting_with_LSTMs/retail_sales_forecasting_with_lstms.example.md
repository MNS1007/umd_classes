<!-- toc -->

- [Retail Sales Forecasting with LSTMs — Example Walkthrough](#retail-sales-forecasting-with-lstms--example-walkthrough)
  * [Audience & Data Inputs](#audience--data-inputs)
  * [Storyboard](#storyboard)
  * [60-Minute Walkthrough](#60-minute-walkthrough)
  * [Data Pipeline](#data-pipeline)
  * [Modeling Blueprint](#modeling-blueprint)
  * [Notebook Outline](#notebook-outline)
  * [Current Progress & Next Steps](#current-progress--next-steps)

<!-- tocstop -->

# Retail Sales Forecasting with LSTMs — Example Walkthrough

This document implements the "Learn X in 60 Minutes" style requested in
`tutorials/docs/all.learn_X_in_60_minutes.how_to_guide.md`. It aligns with the
MSML610 Fall 2025 project brief (Difficulty 3) and demonstrates how to apply the
utilities in `retail_sales_forecasting_utils.py` to the Kaggle **Store Sales –
Time Series Forecasting** dataset (or the bundled synthetic fallback).

## Audience & Data Inputs

- Targeted at MSML610 classmates or interns who need a reproducible demand
  forecasting workflow.
- Requires a basic understanding of pandas/JAX and a local Docker setup.
- Raw files mirrored from Kaggle: `train.csv`, `test.csv`, `oil.csv`,
  `holidays_events.csv`, `transactions.csv` (or synthetic fallback).

## Storyboard

1. **Frame the business problem**: deliver weekly demand forecasts per
   `(store_nbr, family)` while accounting for seasonal effects and promotions.
2. **Ingest production-like data**: pull parquet and CSV files into a unified
  feature table, preserving hierarchical indices.
3. **Engineer temporal signals**: encode seasonalities, promotions, and optional
  external regressors (oil price, transactions).
4. **Train RNN models in JAX**: leverage a Flax LSTM/GRU backbone with JIT-compiled
   optimization for fast experimentation.
5. **Evaluate and visualize**: compare MAE/RMSE/MAPE across validation windows,
   and plot forecast curves to highlight holiday/promotion effects.
6. **Extend to multivariate regressors**: demonstrate optional inclusion of
   macroeconomic drivers in the same workflow.

## 60-Minute Walkthrough

| Minute | Segment | Objective |
|--------|---------|-----------|
| 0–10   | Setup    | Start the Docker container, clone repo, skim README. |
| 10–20  | Data     | Explain Kaggle schema, run `ensure_data_root`, preview tables. |
| 20–30  | Features | Compose `build_feature_pipeline`, visualize engineered columns. |
| 30–45  | Training | Instantiate `ModelConfig`, kick off `train_model`, log metrics. |
| 45–55  | Evaluation | Use `evaluate_model`, chart MAE/RMSE lines, inspect predictions. |
| 55–60  | Wrap-up  | Discuss extensions, homework ideas, and how to swap real data. |

## Data Pipeline

- **Raw Files**: `train.csv`, `test.csv`, `oil.csv`, `holidays_events.csv`,
  `transactions.csv`.
- **Preprocessing Steps**:
  1. Normalize column names and parse dates.
  2. Join auxiliary tables on `date` and `store_nbr`.
  3. Aggregate to daily totals per `(store_nbr, family)`.
  4. Generate supervised learning windows using sliding look-back sequences.
  5. Split into training/validation sets respecting chronological order.

```mermaid
flowchart TD
  A[Raw Kaggle Files] --> B[Clean & Join]
  B --> C[Aggregate by Store + Family]
  C --> D[Window Generation]
  D --> E1[Train Dataset]
  D --> E2[Validation Dataset]
```

## Modeling Blueprint

- **Architecture**: stacked LSTM or GRU layers with layer normalization and
  dropout. Sequence length defaults to 120 days; prediction horizon is 28 days.
- **Loss Function**: SMAPE baseline with optional quantile loss extension.
- **Optimizer**: AdamW via `optax` with cosine decay schedule.
- **Metrics**: MAE, RMSE, and MAPE computed per `(store_nbr, family)` and
  aggregated at the national level.

```text
Input: [batch_size, time_steps, feature_dim]
RNN -> Dense(projection) -> Forecast Horizon
```

## Notebook Outline

1. **Section 1 — Environment Setup**
   - Confirm JAX backend, ensure deterministic seeds, import helpers.
2. **Section 2 — Data Download & Caching**
   - Call `ensure_data_root()` (or future Kaggle helper) to fetch data if missing.
3. **Section 3 — Feature Engineering**
   - Use `build_feature_pipeline()` to transform raw DataFrames.
   - Visualize seasonal features and holiday encodings.
4. **Section 4 — Model Training**
   - Instantiate `ModelConfig`, run `train_model()` with training curves plot.
5. **Section 5 — Evaluation & Visualization**
   - Render metrics table, confusion-style matrix of error per store, and line
     charts around major holidays.
6. **Section 6 — What-If Scenarios**
   - Toggle promotions or oil price regressors to show impact on metrics.

## Current Progress & Next Steps

- ✅ Repository structure aligned to class instructions and learn-in-60 format.
- ✅ Synthetic data pipeline + evaluation artifacts ready for notebooks.
- 🔄 Wire up real Kaggle dataset ingestion and expand event-driven feature coverage.
- 🔄 Benchmark against naive baselines and add hierarchical roll-up metrics.
- 🔄 Enhance visualizations (per-store panels, residual diagnostics).
