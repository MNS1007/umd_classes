# Retail Sales Forecasting with LSTMs — Project Workspace

<!-- toc -->

- [Project Intent](#project-intent)
- [Deliverables & Templates](#deliverables--templates)
- [60-Minute Tutorial Blueprint](#60-minute-tutorial-blueprint)
- [Directory Layout](#directory-layout)
- [Environment Setup (Simple Docker)](#environment-setup-simple-docker)
- [Next Milestones](#next-milestones)

<!-- tocstop -->

## Project Intent

This directory tracks the MSML610 Fall 2025 project focused on multi-store,
multi-product retail sales forecasting with recurrent neural networks in JAX.
For the midterm PR the expectation is to deliver a **runnable scaffold** that
demonstrates the end-to-end workflow on the synthetic Kaggle replica while
following the official class template.

## Deliverables & Templates

- The class requirements are captured in
  `class_project/instructions/README.md`. They define branch naming, Docker
  expectations, and the `{project}.{API,example}.{md,ipynb,py}` artifacts.
- The tutorial-writing best practices live in
  `tutorials/docs/all.learn_X_in_60_minutes.how_to_guide.md`
  (mirrored online at
  <https://github.com/causify-ai/tutorials/blob/master/docs/all.learn_X_in_60_minutes.how_to_guide.md>).
  The API/example markdowns in this repo follow that structure with a clear
  hierarchy, TOC, and a 60-minute teaching plan.
- Current status:
  1. **Planning & Scope** — outlined in the API/example markdowns.
  2. **Executable Notebooks** — notebooks exist and will call into the utils
     module; coding comes next.
  3. **Utility Module** — shared logic lives in `retail_sales_forecasting_utils.py`.
  4. **Docker Scripts** — `docker_simple/` ready for day-to-day dev; thin env
     placeholder checked in for later.

## 60-Minute Tutorial Blueprint

| Minute        | Focus                                   | Artifact |
|---------------|-----------------------------------------|----------|
| 0–10 min      | Context, target audience, prerequisites | README   |
| 10–25 min     | API concepts + configuration dataclasses| `*.API.md` / notebook |
| 25–45 min     | Hands-on example + data ingestion       | `*.example.md` / notebook |
| 45–55 min     | Evaluation + discussion of results      | Example notebook |
| 55–60 min     | Next steps + homework ideas             | README/Example doc |

Each section in the markdowns calls out the expected talking points so the
notebooks can stay focused on concise executable cells.

## Directory Layout

- `retail_sales_forecasting_with_lstms.API.*`: interface-first documentation,
  notebook, and helper module describing the reusable forecasting API surface.
- `retail_sales_forecasting_with_lstms.example.*`: end-to-end tutorial material
  showing how to pull data, train the JAX LSTM/GRU models, and evaluate results.
- `retail_sales_forecasting_utils.py`: shared utility functions (data loading,
  feature engineering, model builders) imported by the notebooks.
- `docker_simple/`: DATA605-style container scripts for development.
- `docker_causify_dev_system/`: placeholder for the thin-environment setup should
  we migrate to the advanced workflow later in the semester.
- `tests/`: pytest suite that smoke-tests the utils module with synthetic data.
- `cluster_jobs/`: SLURM scripts for training on the UMD Nexus GPU cluster.

Please keep new assets inside this directory per the instructions.

## Environment Setup (Simple Docker)

```bash
cd /Users/mns/Documents/umd_classes/class_project/MSML610/Fall2025/projects/UmdTask77_Retail_Sales_Forecasting_with_LSTMs/docker_simple
bash docker_build.sh   # builds the Jupyter-ready image with JAX/Flax dependencies
bash docker_jupyter.sh # launches JupyterLab with the project mounted at /app/project
```

The Dockerfile already bundles CPU-enabled JAX, Flax, Optax, pandas/polars,
scikit-learn, and plotting libraries. If you have an NVIDIA GPU available add
the appropriate `jaxlib` wheel before rebuilding.

## GPU Training on Nexus

- `analysis.ipynb` now saves normalized sliding windows (`train_X.npy`, etc.) to
  the Kaggle data directory so you can reuse preprocessing output on the
  cluster. Update `DATA_ROOT` inside the SLURM scripts if your dataset or `.npy`
  files live elsewhere.
- `cluster_jobs/nexus_train_lstm.sh` and `cluster_jobs/nexus_train_gru.sh` are
  templates copied from the Nexus scavenger queue configuration. Customize:
  - `PROJECT_ROOT`, `DATA_ROOT`, and `VENV_PATH` (virtualenv/conda) for your
    account.
  - `#SBATCH` resources (GPUs, memory, wall clock) according to availability.
  - CLI flags after `python -m retail_sales_forecasting_with_lstms.example` to
    adjust lookback, epochs, metrics, etc.
- Submit with `sbatch cluster_jobs/nexus_train_lstm.sh` (or the GRU variant).
  Logs are written to `logs/rsf_*.{out,err}` inside the project directory.

## Next Milestones

1. Integrate the Kaggle Store Sales dataset ingestion pipeline with holiday and
   promotion feature encoding.
2. Add store/family-level aggregation metrics and baseline comparisons.
3. Expand exploratory plots (holiday overlays, per-store drill-downs).
4. Package trained parameters and scalers for reuse outside the notebooks.

Please do not merge this branch without TA approval. Branch and folder naming
follow the `UmdTask77_Retail_Sales_Forecasting_with_LSTMs` convention required
by the course instructions.
