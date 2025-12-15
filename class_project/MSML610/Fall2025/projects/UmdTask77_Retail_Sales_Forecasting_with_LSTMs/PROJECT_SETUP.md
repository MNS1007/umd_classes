<!-- toc -->

- [Project Setup Checklist](#project-setup-checklist)
  * [References](#references)
  * [File Map](#file-map)
  * [Workflow TODO](#workflow-todo)
  * [Development Notes](#development-notes)

<!-- tocstop -->

# Project Setup Checklist

This document captures how the repository follows the class instructions in
`class_project/instructions/README.md` and the learn-in-60-minutes format
described in `tutorials/docs/all.learn_X_in_60_minutes.how_to_guide.md`
(mirrored online at
<https://github.com/causify-ai/tutorials/blob/master/docs/all.learn_X_in_60_minutes.how_to_guide.md>).

## References

- `README.md` — overview, template compliance, Docker instructions.
- `retail_sales_forecasting_with_lstms.API.*` — API contract docs/notebooks/code.
- `retail_sales_forecasting_with_lstms.example.*` — hands-on tutorial docs/notebooks/code.
- `retail_sales_forecasting_utils.py` — shared implementation imported by both notebooks.
- `docker_simple/` — DATA605-style Docker runtime for notebooks/tests.
- `tests/` — pytest suite hitting the utils module with synthetic data.

## File Map

| Asset | Purpose | Status | Next Step |
|-------|---------|--------|-----------|
| `retail_sales_forecasting_with_lstms.API.md` | Explains the reusable API surface in a 60-minute format | ✅ | Keep in sync with notebooks |
| `retail_sales_forecasting_with_lstms.API.ipynb` | Walks through API usage interactively | ⏳ | Flesh out cells once coding resumes |
| `retail_sales_forecasting_with_lstms.API.py` | Re-exports typed contracts from the utils module | ✅ | Consider adding Protocols for downstream adapters |
| `retail_sales_forecasting_with_lstms.example.md` | End-to-end project narrative | ✅ | Update screenshots/plots after real data hookup |
| `retail_sales_forecasting_with_lstms.example.ipynb` | Runnable example notebook | ⏳ | Wire to real Kaggle data + plots |
| `retail_sales_forecasting_with_lstms.example.py` | CLI entrypoint mirroring the example notebook | ✅ | Add CLI args for quick experiments |
| `retail_sales_forecasting_utils.py` | Data prep, feature engineering, model build/eval | ✅ | Harden data download + caching |
| `docker_simple/` | Docker build/run scripts per instructions | ✅ | Add GPU notes if hardware available |

✅ = ready for midterm scaffold, ⏳ = placeholder awaiting implementation details.

## Workflow TODO

- [x] Copy tutorial template and rename artifacts to the project slug.
- [x] Document the 60-minute teaching plan (README + markdowns).
- [x] Wire Docker instructions for the simple setup.
- [ ] Finish notebooks so they call into the utils module.
- [ ] Add thin-environment instructions once the Causify dev-system is required.
- [ ] Flesh out CI hooks (pre-commit, lint, pytest) before final submission.

## Development Notes

- Keep new artifacts under `class_project/MSML610/Fall2025/projects/UmdTask77_Retail_Sales_Forecasting_with_LSTMs/`
  per the instructions.
- For any shared logic, prefer extending `retail_sales_forecasting_utils.py`
  and import the functions inside notebooks. This keeps notebooks clean and
  matches the tutorial template.
- The learn-in-60 style expects each notebook/markdown to be runnable within a
  tight schedule; keep the timelines in README.md synced with actual runtime.
