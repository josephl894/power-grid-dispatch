# Power Grid Dispatch Model

A learning-to-optimize pipeline for New York's grid. A model learns generation costs from public load, price, and outage data, and an optimizer sets the hourly plan. Adding transmission limits and a reserve margin cut the extra cost above the cheapest possible hourly plan by 47% versus the old schedule.

The code calls the public data by its source name, NYISO. This is a public-data proxy of hourly dispatch, not the grid operator's real unit-commitment model.

## Where to look

- `run_zonal_cost_lto.py` runs the learning-to-optimize pipeline
- `final_scuc_proxy_results.md` records the compared plans
- `nyiso_processing.py` and `nyiso_cost_processing.py` build the public hourly inputs

The hourly tables are large and are not in this repository. The scripts expect them under `data/processed/`.
