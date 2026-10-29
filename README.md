# Power Grid Dispatch Model

A learning-to-optimize pipeline for New York's hourly generation plan.

A model learns generation costs from public load, price, and outage data. An optimizer sets the hourly plan from those costs. This is a public-data proxy. It is not the grid operator's real unit-commitment model, because bids and the full security-constrained problem are not public.

Adding transmission limits and a reserve margin cut the extra cost above the cheapest possible hourly plan by 47% versus the old schedule. On the 2023–2024 test hours that gap fell from $52,755 per hour to $28,189 per hour. The whole generation bill fell only about 5%. The 47% is the shrink in the extra cost, not in the total bill.

## Code

- `run_zonal_cost_lto.py` trains the cost model and solves each hour
- `nyiso_processing.py` and `nyiso_cost_processing.py` build the hourly inputs from the public files
- `final_scuc_proxy_results.md` is the stage-by-stage result
- `figures/` holds the cost and fuel-mix charts

The hourly tables are large and are not in this repository. The scripts look for them under `data/processed/`.

## Setup

```bash
pip install -r requirements.txt
```
