# Power Grid Dispatch Model

A learning-to-optimize pipeline for New York's hourly generation plan.

Public load, price, and outage data go in. A model learns generation costs from that data. A linear program then sets the hourly plan. This is a public-data proxy. It is not the grid operator's real unit-commitment model, because bids and the full security-constrained problem are not public.

The comparison that matters is the extra cost above the cheapest feasible plan on the same public data. On the 2023–2024 test hours, adding transmission limits and a reserve margin cut that extra cost by 47%, from $52,755 per hour to $28,189 per hour. The whole generation bill only fell from $463,109 to $437,598 per hour, about 5.5%. Every selected run solved every hour.

## Result

The chosen plan is A6c: updated fleet capacity, transmission limits between zones, day-ahead constraint and outage stress, and a fixed 2% reserve margin.

| Stage | What was added | Generation bill ($/hr) | Extra cost ($/hr) |
| --- | --- | ---: | ---: |
| A0 | Old capacity schedule | 463,109 | 52,755 |
| A1 | Updated public fleet capacity | 465,540 | 55,674 |
| A2 | Transmission limits between zones | 445,503 | 35,637 |
| A3 | Day-ahead transmission stress | 449,337 | 39,470 |
| A5b | Scheduled outages | 442,472 | 32,606 |
| **A6c** | **Fixed 2% reserve margin** | **437,598** | **28,189** |
| A7b | Learned reserve instead of the fixed 2% | 438,679 | 29,270 |

A later variant matched the published fuel mix more closely, but its extra cost was about $20,500 per hour worse than A6c, so it was not the plan that was kept. The charts below stop at A7b. The full table, including that fuel-mix variant, is in `final_scuc_proxy_results.md`.

### Extra cost by stage

The orange bar is the plan that was kept. The chart title calls this gap regret. It is the extra cost above the cheapest feasible plan, not the generation bill itself.

![Extra cost above the cheapest feasible plan, by stage. A6c is lowest.](figures/final_scuc_proxy/01_financial_regret_by_stage.png)

### Generation bill by stage

The bars are the generation bill. The red line is the cheapest feasible plan. The bars look close because the extra cost is small next to a bill of about $440,000 per hour. A6c is the shortest bar.

![Generation bill by stage, with the cheapest feasible plan as a line.](figures/final_scuc_proxy/02_financial_cost_by_stage.png)

### Where the reserve margin helps

A5b is the same plan without the reserve margin. A6c is lower in most months. It is not lower in every month. Summer 2023 and summer 2024 are the expensive months for both.

![Monthly extra cost, plan without a reserve margin versus the plan that was kept.](figures/final_scuc_proxy/05_monthly_regret_a5b_vs_a6c.png)

### Fuel mix

A6c also reduced how far the plan's fuel mix sits from the published mix, from 292 MW average error to 261 MW. Letting the model choose the reserve (A7b) matches the fuel mix a bit better, at a slightly higher bill. That is why A6c was kept.

![Average fuel-mix error by stage. A7b is lowest. A6c is next.](figures/final_scuc_proxy/03_total_fuel_mix_mae_by_stage.png)

![Monthly fuel-mix error. The learned reserve is usually the lowest line.](figures/final_scuc_proxy/06_monthly_total_fuel_mix_mae.png)

## Code

- `run_zonal_cost_lto.py` trains the cost model and solves each hour
- `nyiso_processing.py` and `nyiso_cost_processing.py` build the hourly inputs
- `final_scuc_proxy_results.md` is the stage-by-stage write-up
- `figures/final_scuc_proxy/` holds the charts above

The other PNGs in `figures/` are earlier model variants. They predict the next fuel mix and then repair that prediction. They are not the cost-learning pipeline in the table.

The hourly tables are large and are not in this repository. The scripts look for them under `data/processed/`.

```bash
pip install -r requirements.txt
```
