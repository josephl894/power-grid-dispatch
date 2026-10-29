# Hourly Generation Planning from Public Grid Data

Plant owners submit private bids, and New York uses those bids to decide which plants run each hour. The public record reports something else: demand in each region, fuel prices, scheduled outages, and the flow limits on the lines between regions. This study builds an hourly generation plan from that record and measures it against the cheapest plan the same public data will allow.

## Constraints

Every plan in the study has to be a plan the grid could run. Four constraints define that.

1. **Demand.** Each of eleven regions receives its expected demand for the hour. Power may move between regions, and energy is conserved: what leaves one region arrives in another.
2. **Line limits.** Flow between regions stays inside the published limits on those lines.
3. **Plant limits.** Output stays within available capacity. The change from the previous hour stays within the historical ramp rate for that fuel.
4. **Reserve.** Spare capacity equal to 2% of that hour's demand is held on plants that can provide it. The selected plan uses this fixed fraction, so the schedule cannot run every plant flat out.

## Method

The pipeline is learning-to-optimize. A model predicts costs. An optimizer writes the plan and enforces the four constraints. The model never emits megawatts directly.

Each hour proceeds as follows.

1. The hour is represented by demand in each region, weather, fuel prices, scheduled outages, and recent stress on the lines.
2. A linear model maps that hour to a cost for each of seven fuels in each of eleven regions.
3. A linear program minimizes those costs subject to the four constraints. The solution is the plan for the hour.
4. The same program is solved again with observed public fuel costs, under the same constraints. That solution is the cost benchmark: the cheapest feasible plan the public data allows.
5. The model is updated so the next set of predicted costs produces a plan closer to that benchmark.

Training uses 720 hours from 2020 and 2021, for three passes. 2022 is the check year. 2023 and 2024 are the test years. The selected model learns costs only. The 2% reserve is fixed. A separate variant lets the model choose the reserve. That variant costs more, and it is included in the results below.

## Results

The selected plan enforces the line limits and the 2% reserve. On the 2023–2024 test set it is the cheapest variant in the comparison. Every variant returns a feasible plan for every test hour.

Two numbers describe the result, and they answer different questions.

The generation bill falls from $463,109 an hour to $437,598, about 5.5%. The extra cost above the public-data benchmark falls from $52,755 an hour to $28,189, a 47% reduction. The bill is the cost of producing the power. The extra cost is how far that bill sits above the cheapest feasible plan. The 47% figure is the second number.

Variants that look more sophisticated cost more. Matching the published fuel mix more closely adds about $20,500 an hour of extra cost. Letting the model choose the reserve raises extra cost from $28,189 an hour to $29,270.

A utility or a trading desk that sees the public record and not the private bids can use the same procedure: predict costs, let the optimizer enforce line limits and a small reserve, and score the plan on extra cost against the cheapest feasible plan. Fuel-mix similarity is a separate score. On this test set it points to a more expensive plan. The results are a public-data proxy of hourly scheduling, evaluated under the constraints above.

## Results by variant

Each bar in the charts is one variant. **A0** is the historical schedule. **A6c**, shown in orange, is the selected plan: line limits plus the 2% reserve. The labels in between are the intermediate variants. Where the chart is a cost, shorter is better.

| Variant | Change from the previous plan | Bill ($/hr) | Extra cost ($/hr) |
| --- | --- | ---: | ---: |
| A0 | Historical schedule | 463,109 | 52,755 |
| A1 | Newer public list of plant capacity | 465,540 | 55,674 |
| A2 | Line limits between regions | 445,503 | 35,637 |
| A3 | Day-ahead stress on those lines | 449,337 | 39,470 |
| A5b | Plants known to be offline | 442,472 | 32,606 |
| **A6c** | **Fixed reserve of 2% of demand** | **437,598** | **28,189** |
| A7b | Reserve chosen by the model | 438,679 | 29,270 |

### Extra cost

This chart is the 47% result. Each bar is one variant's extra cost: dollars per hour above the cheapest feasible plan. The figure title uses the word regret for that gap. The orange bar is A6c, the shortest bar, and the selected plan.

![Extra cost by variant. The orange bar is the selected plan.](figures/final_scuc_proxy/01_financial_regret_by_stage.png)

### Generation bill

The same variants, now as the full cost of generating the power. The red line is the cheapest feasible plan. The bars are nearly equal because a gap of about $25,000 is small beside a bill near $440,000. The extra-cost chart is what makes the 47% visible. The orange bar is again A6c.

![Generation bill by variant. The red line is the cheapest feasible plan.](figures/final_scuc_proxy/02_financial_cost_by_stage.png)

### The reserve, month by month

Blue is the plan with line limits and known outages, and no reserve (A5b). Orange is that plan plus the 2% reserve (A6c). The orange line is lower in most months, which is the basis for selecting A6c. Both lines rise in the summer months.

![Monthly extra cost, plan without a reserve versus the selected plan.](figures/final_scuc_proxy/05_monthly_regret_a5b_vs_a6c.png)

### Fuel-mix error

These two charts measure a different objective: average megawatts away from the fuel mix New York published. Lower means a closer match to history. A7b, the learned reserve, is the closest. It is also more expensive than A6c, which is why A6c remains the selected plan. A6c is second on fuel-mix error and first on extra cost.

![Fuel-mix error by variant.](figures/final_scuc_proxy/03_total_fuel_mix_mae_by_stage.png)

![Monthly fuel-mix error. The green line is the learned reserve.](figures/final_scuc_proxy/06_monthly_total_fuel_mix_mae.png)

## Code

`run_zonal_cost_lto.py` trains the cost model and solves each hour. `nyiso_processing.py` and `nyiso_cost_processing.py` build those hours from the public files. The charts above are in `figures/final_scuc_proxy/`.

The other images in `figures/` come from an earlier model that predicted the fuel mix and then repaired the prediction. They are separate from the cost-learning pipeline in the table.

The hourly tables are too large for this repository. The scripts read them from `data/processed/`.

```bash
pip install -r requirements.txt
```
