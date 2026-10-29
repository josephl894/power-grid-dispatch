# Hourly Generation Planning from Public Grid Data

## Context

New York chooses, every hour, which power plants run and how much each one produces. That choice is made from private bids. The public record reports regional demand, fuel prices, scheduled outages, and the flow limits on the lines between regions.

This study builds the hourly plan from the public record and scores it against the cheapest plan those same data allow.

## The Optimization Problem

**Decision.** Megawatts of each fuel in each region, for each hour. Eleven regions, seven fuels.

**Objective.** Minimize the cost of that plan.

**Constraints.**

| Constraint | Requirement |
| --- | --- |
| Demand | Each region receives its expected demand. Power may move between regions, and energy is conserved. |
| Line limits | Flow between regions stays inside the published limits. |
| Plant limits | Output stays within available capacity, and the change from the previous hour stays within the historical ramp rate for that fuel. |
| Reserve | Spare capacity equal to 2% of that hour's demand is held on plants that can provide it. |

**Benchmark.** The same four constraints, solved with observed public fuel costs. This is the cheapest feasible plan the public data allow. Extra cost is a plan's cost minus that benchmark. It is the gap, not the power bill.

## Our Approach

Learning to optimize. A model predicts the costs. A linear program writes the plan, enforces the constraints, and is what produces the megawatts.

1. Represent the hour by regional demand, weather, fuel prices, scheduled outages, and recent stress on the lines.
2. Map that hour, with a linear model, to one cost per fuel per region.
3. Minimize those costs with a linear program under the four constraints. The solution is the hourly plan.
4. Solve the same program with observed public fuel costs. That solution is the benchmark.
5. Update the model so the next predicted costs move the plan toward the benchmark.

Training uses 720 hours from 2020 and 2021, three passes. 2022 is the check year. 2023 and 2024 are the test years. The selected model learns costs only. The 2% reserve stays fixed. A variant that learns the reserve is reported with the results.

`run_zonal_cost_lto.py` trains the cost model and solves each hour. `nyiso_processing.py` and `nyiso_cost_processing.py` build the hours from the public files.

## Our Results

The selected plan enforces the line limits and the fixed 2% reserve. On 2023–2024 it is the cheapest variant tested. Every variant returns a feasible plan for every test hour.

| | Historical schedule | Selected plan |
| --- | ---: | ---: |
| Generation bill | $463,109 / hr | $437,598 / hr |
| Extra cost | $52,755 / hr | $28,189 / hr |

The bill, the cost of producing the power, falls about 5.5%. The extra cost, how far that bill sits above the cheapest feasible plan, falls 47%.

Closer copies of the published fuel mix cost more. The closest fuel-mix variant adds about $20,500 an hour of extra cost. Letting the model choose the reserve raises extra cost from $28,189 an hour to $29,270.

A desk that sees the public record, and not the private bids, can follow the same procedure: predict costs, enforce the line limits and a small reserve inside the optimizer, and score the plan on extra cost.

### Variants

**A0** is the historical schedule. **A6c**, the orange bar in the charts, is the selected plan. Labels in between are the steps from one to the other. On a cost chart, shorter is better.

| Variant | What changed | Bill ($/hr) | Extra cost ($/hr) |
| --- | --- | ---: | ---: |
| A0 | Historical schedule | 463,109 | 52,755 |
| A1 | Newer public list of plant capacity | 465,540 | 55,674 |
| A2 | Line limits between regions | 445,503 | 35,637 |
| A3 | Day-ahead stress on those lines | 449,337 | 39,470 |
| A5b | Plants known to be offline | 442,472 | 32,606 |
| **A6c** | **Fixed reserve, 2% of demand** | **437,598** | **28,189** |
| A7b | Reserve chosen by the model | 438,679 | 29,270 |

**Extra cost.** Dollars per hour above the cheapest feasible plan. The figure title calls this gap regret. The orange bar is A6c.

![Extra cost by variant. The orange bar is the selected plan.](figures/final_scuc_proxy/01_financial_regret_by_stage.png)

**Generation bill.** The same variants as the full cost of generating the power. The red line is the cheapest feasible plan. The bars look level because a gap near $25,000 is small beside a bill near $440,000.

![Generation bill by variant. The red line is the cheapest feasible plan.](figures/final_scuc_proxy/02_financial_cost_by_stage.png)

**The reserve, by month.** Blue is line limits and known outages, with no reserve (A5b). Orange adds the 2% reserve (A6c). Orange is lower in most months. Both lines rise in the summer.

![Monthly extra cost, no reserve versus the selected plan.](figures/final_scuc_proxy/05_monthly_regret_a5b_vs_a6c.png)

**Fuel-mix error.** Average megawatts away from the fuel mix New York published. Lower is a closer match to history. A7b, the learned reserve, is closest, and it costs more than A6c. A6c is second on this score and first on extra cost.

![Fuel-mix error by variant.](figures/final_scuc_proxy/03_total_fuel_mix_mae_by_stage.png)

![Monthly fuel-mix error. The green line is the learned reserve.](figures/final_scuc_proxy/06_monthly_total_fuel_mix_mae.png)

## Our Challenges

- **Hidden bids.** The operator schedules from private bids and a fuller network model. This study has the public record and the four constraints above, so the plans are a proxy of that schedule.
- **Two different scores.** A 5.5% bill reduction and a 47% extra-cost reduction describe the same plan. The bill chart hides the result that the extra-cost chart shows.
- **Fuel mix against cost.** Matching the published mix of gas, hydro, and wind more closely raises extra cost by about $20,500 an hour. The cheaper plan is the worse historical copy.
- **The reserve is uneven.** The 2% reserve lowers extra cost in most months, and both plans spike in summer.
- **What a clone can run.** The hourly tables are too large for this repository. Scripts expect them under `data/processed/`. Images elsewhere in `figures/` belong to an earlier model that predicted the fuel mix and then repaired the prediction.

```bash
pip install -r requirements.txt
```
