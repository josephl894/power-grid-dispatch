# Power Grid Dispatch Model

Every hour, New York has to decide which power plants turn on. The people who actually make that decision see private bids from the plant owners. I did not. I could see the public record: how much electricity people used, what prices did, and which plants were offline.

The job was to build an hourly plan from that public record and see if it could beat the old schedule on cost.

I trained a model to estimate what it costs to run each kind of plant. An optimizer then picked the plan for each hour. I did not let it treat the state like one big copper plate. Power lines between regions have limits, and some capacity has to sit in reserve instead of running flat out.

## Conclusion

The plan I kept is the ordinary one. It respects the power lines, and it holds back 2% of capacity as reserve.

On 2023 and 2024, that plan left less money on the table than the old schedule. The gap versus the cheapest plan the public data would allow fell from $52,755 an hour to $28,189 an hour. That is a 47% smaller gap. The bill for generating the power barely moved, from $463,109 an hour to $437,598. So this is not "the power bill fell by half." The old plan was quietly expensive relative to the best plan available. The new one is less quietly expensive. Every version I compared still produced a workable schedule for every hour.

The versions that look smarter were worse business. A plan that copied the historical fuel mix more closely cost about $20,500 more per hour. Letting the model pick its own reserve also cost more than just using 2%. Matching the past, and a fancier model, were not the cheaper way to run the system.

A utility or a trading desk that cannot see private bids can use this the same way: draft the hourly plan from the public record, force it to obey the lines and a small reserve, and grade it on dollars left above the cheapest workable plan. Do not grade it on whether the mix of gas, hydro, and wind looks like last year. This does not replace the software the grid operator actually runs.

## What the charts are

Each bar or line is one version of the plan. **A0** is the old schedule. **A6c**, the orange bar, is the plan I kept: line limits plus a 2% reserve. The letters in between are the versions I tried on the way there. Shorter is better when the chart is about cost.

| Version | What I changed | Bill ($/hr) | Left on the table ($/hr) |
| --- | --- | ---: | ---: |
| A0 | Old schedule | 463,109 | 52,755 |
| A1 | Newer public list of plant capacity | 465,540 | 55,674 |
| A2 | Limits on the lines between regions | 445,503 | 35,637 |
| A3 | Day-ahead stress on those lines | 449,337 | 39,470 |
| A5b | Plants known to be offline | 442,472 | 32,606 |
| **A6c** | **Hold back 2% as reserve** | **437,598** | **28,189** |
| A7b | Let the model choose the reserve | 438,679 | 29,270 |

### Money left on the table

This is the chart behind the 47%. Each bar is one version. The orange bar is the plan I kept, and it is the shortest. The chart's own title says "regret." That word means the same thing as the column above: dollars per hour above the cheapest workable plan. It is not the power bill.

![Money left on the table, by version. The orange bar is the plan I kept.](figures/final_scuc_proxy/01_financial_regret_by_stage.png)

### The actual bill

Same versions, but this time the bars are the full cost of generating the power. The red line is the cheapest workable plan. The bars look almost the same height because a $25,000 gap is small next to a $440,000 bill. That is why the first chart exists. The bill chart hides the result. The orange bar is still the one I kept.

![Full generation bill by version. The red line is the cheapest workable plan.](figures/final_scuc_proxy/02_financial_cost_by_stage.png)

### Did the reserve help every month?

Blue is the plan with line limits and outages, and no reserve. Orange is that same plan plus the 2% reserve. Orange is lower most months, which is why I kept it. It is not lower every month. Both lines spike in the summers.

![Month by month, plan without a reserve versus the plan I kept.](figures/final_scuc_proxy/05_monthly_regret_a5b_vs_a6c.png)

### Copying the fuel mix is a different goal

These two charts are not about the bill. They ask how many megawatts off the plan was from the mix of fuels New York actually published. Lower means a closer copy of history. Letting the model choose the reserve (A7b) wins that contest. It also costs more, which is why it is not the plan I kept. A6c is the next best on this chart and the best on the money chart.

![How far each version sits from the published fuel mix.](figures/final_scuc_proxy/03_total_fuel_mix_mae_by_stage.png)

![Same fuel-mix gap, month by month. The green line is the learned reserve.](figures/final_scuc_proxy/06_monthly_total_fuel_mix_mae.png)

## Code

`run_zonal_cost_lto.py` trains the cost model and solves each hour. `nyiso_processing.py` and `nyiso_cost_processing.py` turn the public files into those hours. The charts above live in `figures/final_scuc_proxy/`.

The other pictures in `figures/` are an earlier attempt. That attempt guessed the next fuel mix and then patched the guess. It is not the plan in the table.

The hourly data files are large and are not in this repository. The scripts expect them under `data/processed/`.

```bash
pip install -r requirements.txt
```
