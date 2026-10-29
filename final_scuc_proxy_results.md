# Final Public NYISO SCUC Proxy Results

## Research Framing

This project uses a Parmentier-style learning-to-optimize framing for a public-data approximation of NYISO dispatch. The hard problem is the hidden NYISO SCUC and real dispatch process. Public data does not reveal the full unit commitment state, private offers, complete security constraints, reserve requirements, outage semantics, or operator actions.

The easy problem is therefore a public SCUC-inspired LP using NYISO zones, fuel categories, Gold Book/P-19 capacity, P-32 interface limits, financial costs, constraint and outage stress features, ancillary stress, and reserve/headroom proxies. The learned transformation `h_theta(x_t)` predicts LP costs or reserve proxy terms. Later A8 variants test richer Parmentier-style transformations that also learn reserve terms and conservative derates. The results below should be described as a public SCUC proxy, not an exact NYISO SCUC reconstruction.

## Final Candidate Set

The existing generated reporting table is:

```text
data/processed/final_submission/final_scuc_proxy_ablation_table.csv
```

The A8b result below comes from the direct A8b artifact and should be added to
the generated final table/figures if those artifacts are rebuilt.

The selected stages are:

```text
A0  Legacy capacity baseline
A1  Gold Book/P-19 capacity
A2  Enforced P-32 interface baseline
A3  Enhanced P-511A day-ahead constraint stress
A5b P-511A + P-54A scheduled outage stress
A6c P-511A + P-54A + P-6B + fixed 2% reserve/headroom proxy
A7b A6c structure + learned reserve requirement fraction
A8b P-5 day-ahead ancillary stress + learned reserve fraction/penalty
```

Rejected exploratory variants are kept in:

```text
data/processed/final_submission/final_scuc_proxy_appendix_table.csv
```

These include A4, A4r, A7a, A8a, and A8c.

## Main Results

All selected runs use the 2023-2024 test split, Gold Book capacity for A1 and later, enforced real P-32 interface data where applicable, and 100% solver success.

```text
Stage  LtO cost ($/hr)  Regret ($/hr)  Overall P-63 MAE  Total fuel MAE
A0          463,109          52,755           270.263          291.789
A1          465,540          55,674           278.972          281.432
A2          445,503          35,637           273.550          276.727
A3          449,337          39,470           287.657          266.187
A5b         442,472          32,606           278.326          286.144
A6c         437,598          28,189           286.324          261.193
A7b         438,679          29,270           277.109          253.969
A8b         458,106          48,698           272.203          250.437
```

The primary financial/stress-aware candidate is A6c. It has the lowest financial dispatch cost and regret among the final selected variants.

The best fuel-mix realism candidate is now A8b. It has the lowest overall P-63 MAE and total fuel-mix MAE among the tested variants, but its financial regret is about 20,509 dollars/hour worse than A6c.

## Interpretation

A1 improves the public evidence basis by replacing legacy capacity with Gold Book/P-19 fleet data, but the first learned-cost model does not exploit it by itself.

A2 is the first defensible interface-constrained public proxy baseline because P-32 interface data is enforced rather than silently omitted.

A3 shows that P-511A day-ahead limiting-constraint stress is useful diagnostic context, but adding it only as learned-cost features improves fuel-mix MAE while worsening financial regret.

A5b shows that scheduled outage stress is useful when combined with P-511A stress. It is the clean no-reserve baseline.

A6c shows the strongest financial improvement. P-6B ancillary price stress becomes useful when paired with a soft ramp-accessible reserve/headroom proxy. A6c is therefore the main result if the project prioritizes financial regret against the public oracle.

A7b shows that learning a bounded reserve requirement fraction improves fuel-mix realism with only a small financial penalty relative to A6c. It remains the best balanced reserve-learning comparison, but A8b is the stronger pure fuel-realism variant.

A8a/A6f shows that adding all cleaner day-ahead feature families with default gates does not improve financial regret. A8b shows that a richer Parmentier reserve transformation using P-5 day-ahead ancillary stress substantially improves P-63 realism, but worsens financial regret. A8c is not promoted because conservative learned derates still hit the 20% cap and worsen regret.

## Rejected Variants

A4 and A4r learned thermal derates are not promoted. They improve some fuel-mix diagnostics but worsen financial regret and show problematic derate behavior.

A7a learned reserve slack penalty is not promoted. It improves total fuel-mix MAE versus A6c but worsens financial regret, while reserve slack remains zero and the learned penalty reaches the upper bound in a nontrivial share of hours.

A8c learned derates are not promoted. Even under conservative caps and regularization, the maximum learned thermal derate reaches the 20% cap and financial regret worsens to about 53,705 dollars/hour.

## Figures

Final plots are written to:

```text
figures/final_scuc_proxy/
```

Generated figures from the current report builder:

```text
01_financial_regret_by_stage.png
02_financial_cost_by_stage.png
03_total_fuel_mix_mae_by_stage.png
04_overall_p63_mae_by_stage.png
05_monthly_regret_a5b_vs_a6c.png
06_monthly_total_fuel_mix_mae.png
07_a7b_learned_reserve_fraction_distribution.png
```

An A8b comparison figure is still pending if the generated artifact set is
refreshed.

Monthly diagnostics are written to:

```text
data/processed/final_submission/final_scuc_proxy_monthly_diagnostics.csv
```

## Limitations

This is not exact NYISO SCUC. The public data is incomplete, and the LP is a structured proxy.

Gold Book capacity is a fleet snapshot, not a historically exact annual capacity series. P-27 masked IDs are not treated as generator identity truth. P-54A outage records are used as scheduled outage stress features, not as direct topology or generator derates. P-6B ancillary prices are used as public reserve stress context, not as the true NYISO reserve procurement model. The reserve/headroom constraint is a proxy with sensitivity-tested assumptions.

## Final Recommendation

Report A6c as the primary public SCUC proxy result:

```text
A6c: Gold Book + P-32 + P-511A + P-54A + P-6B + fixed 2% reserve/headroom
```

Report A8b as the secondary fuel-mix realism result:

```text
A8b: P-5 day-ahead ancillary stress + learned reserve fraction/penalty
```

Keep A7b as a useful earlier reserve-learning comparison and A5b as the clean no-reserve control. Use A4/A4r/A7a/A8c only as appendix evidence that not every learned transformation improves the proxy.
