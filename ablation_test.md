# Public NYISO SCUC Proxy Ablation Tests

## Purpose

This document is the final ablation record for the public-data NYISO SCUC proxy.
The goal is not to reconstruct exact NYISO SCUC. Public data does not expose the
full commitment state, private offers, complete network-security model, reserve
procurement model, or operator actions. The goal is to test which public NYISO
evidence sources make the easy LP proxy more useful under a Parmentier-style
learning-to-optimize setup.

The hard problem is the hidden NYISO SCUC and real dispatch process. The easy
problem is a public SCUC-inspired zonal LP using NYISO zones, fuel categories,
Gold Book/P-19 capacity, P-32 interface limits, financial costs, outage and
constraint stress, ancillary stress, and reserve/headroom proxies. The learned
transformation `h_theta(x_t)` predicts costs or proxy terms for that easy LP.

## Common Evaluation Design

Primary split:

```text
train:      2020-2021 source period, capped for structured training
validation: 2022
test:       2023-2024
holdout:    2025, not used for the main final table
```

The main comparable test runs use:

```text
split:              test
test period:        2023-2024
structured epochs:  3 unless noted otherwise
train cap:          720 hours unless noted otherwise
capacity source:    Gold Book/P-19 for A1 and later
interfaces:         real P-32 enforced for interface-enabled claims
target diagnostic:  P-63 fuel mix
financial oracle:   public-data LP oracle under the same feasible set
```

Reported metrics:

- `oracle_financial_cost_per_hour`: mean public LP oracle cost.
- `lto_financial_cost_per_hour`: mean LtO dispatch cost evaluated under public financial costs.
- `regret_to_oracle_per_hour`: LtO cost minus oracle cost.
- `p95_regret_to_oracle_per_hour`: upper-tail financial regret.
- `overall_p63_mae_mw`: mean absolute error across P-63 fuel categories.
- `total_fuel_mix_mae_mw`: mean absolute error in total fuel generation.
- `success_rate`: LP solve success rate.
- `mean_balance_residual_mw`: balance residual.
- `mean_interface_slack_mw`: interface slack.
- `mean_binding_interface_count`: average binding interface count where available.
- `mean_transfer_magnitude_mw`: average zonal transfer magnitude.
- `mean_reserve_slack_mw`: reserve proxy slack where applicable.

## Final Selected Ablation Table

Source artifact:

```text
data/processed/final_submission/final_scuc_proxy_ablation_table.csv
```

| Stage | Goal | LtO cost ($/hr) | Regret ($/hr) | P95 regret ($/hr) | Overall P-63 MAE (MW) | Total fuel MAE (MW) | Success | Interface slack (MW) | Binding interfaces | Transfer MW | Reserve slack MW |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A0 | Legacy capacity baseline | 463108.739 | 52754.891 | 118414.836 | 270.263 | 291.789 | 1.000 | 0.000 | n/a | 3950.108 | n/a |
| A1 | Gold Book/P-19 capacity | 465540.118 | 55673.920 | 138184.356 | 278.972 | 281.432 | 1.000 | 0.000 | n/a | 4154.123 | n/a |
| A2 | Enforced P-32 interface baseline | 445503.086 | 35636.888 | 105534.757 | 273.550 | 276.727 | 1.000 | 0.000 | 0.377 | 4471.521 | n/a |
| A3 | Enhanced P-511A stress features | 449336.516 | 39470.317 | 116159.639 | 287.657 | 266.187 | 1.000 | 0.000 | 0.357 | 4319.463 | n/a |
| A5b | P-511A + P-54A, no reserve | 442471.710 | 32605.512 | 115652.003 | 278.326 | 286.144 | 1.000 | 0.000 | 0.167 | 3654.159 | 0.000 |
| A6c | P-6B + fixed 2% reserve/headroom | 437597.617 | 28188.898 | 124462.203 | 286.324 | 261.193 | 1.000 | 0.000 | 0.349 | 4476.827 | 0.000 |
| A7b | Learned reserve requirement fraction | 438678.942 | 29270.223 | 112659.265 | 277.109 | 253.969 | 1.000 | 0.000 | 0.185 | 4260.095 | 0.000 |
| A8a | Causal feature gates, default weights | 460283.290 | 50874.571 | n/a | 284.288 | 279.768 | 1.000 | 0.000 | 0.131 | 3870.953 | 0.000 |
| A8b | P-5 learned reserve terms | 458106.225 | 48697.506 | n/a | 272.203 | 250.437 | 1.000 | 0.000 | 0.124 | 4189.306 | 0.000 |
| A8c | A8b + P-54C learned derates | 463113.522 | 53704.803 | n/a | 288.646 | 267.702 | 1.000 | 0.000 | 0.292 | 4699.620 | 0.000 |

## Stage Results And Interpretation

### A0: Legacy Capacity Baseline

Goal: Preserve the original cost-aware zonal LtO baseline using legacy capacity.
This is the historical control, not the preferred public SCUC proxy.

Result: A0 has regret of about `52755 $/hr`. It solves reliably, but the
capacity basis is weaker than Gold Book/P-19. It remains useful only as an
ablation showing why we moved to public fleet evidence.

Decision: Do not use as the final model. Keep as the legacy control.

### A1: Gold Book/P-19 Capacity

Goal: Replace legacy capacity with a stronger NYISO public fleet snapshot. Gold
Book capacity is joined to P-19 generator names by PTID, unmatched rows are
preserved, fuel categories are standardized, and zone/fuel capacities are
aggregated.

Result: A1 improves the evidence basis but not the FY learner by itself. Regret
rises to about `55674 $/hr`, although total fuel MAE improves versus A0.

Interpretation: Better public capacity data does not automatically improve the
learned marginal-cost transformation. The LP needs stronger public stress
context.

Decision: Keep Gold Book/P-19 as the default capacity source despite A1 regret.
It is more defensible for the final public SCUC proxy.

### A2: Enforced P-32 Interface Baseline

Goal: Formalize interface-constrained feasibility. A2 requires
`nyiso_interface_hour.parquet`, fails on missing or empty interface data, and
reports P-32 coverage, finite limits, binding interface count, slack, and
transfer magnitude.

Result: Regret falls sharply to about `35637 $/hr`, with 100% solver success,
near-zero balance residual, and zero mean interface slack. Mean binding interface
count is about `0.377`.

Interpretation: A2 is the first defensible interface-constrained public proxy
baseline. It is the right comparison point for later public constraint features.

Decision: Keep A2 as the interface baseline.

### A3: Enhanced P-511A Day-Ahead Constraint Stress

Goal: Add P-511A day-ahead limiting-constraint features as learner context, not
as hard constraints. Added features include active flags, positive/negative cost
summaries, log-scaled counts and cost features, and causal rolling/lagged stress.

Result: A3 improves total fuel MAE from `276.727 MW` to `266.187 MW`, but
worsens financial regret from `35637 $/hr` to `39470 $/hr`.

Interpretation: P-511A is informative, but using it only as marginal-cost
features does not guarantee better financial dispatch. It appears useful as
constraint-stress context rather than a complete missing-constraint substitute.

Decision: Keep P-511A in later candidate models, but do not promote A3 as the
final baseline.

### A4: Learned Thermal Derates

Goal: Test whether P-511A stress can drive learned thermal derates, not just
learned costs. This is a step toward learning penalties or feasible-set
adjustments.

Result: A4 regret is about `40296 $/hr`, worse than A2 and A3. Total fuel MAE
improves slightly versus A3 to `265.745 MW`, but derates show saturation risk.

Interpretation: Derates are the right type of transformation for public stress
signals, but the first update rule is too blunt.

Decision: Reject A4 as a final candidate. Keep in appendix.

### A4r: Regularized Learned Thermal Derates

Goal: Add regularization to reduce A4 derate saturation.

Result: Regret worsens to about `44702 $/hr`, and total fuel MAE worsens to
`281.228 MW`. Mean derate falls, but the maximum still reaches the cap.

Interpretation: Regularization does not rescue this heuristic learned-derate
path.

Decision: Reject A4r. Future derates would need a better supervised or
repair-derived target.

### A5a: Scheduled Outage Stress Only

Goal: Add P-54A scheduled outage stress as an A2-based public feature ablation.
P-54A is treated as scheduled outage context, not as exact generator derating.

Result: A5a worsens financial regret to about `43159 $/hr`.

Interpretation: Scheduled outage stress alone is not enough. It likely needs
constraint-stress context to be useful.

Decision: Do not promote A5a.

### A5b: P-511A + P-54A Clean No-Reserve Baseline

Goal: Combine P-511A constraint stress with P-54A scheduled outage stress while
leaving reserve/headroom out. This tests whether public constraint and outage
stress improve the learned cost proxy before adding reserve assumptions.

Result: A5b improves regret to about `32606 $/hr`, better than A2 and A3. It is
the clean no-reserve baseline. The tradeoff is worse total fuel MAE
(`286.144 MW`).

Interpretation: P-54A becomes useful when combined with P-511A. The model gets
closer to the public financial oracle but less close to P-63 total fuel mix.

Decision: Keep A5b as the clean no-reserve control.

### A6a: P-6B Ancillary Stress Features

Goal: Add P-6B ancillary/reserve price stress features to A5b without adding a
reserve/headroom constraint.

Result: A6a improves total fuel MAE but worsens financial regret relative to
A5b.

Interpretation: P-6B is informative for fuel-mix realism, but as features alone
it does not improve the financial objective.

Decision: Do not promote A6a alone.

### A6b: Reserve/Headroom Proxy

Goal: Add a soft ramp-accessible reserve/headroom proxy without P-6B ancillary
stress.

Result: A6b improves total fuel MAE versus A5b but worsens financial cost.
Reserve slack remains zero, so the constraint is feasible under the tested
setting.

Interpretation: Reserve/headroom structure helps realism but needs ancillary
stress context to improve financial regret.

Decision: Do not promote A6b alone.

### A6c: P-6B + Reserve/Headroom Proxy

Goal: Combine P-511A, P-54A, P-6B, and a fixed reserve/headroom proxy. This tests
whether ancillary stress and reserve structure work together.

Result: A6c is the best financial candidate. With the selected 2% reserve
fraction and 720 train hours / 7 epochs, LtO cost is about `437598 $/hr`, regret
is about `28189 $/hr`, and total fuel MAE improves to `261.193 MW`. Solver
success is 100%, balance residual is numerical zero, interface slack is zero,
and reserve slack is zero.

Interpretation: P-6B ancillary stress becomes useful when paired with an
explicit reserve/headroom proxy. A6c is the main financial/stress-aware result.

Decision: Use A6c as the primary financial-regret candidate.

### A6c Sensitivity Tests

Goal: Test whether the A6c improvement is robust to training budget, epochs, and
reserve fraction.

Observed training/epoch sensitivity:

| Run | LtO cost ($/hr) | Regret ($/hr) | Overall P-63 MAE (MW) | Total fuel MAE (MW) |
|---|---:|---:|---:|---:|
| A6c 168 train, 3 epochs | 475920.258 | 66511.539 | 262.844 | 306.581 |
| A6c 720 train, 3 epochs | 440224.716 | 30815.997 | 286.010 | 260.437 |
| A6c 720 train, 5 epochs | 439418.953 | 30010.234 | 283.315 | 260.844 |
| A6c 720 train, 7 epochs | 437597.617 | 28188.898 | 286.324 | 261.193 |
| A6c 1440 train, 3 epochs | 448132.089 | 38723.369 | 271.174 | 254.359 |
| A6c 1440 train, 5 epochs | 446469.699 | 37060.980 | 276.790 | 271.112 |

Interpretation: More hours are not automatically better. The 720-hour budget
with more structured epochs is strongest financially. The 1440-hour runs improve
some fuel diagnostics but lose financial regret. The 168-hour run is too small
for stable financial learning.

Observed monthly comparison against A5b:

```text
A6c beats A5b on monthly financial regret in 16 of 25 observed months.
A6c beats A5b on monthly total fuel MAE in 12 of 25 observed months.
```

Decision: Use A6c 720/e7 with the 2% reserve fraction as the primary A6c result.

### A7a: Learned Reserve Slack Penalty

Goal: Learn the reserve slack penalty instead of using a fixed penalty.

Result: A7a improves total fuel MAE to `257.676 MW`, but worsens regret to about
`36078 $/hr`. Reserve slack remains zero, and the learned penalty reaches its
upper bound in a nontrivial share of hours.

Interpretation: Learning the slack penalty does not improve the active
constraint behavior enough to beat A6c.

Decision: Reject A7a. Keep in appendix.

### A7b: Learned Reserve Requirement Fraction

Goal: Learn a bounded reserve requirement fraction rather than a fixed reserve
fraction. This is a more direct transformation than learning slack penalties.

Result: A7b has LtO cost about `438679 $/hr`, regret about `29270 $/hr`, overall
P-63 MAE `277.109 MW`, and total fuel MAE `253.969 MW`. It improves fuel-mix
realism relative to A6c, but financial regret is about `1081 $/hr` worse.

Interpretation: A7b was the first strong fuel-mix realism candidate. It does
not replace A6c on financial regret, but it remains a useful reserve-learning
comparison because it shows that learning reserve requirement structure can move
dispatch toward observed P-63 fuel totals.

Decision: Keep A7b as an earlier balanced reserve-learning comparison. A8b
supersedes it as the secondary fuel-mix realism candidate.

### A8a: Rich Parmentier Feature Gates

Goal: Test the A6f causal feature set with explicit dataset gate coefficients
for P-511A, P-54A, P-54C, P-5, P-59, weather, and costs. In this first run, all
gates are fixed at `1.0`, so A8a acts as the ungated causal-rich control.

Result: A8a has LtO cost about `460283 $/hr`, regret about `50875 $/hr`,
overall P-63 MAE `284.288 MW`, and total fuel MAE `279.768 MW`.

Interpretation: Simply exposing all causal-clean feature families does not
recover the A6c financial result. Fixed gates are useful infrastructure, but the
default all-on version should not be promoted.

Decision: Keep A8a as a scaffold/control for future gate tuning.

### A8b: P-5 Learned Reserve Terms

Goal: Test a richer Parmentier transformation where P-5 day-ahead ancillary
stress helps learn reserve requirement fraction and reserve slack penalty,
instead of only entering learned fuel costs.

Result: A8b has LtO cost about `458106 $/hr`, regret about `48698 $/hr`,
overall P-63 MAE `272.203 MW`, and total fuel MAE `250.437 MW`. Mean learned
reserve requirement fraction falls to about `0.013`, with median about `0.009`.

Interpretation: A8b is not financially competitive with A6c, but it is the best
fuel-realism result so far. This supports the richer-transformation idea:
reserve learning can move the public proxy closer to observed P-63 behavior,
even though the current reserve-learning objective does not reduce financial
regret.

Decision: Report A8b as the secondary fuel-mix realism result, replacing A7b in
that role. Keep A6c as the primary financial-regret result.

### A8c: P-54C Learned Derates

Goal: Add P-54C day-ahead scheduled outage stress and conservative learned
thermal derates to A8b. The runner caps A8c derates at `0.20` and applies
stronger default regularization than A4/A4r.

Result: A8c has LtO cost about `463114 $/hr`, regret about `53705 $/hr`,
overall P-63 MAE `288.646 MW`, and total fuel MAE `267.702 MW`. Mean thermal
derate is about `0.036`, but max thermal derate reaches the `0.200` cap.

Interpretation: Learned derates remain unstable. Even conservative A8c derates
hit the cap and worsen financial regret.

Decision: Do not promote A8c. Keep it as evidence that derate learning needs a
better target or formulation before it can be used in the final proxy.

## Final Model Roles

Use three models in the final narrative:

- A5b: clean no-reserve control.
- A6c: primary financial/stress-aware public SCUC proxy.
- A8b: secondary learned-reserve fuel-mix realism proxy.

Do not claim that A6c or A8b reconstruct exact NYISO SCUC. The correct claim is
that public SCUC-inspired constraints and stress features improve the learned
optimization proxy under selected metrics.

## Current Recommended Commands

Smoke run:

```bash
python3 run_zonal_cost_lto.py --processed-dir data/processed/final_submission --output-dir data/processed/final_submission/zonal_cost_lto_smoke_a6c --proxy-stage a6c --reserve-requirement-fraction 0.02 --max-train-hours 168 --max-eval-hours 24 --structured-epochs 1 --split test
```

Primary A6c run:

```bash
python3 run_zonal_cost_lto.py --processed-dir data/processed/final_submission --output-dir data/processed/final_submission/zonal_cost_lto_A6c_frac02_train720_e7 --proxy-stage a6c --reserve-requirement-fraction 0.02 --max-train-hours 720 --max-eval-hours 0 --structured-epochs 7 --split test
```

Reserve-learning comparison A7b run:

```bash
python3 run_zonal_cost_lto.py --processed-dir data/processed/final_submission --output-dir data/processed/final_submission/zonal_cost_lto_A7b_learned_reserve_fraction_train720_e7 --proxy-stage a7b --reserve-requirement-fraction 0.02 --reserve-fraction-max 0.06 --max-train-hours 720 --max-eval-hours 0 --structured-epochs 7 --split test
```

Secondary A8b run:

```bash
python3 run_zonal_cost_lto.py --processed-dir data/processed/final_submission --output-dir data/processed/final_submission/zonal_cost_lto_A8b_train720_e7 --proxy-stage a8b --reserve-requirement-fraction 0.02 --max-train-hours 720 --max-eval-hours 0 --structured-epochs 7 --split test
```

Final report artifacts:

```bash
python3 build_final_scuc_proxy_report.py
```

Compile validation:

```bash
python3 -m py_compile nyiso_experiments.py run_zonal_cost_lto.py nyiso_processing.py nyiso_cost_processing.py run_scuc_proxy_data_processing.py build_final_scuc_proxy_report.py
```

## Final Conclusion

A6c is the best primary result if the report prioritizes public-oracle financial
regret. A8b is the best secondary result if the report emphasizes observed
P-63 fuel-mix realism. A5b remains the clean no-reserve control. The ablation
sequence supports the final framing: public NYISO evidence helps most when it is
introduced as structured constraint and reserve context, not merely as more
features in a marginal-cost learner.
