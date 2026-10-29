# NYISO Dispatch Design Decisions

This document records modeling decisions for the NYISO weather-aware dispatch imitation project. The filename keeps the requested spelling: `design_descision.md`.

## NYISO System-Level Scope

The current model is a system-level NYISO fuel-mix dispatch imitator. It predicts hourly generation by fuel category:

```text
[Dual Fuel, Natural Gas, Hydro, Nuclear, Wind, Other Renewables, Other Fossil Fuels]
```

The target is P-63 real-time fuel mix. This is treated as the oracle-like realized dispatch signal for supervised imitation and structured learning.

The v1 dispatch LP is still a system-level copperplate model. It does not yet enforce zonal nodal balances, transmission-interface limits, or generator-unit commitment.

## Day-Ahead Causal Feature Policy

The default experiment mode is now `causal_day_ahead`. Features must be information that would plausibly be available before or at the day-ahead decision point.

Allowed same-hour features include:

```text
forecast_load_mw
day-ahead LBMP and congestion summaries
day-ahead limiting-constraint summaries
scheduled outage counts
weather features
calendar features
static fuel capacity
```

Realized quantities can only enter as lagged state:

```text
actual_load_mw_lag_1, actual_load_mw_lag_24
rt_lbmp_*_lag_1, rt_lbmp_*_lag_24
rt_constraint_*_lag_1, rt_constraint_*_lag_24
gen_<fuel>_mw_lag_1, gen_<fuel>_mw_lag_24
```

The default feature set excludes same-hour ex-post fields:

```text
actual_load_mw
load_forecast_error_mw
rt_lbmp_*
rt_constraint_*
realtime_outage_count
```

## Diagnostic Feature Policy

The previous ex-post feature set is retained only as `diagnostic_ex_post`. It can be used for debugging or upper-bound comparisons, but it is not the default experiment mode because it uses information that would not be known in a real day-ahead prediction setting.

## Training Window Definition

The default neural input window is:

```text
history_window_hours = 24
```

For prediction at hour `t`, the model sees:

```text
X_t = [x_{t-23}, x_{t-22}, ..., x_t]
```

This is causal because the current row `x_t` contains forecast/known values only. Realized information appears only through lagged columns.

## Solver Load Policy

The default day-ahead LP mode is now `internal_generation`, because the P-63 target is internal fuel generation, not raw NYISO load. The processed system table records:

```text
target_internal_generation_mw = sum_f P63_fuel_generation_f
forecast_net_import_residual_mw = forecast_load_mw - target_internal_generation_mw
actual_net_import_residual_mw = actual_load_mw - target_internal_generation_mw
```

For prediction, the v1 residual forecast is the training median by hour-of-week:

```text
predicted_net_import_residual_t = median_train(forecast_net_import_residual | hour_of_week)
solver_balance_mw_t = forecast_load_mw_t - predicted_net_import_residual_t
```

The LP balances `solver_balance_mw` in the default mode. `forecast_load` remains available as a comparison mode.

The realized P-63 fuel mix remains the evaluation target:

```text
y_t = P63_fuel_mix_t
```

## Weather Treatment

Weather is joined by UTC timestamp and zone code. Zone weather fields are cleaned and converted into system-level aggregates:

```text
wind: capacity-weighted by wind capacity
solar/GHI: capacity-weighted by solar capacity
temperature: weighted by average zonal load share
```

The v1 weather features are:

```text
system_wind_speed_50m_mps
system_wind_speed_cubed
system_ghi_wh_m2
system_ghi_rolling_3h
system_temp_2m_c
system_temp_rolling_24h
```

Precipitation is excluded for v1 because the available weather files have empty or inconsistent precipitation fields.

## Availability Bounds

Weather is now also used to create zonal availability bounds for fuels that should not be dispatched freely to static nameplate capacity:

```text
wind_available_z,t = wind_capacity_z * normalized_wind_power_curve(WS50M_z,t)
solar_available_z,t = other_renewables_capacity_z * normalized_GHI_z,t
other_renewables_available_z,t = other_renewables_capacity_z * (0.20 + 0.80 * normalized_GHI_z,t)
hydro_available_z,t = hydro_capacity_z
nuclear_available_z,t = nuclear_capacity_z
```

Hydro and nuclear also receive empirical system caps from the training period:

```text
hydro_cap_t <= min(static_hydro_capacity, 1.05 * Q_0.99(train_hydro_generation))
nuclear_cap_t <= min(static_nuclear_capacity, 1.01 * Q_0.99(train_nuclear_generation))
```

These are first-pass proxy bounds, not generator-specific availability or unit commitment. When availability and ramp-down lower bounds conflict, the zonal LP clips the ramp lower bound to the hour's feasible available capacity. This treats weather-driven renewable loss as an availability event rather than an operator-controlled ramp violation.

## Outage Feature Policy

Scheduled P-54A outages are causal day-ahead features. They are used as network stress descriptors, not generator derates, because PTID-to-generator mapping is not yet reliable.

Current scheduled outage features:

```text
scheduled_outage_count
scheduled_outage_ptid_count
scheduled_outage_equipment_count
scheduled_outage_345kv_count
scheduled_outage_by_interface_hint
```

P-54B real-time outages remain diagnostic or lagged-only in causal experiments.

## Normalization Policy

Neural model inputs are standardized with `StandardScaler` fitted on the training split only. Direct neural targets are also standardized during training and inverse-transformed for evaluation.

LP quantities remain in physical MW units. Load, generation, capacity, residual supply, ENS, spill, and ramp limits are not normalized before entering the solver.

## Capacity Assumptions

Fuel-category capacity bounds use:

```text
CAP_f = max(static_capacity_f, 1.05 * train_max_observed_generation_f)
```

This protects the LP from unrealistically tight metadata capacity when the observed P-63 training data already exceeds the static facility total.

## Empirical Ramping

The LP now supports fuel-level empirical ramp constraints. Ramping is estimated from observed training-period P-63 fuel mix.

For each fuel category `f`:

```text
Delta g_{t,f} = g_{t,f} - g_{t-1,f}
RU_f = Q_0.99(max(Delta g_{t,f}, 0))
RD_f = Q_0.99(max(-Delta g_{t,f}, 0))
```

The LP imposes:

```text
max(0, g_{t-1,f} - RD_f) <= g_{t,f} <= min(CAP_f, g_{t-1,f} + RU_f)
```

The first solved hour uses the observed lagged P-63 dispatch as the previous state. Subsequent solved hours use the previous LP prediction, making the LP rollout sequential.

This is an empirical estimator of aggregate flexibility, not a unit-level commitment model.

## Structured Learning Decision

The learned-cost model follows the learning-to-optimize idea: the neural network predicts costs, and the LP maps those costs into a feasible dispatch.

```text
c_t = h_theta(X_t)
g_hat_t = argmin_g c_t^T g
```

Candidate dispatches include a ramp-feasible projected oracle, a fixed-cost LP rollout, and perturbed-cost LP rollouts. Candidate generation uses the same capacity and ramp restrictions as prediction.

## Geographic and Grid-Constraint Limitations

Weather and capacity are geographically resolved by NYISO zone, but the current LP is system-level. It does not use zone coordinates to estimate energy travel distance, and it does not assume that distance is a reliable proxy for transmission feasibility.

Current grid information is used as features:

```text
day-ahead limiting constraints
lagged real-time limiting constraints
day-ahead congestion summaries
lagged real-time congestion summaries
```

Hard zonal/interface constraints are deferred to a later model version. A defensible zonal model should use NYISO interface limits and zonal balances rather than coordinate distance alone.

## Zonal Interface Dispatch

The first zonal dispatch version uses NYISO zones A-K as nodes and P-32 internal transfer interfaces as cut constraints inside the LP. This is a graph/cut representation, not a graph neural network.

P-32 limit cleaning:

```text
raw_positive_limit >= 9000  -> no finite upper bound
raw_positive_limit <= 0     -> no finite upper bound
raw_negative_limit <= -9000 -> no finite lower bound
raw_negative_limit >= 0     -> no finite lower bound
```

Hourly aggregation is conservative:

```text
upper_limit_mw = min finite positive limit within hour
lower_limit_mw = max finite negative limit within hour
flow_mw_mean   = mean observed flow within hour
```

The v1 internal cut matrix is:

```text
DYSINGER EAST      <- source side A, B
MOSES SOUTH        <- source side D
CENTRAL EAST - VC  <- source side A, B, C, D, E
TOTAL EAST         <- source side A, B, C, D, E
UPNY CONED         <- source side A, B, C, D, E, F
SPR/DUN-SOUTH      <- source side A, B, C, D, E, F, G, H
```

Positive modeled flow means export from the listed source-side zones to the rest of the NYISO system:

```text
F_k,t = sum_{z in source_side(k)} n_z,t
```

Interface bounds are enforced with high-penalty slack:

```text
F_k,t <= upper_k,t + s^+_k,t
F_k,t >= lower_k,t - s^-_k,t
```

Net zonal injections are no longer free. The LP uses positive/negative split variables:

```text
n_z,t = n^+_z,t - n^-_z,t
```

and penalizes transfer magnitude:

```text
p_transfer * sum_z (n^+_z,t + n^-_z,t)
```

The first smoke-test value is:

```text
p_transfer = 5
```

The zonal balance equation is:

```text
sum_f g_z,f,t + residual_z,t + ens_z,t - spill_z,t - L_z,t = n_z,t
```

In `forecast_load` mode, `L_z,t = forecast_load_z,t`. In default `internal_generation` mode, the system `solver_balance_mw_t` is allocated to zones in proportion to zonal forecast load shares.

and system balance is:

```text
sum_z n_z,t = 0
```

External schedule interfaces are excluded from hard internal constraints in v1:

```text
all SCH - * interfaces
WEST CENTRAL
```

`SCH - *` rows describe external scheduled interfaces rather than internal NY zone-to-zone transfer limits. `WEST CENTRAL` is mostly sentinel-limited in the current P-32 data and is retained as diagnostic information rather than a hard cut.

`TOTAL EAST` overlaps with `CENTRAL EAST - VC`. It is kept with slack penalties as an aggregate consistency constraint; slack diagnostics must be inspected for infeasibility or over-constraining.

Because P-63 fuel mix is system-level, not zonal, zonal dispatch is evaluated after summing zone decisions:

```text
G_f,t = sum_z g_z,f,t
```

This is compared against the same system P-63 target. There is currently no zonal generation oracle target.

## Zonal Learned-Cost Baseline

The first learned zonal baseline reuses the existing Parmentier-style system cost learner:

```text
c_f,t = h_theta(X_t)
```

The learned fuel costs are broadcast across zones and repaired through the zonal/interface LP:

```text
c_f,t -> zonal LP -> g_z,f,t -> sum_z g_z,f,t
```

This is intentionally conservative. It tests whether learned costs plus zonal feasibility improve over fixed zonal costs before adding a full zone-fuel cost model:

```text
c_z,f,t = h_theta(zone_features_z,t, system_features_t)
```

## Persistence Baselines

The previous `persistence` row is now named explicitly:

```text
persistence_previous_hour
```

This is a real-time style baseline and performs well because hourly fuel mix is strongly autocorrelated. Stricter alternatives are now included:

```text
persistence_previous_day_hour
persistence_previous_week_hour
day_ahead_persistence_rollout
```

## Residual-Aware Repair LP

The direct neural baseline now supports residual learning:

```text
Delta g_hat_t = h_theta(X_t)
g_raw_t = g_{t-1} + Delta g_hat_t
```

The residual-aware repair LP treats `g_raw_t` as a learned decision target, not as final dispatch. The optimizer projects it into the feasible set:

```text
minimize sum_f |g_f,t - g_raw_f,t|
```

subject to balance, capacity, empirical ramping, and nonnegative residual/ENS/spill variables. The zonal version allocates `g_raw_f,t` to zones by zonal fuel-capacity share, then applies the zonal/interface LP with availability bounds and transfer penalties.

This remains an LtO-style model because the neural model supplies decision-guiding quantities and the LP produces the final feasible dispatch.

The first smoke result shows an important tradeoff: repair feasibility is excellent, but MAE can worsen because the LP enforces `solver_balance_mw`, which is based on the current hour-of-week net-import residual estimate. This exposes the next modeling target: improve the internal-generation/residual forecast or let the repair LP include a calibrated soft balance term.

## Feasibility-Gap Diagnostics

The first diagnostic track projects the P-63 oracle dispatch into progressively stricter easy-LP feasible sets. It measures:

```text
projection_gap_t = y_projected_t - y_P63_t
```

The January 2024 smoke diagnostic shows:

```text
capacity_only gap                  = 0.000 MW
capacity_plus_ramp gap             = 0.680 MW
system solver_balance gap          = 114.742 MW
system oracle_total_balance gap    = 1.359 MW
zonal solver_balance gap           = 211.724 MW
zonal oracle_total_balance gap     = 199.212 MW
```

This indicates that system-level capacity and ramp assumptions are not the main bottleneck. The large jump from oracle-total balance to solver-balance confirms that the internal-generation/net-import residual target is currently a major source of repair distortion. The zonal oracle-total gap also shows that zonal allocation, availability, and interface approximations remain restrictive even when the system total is correct.

## Balance-Target Diagnostics

The second diagnostic track compares alternative LP balance targets against P-63 total internal generation:

```text
oracle_total_t = sum_f P63_fuel_generation_f,t
balance_error_t = candidate_balance_t - oracle_total_t
```

Each candidate balance is then used as the hard balance in the system repair LP while the raw dispatch being repaired is the P-63 oracle itself. This isolates balance-target error from neural prediction error.

The January 2024 smoke diagnostic showed that the original hour-of-week `solver_balance_mw` was weak:

```text
solver_balance_mw absolute target error      = 816.488 MW
solver_balance_mw repair overall MAE         = 114.742 MW
lag-1 residual balance absolute target error = 235.849 MW
lag-1 residual balance repair overall MAE    = 33.844 MW
oracle-total balance repair overall MAE      = 1.359 MW
```

This means the repair LP itself is not the main issue at system level. The dominant system-level repair problem is that the hard internal-generation balance target is too far from the P-63 total. A causal lag-1 net-import residual target is currently the strongest simple candidate and should be tested as the next default balance policy before more complex loss-function experiments.

The Task 4 refactor makes lag-1 net-import residual the default repair balance policy:

```text
solver_balance_mw = forecast_load_mw - forecast_net_import_residual_mw_lag_1
```

The January 2024 smoke comparison confirms that this substantially improves repair:

```text
residual_repair_lp_with_weather_hour_of_week_balance MAE = 139.827 MW
residual_repair_lp_with_weather_lag1_balance MAE         = 75.666 MW
direct_residual_nn_with_weather MAE                       = 62.263 MW
```

So lag-1 balance should replace hour-of-week balance as the default, but hard repair still does not beat the raw residual neural model on the January smoke split. The next repair improvement should be a soft-balance or differentiable repair objective, not another hard-balance target.

## Soft-Balance Residual Repair

Task 5 adds a calibrated soft-balance repair LP because the lag-1 hard balance target is better than hour-of-week balance but still imperfect. The neural model still predicts a raw residual dispatch:

```text
g_raw_t = g_{t-1} + Delta g_hat_t
```

The soft repair LP solves:

```text
minimize sum_f |g_f,t - g_raw_f,t|
       + p_soft_balance * (balance_under_t + balance_over_t)
```

subject to capacity, empirical ramping, and:

```text
sum_f g_f,t + balance_under_t - balance_over_t = solver_balance_mw_t
```

This keeps repair feasible while explicitly measuring how much the optimizer deviates from the uncertain internal-generation balance target. Task 5 is system-level only; zonal soft repair is deferred until the system behavior is understood.

The January 2024 smoke comparison shows that low-penalty soft repair is the first repair variant to slightly beat the raw residual neural model:

```text
direct_residual_nn_with_weather MAE              = 62.263 MW
hard_repair_lag1_with_weather MAE                = 75.666 MW
soft_repair_lag1_with_weather, p = 0.1, MAE      = 61.999 MW
soft_repair_lag1_with_weather, p = 0.1, balance deviation = 268.144 MW
```

This is promising but should not be over-claimed yet. The best soft repair mostly preserves the neural dispatch and uses balance relaxation to avoid damaging a strong residual prediction. The next validation step is to test whether this holds on the full 2023-2024 test period and then decide whether soft repair is the final system-level LtO candidate.

The year-aligned 2020-2025 pilot gives stronger evidence. Using:

```text
train:      2020-2021
validation: 2022
test:       2023-2024
holdout:    2025
```

the best test and holdout rows are:

```text
test direct residual with weather MAE       = 79.372 MW
test soft repair with weather p=0.1 MAE     = 78.841 MW
holdout direct residual with weather MAE    = 117.997 MW
holdout soft repair with weather p=0.1 MAE  = 114.034 MW
```

This means the low-penalty soft repair improvement does generalize beyond January, and the improvement is larger on 2025 holdout than in the January smoke. Hard repair remains worse for fuel-category MAE but improves total fuel-mix MAE, confirming the tradeoff:

```text
hard repair improves aggregate balance but damages fuel allocation.
soft repair preserves fuel allocation while reporting balance uncertainty.
```

## Loss Function Experiments

The third diagnostic track compares LtO training losses under the corrected lag-1 residual balance target:

```text
loss_experiment_balance_mw = forecast_load_mw - forecast_net_import_residual_mw_lag_1
```

The comparison includes:

```text
direct residual NN with L2 training loss
current candidate-softmax structured loss
Fenchel-Young structured loss
fixed-cost LP reference
projected-oracle reference
```

The Fenchel-Young trainer uses the projected oracle as the hard-solution comparator:

```text
L_FY(c, y_proj) = c^T y_proj - min_g c^T g
dL/dc = y_proj - g_pred
```

The January 2024 smoke result shows that the structured cost learners are feasible but not yet competitive:

```text
projected oracle MAE                 = 33.844 MW
direct residual NN with weather MAE  = 62.263 MW
candidate-softmax with weather MAE   = 1331.880 MW
Fenchel-Young with weather MAE       = 1632.327 MW
```

This does not reject Fenchel-Young as a research direction; it shows that the current fuel-level cost parameterization is too weak or too poorly conditioned to imitate dispatch through a linear LP objective. The next LtO improvement should focus on either a residual-aware repair objective, a zone/fuel cost parameterization, or a differentiable QP/repair layer before promoting FY to the main model.

## Smoke Experiment Profile

The default smoke output now uses the `current_research` profile. It reports persistence baselines, fixed zonal LP, residual direct neural models, and residual repair LPs. Older learned-cost grids, no-lag variants, and system fixed-cost LP runs are retained behind config toggles rather than removed.

## Task 8 Zonal Diagnostics

After the system soft-balance repair showed stable full-period behavior, zonal work is diagnosed before it is promoted into the main leaderboard. The diagnostic runner compares fixed-cost zonal LP variants with interface constraints on/off, availability bounds on/off, and transfer penalties varied. This isolates whether zonal error is coming from capacity-share allocation, weather availability bounds, P-32 interface cuts, or the fact that net zonal transfers are otherwise too cheap.

The diagnostic is intentionally bounded by `max_hours` so it can be run on a small sample of the year-aligned split. Interface-enabled variants now require `nyiso_interface_hour.parquet` with finite P-32 limits. If the selected processed directory is missing that table, interface-constrained results must not be reported because the LP would otherwise fall back to unconstrained transfers.

## Public SCUC Proxy Capacity Snapshot

The next zonal LtO path uses a public SCUC-inspired proxy, not a reconstruction of NYISO SCUC or real dispatch. The hard industrial process remains hidden: public data does not expose the complete unit commitment state, private offer stack, security constraints, outage semantics, reserve requirements, or operator overrides. The easy problem is therefore a defensible public LP using NYISO zones, fuel categories, capacity, ramping, availability, interface constraints, financial costs, outages/stress features, and later reserve/ancillary proxies.

Gold Book processing writes:

```text
data/processed/scuc_proxy/generator_facilities.parquet
data/processed/scuc_proxy/zonal_fuel_capacity.parquet
data/processed/scuc_proxy/zonal_tech_capacity.parquet
```

The Gold Book table is treated as a fleet snapshot rather than historically exact annual capacity. Capacity uses cleaned numeric fields with commas removed. Summer capability prefers `sum_cris_MW` and falls back to `name_plate_rating_MW`; winter capability prefers `win_cris_MW`, then `win_cap_MW`, then `name_plate_rating_MW`. The dispatch capacity column used by the A1 proxy is the summer capability aggregate.

Gold Book `PTID` is left-joined to P-19 `Generator PTID` after deduplicating P-19 by generator PTID. Unmatched Gold Book rows are preserved with `ptid_match_status` because a failed public-name join is not evidence that a facility should be dropped.

Fuel categories remain the seven project categories:

```text
Dual Fuel
Natural Gas
Hydro
Nuclear
Wind
Other Renewables
Other Fossil Fuels
```

The mapping uses `fuel_type_1`, `fuel_type_2`, `Dual`, and `unit_type`. P-27 masked generator IDs are not generator truth and are not used to define the fleet. P-61 is omitted because the locally available data only extends to roughly 2005, which is mismatched with the 2020-2025 experiment period. P-54C and P-59 are also excluded unless local files beyond `.DS_Store` are verified.

The zonal cost LtO runner now has `--zonal-capacity-source legacy|gold_book`. The legacy path preserves `nyiso_zonal_capacity.parquet` for ablation A0. The Gold Book path uses `data/processed/scuc_proxy/zonal_fuel_capacity.parquet` for A1 and later public SCUC proxy runs.

Initial A0/A1 test ablation with real P-32 interface data:

```text
A0 legacy capacity, test 2023-2024:
  oracle mean financial cost = 410,353.848 dollars/hour
  LtO mean financial cost    = 463,108.739 dollars/hour
  mean regret                = 52,754.891 dollars/hour
  P-63 overall MAE           = 270.263 MW for LtO
  total fuel-mix MAE         = 291.789 MW for LtO
  transfer magnitude         = 3,950.108 MW

A1 Gold Book capacity, test 2023-2024:
  oracle mean financial cost = 409,866.198 dollars/hour
  LtO mean financial cost    = 465,540.118 dollars/hour
  mean regret                = 55,673.920 dollars/hour
  P-63 overall MAE           = 278.972 MW for LtO
  total fuel-mix MAE         = 281.432 MW for LtO
  transfer magnitude         = 4,154.123 MW
```

Both runs had 100% solver success, numerical balance residuals, and 0 MW mean interface slack. The A1 oracle is slightly cheaper, but this first Gold Book run did not improve learned-policy regret. This supports keeping Gold Book as the stronger public proxy input while shifting the next LtO step toward constraint-stress features and learned penalties or derates instead of only zone-fuel costs.

## A2/A3 Public Proxy Ablation

A2 formalizes the interface-constrained A1 baseline. The runner now writes `zonal_cost_lto_interface_summary.csv`, `zonal_cost_lto_feature_metadata.csv`, and optional `zonal_cost_lto_interface_ablation_metrics.csv`. Interface-enabled runs still hard-fail if P-32 data are missing, empty, or contain no finite limits.

A3 adds enhanced P-511A day-ahead limiting-constraint features as learned context only, not as hard LP constraints. The added features include active flags, positive/negative constraint counts and means, absolute-cost means, log-scaled count/cost features, 24-hour rolling means, and 24-hour lags. These are causal day-ahead context features because P-511A is a day-ahead market artifact; they are not used to claim exact NYISO SCUC constraint reconstruction.

Refreshed A1/A2/A3 test comparison, 2023-2024, 17,540 evaluated hours, Gold Book capacity, real P-32 interface data, 720 training-hour cap, and 3 structured epochs:

```text
A1 reference / A2 interface main:
  oracle mean financial cost = 409,866.198 dollars/hour
  LtO mean financial cost    = 445,503.086 dollars/hour
  mean regret                = 35,636.888 dollars/hour
  P-63 overall MAE           = 273.550 MW for LtO
  total fuel-mix MAE         = 276.727 MW for LtO
  mean binding interfaces    = 0.377
  transfer magnitude         = 4,471.521 MW

A3 + P-511A enhanced features:
  oracle mean financial cost = 409,866.198 dollars/hour
  LtO mean financial cost    = 449,336.516 dollars/hour
  mean regret                = 39,470.317 dollars/hour
  P-63 overall MAE           = 287.657 MW for LtO
  total fuel-mix MAE         = 266.187 MW for LtO
  mean binding interfaces    = 0.357
  transfer magnitude         = 4,319.463 MW
```

A2 interface-off diagnostic, using the same learned-cost model but solving without interface limits, gives LtO regret about 34,970 dollars/hour versus 35,637 dollars/hour with interfaces on. This is useful as a diagnostic only; the reported public SCUC proxy result should use interface-on.

Interpretation: A3 improves total fuel-mix MAE but worsens financial regret and overall fuel MAE. Under the current interpretation rule, retain A2/A1 as the main public SCUC proxy baseline and keep A3 as evidence that constraint-stress information is useful but needs a richer learned transformation, likely learned penalties or derates rather than more zone-fuel marginal-cost features.

## A4 Learned Thermal Derates

A4 tests whether P-511A stress context is more useful when the learned transformation can reduce thermal availability rather than only changing zone-fuel marginal costs. The A4 implementation predicts derates for:

```text
Dual Fuel
Natural Gas
Other Fossil Fuels
```

The derates are bounded by `thermal_derate_max` and applied as:

```text
effective_capacity_z,f,t = base_capacity_z,f * (1 - derate_z,f,t)
```

The public financial oracle remains the no-derate A2/A3 oracle, so regret remains comparable. A4 is explicit opt-in via `--proxy-stage a4`; the runner default remains A2 because A4 is not promoted.

Initial A4 test, same 2023-2024 split and training budget:

```text
A4 + P-511A + learned thermal derates:
  oracle mean financial cost = 409,866.198 dollars/hour
  LtO mean financial cost    = 450,162.329 dollars/hour
  mean regret                = 40,296.130 dollars/hour
  P-63 overall MAE           = 285.651 MW for LtO
  total fuel-mix MAE         = 265.745 MW for LtO
  mean thermal derate        = 0.157
  max thermal derate         = 0.500
```

Interpretation: learned thermal derates slightly improve total fuel-mix MAE versus A3, but they worsen financial regret versus A2 and A3 and frequently saturate at the derate cap. This suggests the current heuristic derate update is too blunt. The next derate version should either use stronger regularization/monotonic stress calibration or move to a supervised/repair-derived derate target before being considered for promotion.

An A4r regularization test added a mean-derate penalty and L2 shrinkage on derate weights. The selected full test used `thermal_derate_mean_penalty=5.0` and `thermal_derate_l2=0.05`:

```text
A4r regularized derates:
  LtO mean financial cost    = 454,567.752 dollars/hour
  mean regret                = 44,701.554 dollars/hour
  P-63 overall MAE           = 284.790 MW for LtO
  total fuel-mix MAE         = 281.228 MW for LtO
  mean thermal derate        = 0.090
  max thermal derate         = 0.500
```

Regularization reduced the mean derate but did not eliminate cap hits or improve regret. A lower-cap smoke also failed to beat the A3 smoke. This closes the current heuristic-derate branch: learned derates should not be promoted without a better target or monotonic calibration.

## A5 Scheduled Outage Stress Features

A5 returns to the more stable A2/A3 learned-cost formulation and adds P-54A scheduled outage stress as causal context features only. P-54A rows are not converted into hard generator derates or topology constraints because the public records do not provide a complete mapping from outage records to the current zonal LP cut matrix. This preserves the public SCUC proxy claim: the model uses public outage stress evidence, but does not reconstruct exact NYISO SCUC.

A5 is split into two ablations:

```text
A5a = A2 + scheduled outage stress features
A5b = A3 + scheduled outage stress features
```

The scheduled outage feature block includes outage active flags, log-scaled counts, PTID/equipment/345 kV/interface-hint summaries, and 24-hour rolling and lagged stress summaries. These are written to `zonal_cost_lto_scheduled_outage_feature_summary.csv` and marked in `zonal_cost_lto_feature_metadata.csv`.

Full A5 test comparison, 2023-2024, 17,540 evaluated hours, Gold Book capacity, real P-32 interface data, 720 training-hour cap, and 3 structured epochs:

```text
A2 interface baseline:
  LtO mean financial cost    = 445,503.086 dollars/hour
  mean regret                = 35,636.888 dollars/hour
  P-63 overall MAE           = 273.550 MW for LtO
  total fuel-mix MAE         = 276.727 MW for LtO
  mean binding interfaces    = 0.377
  transfer magnitude         = 4,471.521 MW

A3 + P-511A enhanced features:
  LtO mean financial cost    = 449,336.516 dollars/hour
  mean regret                = 39,470.317 dollars/hour
  P-63 overall MAE           = 287.657 MW for LtO
  total fuel-mix MAE         = 266.187 MW for LtO
  mean binding interfaces    = 0.357
  transfer magnitude         = 4,319.463 MW

A5a + scheduled outage stress:
  LtO mean financial cost    = 453,025.070 dollars/hour
  mean regret                = 43,158.872 dollars/hour
  P-63 overall MAE           = 288.865 MW for LtO
  total fuel-mix MAE         = 284.867 MW for LtO
  mean binding interfaces    = 0.188
  transfer magnitude         = 3,754.815 MW

A5b + P-511A + scheduled outage stress:
  LtO mean financial cost    = 442,471.710 dollars/hour
  mean regret                = 32,605.512 dollars/hour
  P-63 overall MAE           = 278.326 MW for LtO
  total fuel-mix MAE         = 286.144 MW for LtO
  mean binding interfaces    = 0.167
  transfer magnitude         = 3,654.159 MW
```

All A5 full runs had 100% solver success, numerical balance residuals, and 0 MW mean interface slack. The financial oracle remains unchanged at 409,866.198 dollars/hour because A5 changes learned context features, not the oracle LP.

Interpretation: scheduled outage stress alone is harmful in A5a. The same outage information becomes useful when combined with P-511A constraint stress in A5b, reducing mean financial regret from 35,637 dollars/hour in A2 to 32,606 dollars/hour. A5b is therefore the current best public SCUC proxy variant on financial regret. The tradeoff is that A5b worsens total fuel-mix MAE versus both A2 and A3, so it should be promoted only for financial-regret-oriented reporting while the next ablation targets the fuel-mix/financial tradeoff.

## A6 Ancillary Stress And Reserve Headroom

A6 tests whether public reserve/ancillary evidence can improve the A5b tradeoff without changing the core learning-to-optimize idea. The model remains a public SCUC proxy: P-6B real-time ancillary service prices are learned context features, and reserve headroom is a soft proxy constraint, not a reconstruction of NYISO reserve procurement.

The A6 variants are:

```text
A6a = A5b + P-6B ancillary stress features
A6b = A5b + soft ramp-accessible reserve/headroom proxy
A6c = A5b + P-6B ancillary stress features + soft ramp-accessible reserve/headroom proxy
```

P-6B is aggregated to hourly stress features from the public 5-minute ancillary service price data: reserve price means/maxes, regulation price means/maxes, stress maxes, scarcity flags, log-scaled values, 24-hour rolling means, and 24-hour lags. These features are written to `zonal_cost_lto_ancillary_feature_summary.csv` and marked in `zonal_cost_lto_feature_metadata.csv`.

The first headroom version using full static fleet capacity was effectively nonbinding, so A6b/A6c use ramp-accessible reserve capacity when ramp limits and previous generation are available. The soft proxy is:

```text
sum_dispatchable_generation_t - reserve_slack_t
  <= ramp_accessible_dispatchable_capacity_t - reserve_requirement_t
```

with dispatchable fuels:

```text
Dual Fuel
Natural Gas
Hydro
Other Fossil Fuels
```

The first headroom test used `reserve_requirement_fraction = 0.04`, i.e. 4% of solver load, and reports reserve requirement, reserve headroom, reserve slack, and reserve shortfall after slack. A follow-up A6c sensitivity also tested 2% and 6%.

Full A5b/A6 test comparison, 2023-2024, 17,540 evaluated hours, Gold Book capacity, real P-32 interface data, 720 training-hour cap, and 3 structured epochs:

```text
A5b reference:
  oracle mean financial cost = 409,866.198 dollars/hour
  LtO mean financial cost    = 442,471.710 dollars/hour
  mean regret                = 32,605.512 dollars/hour
  P-63 overall MAE           = 278.326 MW for LtO
  total fuel-mix MAE         = 286.144 MW for LtO
  mean reserve slack         = 0.000 MW

A6a + P-6B ancillary stress:
  oracle mean financial cost = 409,866.198 dollars/hour
  LtO mean financial cost    = 450,358.535 dollars/hour
  mean regret                = 40,492.337 dollars/hour
  P-63 overall MAE           = 286.611 MW for LtO
  total fuel-mix MAE         = 266.318 MW for LtO
  mean reserve slack         = 0.000 MW

A6b + reserve headroom:
  oracle mean financial cost = 408,484.118 dollars/hour
  LtO mean financial cost    = 445,383.357 dollars/hour
  mean regret                = 36,899.240 dollars/hour
  P-63 overall MAE           = 280.357 MW for LtO
  total fuel-mix MAE         = 269.995 MW for LtO
  mean reserve requirement   = 583.069 MW
  mean reserve headroom      = 1,620.841 MW for LtO
  mean reserve slack         = 0.000 MW

A6c + P-6B ancillary stress + reserve headroom, 4% reserve:
  oracle mean financial cost = 408,484.118 dollars/hour
  LtO mean financial cost    = 440,746.491 dollars/hour
  mean regret                = 32,262.374 dollars/hour
  P-63 overall MAE           = 284.456 MW for LtO
  total fuel-mix MAE         = 267.005 MW for LtO
  mean reserve requirement   = 583.069 MW
  mean reserve headroom      = 1,636.425 MW for LtO
  mean reserve slack         = 0.000 MW
```

All A6 full runs had 100% solver success, numerical balance residuals, 0 MW mean interface slack, and 0 MW reserve slack. A6a by itself is not a financial improvement. A6b by itself improves total fuel-mix MAE but worsens financial cost versus A5b. A6c is the useful combined variant: at 4% reserve it slightly improves financial regret against its reserve-constrained oracle and lowers absolute LtO financial cost by about 1,725 dollars/hour versus A5b while also reducing total fuel-mix MAE from 286.144 MW to 267.005 MW.

A6c reserve-fraction sensitivity:

```text
A6c, 2% reserve:
  LtO mean financial cost    = 440,224.716 dollars/hour
  mean regret                = 30,815.997 dollars/hour
  P-63 overall MAE           = 286.010 MW for LtO
  total fuel-mix MAE         = 260.437 MW for LtO
  mean reserve requirement   = 291.534 MW
  mean reserve headroom      = 1,614.294 MW for LtO
  mean reserve slack         = 0.000 MW

A6c, 4% reserve:
  LtO mean financial cost    = 440,746.491 dollars/hour
  mean regret                = 32,262.374 dollars/hour
  total fuel-mix MAE         = 267.005 MW for LtO

A6c, 6% reserve:
  LtO mean financial cost    = 446,549.328 dollars/hour
  mean regret                = 40,248.188 dollars/hour
  total fuel-mix MAE         = 282.526 MW for LtO
```

Interpretation: public ancillary price stress is useful only when paired with an operational headroom proxy. A6c at 2% reserve is the current best balanced public SCUC proxy candidate: it improves financial cost and total fuel-mix MAE versus A5b, and it beats the 4% and 6% A6c sensitivities. A5b remains the cleanest no-reserve baseline.

## A7 Learned Reserve Penalty

A7a tests whether the A6c reserve/headroom proxy should use a learned reserve slack penalty instead of a fixed penalty. The LP still uses the same public ramp-accessible headroom proxy and a 2% reserve requirement, but the learned transformation predicts an hourly reserve slack penalty:

```text
reserve_slack_penalty_t = clip(base_penalty * exp(h_theta(x_t)), 5,000, 200,000)
```

This remains a public proxy penalty. It is not a claim that the model learns the true NYISO reserve requirement or reserve procurement process.

Initial A7a test, 2023-2024 split, Gold Book capacity, real P-32 interface data, P-511A/P-54A/P-6B stress features, 720 training-hour cap, 7 structured epochs, and 2% reserve requirement:

```text
A6c fixed reserve penalty, 2% reserve, 720h/e7:
  LtO mean financial cost    = 437,597.617 dollars/hour
  mean regret                = 28,188.898 dollars/hour
  P-63 overall MAE           = 286.324 MW for LtO
  total fuel-mix MAE         = 261.193 MW for LtO
  mean reserve slack         = 0.000 MW

A7a learned reserve penalty, 2% reserve, 720h/e7:
  LtO mean financial cost    = 445,486.579 dollars/hour
  mean regret                = 36,077.860 dollars/hour
  P-63 overall MAE           = 278.849 MW for LtO
  total fuel-mix MAE         = 257.676 MW for LtO
  mean reserve slack         = 0.000 MW
  mean learned penalty       = 70,680.286
  p50 learned penalty        = 53,071.616
  p95 learned penalty        = 200,000.000
  fraction at min penalty    = 0.002
  fraction at max penalty    = 0.067
```

Interpretation: A7a improves total fuel-mix MAE versus A6c but worsens financial regret by about 7,889 dollars/hour. The learned penalty often moves upward and reaches the max bound in about 6.7% of hours, while reserve slack remains zero. This indicates that the current penalty-learning signal is not improving the financial LtO objective; it mostly changes dispatch through a more conservative penalty profile. A7a should not replace A6c. The next A7 variant should learn a reserve requirement adjustment or use a supervised/diagnostic target rather than only learning the slack penalty.

A7b tests the alternative reserve lever: learn the effective reserve requirement fraction while keeping the slack penalty fixed. The public proxy constraint remains the same, but the reserve fraction is bounded:

```text
reserve_fraction_t = clip(f_theta(x_t), 0.00, 0.06)
reserve_requirement_t = reserve_fraction_t * solver_load_t
```

The first implementation initializes around the A6c 2% reserve assumption and uses the A6c feature set. It does not claim to infer true NYISO reserve requirements.

Initial A7b test, same 2023-2024 split and training budget as A6c/A7a:

```text
A7b learned reserve fraction, 720h/e7:
  LtO mean financial cost    = 438,678.942 dollars/hour
  mean regret                = 29,270.223 dollars/hour
  P-63 overall MAE           = 277.109 MW for LtO
  total fuel-mix MAE         = 253.969 MW for LtO
  mean reserve slack         = 0.000 MW
  mean reserve fraction      = 0.012
  p50 reserve fraction       = 0.007
  p95 reserve fraction       = 0.043
  fraction at min bound      = 0.000
  fraction at max bound      = 0.000
```

Interpretation: A7b does not beat A6c on financial regret, but it is much better than A7a and gives a strong reserve-aware fuel-mix diagnostic. The learned reserve fraction is interpretable and does not saturate, but it shifts the average reserve requirement below the fixed 2% A6c setting. Keep A6c 2%, 720h/e7 as the best financial/stress-aware candidate. After the A8 tests, keep A7b as an earlier reserve-learning comparison rather than the final secondary realism result.

## A8 Rich Parmentier-Style Transformations

A8 tests whether the learned transformation should modify more of the public LP
instance than zone/fuel costs. The current A8 stages keep the same public SCUC
proxy framing but expose richer levers:

```text
A8a: A6f causal feature path with fixed dataset gates
A8b: P-5 day-ahead ancillary stress with learned reserve fraction and penalty
A8c: A8b plus P-54C day-ahead outage stress and conservative learned derates
```

The dataset gates are currently fixed hyperparameters, not learned coefficients.
They are included to make later data-family weighting explicit:

```text
gate_p511a, gate_p54a, gate_p54c, gate_p5, gate_p59, gate_weather, gate_cost
```

Full 2023-2024 A8 tests, Gold Book capacity, real P-32 interface data, 720
training-hour cap, 7 structured epochs:

```text
A8a causal feature gates, default weights:
  LtO mean financial cost    = 460,283.290 dollars/hour
  mean regret                = 50,874.571 dollars/hour
  P-63 overall MAE           = 284.288 MW for LtO
  total fuel-mix MAE         = 279.768 MW for LtO

A8b P-5 learned reserve terms:
  LtO mean financial cost    = 458,106.225 dollars/hour
  mean regret                = 48,697.506 dollars/hour
  P-63 overall MAE           = 272.203 MW for LtO
  total fuel-mix MAE         = 250.437 MW for LtO
  mean reserve fraction      = 0.013
  p50 reserve fraction       = 0.009

A8c P-54C learned derates:
  LtO mean financial cost    = 463,113.522 dollars/hour
  mean regret                = 53,704.803 dollars/hour
  P-63 overall MAE           = 288.646 MW for LtO
  total fuel-mix MAE         = 267.702 MW for LtO
  mean thermal derate        = 0.036
  max thermal derate         = 0.200
```

Interpretation: A8b supports the richer-transformation idea because learning
reserve terms from cleaner P-5 day-ahead ancillary stress gives the best
observed P-63 fuel-mix realism so far. It does not beat A6c on financial regret,
so it should be reported as a secondary realism result, not the primary model.
A8c shows that derate learning remains unstable even with conservative caps:
the max derate reaches the 20% cap and financial regret worsens. The current
final hierarchy is A6c for financial regret, A8b for secondary fuel realism,
A5b as the clean no-reserve control, and A7b as an earlier reserve-learning
comparison.

## Task 9 Residual-Aware LtO

The first richer LtO parameterization is system-level and residual-aware. Instead of predicting only fuel-level generation costs, the neural model predicts separate costs for upward and downward dispatch adjustments:

```text
c_up_f,t, c_down_f,t = h_theta(X_t, g_t-1)
```

The LP then chooses:

```text
g_t = g_t-1 + u_t - d_t
```

with capacity, empirical ramping, and soft internal-generation balance constraints. The objective is:

```text
minimize sum_f c_up_f,t u_f,t
       + sum_f c_down_f,t d_f,t
       + p_balance (balance_under_t + balance_over_t)
```

Training currently uses a Fenchel-Young style gradient against the projected P-63 oracle in adjustment space:

```text
dL / dc_up   = u_projected_oracle - u_pred
dL / dc_down = d_projected_oracle - d_pred
```

This is closer to the dispatch process than the older fuel-cost LP because it learns the cost of changing the system state rather than the cost of absolute generation from scratch. The January 2024 smoke result is promising:

```text
direct residual NN with weather MAE             = 62.263 MW
soft-balance repair with weather p=0.1 MAE      = 61.999 MW
residual-aware FY LtO with weather MAE          = 58.908 MW
```

However, the residual-aware LtO row had worse total fuel-mix MAE than soft repair in the same smoke run:

```text
soft-balance repair with weather total MAE      = 237.903 MW
residual-aware FY LtO with weather total MAE    = 293.647 MW
```

So this model should not be promoted yet. The next check is a year-aligned 2020-2025 run and a balance-penalty sensitivity test.

## Current Exclusions and Future Decisions

Not included in v1:

```text
unit commitment
minimum up/down times
generator-specific ramp rates
reserve products
DC power flow
zonal dispatch targets
generator/PTID outage derates
differentiable repair/QP layer
full zone-fuel learned costs
calibrated soft balance in repair LP
partial 2026 final metrics
```

Future work should prioritize zonal balance and interface constraints before attempting unit-level commitment, because the current target is fuel-category system dispatch rather than generator-level output.
