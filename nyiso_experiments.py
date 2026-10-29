from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from pandas.api.types import is_datetime64tz_dtype
from scipy.optimize import linprog
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

from nyiso_processing import (
    ANCILLARY_STRESS_FEATURES,
    DA_ANCILLARY_STRESS_FEATURES,
    FUEL_CATEGORIES,
    FUEL_TARGET_COLUMNS,
    OUTAGE_STRESS_FEATURES,
    ZONE_CODE_TO_NAME,
    ZONAL_BID_LOAD_FEATURES,
    add_ancillary_stress_features,
    add_causal_lag_features,
    add_constraint_stress_features,
    add_day_ahead_ancillary_stress_features,
    add_outage_stress_features,
    add_zonal_bid_load_features,
)

TARGET_COLUMNS = [FUEL_TARGET_COLUMNS[fuel] for fuel in FUEL_CATEGORIES]
ZONE_CODES = sorted(ZONE_CODE_TO_NAME)
THERMAL_DERATE_FUELS = ["Dual Fuel", "Natural Gas", "Other Fossil Fuels"]
THERMAL_DERATE_FUEL_INDICES = [FUEL_CATEGORIES.index(fuel) for fuel in THERMAL_DERATE_FUELS]
RESERVE_HEADROOM_FUELS = ["Dual Fuel", "Natural Gas", "Hydro", "Other Fossil Fuels"]
RESERVE_HEADROOM_FUEL_INDICES = [FUEL_CATEGORIES.index(fuel) for fuel in RESERVE_HEADROOM_FUELS]

INTERNAL_INTERFACE_CUTS = {
    "DYSINGER EAST": ["A", "B"],
    "MOSES SOUTH": ["D"],
    "CENTRAL EAST - VC": ["A", "B", "C", "D", "E"],
    "TOTAL EAST": ["A", "B", "C", "D", "E"],
    "UPNY CONED": ["A", "B", "C", "D", "E", "F"],
    "SPR/DUN-SOUTH": ["A", "B", "C", "D", "E", "F", "G", "H"],
}

DIAGNOSTIC_FEATURE_COLUMNS = [
    "actual_load_mw",
    "forecast_load_mw",
    "load_forecast_error_mw",
    "da_lbmp_mean",
    "da_lbmp_max",
    "da_congestion_mean",
    "rt_lbmp_mean",
    "rt_lbmp_max",
    "rt_congestion_mean",
    "interface_stress_mean",
    "interface_stress_max",
    "interface_stressed_count",
    "rt_constraint_count",
    "rt_constraint_cost_mean",
    "rt_constraint_abs_cost_max",
    "da_constraint_count",
    "da_constraint_cost_mean",
    "da_constraint_abs_cost_max",
    "scheduled_outage_count",
    "realtime_outage_count",
    "hour",
    "dayofweek",
    "month",
    "is_weekend",
    "sin_hour",
    "cos_hour",
    "sin_dow",
    "cos_dow",
    "sin_month",
    "cos_month",
]

CAUSAL_DAY_AHEAD_FEATURE_COLUMNS = [
    "forecast_load_mw",
    "da_lbmp_mean",
    "da_lbmp_max",
    "da_congestion_mean",
    "da_constraint_count",
    "da_constraint_cost_mean",
    "da_constraint_abs_cost_max",
    "scheduled_outage_count",
    "scheduled_outage_ptid_count",
    "scheduled_outage_equipment_count",
    "scheduled_outage_345kv_count",
    "scheduled_outage_by_interface_hint",
    "hour",
    "dayofweek",
    "month",
    "is_weekend",
    "sin_hour",
    "cos_hour",
    "sin_dow",
    "cos_dow",
    "sin_month",
    "cos_month",
    "actual_load_mw_lag_1",
    "actual_load_mw_lag_24",
    "rt_lbmp_mean_lag_1",
    "rt_lbmp_mean_lag_24",
    "rt_lbmp_max_lag_1",
    "rt_lbmp_max_lag_24",
    "rt_congestion_mean_lag_1",
    "rt_congestion_mean_lag_24",
    "rt_constraint_count_lag_1",
    "rt_constraint_count_lag_24",
    "rt_constraint_cost_mean_lag_1",
    "rt_constraint_cost_mean_lag_24",
    "rt_constraint_abs_cost_max_lag_1",
    "rt_constraint_abs_cost_max_lag_24",
    *[f"{col}_lag_1" for col in TARGET_COLUMNS],
    *[f"{col}_lag_24" for col in TARGET_COLUMNS],
]

BASE_FEATURE_COLUMNS = DIAGNOSTIC_FEATURE_COLUMNS

WEATHER_FEATURE_COLUMNS = [
    "system_wind_speed_50m_mps",
    "system_wind_speed_cubed",
    "system_ghi_wh_m2",
    "system_ghi_rolling_3h",
    "system_temp_2m_c",
    "system_temp_rolling_24h",
]

STATIC_CAPACITY_COLUMNS = [
    "static_capacity_dual_fuel_mw",
    "static_capacity_natural_gas_mw",
    "static_capacity_hydro_mw",
    "static_capacity_nuclear_mw",
    "static_capacity_wind_mw",
    "static_capacity_other_renewables_mw",
    "static_capacity_other_fossil_fuels_mw",
]

COST_FEATURE_COLUMNS = [
    "fuel_price_natural_gas_dollars_per_mmbtu",
    "fuel_price_petroleum_liquids_dollars_per_mmbtu",
    "financial_cost_dual_fuel_usd_per_mwh",
    "financial_cost_natural_gas_usd_per_mwh",
    "financial_cost_hydro_usd_per_mwh",
    "financial_cost_nuclear_usd_per_mwh",
    "financial_cost_wind_usd_per_mwh",
    "financial_cost_other_renewables_usd_per_mwh",
    "financial_cost_other_fossil_fuels_usd_per_mwh",
    "p27_dam_dispatch_price_mean",
    "p27_dam_dispatch_price_curve_max_mean",
    "p27_dam_uol_sum",
    "p27_dam_fixed_min_gen_cost_mean",
    "p27_uc_startup_cost_median",
    "p27_uc_min_run_time_median",
]

DA_CONSTRAINT_STRESS_FEATURE_COLUMNS = [
    "da_constraint_active",
    "da_constraint_positive_count",
    "da_constraint_negative_count",
    "da_constraint_cost_positive_mean",
    "da_constraint_cost_negative_mean",
    "da_constraint_cost_abs_mean",
    "da_constraint_count_log1p",
    "da_constraint_abs_cost_max_log1p",
    "da_constraint_abs_cost_mean_log1p",
    "da_constraint_count_rolling_24h_mean",
    "da_constraint_abs_cost_max_rolling_24h_mean",
    "da_constraint_count_lag_24",
    "da_constraint_abs_cost_max_lag_24",
]

SCHEDULED_OUTAGE_STRESS_FEATURE_COLUMNS = [
    "scheduled_outage_active",
    "scheduled_outage_count_log1p",
    "scheduled_outage_ptid_count_log1p",
    "scheduled_outage_equipment_count_log1p",
    "scheduled_outage_345kv_count_log1p",
    "scheduled_outage_by_interface_hint_log1p",
    "scheduled_outage_count_rolling_24h_mean",
    "scheduled_outage_ptid_count_rolling_24h_mean",
    "scheduled_outage_equipment_count_rolling_24h_mean",
    "scheduled_outage_345kv_count_rolling_24h_mean",
    "scheduled_outage_by_interface_hint_rolling_24h_mean",
    "scheduled_outage_count_lag_24",
    "scheduled_outage_ptid_count_lag_24",
    "scheduled_outage_equipment_count_lag_24",
    "scheduled_outage_345kv_count_lag_24",
    "scheduled_outage_by_interface_hint_lag_24",
]

ANCILLARY_STRESS_FEATURE_COLUMNS = ANCILLARY_STRESS_FEATURES.copy()
DAY_AHEAD_ANCILLARY_STRESS_FEATURE_COLUMNS = DA_ANCILLARY_STRESS_FEATURES.copy()
DAY_AHEAD_SCHEDULED_OUTAGE_STRESS_FEATURE_COLUMNS = [
    f"da_scheduled_{feature}" for feature in OUTAGE_STRESS_FEATURES
]
ZONAL_BID_LOAD_FEATURE_COLUMNS = ZONAL_BID_LOAD_FEATURES.copy()

FINANCIAL_COST_COLUMNS_BY_FUEL = {
    "Dual Fuel": "financial_cost_dual_fuel_usd_per_mwh",
    "Natural Gas": "financial_cost_natural_gas_usd_per_mwh",
    "Hydro": "financial_cost_hydro_usd_per_mwh",
    "Nuclear": "financial_cost_nuclear_usd_per_mwh",
    "Wind": "financial_cost_wind_usd_per_mwh",
    "Other Renewables": "financial_cost_other_renewables_usd_per_mwh",
    "Other Fossil Fuels": "financial_cost_other_fossil_fuels_usd_per_mwh",
}

AVAILABILITY_COLUMNS_BY_FUEL = {
    "Wind": "wind_available_mw",
    "Other Renewables": "other_renewables_available_mw",
    "Hydro": "hydro_available_mw",
    "Nuclear": "nuclear_available_mw",
}

SOLVER_BALANCE_COLUMN = "solver_balance_mw"
HOUR_OF_WEEK_BALANCE_COLUMN = "solver_balance_hour_of_week_mw"
LAG1_BALANCE_COLUMN = "solver_balance_lag_1_residual_mw"

FIXED_FUEL_COSTS = np.array([55.0, 45.0, 12.0, 8.0, 1.0, 4.0, 80.0], dtype=np.float64)

COMPACT_METRIC_COLUMNS = [
    "model",
    "overall_mae",
    "total_fuel_mix_mae",
    "renewable_mae",
    "thermal_mae",
    "mae_wind",
    "mean_abs_balance_residual",
    "max_ramp_violation_mw",
    "mean_interface_slack_mw",
    "mean_transfer_magnitude_mw",
    "mean_runtime_seconds",
]

COMPACT_DIAGNOSTIC_COLUMNS = [
    "model",
    "rows",
    "success_rate",
    "mean_abs_balance_residual",
    "max_ramp_violation_mw",
    "mean_interface_slack_mw",
    "mean_transfer_magnitude_mw",
    "mean_runtime_seconds",
]

COMPACT_FEASIBILITY_COLUMNS = [
    "variant",
    "overall_mae_to_oracle",
    "projection_l1_mw",
    "projection_l2_mw",
    "total_generation_gap_mw",
    "success_rate",
]

COMPACT_BALANCE_TARGET_COLUMNS = [
    "target",
    "causal_policy",
    "abs_error_mw",
    "rmse_mw",
    "bias_mw",
    "repair_overall_mae",
    "repair_total_fuel_mix_mae",
    "success_rate",
]

COMPACT_LOSS_EXPERIMENT_COLUMNS = [
    "model",
    "overall_mae",
    "total_fuel_mix_mae",
    "solution_l1_to_projected_oracle",
    "solution_l2_to_projected_oracle",
    "fenchel_young_loss",
    "objective_gap_under_learned_cost",
    "balance_target_abs_error_mw",
    "success_rate",
    "mean_abs_balance_residual",
    "max_ramp_violation_mw",
    "mean_runtime_seconds",
]

COMPACT_REPAIR_COMPARISON_COLUMNS = [
    "model",
    "overall_mae",
    "total_fuel_mix_mae",
    "mean_abs_repair_delta_mw",
    "mean_abs_balance_deviation_mw",
    "max_ramp_violation_mw",
    "mean_runtime_seconds",
]

COMPACT_ZONAL_DIAGNOSTIC_COLUMNS = [
    "variant",
    "overall_mae",
    "total_fuel_mix_mae",
    "renewable_mae",
    "thermal_mae",
    "mae_wind",
    "mae_hydro",
    "mae_natural_gas",
    "mean_residual_supply_mw",
    "mean_interface_slack_mw",
    "mean_transfer_magnitude_mw",
    "max_zonal_balance_residual",
    "success_rate",
    "mean_runtime_seconds",
]

ORACLE_TOTAL_COLUMN = "oracle_total_balance_mw"
LOSS_EXPERIMENT_BALANCE_COLUMN = "loss_experiment_balance_mw"

@dataclass
class ExperimentConfig:
    history_window_hours: int = 24
    history_window: int | None = None
    feature_mode: str = "causal_day_ahead"
    load_balance_mode: str = "internal_generation"
    balance_target_policy: str = "lag_1_residual"
    solver_load_column: str = "forecast_load_mw"
    use_ramp_constraints: bool = True
    ramp_quantile: float = 0.99
    direct_epochs: int = 80
    structured_epochs: int = 3
    structured_lr: float = 0.03
    structured_loss: str = "candidate_softmax"
    repair_mode: str = "hard"
    soft_balance_penalty_grid: list[float] = field(default_factory=lambda: [0.1, 1.0, 5.0, 10.0, 50.0, 100.0])
    hidden_dim: int = 48
    candidate_count: int = 5
    candidate_noise_scale: float = 0.35
    temperature: float = 1.0
    ens_penalty: float = 1e6
    interface_penalty: float = 1e5
    transfer_penalty: float = 5.0
    flow_match_penalty: float = 0.0
    use_zonal_interface_constraints: bool = True
    use_zonal_availability_bounds: bool = True
    residual_penalty: float = 1e3
    spill_penalty: float = 1.0
    seed: int = 42
    include_lags: bool = True
    experiment_profile: str = "current_research"
    display_metric_profile: str = "compact"
    run_antiquated_pipelines: bool = False
    run_learned_cost_grid: bool = False
    run_no_lag_direct_models: bool = False
    run_system_fixed_lp: bool = False
    include_cost_features: bool = False
    zonal_capacity_source: str = "legacy"
    include_da_constraint_stress_features: bool = False
    include_scheduled_outage_stress_features: bool = False
    include_ancillary_stress_features: bool = False
    include_day_ahead_ancillary_stress_features: bool = False
    include_day_ahead_scheduled_outage_stress_features: bool = False
    include_zonal_bid_load_features: bool = False
    gate_p511a: float = 1.0
    gate_p54a: float = 1.0
    gate_p54c: float = 1.0
    gate_p5: float = 1.0
    gate_p59: float = 1.0
    gate_weather: float = 1.0
    gate_cost: float = 1.0
    use_reserve_headroom_constraint: bool = False
    reserve_requirement_fraction: float = 0.04
    reserve_slack_penalty: float = 5e4
    use_learned_reserve_penalty: bool = False
    reserve_penalty_min: float = 5e3
    reserve_penalty_max: float = 2e5
    reserve_penalty_lr: float = 0.02
    reserve_penalty_l2: float = 0.0
    use_learned_reserve_fraction: bool = False
    reserve_fraction_min: float = 0.0
    reserve_fraction_max: float = 0.06
    reserve_fraction_lr: float = 0.01
    reserve_fraction_l2: float = 0.001
    use_learned_thermal_derates: bool = False
    thermal_derate_max: float = 0.5
    thermal_derate_offset: float = 4.0
    thermal_derate_lr: float = 0.05
    thermal_derate_l2: float = 0.0
    thermal_derate_mean_penalty: float = 0.0

    @property
    def resolved_history_window(self) -> int:
        return self.history_window if self.history_window is not None else self.history_window_hours
@dataclass(frozen=True)
class RampLimits:
    up: np.ndarray
    down: np.ndarray
    quantile: float
@dataclass
class DispatchResult:
    generation: np.ndarray
    residual_supply: float
    ens: float
    spill: float
    objective: float
    success: bool
    runtime_seconds: float
    max_ramp_violation_mw: float = 0.0
    balance_under: float = 0.0
    balance_over: float = 0.0
@dataclass
class ZonalDispatchResult:
    generation_by_zone: np.ndarray
    net_injection: np.ndarray
    residual_supply: np.ndarray
    ens: np.ndarray
    spill: np.ndarray
    interface_slack_pos: np.ndarray
    interface_slack_neg: np.ndarray
    interface_flows: np.ndarray
    objective: float
    success: bool
    runtime_seconds: float
    max_zonal_balance_residual: float
    system_net_injection_residual: float
    max_interface_violation_after_slack: float
    max_ramp_violation_mw: float
    transfer_magnitude_mw: float
    reserve_requirement_mw: float = 0.0
    reserve_headroom_mw: float = 0.0
    reserve_slack_mw: float = 0.0

def feature_columns(
    use_weather: bool,
    feature_mode: str = "causal_day_ahead",
    include_lag=True,
    include_cost_features: bool = False,
    include_da_constraint_stress_features: bool = False,
    include_scheduled_outage_stress_features: bool = False,
    include_ancillary_stress_features: bool = False,
    include_day_ahead_ancillary_stress_features: bool = False,
    include_day_ahead_scheduled_outage_stress_features: bool = False,
    include_zonal_bid_load_features: bool = False,
) -> list[str]:
    if feature_mode == "causal_day_ahead":
        cols = CAUSAL_DAY_AHEAD_FEATURE_COLUMNS.copy()
        if not include_lag:
            cols = [col for col in cols if "_lag_" not in col]
    elif feature_mode == "diagnostic_ex_post":
        cols = DIAGNOSTIC_FEATURE_COLUMNS.copy()
    else:
        raise ValueError(f"Unknown feature mode: {feature_mode}")
    if use_weather:
        cols += WEATHER_FEATURE_COLUMNS
    cols += STATIC_CAPACITY_COLUMNS
    if include_da_constraint_stress_features:
        cols += DA_CONSTRAINT_STRESS_FEATURE_COLUMNS
    if include_scheduled_outage_stress_features:
        cols += SCHEDULED_OUTAGE_STRESS_FEATURE_COLUMNS
    if include_ancillary_stress_features:
        cols += ANCILLARY_STRESS_FEATURE_COLUMNS
    if include_day_ahead_ancillary_stress_features:
        cols += DAY_AHEAD_ANCILLARY_STRESS_FEATURE_COLUMNS
    if include_day_ahead_scheduled_outage_stress_features:
        cols += DAY_AHEAD_SCHEDULED_OUTAGE_STRESS_FEATURE_COLUMNS
    if include_zonal_bid_load_features:
        cols += ZONAL_BID_LOAD_FEATURE_COLUMNS
    if include_cost_features:
        cols += COST_FEATURE_COLUMNS
    return cols

def validate_feature_policy(cols: list[str], feature_mode: str) -> None:
    if feature_mode != "causal_day_ahead":
        return
    forbidden = {
        "actual_load_mw",
        "load_forecast_error_mw",
        "rt_lbmp_mean",
        "rt_lbmp_max",
        "rt_congestion_mean",
        "rt_constraint_count",
        "rt_constraint_cost_mean",
        "rt_constraint_abs_cost_max",
        "realtime_outage_count",
    }
    leaked = sorted(set(cols).intersection(forbidden))
    if leaked:
        raise ValueError(f"Causal feature set contains same-hour ex-post columns: {leaked}")

def solver_load_column(config: ExperimentConfig) -> str:
    if config.load_balance_mode == "forecast_load":
        return config.solver_load_column
    if config.load_balance_mode == "internal_generation":
        return SOLVER_BALANCE_COLUMN
    raise ValueError(f"Unknown load balance mode: {config.load_balance_mode}")


def resolve_experiment_config(config: ExperimentConfig) -> ExperimentConfig:
    if config.experiment_profile == "current_research":
        return config
    if config.experiment_profile in {"full", "all"}:
        return replace(
            config,
            run_antiquated_pipelines=True,
            run_learned_cost_grid=True,
            run_no_lag_direct_models=True,
            run_system_fixed_lp=True,
        )
    raise ValueError(f"Unknown experiment profile: {config.experiment_profile}")


def add_solver_balance_targets(
    train_df: pd.DataFrame,
    *frames: pd.DataFrame,
    balance_target_policy: str = "lag_1_residual",
) -> tuple[pd.DataFrame, ...]:
    if balance_target_policy not in {"lag_1_residual", "hour_of_week_median"}:
        raise ValueError(f"Unknown balance target policy: {balance_target_policy}")
    train = train_df.copy()
    if "forecast_net_import_residual_mw" not in train.columns:
        train["forecast_net_import_residual_mw"] = train["forecast_load_mw"] - train[TARGET_COLUMNS].sum(axis=1)
    train["hour_of_week"] = train["dayofweek"] * 24 + train["hour"]
    residual_medians = train.groupby("hour_of_week")["forecast_net_import_residual_mw"].median()
    global_residual = float(train["forecast_net_import_residual_mw"].median())

    lag_source_parts = []
    for frame in frames:
        part = frame[["timestamp_utc", "forecast_load_mw", *TARGET_COLUMNS]].copy()
        if "forecast_net_import_residual_mw" in frame.columns:
            part["forecast_net_import_residual_mw"] = frame["forecast_net_import_residual_mw"].to_numpy(dtype=np.float64)
        else:
            part["forecast_net_import_residual_mw"] = part["forecast_load_mw"] - part[TARGET_COLUMNS].sum(axis=1)
        lag_source_parts.append(part[["timestamp_utc", "forecast_net_import_residual_mw"]])
    lag_source = (
        pd.concat(lag_source_parts, ignore_index=True)
        .drop_duplicates("timestamp_utc")
        .sort_values("timestamp_utc")
        .reset_index(drop=True)
    )
    lag_source["forecast_net_import_residual_mw_lag_1"] = lag_source["forecast_net_import_residual_mw"].shift(1)
    lag_residual_by_timestamp = lag_source.set_index("timestamp_utc")["forecast_net_import_residual_mw_lag_1"]

    out = []
    for frame in frames:
        df = frame.copy()
        if "forecast_net_import_residual_mw" not in df.columns:
            df["forecast_net_import_residual_mw"] = df["forecast_load_mw"] - df[TARGET_COLUMNS].sum(axis=1)
        df["hour_of_week"] = df["dayofweek"] * 24 + df["hour"]
        hour_of_week_residual = (
            df["hour_of_week"].map(residual_medians).fillna(global_residual).astype(float)
        )
        df["forecast_net_import_residual_mw_lag_1"] = pd.to_datetime(df["timestamp_utc"], utc=True).map(
            lag_residual_by_timestamp
        )
        lag_1_residual = df["forecast_net_import_residual_mw_lag_1"].fillna(hour_of_week_residual).astype(float)
        df[HOUR_OF_WEEK_BALANCE_COLUMN] = (df["forecast_load_mw"] - hour_of_week_residual).clip(lower=0.0)
        df[LAG1_BALANCE_COLUMN] = (df["forecast_load_mw"] - lag_1_residual).clip(lower=0.0)
        if balance_target_policy == "lag_1_residual":
            df["predicted_net_import_residual_mw"] = lag_1_residual
            df[SOLVER_BALANCE_COLUMN] = df[LAG1_BALANCE_COLUMN]
        else:
            df["predicted_net_import_residual_mw"] = hour_of_week_residual
            df[SOLVER_BALANCE_COLUMN] = df[HOUR_OF_WEEK_BALANCE_COLUMN]
        df = df.drop(columns=["hour_of_week"])
        out.append(df)
    return tuple(out)

def load_system_hour(path: str | Path = "data/processed/nyiso_system_hour.parquet") -> pd.DataFrame:
    df = pd.read_parquet(path)
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True)
    if not is_datetime64tz_dtype(df["timestamp_local"]):
        df["timestamp_local"] = pd.to_datetime(df["timestamp_local"], utc=True).dt.tz_convert("America/New_York")
    df = df.sort_values("timestamp_utc").reset_index(drop=True)
    if "target_internal_generation_mw" not in df.columns:
        df["target_internal_generation_mw"] = df[TARGET_COLUMNS].sum(axis=1)
    if "forecast_net_import_residual_mw" not in df.columns:
        df["forecast_net_import_residual_mw"] = df["forecast_load_mw"] - df["target_internal_generation_mw"]
    if "actual_net_import_residual_mw" not in df.columns and "actual_load_mw" in df.columns:
        df["actual_net_import_residual_mw"] = df["actual_load_mw"] - df["target_internal_generation_mw"]
    df = add_constraint_stress_features(df, "da")
    df = add_outage_stress_features(df, "scheduled")
    df = add_outage_stress_features(df, "da_scheduled")
    df = add_ancillary_stress_features(df)
    df = add_day_ahead_ancillary_stress_features(df)
    df = add_zonal_bid_load_features(df)
    for col in [
        "scheduled_outage_ptid_count",
        "scheduled_outage_equipment_count",
        "scheduled_outage_345kv_count",
        "scheduled_outage_by_interface_hint",
    ]:
        if col not in df.columns:
            df[col] = 0.0
    required_lag_cols = [f"{col}_lag_{lag}" for col in ["actual_load_mw", *TARGET_COLUMNS] for lag in [1, 24]]
    if any(col not in df.columns for col in required_lag_cols):
        df = add_causal_lag_features(df)
    return df

def load_zone_hour(path: str | Path = "data/processed/nyiso_zone_hour.parquet") -> pd.DataFrame:
    df = pd.read_parquet(path)
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True)
    if "timestamp_local" in df.columns and not is_datetime64tz_dtype(df["timestamp_local"]):
        df["timestamp_local"] = pd.to_datetime(df["timestamp_local"], utc=True).dt.tz_convert("America/New_York")
    return df.sort_values(["timestamp_utc", "zone_code"]).reset_index(drop=True)

def load_interface_hour(path: str | Path = "data/processed/nyiso_interface_hour.parquet") -> pd.DataFrame:
    df = pd.read_parquet(path)
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True)
    return df.sort_values(["timestamp_utc", "interface_name"]).reset_index(drop=True)

def empty_interface_hour() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "timestamp_utc",
            "interface_name",
            "point_id",
            "flow_mw_mean",
            "flow_mw_min",
            "flow_mw_max",
            "upper_limit_mw",
            "lower_limit_mw",
            "upper_is_finite",
            "lower_is_finite",
            "active_side_sentinel_count",
            "stress_mean",
            "stress_max",
        ]
    )

def require_zonal_interface_hour(interface_hour: pd.DataFrame, *, context: str) -> None:
    if interface_hour.empty or "timestamp_utc" not in interface_hour.columns:
        raise ValueError(
            f"{context} requires non-empty nyiso_interface_hour data because "
            "use_zonal_interface_constraints=True."
        )
    finite_rows = int((interface_hour["upper_is_finite"] | interface_hour["lower_is_finite"]).sum())
    if finite_rows == 0:
        raise ValueError(
            f"{context} received nyiso_interface_hour data with no finite P-32 limits; "
            "interface-constrained diagnostics would be meaningless."
        )

def load_zonal_capacity(path: str | Path = "data/processed/nyiso_zonal_capacity.parquet") -> pd.DataFrame:
    return pd.read_parquet(path).sort_values(["zone_code", "fuel_category"]).reset_index(drop=True)


def _cost_month(series: pd.Series) -> pd.Series:
    timestamp = pd.to_datetime(series, utc=True)
    local_naive = timestamp.dt.tz_convert("America/New_York").dt.tz_localize(None)
    return local_naive.dt.to_period("M").dt.to_timestamp()


def _atb_value(curated: pd.DataFrame, assumption_name: str, parameter: str, default: float) -> float:
    mask = curated["assumption_name"].eq(assumption_name) & curated["core_metric_parameter"].eq(parameter)
    if mask.any():
        value = pd.to_numeric(curated.loc[mask, "value"], errors="coerce").dropna()
        if not value.empty:
            return float(value.iloc[0])
    return float(default)


def _fuel_price_series(
    fossil_prices: pd.DataFrame,
    eia923_costs: pd.DataFrame,
    months: pd.DatetimeIndex,
    *,
    fossil_fuel_name: str,
    eia923_fuel_group: str,
) -> pd.Series:
    fallback = (
        fossil_prices[fossil_prices["fuel"].eq(fossil_fuel_name)]
        .set_index("month")["price_dollars_per_mmbtu"]
        .reindex(months)
    )
    if eia923_costs.empty:
        combined = fallback
    else:
        receipts = (
            eia923_costs[eia923_costs["fuel_group"].eq(eia923_fuel_group)]
            .set_index("month")["eia923_weighted_cost_dollars_per_mmbtu"]
            .reindex(months)
        )
        combined = receipts.combine_first(fallback)
    combined = combined.astype(float).ffill().bfill()
    return combined


def build_monthly_financial_fuel_costs(
    cost_input_dir: str | Path = "data/processed/cost_inputs",
) -> pd.DataFrame:
    cost_dir = Path(cost_input_dir)
    fossil = pd.read_parquet(cost_dir / "eia_fossil_monthly_prices.parquet").copy()
    eia923_path = cost_dir / "eia923_monthly_fuel_cost.parquet"
    eia923 = pd.read_parquet(eia923_path).copy() if eia923_path.exists() else pd.DataFrame()
    atb = pd.read_parquet(cost_dir / "atb_curated_dispatch_assumptions.parquet").copy()

    fossil["month"] = pd.to_datetime(fossil["month"])
    if not eia923.empty:
        eia923["month"] = pd.to_datetime(eia923["month"])
    min_month = fossil["month"].min()
    max_month = fossil["month"].max()
    if not eia923.empty:
        min_month = min(min_month, eia923["month"].min())
        max_month = max(max_month, eia923["month"].max())
    months = pd.date_range(min_month, max_month, freq="MS")

    gas_price = _fuel_price_series(
        fossil,
        eia923,
        months,
        fossil_fuel_name="natural_gas",
        eia923_fuel_group="Natural Gas",
    )
    petroleum_price = _fuel_price_series(
        fossil,
        eia923,
        months,
        fossil_fuel_name="petroleum_liquids",
        eia923_fuel_group="Petroleum",
    )

    gas_cc_heat_rate = _atb_value(atb, "natural_gas_combined_cycle", "Heat Rate", 6.8)
    gas_ct_heat_rate = _atb_value(atb, "natural_gas_combustion_turbine", "Heat Rate", 10.0)
    gas_cc_vom = _atb_value(atb, "natural_gas_combined_cycle", "Variable O&M", 2.0)
    gas_ct_vom = _atb_value(atb, "natural_gas_combustion_turbine", "Variable O&M", 7.0)
    gas_blend_heat_rate = 0.70 * gas_cc_heat_rate + 0.30 * gas_ct_heat_rate
    gas_blend_vom = 0.70 * gas_cc_vom + 0.30 * gas_ct_vom

    nuclear_vom = _atb_value(atb, "nuclear_large", "Variable O&M", 3.0)
    nuclear_fuel = _atb_value(atb, "nuclear_large", "Fuel", 10.0)
    hydro_vom = _atb_value(atb, "pumped_storage_hydro", "Variable O&M", 1.0)
    biopower_vom = _atb_value(atb, "biopower_dedicated", "Variable O&M", 5.0)
    biopower_fuel = _atb_value(atb, "biopower_dedicated", "Fuel", 70.0)

    oil_heat_rate = gas_ct_heat_rate
    oil_vom = gas_ct_vom
    gas_cost = gas_price * gas_blend_heat_rate + gas_blend_vom
    oil_cost = petroleum_price * oil_heat_rate + oil_vom

    out = pd.DataFrame({"month": months})
    out["fuel_price_natural_gas_dollars_per_mmbtu"] = gas_price.to_numpy(dtype=np.float64)
    out["fuel_price_petroleum_liquids_dollars_per_mmbtu"] = petroleum_price.to_numpy(dtype=np.float64)
    out["financial_cost_dual_fuel_usd_per_mwh"] = 0.70 * gas_cost.to_numpy(dtype=np.float64) + 0.30 * oil_cost.to_numpy(dtype=np.float64)
    out["financial_cost_natural_gas_usd_per_mwh"] = gas_cost.to_numpy(dtype=np.float64)
    out["financial_cost_hydro_usd_per_mwh"] = hydro_vom
    out["financial_cost_nuclear_usd_per_mwh"] = nuclear_vom + nuclear_fuel
    out["financial_cost_wind_usd_per_mwh"] = 0.0
    out["financial_cost_other_renewables_usd_per_mwh"] = biopower_vom + biopower_fuel
    out["financial_cost_other_fossil_fuels_usd_per_mwh"] = oil_cost.to_numpy(dtype=np.float64)
    return out


def add_financial_cost_features(
    system_df: pd.DataFrame,
    *,
    cost_input_dir: str | Path = "data/processed/cost_inputs",
) -> pd.DataFrame:
    df = system_df.copy()
    monthly_costs = build_monthly_financial_fuel_costs(cost_input_dir)
    df["cost_month"] = _cost_month(df["timestamp_utc"])
    df = df.merge(monthly_costs.rename(columns={"month": "cost_month"}), on="cost_month", how="left")
    for col in [*COST_FEATURE_COLUMNS, *FINANCIAL_COST_COLUMNS_BY_FUEL.values()]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    cost_dir = Path(cost_input_dir)
    p27_path = cost_dir / "p27_gen_bid_hour.parquet"
    if p27_path.exists():
        p27 = pd.read_parquet(p27_path)
        p27["timestamp_utc"] = pd.to_datetime(p27["timestamp_utc"], utc=True)
        p27_dam = p27[p27["market"].astype(str).str.upper().eq("DAM")].copy()
        keep = [
            "timestamp_utc",
            "dispatch_price_mean",
            "dispatch_price_curve_max_mean",
            "uol_sum",
            "fixed_min_gen_cost_mean",
        ]
        p27_dam = p27_dam[[col for col in keep if col in p27_dam.columns]].rename(
            columns={
                "dispatch_price_mean": "p27_dam_dispatch_price_mean",
                "dispatch_price_curve_max_mean": "p27_dam_dispatch_price_curve_max_mean",
                "uol_sum": "p27_dam_uol_sum",
                "fixed_min_gen_cost_mean": "p27_dam_fixed_min_gen_cost_mean",
            }
        )
        df = df.merge(p27_dam, on="timestamp_utc", how="left")

    p27_uc_path = cost_dir / "p27_uc_month.parquet"
    if p27_uc_path.exists():
        uc = pd.read_parquet(p27_uc_path)
        uc["cost_month"] = pd.to_datetime(uc["month"]).dt.to_period("M").dt.to_timestamp()
        uc = uc.rename(
            columns={
                "startup_cost_median": "p27_uc_startup_cost_median",
                "min_run_time_median": "p27_uc_min_run_time_median",
            }
        )
        df = df.merge(
            uc[["cost_month", "p27_uc_startup_cost_median", "p27_uc_min_run_time_median"]],
            on="cost_month",
            how="left",
        )

    for col in COST_FEATURE_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan
        values = pd.to_numeric(df[col], errors="coerce")
        if values.notna().any():
            df[col] = values.ffill().bfill().fillna(float(values.median()))
        else:
            df[col] = 0.0
    return df


def financial_cost_matrix_from_frame(df: pd.DataFrame, *, zonal: bool = False) -> np.ndarray:
    missing = [col for col in FINANCIAL_COST_COLUMNS_BY_FUEL.values() if col not in df.columns]
    if missing:
        raise KeyError(f"Missing financial cost columns: {missing}")
    matrix = df[[FINANCIAL_COST_COLUMNS_BY_FUEL[fuel] for fuel in FUEL_CATEGORIES]].to_numpy(dtype=np.float64)
    if not np.isfinite(matrix).all():
        raise ValueError("Financial cost matrix contains non-finite values.")
    if zonal:
        return np.repeat(matrix[:, None, :], len(ZONE_CODES), axis=1)
    return matrix


def chronological_split(
    df: pd.DataFrame,
    *,
    train_end: str = "2024-01-21",
    val_end: str = "2024-01-26",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    local_dates = pd.to_datetime(df["timestamp_local"]).dt.date
    train_end_date = pd.to_datetime(train_end).date()
    val_end_date = pd.to_datetime(val_end).date()
    train = df[local_dates <= train_end_date].copy()
    val = df[(local_dates > train_end_date) & (local_dates <= val_end_date)].copy()
    test = df[local_dates > val_end_date].copy()
    return train, val, test

def main_project_split(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    local_year = pd.to_datetime(df["timestamp_local"]).dt.year
    return {
        "train": df[(local_year >= 2016) & (local_year <= 2021)].copy(),
        "validation": df[local_year == 2022].copy(),
        "test": df[(local_year >= 2023) & (local_year <= 2024)].copy(),
        "holdout": df[local_year == 2025].copy(),
    }

def adaptive_chronological_split(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    ordered = df.sort_values("timestamp_utc").reset_index(drop=True)
    n = len(ordered)
    train_end = int(n * 0.60)
    validation_end = int(n * 0.75)
    test_end = int(n * 0.90)
    return {
        "train": ordered.iloc[:train_end].copy(),
        "validation": ordered.iloc[train_end:validation_end].copy(),
        "test": ordered.iloc[validation_end:test_end].copy(),
        "holdout": ordered.iloc[test_end:].copy(),
    }

def make_window_matrix(
    df: pd.DataFrame,
    *,
    features: list[str],
    history_window: int,
    load_column: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.Series, np.ndarray]:
    missing = [col for col in [*features, *TARGET_COLUMNS, load_column] if col not in df.columns]
    if missing:
        raise KeyError(f"Missing columns required for window matrix: {missing}")
    x_rows = []
    y_rows = []
    loads = []
    timestamps = []
    row_indices = []
    values = df[features].to_numpy(dtype=np.float64)
    targets = df[TARGET_COLUMNS].to_numpy(dtype=np.float64)
    load_values = df[load_column].to_numpy(dtype=np.float64)
    for i in range(history_window, len(df)):
        window = values[i - history_window + 1 : i + 1]
        if not (
            np.isfinite(window).all()
            and np.isfinite(targets[i]).all()
            and np.isfinite(load_values[i])
        ):
            continue
        x_rows.append(window.reshape(-1))
        y_rows.append(targets[i])
        loads.append(load_values[i])
        timestamps.append(df["timestamp_utc"].iloc[i])
        row_indices.append(i)
    if not x_rows:
        raise ValueError("Not enough rows for requested history window.")
    return np.asarray(x_rows), np.asarray(y_rows), np.asarray(loads), pd.Series(timestamps), np.asarray(row_indices)

def make_aligned_test_frame(train_df: pd.DataFrame, test_df: pd.DataFrame, history_window: int) -> pd.DataFrame:
    return pd.concat([train_df.tail(history_window), test_df], ignore_index=True)

def compute_available_capacities(train_df: pd.DataFrame) -> np.ndarray:
    static_caps = train_df[STATIC_CAPACITY_COLUMNS].iloc[0].to_numpy(dtype=np.float64)
    observed_caps = train_df[TARGET_COLUMNS].max().to_numpy(dtype=np.float64) * 1.05
    return np.maximum(static_caps, observed_caps)

def compute_zonal_capacities(zonal_capacity: pd.DataFrame, system_capacities: np.ndarray) -> np.ndarray:
    matrix = np.zeros((len(ZONE_CODES), len(FUEL_CATEGORIES)), dtype=np.float64)
    for f_idx, fuel in enumerate(FUEL_CATEGORIES):
        fuel_rows = (
            zonal_capacity[zonal_capacity["fuel_category"] == fuel]
            .set_index("zone_code")
            .reindex(ZONE_CODES)
            .fillna({"capacity_mw": 0.0})
        )
        static = fuel_rows["capacity_mw"].to_numpy(dtype=np.float64)
        target_total = float(system_capacities[f_idx])
        static_total = float(static.sum())
        if target_total <= 0.0:
            scaled = np.zeros_like(static)
        elif static_total > 0.0:
            scaled = static / static_total * target_total
        else:
            scaled = np.full(len(ZONE_CODES), target_total / len(ZONE_CODES), dtype=np.float64)
        matrix[:, f_idx] = scaled
    return matrix

def compute_empirical_system_availability_caps(train_df: pd.DataFrame, capacities: np.ndarray) -> np.ndarray:
    caps = np.full(len(FUEL_CATEGORIES), np.inf, dtype=np.float64)
    fuel_to_idx = {fuel: idx for idx, fuel in enumerate(FUEL_CATEGORIES)}
    hydro_idx = fuel_to_idx["Hydro"]
    nuclear_idx = fuel_to_idx["Nuclear"]
    caps[hydro_idx] = min(
        float(capacities[hydro_idx]),
        float(train_df[TARGET_COLUMNS[hydro_idx]].quantile(0.99) * 1.05),
    )
    caps[nuclear_idx] = min(
        float(capacities[nuclear_idx]),
        float(train_df[TARGET_COLUMNS[nuclear_idx]].quantile(0.99) * 1.01),
    )
    return caps

def availability_limits_for_hour(
    zone_rows: pd.DataFrame,
    zonal_capacities: np.ndarray,
    system_availability_caps: np.ndarray | None = None,
) -> np.ndarray:
    limits = np.asarray(zonal_capacities, dtype=np.float64).copy()
    for fuel, col in AVAILABILITY_COLUMNS_BY_FUEL.items():
        if col not in zone_rows.columns:
            continue
        f_idx = FUEL_CATEGORIES.index(fuel)
        values = zone_rows[col].to_numpy(dtype=np.float64)
        values = np.nan_to_num(values, nan=np.inf, posinf=np.inf, neginf=0.0)
        limits[:, f_idx] = np.minimum(limits[:, f_idx], values)

    if system_availability_caps is not None:
        for f_idx, cap in enumerate(system_availability_caps):
            if not np.isfinite(cap):
                continue
            static_total = float(zonal_capacities[:, f_idx].sum())
            if static_total > 0.0:
                zonal_cap = zonal_capacities[:, f_idx] / static_total * cap
            else:
                zonal_cap = np.full(len(ZONE_CODES), cap / len(ZONE_CODES), dtype=np.float64)
            limits[:, f_idx] = np.minimum(limits[:, f_idx], zonal_cap)
    return limits

def compute_empirical_ramp_limits(train_df: pd.DataFrame, quantile: float = 0.99) -> RampLimits:
    dispatch = train_df.sort_values("timestamp_utc")[TARGET_COLUMNS].to_numpy(dtype=np.float64)
    if dispatch.shape[0] < 2:
        raise ValueError("Need at least two training rows to estimate empirical ramp limits.")
    deltas = np.diff(dispatch, axis=0)
    ramp_up = np.nanquantile(np.clip(deltas, 0.0, None), quantile, axis=0)
    ramp_down = np.nanquantile(np.clip(-deltas, 0.0, None), quantile, axis=0)
    observed_max = np.nanmax(dispatch, axis=0)
    floor = np.maximum(1.0, observed_max * 0.001)
    ramp_up = np.maximum(np.nan_to_num(ramp_up, nan=0.0, posinf=0.0, neginf=0.0), floor)
    ramp_down = np.maximum(np.nan_to_num(ramp_down, nan=0.0, posinf=0.0, neginf=0.0), floor)
    return RampLimits(up=ramp_up.astype(np.float64), down=ramp_down.astype(np.float64), quantile=quantile)

def _generation_bounds(
    capacities: np.ndarray,
    previous_generation: np.ndarray | None = None,
    ramp_limits: RampLimits | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    lower = np.zeros(len(TARGET_COLUMNS), dtype=np.float64)
    upper = np.asarray(capacities, dtype=np.float64).copy()
    if previous_generation is not None and ramp_limits is not None:
        previous = np.asarray(previous_generation, dtype=np.float64)
        lower = np.maximum(lower, previous - ramp_limits.down)
        upper = np.minimum(upper, previous + ramp_limits.up)
        lower = np.minimum(lower, upper)
    return lower, upper

def _previous_generation_from_row(row: pd.Series) -> np.ndarray | None:
    lag_cols = [f"{col}_lag_1" for col in TARGET_COLUMNS]
    if not all(col in row.index for col in lag_cols):
        return None
    values = row[lag_cols].to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        return None
    return values

def _max_ramp_violation(
    generation: np.ndarray,
    previous_generation: np.ndarray | None,
    ramp_limits: RampLimits | None,
) -> float:
    if previous_generation is None or ramp_limits is None:
        return 0.0
    delta = np.asarray(generation, dtype=np.float64) - np.asarray(previous_generation, dtype=np.float64)
    up_violation = np.maximum(delta - ramp_limits.up, 0.0)
    down_violation = np.maximum(-delta - ramp_limits.down, 0.0)
    return float(max(up_violation.max(initial=0.0), down_violation.max(initial=0.0)))

def _max_ramp_violation_with_capacity_clip(
    generation: np.ndarray,
    previous_generation: np.ndarray | None,
    ramp_limits: RampLimits | None,
    fuel_upper: np.ndarray,
) -> float:
    if previous_generation is None or ramp_limits is None:
        return 0.0
    generation = np.asarray(generation, dtype=np.float64)
    previous = np.asarray(previous_generation, dtype=np.float64)
    effective_lower = np.minimum(np.maximum(0.0, previous - ramp_limits.down), np.asarray(fuel_upper, dtype=np.float64))
    effective_upper = previous + ramp_limits.up
    up_violation = np.maximum(generation - effective_upper, 0.0)
    down_violation = np.maximum(effective_lower - generation, 0.0)
    return float(max(up_violation.max(initial=0.0), down_violation.max(initial=0.0)))

def solve_fuel_dispatch(
    load_mw: float,
    costs: np.ndarray,
    capacities: np.ndarray,
    config: ExperimentConfig,
    *,
    previous_generation: np.ndarray | None = None,
    ramp_limits: RampLimits | None = None,
) -> DispatchResult:
    start = perf_counter()
    n = len(TARGET_COLUMNS)
    c = np.concatenate(
        [
            np.asarray(costs, dtype=np.float64),
            np.array([config.residual_penalty, config.ens_penalty, config.spill_penalty], dtype=np.float64),
        ]
    )
    a_eq = np.zeros((1, n + 3), dtype=np.float64)
    a_eq[0, :n] = 1.0
    a_eq[0, n] = 1.0
    a_eq[0, n + 1] = 1.0
    a_eq[0, n + 2] = -1.0
    b_eq = np.array([float(load_mw)], dtype=np.float64)
    max_aux = max(float(load_mw) * 2.0, float(np.sum(capacities)) * 2.0, 1.0)
    lower, upper = _generation_bounds(capacities, previous_generation, ramp_limits)
    bounds = [(float(lb), float(ub)) for lb, ub in zip(lower, upper)] + [
        (0.0, max_aux),
        (0.0, max_aux),
        (0.0, max_aux),
    ]
    res = linprog(c=c, A_eq=a_eq, b_eq=b_eq, bounds=bounds, method="highs")
    runtime = perf_counter() - start
    if not res.success:
        return DispatchResult(
            generation=np.zeros(n, dtype=np.float64),
            residual_supply=0.0,
            ens=float(load_mw),
            spill=0.0,
            objective=float(config.ens_penalty * load_mw),
            success=False,
            runtime_seconds=runtime,
            max_ramp_violation_mw=0.0,
        )
    generation = np.asarray(res.x[:n], dtype=np.float64)
    return DispatchResult(
        generation=generation,
        residual_supply=float(res.x[n]),
        ens=float(res.x[n + 1]),
        spill=float(res.x[n + 2]),
        objective=float(res.fun),
        success=True,
        runtime_seconds=runtime,
        max_ramp_violation_mw=_max_ramp_violation(generation, previous_generation, ramp_limits),
    )

def solve_fuel_repair_dispatch(
    raw_generation: np.ndarray,
    load_mw: float,
    capacities: np.ndarray,
    config: ExperimentConfig,
    *,
    previous_generation: np.ndarray | None = None,
    ramp_limits: RampLimits | None = None,
) -> DispatchResult:
    start = perf_counter()
    n = len(TARGET_COLUMNS)
    raw = np.asarray(raw_generation, dtype=np.float64)
    if raw.shape != (n,):
        raise ValueError(f"Expected raw generation shape {(n,)}, received {raw.shape}")

    g0 = 0
    dev_pos0 = g0 + n
    dev_neg0 = dev_pos0 + n
    residual0 = dev_neg0 + n
    ens0 = residual0 + 1
    spill0 = ens0 + 1
    total_vars = spill0 + 1

    c = np.zeros(total_vars, dtype=np.float64)
    c[dev_pos0:dev_neg0] = 1.0
    c[dev_neg0:residual0] = 1.0
    c[residual0] = config.residual_penalty
    c[ens0] = config.ens_penalty
    c[spill0] = config.spill_penalty

    a_eq = np.zeros((1 + n, total_vars), dtype=np.float64)
    b_eq = np.zeros(1 + n, dtype=np.float64)
    a_eq[0, g0:dev_pos0] = 1.0
    a_eq[0, residual0] = 1.0
    a_eq[0, ens0] = 1.0
    a_eq[0, spill0] = -1.0
    b_eq[0] = float(load_mw)
    for f in range(n):
        row = 1 + f
        a_eq[row, g0 + f] = 1.0
        a_eq[row, dev_pos0 + f] = -1.0
        a_eq[row, dev_neg0 + f] = 1.0
        b_eq[row] = raw[f]

    max_aux = max(float(load_mw) * 2.0, float(np.sum(capacities)) * 2.0, float(np.nanmax(np.abs(raw))) * 2.0, 1.0)
    lower, upper = _generation_bounds(capacities, previous_generation, ramp_limits)
    bounds = [(float(lb), float(ub)) for lb, ub in zip(lower, upper)]
    bounds += [(0.0, max_aux) for _ in range(n * 2)]
    bounds += [(0.0, max_aux), (0.0, max_aux), (0.0, max_aux)]

    res = linprog(c=c, A_eq=a_eq, b_eq=b_eq, bounds=bounds, method="highs")
    runtime = perf_counter() - start
    if not res.success:
        return DispatchResult(
            generation=np.zeros(n, dtype=np.float64),
            residual_supply=0.0,
            ens=float(load_mw),
            spill=0.0,
            objective=float(config.ens_penalty * load_mw),
            success=False,
            runtime_seconds=runtime,
            max_ramp_violation_mw=0.0,
        )

    generation = np.asarray(res.x[g0:dev_pos0], dtype=np.float64)
    return DispatchResult(
        generation=generation,
        residual_supply=float(res.x[residual0]),
        ens=float(res.x[ens0]),
        spill=float(res.x[spill0]),
        objective=float(res.fun),
        success=True,
        runtime_seconds=runtime,
        max_ramp_violation_mw=_max_ramp_violation(generation, previous_generation, ramp_limits),
    )


def solve_fuel_soft_repair_dispatch(
    raw_generation: np.ndarray,
    load_mw: float,
    capacities: np.ndarray,
    config: ExperimentConfig,
    *,
    soft_balance_penalty: float,
    previous_generation: np.ndarray | None = None,
    ramp_limits: RampLimits | None = None,
) -> DispatchResult:
    start = perf_counter()
    n = len(TARGET_COLUMNS)
    raw = np.asarray(raw_generation, dtype=np.float64)
    if raw.shape != (n,):
        raise ValueError(f"Expected raw generation shape {(n,)}, received {raw.shape}")

    g0 = 0
    dev_pos0 = g0 + n
    dev_neg0 = dev_pos0 + n
    balance_under0 = dev_neg0 + n
    balance_over0 = balance_under0 + 1
    total_vars = balance_over0 + 1

    c = np.zeros(total_vars, dtype=np.float64)
    c[dev_pos0:dev_neg0] = 1.0
    c[dev_neg0:balance_under0] = 1.0
    c[balance_under0] = float(soft_balance_penalty)
    c[balance_over0] = float(soft_balance_penalty)

    a_eq = np.zeros((1 + n, total_vars), dtype=np.float64)
    b_eq = np.zeros(1 + n, dtype=np.float64)
    a_eq[0, g0:dev_pos0] = 1.0
    a_eq[0, balance_under0] = 1.0
    a_eq[0, balance_over0] = -1.0
    b_eq[0] = float(load_mw)
    for f in range(n):
        row = 1 + f
        a_eq[row, g0 + f] = 1.0
        a_eq[row, dev_pos0 + f] = -1.0
        a_eq[row, dev_neg0 + f] = 1.0
        b_eq[row] = raw[f]

    max_aux = max(float(load_mw) * 2.0, float(np.sum(capacities)) * 2.0, float(np.nanmax(np.abs(raw))) * 2.0, 1.0)
    lower, upper = _generation_bounds(capacities, previous_generation, ramp_limits)
    bounds = [(float(lb), float(ub)) for lb, ub in zip(lower, upper)]
    bounds += [(0.0, max_aux) for _ in range(n * 2)]
    bounds += [(0.0, max_aux), (0.0, max_aux)]

    res = linprog(c=c, A_eq=a_eq, b_eq=b_eq, bounds=bounds, method="highs")
    runtime = perf_counter() - start
    if not res.success:
        return DispatchResult(
            generation=np.zeros(n, dtype=np.float64),
            residual_supply=0.0,
            ens=float(load_mw),
            spill=0.0,
            objective=float(config.ens_penalty * load_mw),
            success=False,
            runtime_seconds=runtime,
            max_ramp_violation_mw=0.0,
        )

    generation = np.asarray(res.x[g0:dev_pos0], dtype=np.float64)
    return DispatchResult(
        generation=generation,
        residual_supply=0.0,
        ens=0.0,
        spill=0.0,
        objective=float(res.fun),
        success=True,
        runtime_seconds=runtime,
        max_ramp_violation_mw=_max_ramp_violation(generation, previous_generation, ramp_limits),
        balance_under=float(res.x[balance_under0]),
        balance_over=float(res.x[balance_over0]),
    )


def solve_residual_aware_cost_dispatch(
    previous_generation: np.ndarray,
    load_mw: float,
    capacities: np.ndarray,
    config: ExperimentConfig,
    *,
    up_costs: np.ndarray,
    down_costs: np.ndarray,
    soft_balance_penalty: float,
    ramp_limits: RampLimits | None = None,
) -> DispatchResult:
    start = perf_counter()
    n = len(TARGET_COLUMNS)
    previous = np.asarray(previous_generation, dtype=np.float64)
    if previous.shape != (n,):
        raise ValueError(f"Expected previous generation shape {(n,)}, received {previous.shape}")
    up_costs = np.asarray(up_costs, dtype=np.float64)
    down_costs = np.asarray(down_costs, dtype=np.float64)
    if up_costs.shape != (n,) or down_costs.shape != (n,):
        raise ValueError("Residual-aware costs must be one vector per fuel for up and down adjustments.")

    g0 = 0
    up0 = g0 + n
    down0 = up0 + n
    balance_under0 = down0 + n
    balance_over0 = balance_under0 + 1
    total_vars = balance_over0 + 1

    c = np.zeros(total_vars, dtype=np.float64)
    c[up0:down0] = up_costs
    c[down0:balance_under0] = down_costs
    c[balance_under0] = float(soft_balance_penalty)
    c[balance_over0] = float(soft_balance_penalty)

    a_eq = np.zeros((1 + n, total_vars), dtype=np.float64)
    b_eq = np.zeros(1 + n, dtype=np.float64)
    a_eq[0, g0:up0] = 1.0
    a_eq[0, balance_under0] = 1.0
    a_eq[0, balance_over0] = -1.0
    b_eq[0] = float(load_mw)
    for f in range(n):
        row = 1 + f
        a_eq[row, g0 + f] = 1.0
        a_eq[row, up0 + f] = -1.0
        a_eq[row, down0 + f] = 1.0
        b_eq[row] = previous[f]

    max_aux = max(float(load_mw) * 2.0, float(np.sum(capacities)) * 2.0, float(np.nanmax(np.abs(previous))) * 2.0, 1.0)
    lower, upper = _generation_bounds(capacities, previous, ramp_limits)
    bounds = [(float(lb), float(ub)) for lb, ub in zip(lower, upper)]
    bounds += [(0.0, max_aux) for _ in range(n * 2)]
    bounds += [(0.0, max_aux), (0.0, max_aux)]

    res = linprog(c=c, A_eq=a_eq, b_eq=b_eq, bounds=bounds, method="highs")
    runtime = perf_counter() - start
    if not res.success:
        return DispatchResult(
            generation=np.zeros(n, dtype=np.float64),
            residual_supply=0.0,
            ens=float(load_mw),
            spill=0.0,
            objective=float(config.ens_penalty * load_mw),
            success=False,
            runtime_seconds=runtime,
            max_ramp_violation_mw=0.0,
        )

    generation = np.asarray(res.x[g0:up0], dtype=np.float64)
    return DispatchResult(
        generation=generation,
        residual_supply=0.0,
        ens=0.0,
        spill=0.0,
        objective=float(res.fun),
        success=True,
        runtime_seconds=runtime,
        max_ramp_violation_mw=_max_ramp_violation(generation, previous, ramp_limits),
        balance_under=float(res.x[balance_under0]),
        balance_over=float(res.x[balance_over0]),
    )


def solve_fuel_projection_dispatch(
    raw_generation: np.ndarray,
    capacities: np.ndarray,
    config: ExperimentConfig,
    *,
    previous_generation: np.ndarray | None = None,
    ramp_limits: RampLimits | None = None,
) -> DispatchResult:
    start = perf_counter()
    n = len(TARGET_COLUMNS)
    raw = np.asarray(raw_generation, dtype=np.float64)
    if raw.shape != (n,):
        raise ValueError(f"Expected raw generation shape {(n,)}, received {raw.shape}")

    g0 = 0
    dev_pos0 = g0 + n
    dev_neg0 = dev_pos0 + n
    total_vars = dev_neg0 + n

    c = np.zeros(total_vars, dtype=np.float64)
    c[dev_pos0:dev_neg0] = 1.0
    c[dev_neg0:] = 1.0

    a_eq = np.zeros((n, total_vars), dtype=np.float64)
    b_eq = np.zeros(n, dtype=np.float64)
    for f in range(n):
        a_eq[f, g0 + f] = 1.0
        a_eq[f, dev_pos0 + f] = -1.0
        a_eq[f, dev_neg0 + f] = 1.0
        b_eq[f] = raw[f]

    max_aux = max(float(np.sum(capacities)) * 2.0, float(np.nanmax(np.abs(raw))) * 2.0, 1.0)
    lower, upper = _generation_bounds(capacities, previous_generation, ramp_limits)
    bounds = [(float(lb), float(ub)) for lb, ub in zip(lower, upper)]
    bounds += [(0.0, max_aux) for _ in range(n * 2)]

    res = linprog(c=c, A_eq=a_eq, b_eq=b_eq, bounds=bounds, method="highs")
    runtime = perf_counter() - start
    if not res.success:
        return DispatchResult(
            generation=np.zeros(n, dtype=np.float64),
            residual_supply=0.0,
            ens=0.0,
            spill=0.0,
            objective=float("inf"),
            success=False,
            runtime_seconds=runtime,
            max_ramp_violation_mw=0.0,
        )
    generation = np.asarray(res.x[g0:dev_pos0], dtype=np.float64)
    return DispatchResult(
        generation=generation,
        residual_supply=0.0,
        ens=0.0,
        spill=0.0,
        objective=float(res.fun),
        success=True,
        runtime_seconds=runtime,
        max_ramp_violation_mw=_max_ramp_violation(generation, previous_generation, ramp_limits),
    )


def solve_projection_dispatch_many(
    df: pd.DataFrame,
    raw_predictions: np.ndarray,
    capacities: np.ndarray,
    config: ExperimentConfig,
    *,
    ramp_limits: RampLimits | None = None,
) -> tuple[np.ndarray, pd.DataFrame]:
    raw_predictions = np.asarray(raw_predictions, dtype=np.float64)
    if raw_predictions.shape != (len(df), len(TARGET_COLUMNS)):
        raise ValueError(f"Expected raw predictions shape {(len(df), len(TARGET_COLUMNS))}, received {raw_predictions.shape}")
    predictions = []
    diagnostics = []
    for i, (_, row) in enumerate(df.iterrows()):
        previous_generation = _previous_generation_from_row(row) if ramp_limits is not None else None
        result = solve_fuel_projection_dispatch(
            raw_predictions[i],
            capacities,
            config,
            previous_generation=previous_generation,
            ramp_limits=ramp_limits,
        )
        predictions.append(result.generation)
        diagnostics.append(
            {
                "timestamp_utc": row["timestamp_utc"],
                "solver_load_mw": np.nan,
                "solver_load_column": "",
                "residual_supply": result.residual_supply,
                "ens": result.ens,
                "spill": result.spill,
                "objective": result.objective,
                "solver_success": result.success,
                "runtime_seconds": result.runtime_seconds,
                "max_ramp_violation_mw": result.max_ramp_violation_mw,
                "mean_abs_repair_delta_mw": float(np.mean(np.abs(result.generation - raw_predictions[i]))),
                "balance_residual": np.nan,
            }
        )
    return np.asarray(predictions), pd.DataFrame(diagnostics)


def solve_repair_dispatch_many(
    df: pd.DataFrame,
    raw_predictions: np.ndarray,
    capacities: np.ndarray,
    config: ExperimentConfig,
    *,
    ramp_limits: RampLimits | None = None,
    load_column: str | None = None,
) -> tuple[np.ndarray, pd.DataFrame]:
    load_column = load_column or config.solver_load_column
    if load_column not in df.columns:
        raise KeyError(f"Missing solver load column: {load_column}")
    raw_predictions = np.asarray(raw_predictions, dtype=np.float64)
    if raw_predictions.shape != (len(df), len(TARGET_COLUMNS)):
        raise ValueError(f"Expected raw predictions shape {(len(df), len(TARGET_COLUMNS))}, received {raw_predictions.shape}")

    predictions = []
    diagnostics = []
    for i, (_, row) in enumerate(df.iterrows()):
        previous_generation = _previous_generation_from_row(row) if ramp_limits is not None else None
        load_mw = float(row[load_column])
        result = solve_fuel_repair_dispatch(
            raw_predictions[i],
            load_mw,
            capacities,
            config,
            previous_generation=previous_generation,
            ramp_limits=ramp_limits,
        )
        predictions.append(result.generation)
        diagnostics.append(
            {
                "timestamp_utc": row["timestamp_utc"],
                "solver_load_mw": load_mw,
                "solver_load_column": load_column,
                "residual_supply": result.residual_supply,
                "ens": result.ens,
                "spill": result.spill,
                "objective": result.objective,
                "solver_success": result.success,
                "runtime_seconds": result.runtime_seconds,
                "max_ramp_violation_mw": result.max_ramp_violation_mw,
                "mean_abs_repair_delta_mw": float(np.mean(np.abs(result.generation - raw_predictions[i]))),
                "balance_residual": float(
                    result.generation.sum() + result.residual_supply + result.ens - result.spill - load_mw
                ),
            }
        )
    return np.asarray(predictions), pd.DataFrame(diagnostics)


def solve_soft_repair_dispatch_many(
    df: pd.DataFrame,
    raw_predictions: np.ndarray,
    capacities: np.ndarray,
    config: ExperimentConfig,
    *,
    soft_balance_penalty: float,
    ramp_limits: RampLimits | None = None,
    load_column: str | None = None,
) -> tuple[np.ndarray, pd.DataFrame]:
    load_column = load_column or config.solver_load_column
    if load_column not in df.columns:
        raise KeyError(f"Missing solver load column: {load_column}")
    raw_predictions = np.asarray(raw_predictions, dtype=np.float64)
    if raw_predictions.shape != (len(df), len(TARGET_COLUMNS)):
        raise ValueError(f"Expected raw predictions shape {(len(df), len(TARGET_COLUMNS))}, received {raw_predictions.shape}")

    predictions = []
    diagnostics = []
    for i, (_, row) in enumerate(df.iterrows()):
        previous_generation = _previous_generation_from_row(row) if ramp_limits is not None else None
        load_mw = float(row[load_column])
        result = solve_fuel_soft_repair_dispatch(
            raw_predictions[i],
            load_mw,
            capacities,
            config,
            soft_balance_penalty=soft_balance_penalty,
            previous_generation=previous_generation,
            ramp_limits=ramp_limits,
        )
        balance_deviation = result.balance_under + result.balance_over
        predictions.append(result.generation)
        diagnostics.append(
            {
                "timestamp_utc": row["timestamp_utc"],
                "solver_load_mw": load_mw,
                "solver_load_column": load_column,
                "soft_balance_penalty": float(soft_balance_penalty),
                "residual_supply": result.residual_supply,
                "ens": result.ens,
                "spill": result.spill,
                "objective": result.objective,
                "solver_success": result.success,
                "runtime_seconds": result.runtime_seconds,
                "max_ramp_violation_mw": result.max_ramp_violation_mw,
                "mean_abs_repair_delta_mw": float(np.mean(np.abs(result.generation - raw_predictions[i]))),
                "balance_under_mw": result.balance_under,
                "balance_over_mw": result.balance_over,
                "abs_balance_deviation_mw": balance_deviation,
                "balance_residual": float(
                    result.generation.sum() + result.balance_under - result.balance_over - load_mw
                ),
            }
        )
    return np.asarray(predictions), pd.DataFrame(diagnostics)


def solve_residual_aware_cost_dispatch_many(
    df: pd.DataFrame,
    up_costs: np.ndarray,
    down_costs: np.ndarray,
    capacities: np.ndarray,
    config: ExperimentConfig,
    *,
    soft_balance_penalty: float,
    ramp_limits: RampLimits | None = None,
    load_column: str | None = None,
) -> tuple[np.ndarray, pd.DataFrame]:
    load_column = load_column or config.solver_load_column
    if load_column not in df.columns:
        raise KeyError(f"Missing solver load column: {load_column}")
    up_costs = np.asarray(up_costs, dtype=np.float64)
    down_costs = np.asarray(down_costs, dtype=np.float64)
    expected_shape = (len(df), len(TARGET_COLUMNS))
    if up_costs.shape != expected_shape or down_costs.shape != expected_shape:
        raise ValueError(f"Expected residual cost shapes {expected_shape}, received {up_costs.shape} and {down_costs.shape}")

    predictions = []
    diagnostics = []
    for i, (_, row) in enumerate(df.iterrows()):
        previous_generation = _previous_generation_from_row(row)
        load_mw = float(row[load_column])
        result = solve_residual_aware_cost_dispatch(
            previous_generation,
            load_mw,
            capacities,
            config,
            up_costs=up_costs[i],
            down_costs=down_costs[i],
            soft_balance_penalty=soft_balance_penalty,
            ramp_limits=ramp_limits,
        )
        balance_deviation = result.balance_under + result.balance_over
        predictions.append(result.generation)
        diagnostics.append(
            {
                "timestamp_utc": row["timestamp_utc"],
                "solver_load_mw": load_mw,
                "solver_load_column": load_column,
                "soft_balance_penalty": float(soft_balance_penalty),
                "residual_supply": result.residual_supply,
                "ens": result.ens,
                "spill": result.spill,
                "objective": result.objective,
                "solver_success": result.success,
                "runtime_seconds": result.runtime_seconds,
                "max_ramp_violation_mw": result.max_ramp_violation_mw,
                "balance_under_mw": result.balance_under,
                "balance_over_mw": result.balance_over,
                "abs_balance_deviation_mw": balance_deviation,
                "mean_abs_adjustment_mw": float(np.mean(np.abs(result.generation - previous_generation))),
                "balance_residual": float(
                    result.generation.sum() + result.balance_under - result.balance_over - load_mw
                ),
            }
        )
    return np.asarray(predictions), pd.DataFrame(diagnostics)


def solve_dispatch_many(
    df: pd.DataFrame,
    costs: np.ndarray,
    capacities: np.ndarray,
    config: ExperimentConfig,
    *,
    ramp_limits: RampLimits | None = None,
    load_column: str | None = None,
) -> tuple[np.ndarray, pd.DataFrame]:
    load_column = load_column or config.solver_load_column
    if load_column not in df.columns:
        raise KeyError(f"Missing solver load column: {load_column}")
    predictions = []
    diagnostics = []
    previous_generation: np.ndarray | None = None
    for i, (_, row) in enumerate(df.iterrows()):
        cost_i = costs[i] if np.asarray(costs).ndim == 2 else costs
        if previous_generation is None and ramp_limits is not None:
            previous_generation = _previous_generation_from_row(row)
        load_mw = float(row[load_column])
        result = solve_fuel_dispatch(
            load_mw,
            cost_i,
            capacities,
            config,
            previous_generation=previous_generation,
            ramp_limits=ramp_limits,
        )
        predictions.append(result.generation)
        previous_for_diag = previous_generation
        diagnostics.append(
            {
                "timestamp_utc": row["timestamp_utc"],
                "solver_load_mw": load_mw,
                "solver_load_column": load_column,
                "residual_supply": result.residual_supply,
                "ens": result.ens,
                "spill": result.spill,
                "objective": result.objective,
                "solver_success": result.success,
                "runtime_seconds": result.runtime_seconds,
                "max_ramp_violation_mw": result.max_ramp_violation_mw,
                "balance_residual": float(
                    result.generation.sum() + result.residual_supply + result.ens - result.spill - load_mw
                ),
            }
        )
        if ramp_limits is not None:
            diagnostics[-1]["previous_generation_total_mw"] = (
                float(np.asarray(previous_for_diag).sum()) if previous_for_diag is not None else np.nan
            )
            previous_generation = result.generation
    return np.asarray(predictions), pd.DataFrame(diagnostics)

def _interface_bounds_for_hour(interface_rows: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    by_name = interface_rows.set_index("interface_name") if not interface_rows.empty else pd.DataFrame()
    upper = np.full(len(INTERNAL_INTERFACE_CUTS), np.nan, dtype=np.float64)
    lower = np.full(len(INTERNAL_INTERFACE_CUTS), np.nan, dtype=np.float64)
    upper_finite = np.zeros(len(INTERNAL_INTERFACE_CUTS), dtype=bool)
    lower_finite = np.zeros(len(INTERNAL_INTERFACE_CUTS), dtype=bool)
    for k, name in enumerate(INTERNAL_INTERFACE_CUTS):
        if name not in by_name.index:
            continue
        row = by_name.loc[name]
        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]
        if bool(row.get("upper_is_finite", False)) and np.isfinite(row.get("upper_limit_mw", np.nan)):
            upper[k] = float(row["upper_limit_mw"])
            upper_finite[k] = True
        if bool(row.get("lower_is_finite", False)) and np.isfinite(row.get("lower_limit_mw", np.nan)):
            lower[k] = float(row["lower_limit_mw"])
            lower_finite[k] = True
    return upper, lower, upper_finite, lower_finite

def _interface_incidence_matrix() -> np.ndarray:
    incidence = np.zeros((len(INTERNAL_INTERFACE_CUTS), len(ZONE_CODES)), dtype=np.float64)
    zone_to_idx = {zone: idx for idx, zone in enumerate(ZONE_CODES)}
    for k, source_zones in enumerate(INTERNAL_INTERFACE_CUTS.values()):
        for zone in source_zones:
            incidence[k, zone_to_idx[zone]] = 1.0
    return incidence


def allocate_system_generation_to_zones(system_generation: np.ndarray, zonal_capacities: np.ndarray) -> np.ndarray:
    raw = np.asarray(system_generation, dtype=np.float64)
    capacities = np.asarray(zonal_capacities, dtype=np.float64)
    if raw.shape != (len(FUEL_CATEGORIES),):
        raise ValueError(f"Expected system generation shape {(len(FUEL_CATEGORIES),)}, received {raw.shape}")
    if capacities.shape != (len(ZONE_CODES), len(FUEL_CATEGORIES)):
        raise ValueError(f"Expected zonal capacity shape {(len(ZONE_CODES), len(FUEL_CATEGORIES))}, received {capacities.shape}")
    out = np.zeros_like(capacities, dtype=np.float64)
    for f in range(len(FUEL_CATEGORIES)):
        total = float(capacities[:, f].sum())
        if total > 0.0:
            shares = capacities[:, f] / total
        else:
            shares = np.full(len(ZONE_CODES), 1.0 / len(ZONE_CODES), dtype=np.float64)
        out[:, f] = raw[f] * shares
    return out


def solve_zonal_dispatch(
    zone_loads_mw: np.ndarray,
    interface_rows: pd.DataFrame,
    costs: np.ndarray,
    zonal_capacities: np.ndarray,
    config: ExperimentConfig,
    *,
    availability_limits: np.ndarray | None = None,
    repair_targets: np.ndarray | None = None,
    previous_generation: np.ndarray | None = None,
    ramp_limits: RampLimits | None = None,
    reserve_slack_penalty: float | None = None,
    reserve_requirement_fraction: float | None = None,
) -> ZonalDispatchResult:
    start = perf_counter()
    nz = len(ZONE_CODES)
    nf = len(FUEL_CATEGORIES)
    nk = len(INTERNAL_INTERFACE_CUTS)
    zone_loads = np.asarray(zone_loads_mw, dtype=np.float64)
    capacities = np.asarray(zonal_capacities, dtype=np.float64)
    if zone_loads.shape != (nz,):
        raise ValueError(f"Expected {nz} zonal loads, received shape {zone_loads.shape}")
    if capacities.shape != (nz, nf):
        raise ValueError(f"Expected zonal capacity shape {(nz, nf)}, received {capacities.shape}")
    if availability_limits is not None:
        availability = np.asarray(availability_limits, dtype=np.float64)
        if availability.shape != (nz, nf):
            raise ValueError(f"Expected availability shape {(nz, nf)}, received {availability.shape}")
        capacities = np.minimum(capacities, np.nan_to_num(availability, nan=np.inf, posinf=np.inf, neginf=0.0))

    cost_array = np.asarray(costs, dtype=np.float64)
    if cost_array.shape == (nf,):
        cost_array = np.tile(cost_array, (nz, 1))
    if cost_array.shape != (nz, nf):
        raise ValueError(f"Expected fuel costs shape {(nf,)} or {(nz, nf)}, received {cost_array.shape}")

    repair_array = None
    if repair_targets is not None:
        repair_array = np.asarray(repair_targets, dtype=np.float64)
        if repair_array.shape != (nz, nf):
            raise ValueError(f"Expected repair target shape {(nz, nf)}, received {repair_array.shape}")

    g0 = 0
    if repair_array is not None:
        dev_pos0 = g0 + nz * nf
        dev_neg0 = dev_pos0 + nz * nf
        n0 = dev_neg0 + nz * nf
    else:
        dev_pos0 = None
        dev_neg0 = None
        n0 = g0 + nz * nf
    n_pos0 = n0 + nz
    n_neg0 = n_pos0 + nz
    residual0 = n_neg0 + nz
    ens0 = residual0 + nz
    spill0 = ens0 + nz
    slack_pos0 = spill0 + nz
    slack_neg0 = slack_pos0 + nk
    reserve_slack0 = slack_neg0 + nk
    total_vars = reserve_slack0 + (1 if config.use_reserve_headroom_constraint else 0)

    c = np.zeros(total_vars, dtype=np.float64)
    if repair_array is None:
        c[g0:n0] = cost_array.reshape(-1)
    else:
        c[dev_pos0:dev_neg0] = 1.0
        c[dev_neg0:n0] = 1.0
    c[n_pos0:n_neg0] = config.transfer_penalty
    c[n_neg0:residual0] = config.transfer_penalty
    c[residual0:ens0] = config.residual_penalty
    c[ens0:spill0] = config.ens_penalty
    c[spill0:slack_pos0] = config.spill_penalty
    c[slack_pos0:slack_neg0] = config.interface_penalty
    c[slack_neg0:reserve_slack0] = config.interface_penalty
    if config.use_reserve_headroom_constraint:
        c[reserve_slack0] = float(config.reserve_slack_penalty if reserve_slack_penalty is None else reserve_slack_penalty)

    repair_eq_count = nz * nf if repair_array is not None else 0
    a_eq = np.zeros((nz * 2 + 1 + repair_eq_count, total_vars), dtype=np.float64)
    b_eq = np.zeros(nz * 2 + 1 + repair_eq_count, dtype=np.float64)
    for z in range(nz):
        a_eq[z, g0 + z * nf : g0 + (z + 1) * nf] = 1.0
        a_eq[z, residual0 + z] = 1.0
        a_eq[z, ens0 + z] = 1.0
        a_eq[z, spill0 + z] = -1.0
        a_eq[z, n0 + z] = -1.0
        b_eq[z] = zone_loads[z]
    a_eq[nz, n0 : n0 + nz] = 1.0
    for z in range(nz):
        row_idx = nz + 1 + z
        a_eq[row_idx, n0 + z] = 1.0
        a_eq[row_idx, n_pos0 + z] = -1.0
        a_eq[row_idx, n_neg0 + z] = 1.0
    if repair_array is not None:
        start_row = nz * 2 + 1
        for z in range(nz):
            for f in range(nf):
                row_idx = start_row + z * nf + f
                var_idx = g0 + z * nf + f
                a_eq[row_idx, var_idx] = 1.0
                a_eq[row_idx, dev_pos0 + z * nf + f] = -1.0
                a_eq[row_idx, dev_neg0 + z * nf + f] = 1.0
                b_eq[row_idx] = repair_array[z, f]

    incidence = _interface_incidence_matrix()
    upper, lower, upper_finite, lower_finite = _interface_bounds_for_hour(interface_rows)
    a_ub_rows = []
    b_ub = []
    for k in range(nk):
        if upper_finite[k]:
            row = np.zeros(total_vars, dtype=np.float64)
            row[n0 : n0 + nz] = incidence[k]
            row[slack_pos0 + k] = -1.0
            a_ub_rows.append(row)
            b_ub.append(upper[k])
        if lower_finite[k]:
            row = np.zeros(total_vars, dtype=np.float64)
            row[n0 : n0 + nz] = -incidence[k]
            row[slack_neg0 + k] = -1.0
            a_ub_rows.append(row)
            b_ub.append(-lower[k])

    if previous_generation is not None and ramp_limits is not None:
        previous = np.asarray(previous_generation, dtype=np.float64)
        for f in range(nf):
            row = np.zeros(total_vars, dtype=np.float64)
            for z in range(nz):
                row[g0 + z * nf + f] = 1.0
            a_ub_rows.append(row)
            b_ub.append(previous[f] + ramp_limits.up[f])

            row = np.zeros(total_vars, dtype=np.float64)
            for z in range(nz):
                row[g0 + z * nf + f] = -1.0
            a_ub_rows.append(row)
            feasible_fuel_upper = float(capacities[:, f].sum())
            ramp_lower = min(max(0.0, previous[f] - ramp_limits.down[f]), feasible_fuel_upper)
            b_ub.append(-ramp_lower)

    reserve_requirement = 0.0
    dispatchable_capacity = 0.0
    if config.use_reserve_headroom_constraint:
        active_reserve_fraction = float(
            config.reserve_requirement_fraction if reserve_requirement_fraction is None else reserve_requirement_fraction
        )
        reserve_requirement = float(zone_loads.sum()) * max(active_reserve_fraction, 0.0)
        reserve_capacity_by_fuel = capacities[:, RESERVE_HEADROOM_FUEL_INDICES].sum(axis=0)
        if previous_generation is not None and ramp_limits is not None:
            ramp_accessible = []
            for local_idx, fuel_idx in enumerate(RESERVE_HEADROOM_FUEL_INDICES):
                ramp_accessible.append(
                    min(
                        float(reserve_capacity_by_fuel[local_idx]),
                        max(0.0, float(previous_generation[fuel_idx]) + float(ramp_limits.up[fuel_idx])),
                    )
                )
            dispatchable_capacity = float(np.sum(ramp_accessible))
        else:
            dispatchable_capacity = float(reserve_capacity_by_fuel.sum())
        row = np.zeros(total_vars, dtype=np.float64)
        for z in range(nz):
            for f in RESERVE_HEADROOM_FUEL_INDICES:
                row[g0 + z * nf + f] = 1.0
        row[reserve_slack0] = -1.0
        a_ub_rows.append(row)
        b_ub.append(dispatchable_capacity - reserve_requirement)

    a_ub = np.vstack(a_ub_rows) if a_ub_rows else None
    b_ub_array = np.asarray(b_ub, dtype=np.float64) if b_ub else None

    max_aux = max(float(zone_loads.sum()) * 2.0, float(capacities.sum()) * 2.0, 10000.0)
    bounds = [(0.0, float(cap)) for cap in capacities.reshape(-1)]
    if repair_array is not None:
        bounds += [(0.0, max_aux) for _ in range(nz * nf * 2)]
    bounds += [(-max_aux, max_aux) for _ in range(nz)]
    bounds += [(0.0, max_aux) for _ in range(nz * 2)]
    bounds += [(0.0, max_aux) for _ in range(nz * 3 + nk * 2)]
    if config.use_reserve_headroom_constraint:
        bounds += [(0.0, max_aux)]

    res = linprog(c=c, A_ub=a_ub, b_ub=b_ub_array, A_eq=a_eq, b_eq=b_eq, bounds=bounds, method="highs")
    runtime = perf_counter() - start
    if not res.success:
        generation = np.zeros((nz, nf), dtype=np.float64)
        net = np.zeros(nz, dtype=np.float64)
        residual = np.zeros(nz, dtype=np.float64)
        ens = zone_loads.copy()
        spill = np.zeros(nz, dtype=np.float64)
        slack_pos = np.zeros(nk, dtype=np.float64)
        slack_neg = np.zeros(nk, dtype=np.float64)
        flows = incidence @ net
        reserve_headroom = dispatchable_capacity
        return ZonalDispatchResult(
            generation_by_zone=generation,
            net_injection=net,
            residual_supply=residual,
            ens=ens,
            spill=spill,
            interface_slack_pos=slack_pos,
            interface_slack_neg=slack_neg,
            interface_flows=flows,
            objective=float(config.ens_penalty * zone_loads.sum()),
            success=False,
            runtime_seconds=runtime,
            max_zonal_balance_residual=0.0,
            system_net_injection_residual=0.0,
            max_interface_violation_after_slack=0.0,
            max_ramp_violation_mw=0.0,
            transfer_magnitude_mw=0.0,
            reserve_requirement_mw=reserve_requirement,
            reserve_headroom_mw=reserve_headroom,
            reserve_slack_mw=0.0,
        )

    x = np.asarray(res.x, dtype=np.float64)
    generation = x[g0 : g0 + nz * nf].reshape(nz, nf)
    net = x[n0 : n0 + nz]
    net_pos = x[n_pos0:n_neg0]
    net_neg = x[n_neg0:residual0]
    residual = x[residual0:ens0]
    ens = x[ens0:spill0]
    spill = x[spill0:slack_pos0]
    slack_pos = x[slack_pos0:slack_neg0]
    slack_neg = x[slack_neg0:reserve_slack0]
    reserve_slack = float(x[reserve_slack0]) if config.use_reserve_headroom_constraint else 0.0
    flows = incidence @ net

    balance = generation.sum(axis=1) + residual + ens - spill - zone_loads - net
    interface_violation = 0.0
    for k in range(nk):
        if upper_finite[k]:
            interface_violation = max(interface_violation, float(max(flows[k] - upper[k] - slack_pos[k], 0.0)))
        if lower_finite[k]:
            interface_violation = max(interface_violation, float(max(lower[k] - flows[k] - slack_neg[k], 0.0)))
    system_generation = generation.sum(axis=0)
    reserve_headroom = (
        float(dispatchable_capacity - generation[:, RESERVE_HEADROOM_FUEL_INDICES].sum())
        if config.use_reserve_headroom_constraint
        else 0.0
    )
    return ZonalDispatchResult(
        generation_by_zone=generation,
        net_injection=net,
        residual_supply=residual,
        ens=ens,
        spill=spill,
        interface_slack_pos=slack_pos,
        interface_slack_neg=slack_neg,
        interface_flows=flows,
        objective=float(res.fun),
        success=True,
        runtime_seconds=runtime,
        max_zonal_balance_residual=float(np.abs(balance).max()),
        system_net_injection_residual=float(abs(net.sum())),
        max_interface_violation_after_slack=interface_violation,
        max_ramp_violation_mw=_max_ramp_violation_with_capacity_clip(
            system_generation,
            previous_generation,
            ramp_limits,
            capacities.sum(axis=0),
        ),
        transfer_magnitude_mw=float((net_pos + net_neg).sum()),
        reserve_requirement_mw=reserve_requirement,
        reserve_headroom_mw=reserve_headroom,
        reserve_slack_mw=reserve_slack,
    )

def solve_zonal_dispatch_many(
    system_df: pd.DataFrame,
    zone_hour: pd.DataFrame,
    interface_hour: pd.DataFrame,
    costs: np.ndarray,
    zonal_capacities: np.ndarray,
    config: ExperimentConfig,
    *,
    ramp_limits: RampLimits | None = None,
    system_availability_caps: np.ndarray | None = None,
    load_column: str = "forecast_load_mw",
    repair_predictions: np.ndarray | None = None,
) -> tuple[np.ndarray, pd.DataFrame]:
    if config.use_zonal_interface_constraints:
        require_zonal_interface_hour(interface_hour, context="solve_zonal_dispatch_many")
    zone_groups = {ts: group for ts, group in zone_hour.groupby("timestamp_utc")}
    if "timestamp_utc" in interface_hour.columns and not interface_hour.empty:
        interface_groups = {ts: group for ts, group in interface_hour.groupby("timestamp_utc")}
    else:
        interface_groups = {}
    if repair_predictions is not None:
        repair_predictions = np.asarray(repair_predictions, dtype=np.float64)
        if repair_predictions.shape != (len(system_df), len(FUEL_CATEGORIES)):
            raise ValueError(
                f"Expected repair predictions shape {(len(system_df), len(FUEL_CATEGORIES))}, "
                f"received {repair_predictions.shape}"
            )
    predictions = []
    diagnostics = []
    previous_generation: np.ndarray | None = None
    interface_names = list(INTERNAL_INTERFACE_CUTS)

    for i, (_, row) in enumerate(system_df.iterrows()):
        timestamp = row["timestamp_utc"]
        if repair_predictions is not None and ramp_limits is not None:
            previous_generation = _previous_generation_from_row(row)
        elif previous_generation is None and ramp_limits is not None:
            previous_generation = _previous_generation_from_row(row)
        if timestamp not in zone_groups:
            raise KeyError(f"Missing zone-hour rows for {timestamp}")
        zone_rows = zone_groups[timestamp].set_index("zone_code").reindex(ZONE_CODES)
        if load_column == SOLVER_BALANCE_COLUMN:
            if "forecast_load_mw" not in zone_rows.columns:
                raise KeyError("Missing zonal forecast_load_mw needed to allocate internal-generation balance.")
            forecast_zone_loads = zone_rows["forecast_load_mw"].to_numpy(dtype=np.float64)
            total_forecast = float(np.nansum(forecast_zone_loads))
            if total_forecast <= 0.0:
                zone_loads = np.full(len(ZONE_CODES), float(row[load_column]) / len(ZONE_CODES), dtype=np.float64)
            else:
                zone_loads = forecast_zone_loads / total_forecast * float(row[load_column])
        elif load_column in row.index:
            if "forecast_load_mw" not in zone_rows.columns:
                raise KeyError(f"Missing zonal forecast_load_mw needed to allocate system load column: {load_column}")
            forecast_zone_loads = zone_rows["forecast_load_mw"].to_numpy(dtype=np.float64)
            total_forecast = float(np.nansum(forecast_zone_loads))
            if total_forecast <= 0.0:
                zone_loads = np.full(len(ZONE_CODES), float(row[load_column]) / len(ZONE_CODES), dtype=np.float64)
            else:
                zone_loads = forecast_zone_loads / total_forecast * float(row[load_column])
        elif load_column not in zone_rows.columns:
            raise KeyError(f"Missing zonal load column: {load_column}")
        else:
            zone_loads = zone_rows[load_column].to_numpy(dtype=np.float64)
        if not np.isfinite(zone_loads).all():
            raise ValueError(f"Non-finite zonal loads for {timestamp}")
        interface_rows = (
            interface_groups.get(timestamp, empty_interface_hour())
            if config.use_zonal_interface_constraints
            else empty_interface_hour()
        )
        availability_limits = None
        if config.use_zonal_availability_bounds:
            availability_limits = availability_limits_for_hour(
                zone_rows,
                zonal_capacities,
                system_availability_caps=system_availability_caps,
            )
        cost_i = costs[i] if np.asarray(costs).ndim in {2, 3} and np.asarray(costs).shape[0] == len(system_df) else costs
        repair_targets = (
            allocate_system_generation_to_zones(repair_predictions[i], zonal_capacities)
            if repair_predictions is not None
            else None
        )
        result = solve_zonal_dispatch(
            zone_loads,
            interface_rows,
            cost_i,
            zonal_capacities,
            config,
            availability_limits=availability_limits,
            repair_targets=repair_targets,
            previous_generation=previous_generation,
            ramp_limits=ramp_limits,
        )
        system_generation = result.generation_by_zone.sum(axis=0)
        predictions.append(system_generation)

        interface_slack = result.interface_slack_pos + result.interface_slack_neg
        upper, lower, upper_finite, lower_finite = _interface_bounds_for_hour(interface_rows)
        binding_count = 0
        for k in range(len(interface_names)):
            if upper_finite[k] and (result.interface_flows[k] >= upper[k] - 1e-5 or result.interface_slack_pos[k] > 1e-5):
                binding_count += 1
            if lower_finite[k] and (result.interface_flows[k] <= lower[k] + 1e-5 or result.interface_slack_neg[k] > 1e-5):
                binding_count += 1

        diag = {
            "timestamp_utc": timestamp,
            "solver_load_mw": float(zone_loads.sum()),
            "solver_load_column": load_column,
            "residual_supply": float(result.residual_supply.sum()),
            "ens": float(result.ens.sum()),
            "spill": float(result.spill.sum()),
            "objective": result.objective,
            "solver_success": result.success,
            "runtime_seconds": result.runtime_seconds,
            "balance_residual": result.max_zonal_balance_residual,
            "zonal_balance_max_abs_residual": result.max_zonal_balance_residual,
            "system_net_injection_residual": result.system_net_injection_residual,
            "interface_violation_after_slack_mw": result.max_interface_violation_after_slack,
            "interface_slack_total_mw": float(interface_slack.sum()),
            "interface_slack_max_mw": float(interface_slack.max(initial=0.0)),
            "binding_interface_count": binding_count,
            "max_ramp_violation_mw": result.max_ramp_violation_mw,
            "transfer_magnitude_mw": result.transfer_magnitude_mw,
            "reserve_requirement_mw": result.reserve_requirement_mw,
            "reserve_headroom_mw": result.reserve_headroom_mw,
            "reserve_slack_mw": result.reserve_slack_mw,
            "reserve_shortfall_after_slack_mw": max(
                result.reserve_requirement_mw - result.reserve_headroom_mw - result.reserve_slack_mw,
                0.0,
            ),
        }
        if repair_predictions is not None:
            diag["mean_abs_repair_delta_mw"] = float(np.mean(np.abs(system_generation - repair_predictions[i])))
        for k, name in enumerate(interface_names):
            key = name.lower().replace(" ", "_").replace("-", "_").replace("/", "_")
            diag[f"interface_flow_{key}_mw"] = float(result.interface_flows[k])
            diag[f"interface_slack_{key}_mw"] = float(interface_slack[k])
        diagnostics.append(diag)
        if ramp_limits is not None and repair_predictions is None:
            previous_generation = system_generation
    return np.asarray(predictions), pd.DataFrame(diagnostics)


def _zonal_solver_inputs_for_row(
    row: pd.Series,
    *,
    zone_groups: dict[object, pd.DataFrame],
    interface_groups: dict[object, pd.DataFrame],
    zonal_capacities: np.ndarray,
    config: ExperimentConfig,
    system_availability_caps: np.ndarray | None,
    load_column: str,
) -> tuple[np.ndarray, pd.DataFrame, np.ndarray | None]:
    timestamp = row["timestamp_utc"]
    if timestamp not in zone_groups:
        raise KeyError(f"Missing zone-hour rows for {timestamp}")
    zone_rows = zone_groups[timestamp].set_index("zone_code").reindex(ZONE_CODES)
    if load_column in row.index:
        if "forecast_load_mw" not in zone_rows.columns:
            raise KeyError(f"Missing zonal forecast_load_mw needed to allocate system load column: {load_column}")
        forecast_zone_loads = zone_rows["forecast_load_mw"].to_numpy(dtype=np.float64)
        total_forecast = float(np.nansum(forecast_zone_loads))
        if total_forecast <= 0.0:
            zone_loads = np.full(len(ZONE_CODES), float(row[load_column]) / len(ZONE_CODES), dtype=np.float64)
        else:
            zone_loads = forecast_zone_loads / total_forecast * float(row[load_column])
    elif load_column in zone_rows.columns:
        zone_loads = zone_rows[load_column].to_numpy(dtype=np.float64)
    else:
        raise KeyError(f"Missing zonal load column: {load_column}")
    if not np.isfinite(zone_loads).all():
        raise ValueError(f"Non-finite zonal loads for {timestamp}")

    interface_rows = (
        interface_groups.get(timestamp, empty_interface_hour())
        if config.use_zonal_interface_constraints
        else empty_interface_hour()
    )
    availability_limits = None
    if config.use_zonal_availability_bounds:
        availability_limits = availability_limits_for_hour(
            zone_rows,
            zonal_capacities,
            system_availability_caps=system_availability_caps,
        )
    return zone_loads, interface_rows, availability_limits


def _zonal_financial_cost(costs: np.ndarray, generation_by_zone: np.ndarray) -> float:
    cost_array = np.asarray(costs, dtype=np.float64)
    generation = np.asarray(generation_by_zone, dtype=np.float64)
    if cost_array.shape == (len(FUEL_CATEGORIES),):
        cost_array = np.tile(cost_array, (len(ZONE_CODES), 1))
    if cost_array.shape != generation.shape:
        raise ValueError(f"Cost shape {cost_array.shape} does not match zonal generation shape {generation.shape}")
    return float(np.sum(cost_array * generation))


def solve_zonal_dispatch_many_with_details(
    system_df: pd.DataFrame,
    zone_hour: pd.DataFrame,
    interface_hour: pd.DataFrame,
    costs: np.ndarray,
    zonal_capacities: np.ndarray,
    config: ExperimentConfig,
    *,
    ramp_limits: RampLimits | None = None,
    system_availability_caps: np.ndarray | None = None,
    load_column: str,
    real_financial_costs: np.ndarray | None = None,
    zonal_derates: np.ndarray | None = None,
    reserve_slack_penalties: np.ndarray | None = None,
    reserve_requirement_fractions: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    if config.use_zonal_interface_constraints:
        require_zonal_interface_hour(interface_hour, context="solve_zonal_dispatch_many_with_details")
    zone_groups = {ts: group for ts, group in zone_hour.groupby("timestamp_utc")}
    if "timestamp_utc" in interface_hour.columns and not interface_hour.empty:
        interface_groups = {ts: group for ts, group in interface_hour.groupby("timestamp_utc")}
    else:
        interface_groups = {}
    costs = np.asarray(costs, dtype=np.float64)
    if costs.ndim == 2 and costs.shape == (len(system_df), len(FUEL_CATEGORIES)):
        costs = np.repeat(costs[:, None, :], len(ZONE_CODES), axis=1)
    if costs.ndim == 2 and costs.shape == (len(ZONE_CODES), len(FUEL_CATEGORIES)):
        costs = np.repeat(costs[None, :, :], len(system_df), axis=0)
    if costs.shape != (len(system_df), len(ZONE_CODES), len(FUEL_CATEGORIES)):
        raise ValueError(
            "Expected costs with shape "
            f"{(len(system_df), len(ZONE_CODES), len(FUEL_CATEGORIES))}, received {costs.shape}"
        )
    if real_financial_costs is None:
        real_financial_costs = costs
    real_financial_costs = np.asarray(real_financial_costs, dtype=np.float64)
    if real_financial_costs.ndim == 2 and real_financial_costs.shape == (len(system_df), len(FUEL_CATEGORIES)):
        real_financial_costs = np.repeat(real_financial_costs[:, None, :], len(ZONE_CODES), axis=1)
    if real_financial_costs.shape != costs.shape:
        raise ValueError(f"Financial cost shape {real_financial_costs.shape} does not match costs shape {costs.shape}")
    if zonal_derates is not None:
        zonal_derates = np.asarray(zonal_derates, dtype=np.float64)
        if zonal_derates.shape != (len(system_df), len(ZONE_CODES), len(FUEL_CATEGORIES)):
            raise ValueError(
                "Expected zonal derates with shape "
                f"{(len(system_df), len(ZONE_CODES), len(FUEL_CATEGORIES))}, received {zonal_derates.shape}"
            )
    if reserve_slack_penalties is not None:
        reserve_slack_penalties = np.asarray(reserve_slack_penalties, dtype=np.float64)
        if reserve_slack_penalties.shape != (len(system_df),):
            raise ValueError(
                f"Expected reserve slack penalties with shape {(len(system_df),)}, "
                f"received {reserve_slack_penalties.shape}"
            )
    if reserve_requirement_fractions is not None:
        reserve_requirement_fractions = np.asarray(reserve_requirement_fractions, dtype=np.float64)
        if reserve_requirement_fractions.shape != (len(system_df),):
            raise ValueError(
                f"Expected reserve requirement fractions with shape {(len(system_df),)}, "
                f"received {reserve_requirement_fractions.shape}"
            )

    predictions = []
    zonal_predictions = []
    diagnostics = []
    interface_names = list(INTERNAL_INTERFACE_CUTS)
    for i, (_, row) in enumerate(system_df.iterrows()):
        zone_loads, interface_rows, availability_limits = _zonal_solver_inputs_for_row(
            row,
            zone_groups=zone_groups,
            interface_groups=interface_groups,
            zonal_capacities=zonal_capacities,
            config=config,
            system_availability_caps=system_availability_caps,
            load_column=load_column,
        )
        previous_generation = _previous_generation_from_row(row) if ramp_limits is not None else None
        derates_i = zonal_derates[i] if zonal_derates is not None else None
        effective_capacities = _apply_zonal_derates(zonal_capacities, derates_i)
        result = solve_zonal_dispatch(
            zone_loads,
            interface_rows,
            costs[i],
            effective_capacities,
            config,
            availability_limits=availability_limits,
            previous_generation=previous_generation,
            ramp_limits=ramp_limits,
            reserve_slack_penalty=reserve_slack_penalties[i] if reserve_slack_penalties is not None else None,
            reserve_requirement_fraction=(
                reserve_requirement_fractions[i] if reserve_requirement_fractions is not None else None
            ),
        )
        system_generation = result.generation_by_zone.sum(axis=0)
        real_cost = _zonal_financial_cost(real_financial_costs[i], result.generation_by_zone)
        proxy_cost = _zonal_financial_cost(costs[i], result.generation_by_zone)
        predictions.append(system_generation)
        zonal_predictions.append(result.generation_by_zone)

        interface_slack = result.interface_slack_pos + result.interface_slack_neg
        upper, lower, upper_finite, lower_finite = _interface_bounds_for_hour(interface_rows)
        binding_count = 0
        for k in range(len(interface_names)):
            if upper_finite[k] and (result.interface_flows[k] >= upper[k] - 1e-5 or result.interface_slack_pos[k] > 1e-5):
                binding_count += 1
            if lower_finite[k] and (result.interface_flows[k] <= lower[k] + 1e-5 or result.interface_slack_neg[k] > 1e-5):
                binding_count += 1
        diag = {
            "timestamp_utc": row["timestamp_utc"],
            "solver_load_mw": float(zone_loads.sum()),
            "solver_load_column": load_column,
            "residual_supply": float(result.residual_supply.sum()),
            "ens": float(result.ens.sum()),
            "spill": float(result.spill.sum()),
            "objective": result.objective,
            "proxy_dispatch_cost": proxy_cost,
            "financial_dispatch_cost": real_cost,
            "financial_dispatch_cost_per_mwh": real_cost / max(float(system_generation.sum()), 1.0),
            "solver_success": result.success,
            "runtime_seconds": result.runtime_seconds,
            "balance_residual": result.max_zonal_balance_residual,
            "zonal_balance_max_abs_residual": result.max_zonal_balance_residual,
            "system_net_injection_residual": result.system_net_injection_residual,
            "interface_violation_after_slack_mw": result.max_interface_violation_after_slack,
            "interface_slack_total_mw": float(interface_slack.sum()),
            "interface_slack_max_mw": float(interface_slack.max(initial=0.0)),
            "binding_interface_count": binding_count,
            "max_ramp_violation_mw": result.max_ramp_violation_mw,
            "transfer_magnitude_mw": result.transfer_magnitude_mw,
            "reserve_requirement_mw": result.reserve_requirement_mw,
            "reserve_headroom_mw": result.reserve_headroom_mw,
            "reserve_slack_mw": result.reserve_slack_mw,
            "reserve_slack_penalty": float(
                reserve_slack_penalties[i] if reserve_slack_penalties is not None else config.reserve_slack_penalty
            ),
            "reserve_requirement_fraction": float(
                reserve_requirement_fractions[i]
                if reserve_requirement_fractions is not None
                else config.reserve_requirement_fraction
            ),
            "reserve_shortfall_after_slack_mw": max(
                result.reserve_requirement_mw - result.reserve_headroom_mw - result.reserve_slack_mw,
                0.0,
            ),
            "mean_thermal_derate": float(derates_i[:, THERMAL_DERATE_FUEL_INDICES].mean()) if derates_i is not None else 0.0,
            "max_thermal_derate": float(derates_i[:, THERMAL_DERATE_FUEL_INDICES].max()) if derates_i is not None else 0.0,
        }
        if derates_i is not None:
            for fuel in THERMAL_DERATE_FUELS:
                fuel_idx = FUEL_CATEGORIES.index(fuel)
                key = fuel.lower().replace(" ", "_")
                diag[f"mean_derate_{key}"] = float(derates_i[:, fuel_idx].mean())
                for z_idx, zone in enumerate(ZONE_CODES):
                    diag[f"derate_{zone}_{key}"] = float(derates_i[z_idx, fuel_idx])
        for k, name in enumerate(interface_names):
            key = name.lower().replace(" ", "_").replace("-", "_").replace("/", "_")
            diag[f"interface_flow_{key}_mw"] = float(result.interface_flows[k])
            diag[f"interface_slack_{key}_mw"] = float(interface_slack[k])
        diagnostics.append(diag)
    return np.asarray(predictions), np.asarray(zonal_predictions), pd.DataFrame(diagnostics)


def persistence_baseline(train_df: pd.DataFrame, test_df: pd.DataFrame) -> np.ndarray:
    return persistence_lag_baseline(train_df, test_df, lag_hours=1)

def persistence_lag_baseline(train_df: pd.DataFrame, test_df: pd.DataFrame, *, lag_hours: int) -> np.ndarray:
    all_df = pd.concat([train_df, test_df], ignore_index=True).sort_values("timestamp_utc")
    shifted = all_df[TARGET_COLUMNS].shift(lag_hours).fillna(train_df[TARGET_COLUMNS].mean())
    return shifted.iloc[len(train_df) :].to_numpy(dtype=np.float64)

def day_ahead_persistence_rollout(train_df: pd.DataFrame, test_df: pd.DataFrame) -> np.ndarray:
    if len(train_df) >= 24:
        seed_profile = train_df.sort_values("timestamp_utc").tail(24)[TARGET_COLUMNS].to_numpy(dtype=np.float64)
    else:
        seed_profile = np.tile(train_df[TARGET_COLUMNS].mean().to_numpy(dtype=np.float64), (24, 1))
    repeats = int(np.ceil(len(test_df) / len(seed_profile)))
    return np.tile(seed_profile, (repeats, 1))[: len(test_df)]

def hour_of_week_baseline(train_df: pd.DataFrame, test_df: pd.DataFrame) -> np.ndarray:
    train = train_df.copy()
    test = test_df.copy()
    train["hour_of_week"] = train["dayofweek"] * 24 + train["hour"]
    test["hour_of_week"] = test["dayofweek"] * 24 + test["hour"]
    grouped = train.groupby("hour_of_week")[TARGET_COLUMNS].mean()
    global_mean = train[TARGET_COLUMNS].mean().to_numpy(dtype=np.float64)
    preds = []
    for key in test["hour_of_week"]:
        preds.append(grouped.loc[key].to_numpy(dtype=np.float64) if key in grouped.index else global_mean)
    return np.asarray(preds)

def train_direct_model(
    train_df: pd.DataFrame,
    test_context_df: pd.DataFrame,
    *,
    use_weather: bool,
    config: ExperimentConfig,
    use_residual: bool = True,
    return_solve_frame: bool = False,
) -> tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    cols = feature_columns(use_weather, config.feature_mode, include_lag=config.include_lags)
    validate_feature_policy(cols, config.feature_mode)
    history_window = config.resolved_history_window
    load_column = solver_load_column(config)
    x_train, y_train, _, _, train_row_indices = make_window_matrix(
        train_df,
        features=cols,
        history_window=history_window,
        load_column=load_column,
    )
    lag_cols = [f"{col}_lag_1" for col in TARGET_COLUMNS]
    y_lag_train = train_df.iloc[train_row_indices][lag_cols].to_numpy(dtype=np.float64)
    if use_residual:
        y_train = y_train - y_lag_train   # Δg_t
    x_test, y_test, _, _, test_row_indices = make_window_matrix(
        test_context_df,
        features=cols,
        history_window=history_window,
        load_column=load_column,
    )
    y_lag_test = test_context_df.iloc[test_row_indices][lag_cols].to_numpy(dtype=np.float64)
    sx = StandardScaler().fit(x_train)
    sy = StandardScaler().fit(y_train)
    model = MLPRegressor(
        hidden_layer_sizes=(config.hidden_dim, config.hidden_dim),
        activation="relu",
        solver="adam",
        learning_rate_init=0.001,
        max_iter=config.direct_epochs,
        random_state=config.seed,
        early_stopping=False,
    )
    model.fit(sx.transform(x_train), sy.transform(y_train))
    pred_residual = sy.inverse_transform(model.predict(sx.transform(x_test)))
    if use_residual:
        pred = y_lag_test + pred_residual
    else:
        pred = pred_residual
    pred = np.clip(pred, 0.0, None)
    if return_solve_frame:
        return pred, y_test, test_context_df.iloc[test_row_indices].copy()
    return pred, y_test

def _softplus(x: np.ndarray) -> np.ndarray:
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0)

def _sigmoid(x: np.ndarray) -> np.ndarray:
    positive = x >= 0
    out = np.empty_like(x, dtype=np.float64)
    out[positive] = 1.0 / (1.0 + np.exp(-x[positive]))
    exp_x = np.exp(x[~positive])
    out[~positive] = exp_x / (1.0 + exp_x)
    return out

def project_oracle_to_dispatch(
    oracle: np.ndarray,
    load: float,
    capacities: np.ndarray,
    *,
    previous_generation: np.ndarray | None = None,
    ramp_limits: RampLimits | None = None,
) -> np.ndarray:
    lower, upper = _generation_bounds(capacities, previous_generation, ramp_limits)
    projected = np.clip(oracle.astype(np.float64), lower, upper)
    target_total = max(float(load), float(lower.sum()))
    if projected.sum() > target_total and projected.sum() > lower.sum():
        reducible = projected - lower
        reduction = min(float(projected.sum() - target_total), float(reducible.sum()))
        if reduction > 0:
            projected = projected - reducible / reducible.sum() * reduction
    return projected

def generate_candidate_dispatches(
    load: float,
    oracle: np.ndarray,
    base_costs: np.ndarray,
    capacities: np.ndarray,
    config: ExperimentConfig,
    rng: np.random.Generator,
    *,
    previous_generation: np.ndarray | None = None,
    ramp_limits: RampLimits | None = None,
) -> list[np.ndarray]:
    candidates = [
        project_oracle_to_dispatch(
            oracle,
            load,
            capacities,
            previous_generation=previous_generation,
            ramp_limits=ramp_limits,
        )
    ]
    candidates.append(
        solve_fuel_dispatch(
            load,
            base_costs,
            capacities,
            config,
            previous_generation=previous_generation,
            ramp_limits=ramp_limits,
        ).generation
    )
    for _ in range(max(0, config.candidate_count - 2)):
        noise = rng.normal(0.0, config.candidate_noise_scale, size=base_costs.shape)
        candidate_costs = np.asarray(base_costs) * np.exp(noise)
        candidates.append(
            solve_fuel_dispatch(
                load,
                candidate_costs,
                capacities,
                config,
                previous_generation=previous_generation,
                ramp_limits=ramp_limits,
            ).generation
        )
    return candidates

@dataclass
class StructuredCostModel:
    scaler: StandardScaler
    weights: np.ndarray

    def predict_costs(self, x: np.ndarray) -> np.ndarray:
        x_scaled = self.scaler.transform(x)
        return _softplus(x_scaled @ self.weights) + 1e-3


@dataclass
class ResidualAwareCostModel:
    scaler: StandardScaler
    weights: np.ndarray

    def predict_adjustment_costs(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        x_scaled = self.scaler.transform(x)
        raw = _softplus(x_scaled @ self.weights) + 1e-3
        raw = raw.reshape(len(x_scaled), 2, len(TARGET_COLUMNS))
        return raw[:, 0, :], raw[:, 1, :]

def train_structured_cost_model(
    train_df: pd.DataFrame,
    *,
    use_weather: bool,
    capacities: np.ndarray,
    config: ExperimentConfig,
    ramp_limits: RampLimits | None = None,
) -> tuple[StructuredCostModel, list[float]]:
    if config.structured_loss not in {"candidate_softmax", "fenchel_young"}:
        raise ValueError(f"Unknown structured loss: {config.structured_loss}")
    cols = feature_columns(use_weather, config.feature_mode, include_lag=config.include_lags)
    validate_feature_policy(cols, config.feature_mode)
    history_window = config.resolved_history_window
    load_column = solver_load_column(config)
    x_train, y_train, loads, _, row_indices = make_window_matrix(
        train_df,
        features=cols,
        history_window=history_window,
        load_column=load_column,
    )
    scaler = StandardScaler().fit(x_train)
    x_scaled = scaler.transform(x_train)
    rng = np.random.default_rng(config.seed)
    weights = rng.normal(0.0, 0.01, size=(x_scaled.shape[1], len(TARGET_COLUMNS)))
    history = []
    for _ in range(config.structured_epochs):
        order = rng.permutation(len(x_scaled))
        epoch_loss = 0.0
        for idx in order:
            x = x_scaled[idx]
            z = x @ weights
            costs = _softplus(z) + 1e-3
            previous_generation = _previous_generation_from_row(train_df.iloc[row_indices[idx]])
            if config.structured_loss == "candidate_softmax":
                candidates = generate_candidate_dispatches(
                    loads[idx],
                    y_train[idx],
                    costs,
                    capacities,
                    config,
                    rng,
                    previous_generation=previous_generation,
                    ramp_limits=ramp_limits,
                )
                candidate_matrix = np.stack(candidates)
                scores = -(candidate_matrix @ costs) / config.temperature
                scores = scores - scores.max()
                probs = np.exp(scores) / np.exp(scores).sum()
                loss = -np.log(max(probs[0], 1e-12))
                expected_candidate = probs @ candidate_matrix
                grad_cost = (candidate_matrix[0] - expected_candidate) / config.temperature
            else:
                projected_oracle = solve_fuel_repair_dispatch(
                    y_train[idx],
                    loads[idx],
                    capacities,
                    config,
                    previous_generation=previous_generation,
                    ramp_limits=ramp_limits,
                ).generation
                easy_solution = solve_fuel_dispatch(
                    loads[idx],
                    costs,
                    capacities,
                    config,
                    previous_generation=previous_generation,
                    ramp_limits=ramp_limits,
                ).generation
                loss = float(costs @ projected_oracle - costs @ easy_solution)
                scale = max(float(abs(loads[idx])), 1.0)
                grad_cost = (projected_oracle - easy_solution) / scale
            epoch_loss += float(loss)
            grad_z = grad_cost * _sigmoid(z)
            weights -= config.structured_lr * np.outer(x, grad_z)
        history.append(epoch_loss / len(x_scaled))
    return StructuredCostModel(scaler=scaler, weights=weights), history


def _adjustments_from_previous(dispatch: np.ndarray, previous_generation: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    delta = np.asarray(dispatch, dtype=np.float64) - np.asarray(previous_generation, dtype=np.float64)
    return np.maximum(delta, 0.0), np.maximum(-delta, 0.0)


def train_residual_aware_cost_model(
    train_df: pd.DataFrame,
    *,
    use_weather: bool,
    capacities: np.ndarray,
    config: ExperimentConfig,
    ramp_limits: RampLimits | None = None,
    load_column: str | None = None,
    soft_balance_penalty: float = 0.1,
) -> tuple[ResidualAwareCostModel, list[float]]:
    cols = feature_columns(use_weather, config.feature_mode, include_lag=config.include_lags)
    validate_feature_policy(cols, config.feature_mode)
    train_load_column = load_column or solver_load_column(config)
    x_train, y_train, loads, _, row_indices = make_window_matrix(
        train_df,
        features=cols,
        history_window=config.resolved_history_window,
        load_column=train_load_column,
    )
    scaler = StandardScaler().fit(x_train)
    x_scaled = scaler.transform(x_train)
    rng = np.random.default_rng(config.seed + 17)
    weights = rng.normal(0.0, 0.01, size=(x_scaled.shape[1], len(TARGET_COLUMNS) * 2))

    previous_rows = train_df.iloc[row_indices].reset_index(drop=True)
    previous_generation = np.vstack([_previous_generation_from_row(row) for _, row in previous_rows.iterrows()])
    projected_oracle = []
    projected_up = []
    projected_down = []
    for i, (_, row) in enumerate(previous_rows.iterrows()):
        projected = solve_fuel_repair_dispatch(
            y_train[i],
            loads[i],
            capacities,
            config,
            previous_generation=previous_generation[i],
            ramp_limits=ramp_limits,
        ).generation
        projected_oracle.append(projected)
        up, down = _adjustments_from_previous(projected, previous_generation[i])
        projected_up.append(up)
        projected_down.append(down)
    projected_up_array = np.asarray(projected_up, dtype=np.float64)
    projected_down_array = np.asarray(projected_down, dtype=np.float64)

    history = []
    for _ in range(config.structured_epochs):
        order = rng.permutation(len(x_scaled))
        epoch_loss = 0.0
        for idx in order:
            x = x_scaled[idx]
            z = x @ weights
            cost_pair = (_softplus(z) + 1e-3).reshape(2, len(TARGET_COLUMNS))
            result = solve_residual_aware_cost_dispatch(
                previous_generation[idx],
                loads[idx],
                capacities,
                config,
                up_costs=cost_pair[0],
                down_costs=cost_pair[1],
                soft_balance_penalty=soft_balance_penalty,
                ramp_limits=ramp_limits,
            )
            pred_up, pred_down = _adjustments_from_previous(result.generation, previous_generation[idx])
            oracle_adjust = np.concatenate([projected_up_array[idx], projected_down_array[idx]])
            pred_adjust = np.concatenate([pred_up, pred_down])
            costs = cost_pair.reshape(-1)
            loss = float(costs @ oracle_adjust - costs @ pred_adjust)
            scale = max(float(abs(loads[idx])), 1.0)
            grad_cost = (oracle_adjust - pred_adjust) / scale
            weights -= config.structured_lr * np.outer(x, grad_cost * _sigmoid(z))
            epoch_loss += loss
        history.append(epoch_loss / len(x_scaled))
    return ResidualAwareCostModel(scaler=scaler, weights=weights), history


@dataclass
class ZonalFinancialCostModel:
    scaler: StandardScaler
    weights: np.ndarray
    derate_weights: np.ndarray | None = None
    reserve_penalty_weights: np.ndarray | None = None
    reserve_fraction_weights: np.ndarray | None = None
    feature_gate_vector: np.ndarray | None = None
    thermal_derate_max: float = 0.5
    thermal_derate_offset: float = 4.0
    reserve_slack_penalty: float = 5e4
    reserve_penalty_min: float = 5e3
    reserve_penalty_max: float = 2e5
    reserve_requirement_fraction: float = 0.02
    reserve_fraction_min: float = 0.0
    reserve_fraction_max: float = 0.06

    def _scaled_features(self, x: np.ndarray) -> np.ndarray:
        x_scaled = self.scaler.transform(x)
        if self.feature_gate_vector is not None:
            x_scaled = x_scaled * self.feature_gate_vector
        return x_scaled

    def predict_zonal_costs(self, x: np.ndarray) -> np.ndarray:
        x_scaled = self._scaled_features(x)
        raw = _softplus(x_scaled @ self.weights) + 1e-3
        return raw.reshape(len(x_scaled), len(ZONE_CODES), len(FUEL_CATEGORIES))

    def predict_zonal_derates(self, x: np.ndarray) -> np.ndarray:
        derates = np.zeros((len(x), len(ZONE_CODES), len(FUEL_CATEGORIES)), dtype=np.float64)
        if self.derate_weights is None:
            return derates
        x_scaled = self._scaled_features(x)
        raw = x_scaled @ self.derate_weights
        thermal = self.thermal_derate_max * _sigmoid(raw - self.thermal_derate_offset)
        thermal = thermal.reshape(len(x), len(ZONE_CODES), len(THERMAL_DERATE_FUEL_INDICES))
        for local_idx, fuel_idx in enumerate(THERMAL_DERATE_FUEL_INDICES):
            derates[:, :, fuel_idx] = thermal[:, :, local_idx]
        return derates

    def predict_reserve_slack_penalties(self, x: np.ndarray) -> np.ndarray:
        if self.reserve_penalty_weights is None:
            return np.full(len(x), self.reserve_slack_penalty, dtype=np.float64)
        x_scaled = self._scaled_features(x)
        raw = np.clip(x_scaled @ self.reserve_penalty_weights, -6.0, 6.0)
        penalties = self.reserve_slack_penalty * np.exp(raw)
        return np.clip(penalties, self.reserve_penalty_min, self.reserve_penalty_max)

    def predict_reserve_requirement_fractions(self, x: np.ndarray) -> np.ndarray:
        if self.reserve_fraction_weights is None:
            return np.full(len(x), self.reserve_requirement_fraction, dtype=np.float64)
        x_scaled = self._scaled_features(x)
        span = max(self.reserve_fraction_max - self.reserve_fraction_min, 1e-9)
        base_position = np.clip((self.reserve_requirement_fraction - self.reserve_fraction_min) / span, 1e-4, 1 - 1e-4)
        bias = float(np.log(base_position / (1.0 - base_position)))
        raw = np.clip(bias + x_scaled @ self.reserve_fraction_weights, -8.0, 8.0)
        fractions = self.reserve_fraction_min + span * _sigmoid(raw)
        return np.clip(fractions, self.reserve_fraction_min, self.reserve_fraction_max)


def _apply_zonal_derates(zonal_capacities: np.ndarray, derates: np.ndarray | None) -> np.ndarray:
    capacities = np.asarray(zonal_capacities, dtype=np.float64)
    if derates is None:
        return capacities
    derate_array = np.asarray(derates, dtype=np.float64)
    if derate_array.shape != capacities.shape:
        raise ValueError(f"Expected derate shape {capacities.shape}, received {derate_array.shape}")
    return capacities * (1.0 - np.clip(derate_array, 0.0, 0.95))


def _feature_gate_vector(cols: list[str], config: ExperimentConfig) -> np.ndarray:
    gates = np.ones(len(cols), dtype=np.float64)
    for i, col in enumerate(cols):
        if col in DA_CONSTRAINT_STRESS_FEATURE_COLUMNS or col.startswith("da_constraint_"):
            gates[i] *= float(config.gate_p511a)
        if col in SCHEDULED_OUTAGE_STRESS_FEATURE_COLUMNS or col.startswith("scheduled_outage_"):
            gates[i] *= float(config.gate_p54a)
        if col in DAY_AHEAD_SCHEDULED_OUTAGE_STRESS_FEATURE_COLUMNS or col.startswith("da_scheduled_outage_"):
            gates[i] *= float(config.gate_p54c)
        if col in DAY_AHEAD_ANCILLARY_STRESS_FEATURE_COLUMNS or col.startswith("da_ancillary_"):
            gates[i] *= float(config.gate_p5)
        if col in ZONAL_BID_LOAD_FEATURE_COLUMNS or col.startswith("p59_"):
            gates[i] *= float(config.gate_p59)
        if col in WEATHER_FEATURE_COLUMNS:
            gates[i] *= float(config.gate_weather)
        if col in COST_FEATURE_COLUMNS or col.startswith("financial_cost_") or col.startswith("fuel_price_"):
            gates[i] *= float(config.gate_cost)
    return gates


def _window_feature_gate_vector(cols: list[str], config: ExperimentConfig, width: int) -> np.ndarray:
    base = _feature_gate_vector(cols, config)
    if width == len(base):
        return base
    if width % len(base) != 0:
        raise ValueError(f"Window feature width {width} is not divisible by base feature count {len(base)}.")
    return np.tile(base, width // len(base))


def train_zonal_financial_cost_model(
    train_df: pd.DataFrame,
    *,
    use_weather: bool,
    zone_hour: pd.DataFrame,
    interface_hour: pd.DataFrame,
    zonal_capacities: np.ndarray,
    config: ExperimentConfig,
    ramp_limits: RampLimits | None = None,
    system_availability_caps: np.ndarray | None = None,
    load_column: str | None = None,
    max_train_samples: int = 0,
) -> tuple[ZonalFinancialCostModel, list[float]]:
    if config.use_zonal_interface_constraints:
        require_zonal_interface_hour(interface_hour, context="train_zonal_financial_cost_model")
    cols = feature_columns(
        use_weather,
        config.feature_mode,
        include_lag=config.include_lags,
        include_cost_features=True,
        include_da_constraint_stress_features=config.include_da_constraint_stress_features,
        include_scheduled_outage_stress_features=config.include_scheduled_outage_stress_features,
        include_ancillary_stress_features=config.include_ancillary_stress_features,
        include_day_ahead_ancillary_stress_features=config.include_day_ahead_ancillary_stress_features,
        include_day_ahead_scheduled_outage_stress_features=config.include_day_ahead_scheduled_outage_stress_features,
        include_zonal_bid_load_features=config.include_zonal_bid_load_features,
    )
    validate_feature_policy(cols, config.feature_mode)
    active_load_column = load_column or solver_load_column(config)
    x_train, _, _, _, row_indices = make_window_matrix(
        train_df,
        features=cols,
        history_window=config.resolved_history_window,
        load_column=active_load_column,
    )
    if max_train_samples and max_train_samples > 0 and len(x_train) > max_train_samples:
        selected = np.linspace(0, len(x_train) - 1, max_train_samples, dtype=int)
        x_train = x_train[selected]
        row_indices = row_indices[selected]

    scaler = StandardScaler().fit(x_train)
    x_scaled = scaler.transform(x_train)
    feature_gate_vector = _window_feature_gate_vector(cols, config, x_scaled.shape[1])
    x_scaled = x_scaled * feature_gate_vector
    rng = np.random.default_rng(config.seed + 31)
    weights = rng.normal(0.0, 0.01, size=(x_scaled.shape[1], len(ZONE_CODES) * len(FUEL_CATEGORIES)))
    derate_weights = None
    if config.use_learned_thermal_derates:
        derate_weights = rng.normal(
            0.0,
            0.01,
            size=(x_scaled.shape[1], len(ZONE_CODES) * len(THERMAL_DERATE_FUEL_INDICES)),
        )
    reserve_penalty_weights = None
    if config.use_learned_reserve_penalty:
        reserve_penalty_weights = rng.normal(0.0, 0.01, size=x_scaled.shape[1])
    reserve_fraction_weights = None
    if config.use_learned_reserve_fraction:
        reserve_fraction_weights = rng.normal(0.0, 0.01, size=x_scaled.shape[1])

    rows = train_df.iloc[row_indices].reset_index(drop=True)
    financial_costs = financial_cost_matrix_from_frame(rows, zonal=True)
    zone_groups = {ts: group for ts, group in zone_hour.groupby("timestamp_utc")}
    if "timestamp_utc" in interface_hour.columns and not interface_hour.empty:
        interface_groups = {ts: group for ts, group in interface_hour.groupby("timestamp_utc")}
    else:
        interface_groups = {}

    oracle_zonal = []
    oracle_reserve_slack = []
    oracle_reserve_headroom = []
    contexts = []
    for i, (_, row) in enumerate(rows.iterrows()):
        zone_loads, interface_rows, availability_limits = _zonal_solver_inputs_for_row(
            row,
            zone_groups=zone_groups,
            interface_groups=interface_groups,
            zonal_capacities=zonal_capacities,
            config=config,
            system_availability_caps=system_availability_caps,
            load_column=active_load_column,
        )
        previous_generation = _previous_generation_from_row(row) if ramp_limits is not None else None
        oracle = solve_zonal_dispatch(
            zone_loads,
            interface_rows,
            financial_costs[i],
            zonal_capacities,
            config,
            availability_limits=availability_limits,
            previous_generation=previous_generation,
            ramp_limits=ramp_limits,
        )
        oracle_zonal.append(oracle.generation_by_zone)
        oracle_reserve_slack.append(oracle.reserve_slack_mw)
        oracle_reserve_headroom.append(oracle.reserve_headroom_mw)
        contexts.append((zone_loads, interface_rows, availability_limits, previous_generation))
    oracle_zonal_array = np.asarray(oracle_zonal, dtype=np.float64)
    oracle_reserve_slack_array = np.asarray(oracle_reserve_slack, dtype=np.float64)
    oracle_reserve_headroom_array = np.asarray(oracle_reserve_headroom, dtype=np.float64)

    history = []
    for _ in range(config.structured_epochs):
        order = rng.permutation(len(x_scaled))
        epoch_loss = 0.0
        for idx in order:
            x = x_scaled[idx]
            z = x @ weights
            learned_costs = (_softplus(z) + 1e-3).reshape(len(ZONE_CODES), len(FUEL_CATEGORIES))
            learned_reserve_penalty = None
            reserve_penalty_raw = None
            if reserve_penalty_weights is not None:
                reserve_penalty_raw = float(np.clip(x @ reserve_penalty_weights, -6.0, 6.0))
                learned_reserve_penalty = float(
                    np.clip(
                        config.reserve_slack_penalty * np.exp(reserve_penalty_raw),
                        config.reserve_penalty_min,
                        config.reserve_penalty_max,
                    )
                )
            learned_reserve_fraction = None
            reserve_fraction_raw = None
            reserve_fraction_derivative = 0.0
            if reserve_fraction_weights is not None:
                span = max(config.reserve_fraction_max - config.reserve_fraction_min, 1e-9)
                base_position = np.clip(
                    (config.reserve_requirement_fraction - config.reserve_fraction_min) / span,
                    1e-4,
                    1 - 1e-4,
                )
                reserve_fraction_bias = float(np.log(base_position / (1.0 - base_position)))
                reserve_fraction_raw = float(np.clip(reserve_fraction_bias + x @ reserve_fraction_weights, -8.0, 8.0))
                reserve_fraction_sigmoid = float(1.0 / (1.0 + np.exp(-reserve_fraction_raw)))
                learned_reserve_fraction = float(
                    np.clip(
                        config.reserve_fraction_min + span * reserve_fraction_sigmoid,
                        config.reserve_fraction_min,
                        config.reserve_fraction_max,
                    )
                )
                reserve_fraction_derivative = span * reserve_fraction_sigmoid * (1.0 - reserve_fraction_sigmoid)
            learned_derates = None
            effective_capacities = zonal_capacities
            if derate_weights is not None:
                dz = x @ derate_weights
                thermal_derates = config.thermal_derate_max * _sigmoid(dz - config.thermal_derate_offset)
                learned_derates = np.zeros((len(ZONE_CODES), len(FUEL_CATEGORIES)), dtype=np.float64)
                thermal_derates_matrix = thermal_derates.reshape(len(ZONE_CODES), len(THERMAL_DERATE_FUEL_INDICES))
                for local_idx, fuel_idx in enumerate(THERMAL_DERATE_FUEL_INDICES):
                    learned_derates[:, fuel_idx] = thermal_derates_matrix[:, local_idx]
                effective_capacities = _apply_zonal_derates(zonal_capacities, learned_derates)
            zone_loads, interface_rows, availability_limits, previous_generation = contexts[idx]
            result = solve_zonal_dispatch(
                zone_loads,
                interface_rows,
                learned_costs,
                effective_capacities,
                config,
                availability_limits=availability_limits,
                previous_generation=previous_generation,
                ramp_limits=ramp_limits,
                reserve_slack_penalty=learned_reserve_penalty,
                reserve_requirement_fraction=learned_reserve_fraction,
            )
            oracle_flat = oracle_zonal_array[idx].reshape(-1)
            pred_flat = result.generation_by_zone.reshape(-1)
            learned_flat = learned_costs.reshape(-1)
            loss = float(learned_flat @ oracle_flat - learned_flat @ pred_flat)
            scale = max(float(np.sum(zone_loads)), 1.0)
            grad_cost = (oracle_flat - pred_flat) / scale
            weights -= config.structured_lr * np.outer(x, grad_cost * _sigmoid(z))
            if derate_weights is not None:
                excess = result.generation_by_zone[:, THERMAL_DERATE_FUEL_INDICES] - oracle_zonal_array[idx][
                    :, THERMAL_DERATE_FUEL_INDICES
                ]
                cap_scale = np.maximum(zonal_capacities[:, THERMAL_DERATE_FUEL_INDICES], 1.0)
                derate_signal = (excess / cap_scale).reshape(-1)
                derate_grad = derate_signal * (
                    config.thermal_derate_max
                    * _sigmoid(dz - config.thermal_derate_offset)
                    * (1.0 - _sigmoid(dz - config.thermal_derate_offset))
                )
                if config.thermal_derate_mean_penalty > 0.0:
                    derate_grad -= config.thermal_derate_mean_penalty * (
                        thermal_derates / max(config.thermal_derate_max, 1e-6)
                    ) * (
                        config.thermal_derate_max
                        * _sigmoid(dz - config.thermal_derate_offset)
                        * (1.0 - _sigmoid(dz - config.thermal_derate_offset))
                    )
                derate_weights += config.thermal_derate_lr * np.outer(x, derate_grad)
                if config.thermal_derate_l2 > 0.0:
                    derate_weights *= max(0.0, 1.0 - config.thermal_derate_lr * config.thermal_derate_l2)
            if reserve_penalty_weights is not None and learned_reserve_penalty is not None:
                slack_signal = (oracle_reserve_slack_array[idx] - result.reserve_slack_mw) / scale
                if config.reserve_penalty_min < learned_reserve_penalty < config.reserve_penalty_max:
                    reserve_penalty_grad = slack_signal * learned_reserve_penalty
                    reserve_penalty_weights -= config.reserve_penalty_lr * x * reserve_penalty_grad
                    if config.reserve_penalty_l2 > 0.0:
                        reserve_penalty_weights *= max(0.0, 1.0 - config.reserve_penalty_lr * config.reserve_penalty_l2)
            if reserve_fraction_weights is not None and learned_reserve_fraction is not None:
                headroom_signal = (oracle_reserve_headroom_array[idx] - result.reserve_headroom_mw) / scale
                if config.reserve_fraction_min < learned_reserve_fraction < config.reserve_fraction_max:
                    reserve_fraction_weights += (
                        config.reserve_fraction_lr * x * headroom_signal * reserve_fraction_derivative
                    )
                    if config.reserve_fraction_l2 > 0.0:
                        reserve_fraction_weights *= max(0.0, 1.0 - config.reserve_fraction_lr * config.reserve_fraction_l2)
            epoch_loss += loss
        history.append(epoch_loss / len(x_scaled))
    return (
        ZonalFinancialCostModel(
            scaler=scaler,
            weights=weights,
            derate_weights=derate_weights,
            reserve_penalty_weights=reserve_penalty_weights,
            reserve_fraction_weights=reserve_fraction_weights,
            feature_gate_vector=feature_gate_vector,
            thermal_derate_max=config.thermal_derate_max,
            thermal_derate_offset=config.thermal_derate_offset,
            reserve_slack_penalty=config.reserve_slack_penalty,
            reserve_penalty_min=config.reserve_penalty_min,
            reserve_penalty_max=config.reserve_penalty_max,
            reserve_requirement_fraction=config.reserve_requirement_fraction,
            reserve_fraction_min=config.reserve_fraction_min,
            reserve_fraction_max=config.reserve_fraction_max,
        ),
        history,
    )


def predict_zonal_financial_oracle(
    test_context_df: pd.DataFrame,
    *,
    use_weather: bool,
    zone_hour: pd.DataFrame,
    interface_hour: pd.DataFrame,
    zonal_capacities: np.ndarray,
    config: ExperimentConfig,
    ramp_limits: RampLimits | None = None,
    system_availability_caps: np.ndarray | None = None,
    load_column: str | None = None,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, np.ndarray, pd.DataFrame, np.ndarray]:
    cols = feature_columns(
        use_weather,
        config.feature_mode,
        include_lag=config.include_lags,
        include_cost_features=True,
        include_da_constraint_stress_features=config.include_da_constraint_stress_features,
        include_scheduled_outage_stress_features=config.include_scheduled_outage_stress_features,
        include_ancillary_stress_features=config.include_ancillary_stress_features,
        include_day_ahead_ancillary_stress_features=config.include_day_ahead_ancillary_stress_features,
        include_day_ahead_scheduled_outage_stress_features=config.include_day_ahead_scheduled_outage_stress_features,
        include_zonal_bid_load_features=config.include_zonal_bid_load_features,
    )
    validate_feature_policy(cols, config.feature_mode)
    active_load_column = load_column or solver_load_column(config)
    _, y_test, _, _, row_indices = make_window_matrix(
        test_context_df,
        features=cols,
        history_window=config.resolved_history_window,
        load_column=active_load_column,
    )
    solve_df = test_context_df.iloc[row_indices].copy()
    financial_costs = financial_cost_matrix_from_frame(solve_df, zonal=True)
    pred, zonal_pred, diagnostics = solve_zonal_dispatch_many_with_details(
        solve_df,
        zone_hour,
        interface_hour,
        financial_costs,
        zonal_capacities,
        config,
        ramp_limits=ramp_limits,
        system_availability_caps=system_availability_caps,
        load_column=active_load_column,
        real_financial_costs=financial_costs,
    )
    return pred, y_test, diagnostics, financial_costs, solve_df, zonal_pred


def predict_zonal_financial_cost_lto(
    model: ZonalFinancialCostModel,
    test_context_df: pd.DataFrame,
    *,
    use_weather: bool,
    zone_hour: pd.DataFrame,
    interface_hour: pd.DataFrame,
    zonal_capacities: np.ndarray,
    config: ExperimentConfig,
    ramp_limits: RampLimits | None = None,
    system_availability_caps: np.ndarray | None = None,
    load_column: str | None = None,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, np.ndarray, pd.DataFrame, np.ndarray]:
    cols = feature_columns(
        use_weather,
        config.feature_mode,
        include_lag=config.include_lags,
        include_cost_features=True,
        include_da_constraint_stress_features=config.include_da_constraint_stress_features,
        include_scheduled_outage_stress_features=config.include_scheduled_outage_stress_features,
        include_ancillary_stress_features=config.include_ancillary_stress_features,
        include_day_ahead_ancillary_stress_features=config.include_day_ahead_ancillary_stress_features,
        include_day_ahead_scheduled_outage_stress_features=config.include_day_ahead_scheduled_outage_stress_features,
        include_zonal_bid_load_features=config.include_zonal_bid_load_features,
    )
    validate_feature_policy(cols, config.feature_mode)
    active_load_column = load_column or solver_load_column(config)
    x_test, y_test, _, _, row_indices = make_window_matrix(
        test_context_df,
        features=cols,
        history_window=config.resolved_history_window,
        load_column=active_load_column,
    )
    learned_costs = model.predict_zonal_costs(x_test)
    learned_derates = model.predict_zonal_derates(x_test)
    learned_reserve_penalties = model.predict_reserve_slack_penalties(x_test)
    learned_reserve_fractions = model.predict_reserve_requirement_fractions(x_test)
    solve_df = test_context_df.iloc[row_indices].copy()
    financial_costs = financial_cost_matrix_from_frame(solve_df, zonal=True)
    pred, zonal_pred, diagnostics = solve_zonal_dispatch_many_with_details(
        solve_df,
        zone_hour,
        interface_hour,
        learned_costs,
        zonal_capacities,
        config,
        ramp_limits=ramp_limits,
        system_availability_caps=system_availability_caps,
        load_column=active_load_column,
        real_financial_costs=financial_costs,
        zonal_derates=learned_derates if config.use_learned_thermal_derates else None,
        reserve_slack_penalties=learned_reserve_penalties if config.use_learned_reserve_penalty else None,
        reserve_requirement_fractions=learned_reserve_fractions if config.use_learned_reserve_fraction else None,
    )
    return pred, y_test, diagnostics, learned_costs, solve_df, zonal_pred


def predict_structured_cost_lp(
    model: StructuredCostModel,
    test_context_df: pd.DataFrame,
    *,
    use_weather: bool,
    capacities: np.ndarray,
    config: ExperimentConfig,
    ramp_limits: RampLimits | None = None,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    cols = feature_columns(use_weather, config.feature_mode, include_lag=config.include_lags)
    validate_feature_policy(cols, config.feature_mode)
    history_window = config.resolved_history_window
    load_column = solver_load_column(config)
    x_test, y_test, _, _, row_indices = make_window_matrix(
        test_context_df,
        features=cols,
        history_window=history_window,
        load_column=load_column,
    )
    costs = model.predict_costs(x_test)
    solve_df = test_context_df.iloc[row_indices].copy()
    pred, diagnostics = solve_dispatch_many(
        solve_df,
        costs,
        capacities,
        config,
        ramp_limits=ramp_limits,
        load_column=load_column,
    )
    return pred, y_test, diagnostics

def predict_structured_cost_zonal_lp(
    model: StructuredCostModel,
    test_context_df: pd.DataFrame,
    *,
    use_weather: bool,
    zone_hour: pd.DataFrame,
    interface_hour: pd.DataFrame,
    zonal_capacities: np.ndarray,
    config: ExperimentConfig,
    ramp_limits: RampLimits | None = None,
    system_availability_caps: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    cols = feature_columns(use_weather, config.feature_mode, include_lag=config.include_lags)
    validate_feature_policy(cols, config.feature_mode)
    history_window = config.resolved_history_window
    load_column = solver_load_column(config)
    x_test, y_test, _, _, row_indices = make_window_matrix(
        test_context_df,
        features=cols,
        history_window=history_window,
        load_column=load_column,
    )
    costs = model.predict_costs(x_test)
    solve_df = test_context_df.iloc[row_indices].copy()
    pred, diagnostics = solve_zonal_dispatch_many(
        solve_df,
        zone_hour,
        interface_hour,
        costs,
        zonal_capacities,
        config,
        ramp_limits=ramp_limits,
        system_availability_caps=system_availability_caps,
        load_column=load_column,
    )
    return pred, y_test, diagnostics

def metrics(pred: np.ndarray, target: np.ndarray, diagnostics: pd.DataFrame | None = None) -> dict[str, float]:
    err = pred - target
    result: dict[str, float] = {
        "overall_mae": float(np.mean(np.abs(err))),
        "overall_rmse": float(np.sqrt(np.mean(err**2))),
        "total_fuel_mix_mae": float(np.mean(np.abs(pred.sum(axis=1) - target.sum(axis=1)))),
        "renewable_mae": float(np.mean(np.abs(err[:, [4, 5]]))),
        "thermal_mae": float(np.mean(np.abs(err[:, [0, 1]]))),
    }
    for i, fuel in enumerate(FUEL_CATEGORIES):
        key = fuel.lower().replace(" ", "_")
        result[f"mae_{key}"] = float(np.mean(np.abs(err[:, i])))
        result[f"rmse_{key}"] = float(np.sqrt(np.mean(err[:, i] ** 2)))
    if diagnostics is not None and not diagnostics.empty:
        result["mean_abs_balance_residual"] = float(diagnostics["balance_residual"].abs().mean())
        result["mean_runtime_seconds"] = float(diagnostics["runtime_seconds"].mean())
        result["mean_ens_mw"] = float(diagnostics["ens"].mean())
        result["mean_residual_supply_mw"] = float(diagnostics["residual_supply"].mean())
        if "max_ramp_violation_mw" in diagnostics.columns:
            result["max_ramp_violation_mw"] = float(diagnostics["max_ramp_violation_mw"].max())
        if "interface_slack_total_mw" in diagnostics.columns:
            result["mean_interface_slack_mw"] = float(diagnostics["interface_slack_total_mw"].mean())
        if "interface_slack_max_mw" in diagnostics.columns:
            result["max_interface_slack_mw"] = float(diagnostics["interface_slack_max_mw"].max())
        if "binding_interface_count" in diagnostics.columns:
            result["mean_binding_interface_count"] = float(diagnostics["binding_interface_count"].mean())
        if "zonal_balance_max_abs_residual" in diagnostics.columns:
            result["max_zonal_balance_residual"] = float(diagnostics["zonal_balance_max_abs_residual"].max())
        if "system_net_injection_residual" in diagnostics.columns:
            result["max_system_net_injection_residual"] = float(diagnostics["system_net_injection_residual"].max())
        if "interface_violation_after_slack_mw" in diagnostics.columns:
            result["max_interface_violation_after_slack_mw"] = float(
                diagnostics["interface_violation_after_slack_mw"].max()
            )
        if "transfer_magnitude_mw" in diagnostics.columns:
            result["mean_transfer_magnitude_mw"] = float(diagnostics["transfer_magnitude_mw"].mean())
        if "reserve_requirement_mw" in diagnostics.columns:
            result["mean_reserve_requirement_mw"] = float(diagnostics["reserve_requirement_mw"].mean())
        if "reserve_headroom_mw" in diagnostics.columns:
            result["mean_reserve_headroom_mw"] = float(diagnostics["reserve_headroom_mw"].mean())
        if "reserve_slack_mw" in diagnostics.columns:
            result["mean_reserve_slack_mw"] = float(diagnostics["reserve_slack_mw"].mean())
            result["max_reserve_slack_mw"] = float(diagnostics["reserve_slack_mw"].max())
        if "reserve_slack_penalty" in diagnostics.columns:
            result["mean_reserve_slack_penalty"] = float(diagnostics["reserve_slack_penalty"].mean())
            result["p05_reserve_slack_penalty"] = float(diagnostics["reserve_slack_penalty"].quantile(0.05))
            result["p50_reserve_slack_penalty"] = float(diagnostics["reserve_slack_penalty"].quantile(0.50))
            result["p95_reserve_slack_penalty"] = float(diagnostics["reserve_slack_penalty"].quantile(0.95))
        if "reserve_requirement_fraction" in diagnostics.columns:
            result["mean_reserve_requirement_fraction"] = float(diagnostics["reserve_requirement_fraction"].mean())
            result["p05_reserve_requirement_fraction"] = float(diagnostics["reserve_requirement_fraction"].quantile(0.05))
            result["p50_reserve_requirement_fraction"] = float(diagnostics["reserve_requirement_fraction"].quantile(0.50))
            result["p95_reserve_requirement_fraction"] = float(diagnostics["reserve_requirement_fraction"].quantile(0.95))
        if "reserve_shortfall_after_slack_mw" in diagnostics.columns:
            result["max_reserve_shortfall_after_slack_mw"] = float(
                diagnostics["reserve_shortfall_after_slack_mw"].max()
            )
        if "mean_thermal_derate" in diagnostics.columns:
            result["mean_thermal_derate"] = float(diagnostics["mean_thermal_derate"].mean())
        if "max_thermal_derate" in diagnostics.columns:
            result["max_thermal_derate"] = float(diagnostics["max_thermal_derate"].max())
        for fuel in THERMAL_DERATE_FUELS:
            key = fuel.lower().replace(" ", "_")
            col = f"mean_derate_{key}"
            if col in diagnostics.columns:
                result[col] = float(diagnostics[col].mean())
        if "mean_abs_repair_delta_mw" in diagnostics.columns:
            result["mean_abs_repair_delta_mw"] = float(diagnostics["mean_abs_repair_delta_mw"].mean())
        if "abs_balance_deviation_mw" in diagnostics.columns:
            result["mean_abs_balance_deviation_mw"] = float(diagnostics["abs_balance_deviation_mw"].mean())
        if "balance_under_mw" in diagnostics.columns:
            result["mean_balance_under_mw"] = float(diagnostics["balance_under_mw"].mean())
        if "balance_over_mw" in diagnostics.columns:
            result["mean_balance_over_mw"] = float(diagnostics["balance_over_mw"].mean())
    return result


def _projection_summary_row(
    variant: str,
    pred: np.ndarray,
    target: np.ndarray,
    diagnostics: pd.DataFrame | None = None,
) -> dict[str, float | str]:
    err = np.asarray(pred, dtype=np.float64) - np.asarray(target, dtype=np.float64)
    row: dict[str, float | str] = {
        "variant": variant,
        "overall_mae_to_oracle": float(np.mean(np.abs(err))),
        "projection_l1_mw": float(np.mean(np.abs(err))),
        "projection_l2_mw": float(np.mean(np.sqrt(np.mean(err**2, axis=1)))),
        "total_generation_gap_mw": float(np.mean(np.abs(err.sum(axis=1)))),
    }
    for i, fuel in enumerate(FUEL_CATEGORIES):
        key = fuel.lower().replace(" ", "_")
        row[f"mae_{key}"] = float(np.mean(np.abs(err[:, i])))
    if diagnostics is not None and not diagnostics.empty:
        row["success_rate"] = float(diagnostics["solver_success"].mean()) if "solver_success" in diagnostics else np.nan
        if "balance_residual" in diagnostics.columns:
            row["mean_abs_balance_residual"] = float(diagnostics["balance_residual"].abs().mean())
        if "max_ramp_violation_mw" in diagnostics.columns:
            row["max_ramp_violation_mw"] = float(diagnostics["max_ramp_violation_mw"].max())
        if "ens" in diagnostics.columns:
            row["mean_ens_mw"] = float(diagnostics["ens"].mean())
        if "spill" in diagnostics.columns:
            row["mean_spill_mw"] = float(diagnostics["spill"].mean())
        if "residual_supply" in diagnostics.columns:
            row["mean_residual_supply_mw"] = float(diagnostics["residual_supply"].mean())
        if "interface_slack_total_mw" in diagnostics.columns:
            row["mean_interface_slack_mw"] = float(diagnostics["interface_slack_total_mw"].mean())
        if "transfer_magnitude_mw" in diagnostics.columns:
            row["mean_transfer_magnitude_mw"] = float(diagnostics["transfer_magnitude_mw"].mean())
    else:
        row["success_rate"] = 1.0
    return row


def _projection_hourly_rows(
    variant: str,
    pred: np.ndarray,
    target: np.ndarray,
    frame: pd.DataFrame,
) -> pd.DataFrame:
    err = np.asarray(pred, dtype=np.float64) - np.asarray(target, dtype=np.float64)
    return pd.DataFrame(
        {
            "timestamp_utc": frame["timestamp_utc"].to_numpy(),
            "variant": variant,
            "oracle_total_mw": target.sum(axis=1),
            "projected_total_mw": pred.sum(axis=1),
            "solver_balance_mw": frame[SOLVER_BALANCE_COLUMN].to_numpy(dtype=np.float64),
            "forecast_load_mw": frame["forecast_load_mw"].to_numpy(dtype=np.float64),
            "target_internal_generation_mw": frame["target_internal_generation_mw"].to_numpy(dtype=np.float64),
            "projection_l1_mw": np.mean(np.abs(err), axis=1),
            "projection_l2_mw": np.sqrt(np.mean(err**2, axis=1)),
            "total_generation_gap_mw": err.sum(axis=1),
        }
    )


def _safe_group_median_prediction(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    group_cols: list[str],
    value_col: str,
) -> pd.Series:
    medians = train_df.groupby(group_cols)[value_col].median()
    global_median = float(train_df[value_col].median())
    indexed = test_df.set_index(group_cols).index
    values = pd.Series(indexed.map(medians), index=test_df.index, dtype=np.float64)
    return values.fillna(global_median)


def _train_direct_balance_residual_model(
    train_df: pd.DataFrame,
    test_context_df: pd.DataFrame,
    *,
    use_weather: bool,
    config: ExperimentConfig,
) -> tuple[pd.Series, pd.DataFrame]:
    cols = feature_columns(use_weather, config.feature_mode, include_lag=config.include_lags)
    validate_feature_policy(cols, config.feature_mode)
    history_window = config.resolved_history_window
    load_column = solver_load_column(config)
    x_train, _, _, _, train_row_indices = make_window_matrix(
        train_df,
        features=cols,
        history_window=history_window,
        load_column=load_column,
    )
    x_test, _, _, _, test_row_indices = make_window_matrix(
        test_context_df,
        features=cols,
        history_window=history_window,
        load_column=load_column,
    )
    y_train = train_df.iloc[train_row_indices]["forecast_net_import_residual_mw"].to_numpy(dtype=np.float64).reshape(-1, 1)
    sx = StandardScaler().fit(x_train)
    sy = StandardScaler().fit(y_train)
    model = MLPRegressor(
        hidden_layer_sizes=(config.hidden_dim, config.hidden_dim),
        activation="relu",
        solver="adam",
        learning_rate_init=0.001,
        max_iter=config.direct_epochs,
        random_state=config.seed,
        early_stopping=False,
    )
    model.fit(sx.transform(x_train), sy.transform(y_train).ravel())
    residual_pred = sy.inverse_transform(model.predict(sx.transform(x_test)).reshape(-1, 1)).ravel()
    solve_df = test_context_df.iloc[test_row_indices].copy()
    balance = pd.Series(
        solve_df["forecast_load_mw"].to_numpy(dtype=np.float64) - residual_pred,
        index=solve_df.index,
        dtype=np.float64,
    ).clip(lower=0.0)
    return balance, solve_df


def _balance_target_summary_row(
    target_name: str,
    candidate_balance: np.ndarray,
    oracle_total: np.ndarray,
    causal_policy: str,
    repair_pred: np.ndarray,
    oracle_dispatch: np.ndarray,
    repair_diag: pd.DataFrame,
) -> dict[str, float | str]:
    candidate_balance = np.asarray(candidate_balance, dtype=np.float64)
    oracle_total = np.asarray(oracle_total, dtype=np.float64)
    error = candidate_balance - oracle_total
    abs_error = np.abs(error)
    row: dict[str, float | str] = {
        "target": target_name,
        "causal_policy": causal_policy,
        "abs_error_mw": float(abs_error.mean()),
        "rmse_mw": float(np.sqrt(np.mean(error**2))),
        "bias_mw": float(error.mean()),
        "p50_abs_error_mw": float(np.quantile(abs_error, 0.50)),
        "p90_abs_error_mw": float(np.quantile(abs_error, 0.90)),
        "p95_abs_error_mw": float(np.quantile(abs_error, 0.95)),
        "correlation_with_oracle_total": (
            float(np.corrcoef(candidate_balance, oracle_total)[0, 1])
            if np.std(candidate_balance) > 0.0 and np.std(oracle_total) > 0.0
            else np.nan
        ),
        **{f"repair_{key}": value for key, value in metrics(repair_pred, oracle_dispatch, repair_diag).items()},
        "success_rate": (
            float(repair_diag["solver_success"].mean())
            if "solver_success" in repair_diag.columns and len(repair_diag)
            else np.nan
        ),
    }
    return row


def _balance_target_hourly_rows(
    target_name: str,
    candidate_balance: np.ndarray,
    oracle_total: np.ndarray,
    frame: pd.DataFrame,
    causal_policy: str,
    repair_pred: np.ndarray,
    oracle_dispatch: np.ndarray,
    repair_diag: pd.DataFrame,
) -> pd.DataFrame:
    candidate_balance = np.asarray(candidate_balance, dtype=np.float64)
    oracle_total = np.asarray(oracle_total, dtype=np.float64)
    dispatch_err = np.asarray(repair_pred, dtype=np.float64) - np.asarray(oracle_dispatch, dtype=np.float64)
    rows = pd.DataFrame(
        {
            "timestamp_utc": frame["timestamp_utc"].to_numpy(),
            "target": target_name,
            "causal_policy": causal_policy,
            "oracle_total_mw": oracle_total,
            "candidate_balance_mw": candidate_balance,
            "balance_target_error_mw": candidate_balance - oracle_total,
            "abs_error_mw": np.abs(candidate_balance - oracle_total),
            "forecast_load_mw": frame["forecast_load_mw"].to_numpy(dtype=np.float64),
            "actual_load_mw": frame["actual_load_mw"].to_numpy(dtype=np.float64)
            if "actual_load_mw" in frame.columns
            else np.nan,
            "solver_balance_mw": frame[SOLVER_BALANCE_COLUMN].to_numpy(dtype=np.float64),
            "target_internal_generation_mw": frame["target_internal_generation_mw"].to_numpy(dtype=np.float64),
            "repair_projection_l1_mw": np.mean(np.abs(dispatch_err), axis=1),
            "repair_projection_l2_mw": np.sqrt(np.mean(dispatch_err**2, axis=1)),
            "repair_total_generation_gap_mw": repair_pred.sum(axis=1) - oracle_dispatch.sum(axis=1),
        }
    )
    diag_cols = [
        "solver_success",
        "balance_residual",
        "max_ramp_violation_mw",
        "ens",
        "spill",
        "residual_supply",
        "runtime_seconds",
        "mean_abs_repair_delta_mw",
    ]
    for col in diag_cols:
        if col in repair_diag.columns:
            rows[col] = repair_diag[col].to_numpy()
    return rows


def add_balance_target_candidates(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    *,
    config: ExperimentConfig,
    test_context_df: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, dict[str, str]]:
    train = train_df.copy()
    test = test_df.copy()
    if "hour_of_week" not in train.columns:
        train["hour_of_week"] = train["dayofweek"] * 24 + train["hour"]
    if "hour_of_week" not in test.columns:
        test["hour_of_week"] = test["dayofweek"] * 24 + test["hour"]
    if "forecast_net_import_residual_mw" not in train.columns:
        train["forecast_net_import_residual_mw"] = train["forecast_load_mw"] - train[TARGET_COLUMNS].sum(axis=1)
    if "forecast_net_import_residual_mw" not in test.columns:
        test["forecast_net_import_residual_mw"] = test["forecast_load_mw"] - test[TARGET_COLUMNS].sum(axis=1)

    candidates = pd.DataFrame({"timestamp_utc": test["timestamp_utc"].to_numpy()})
    policies: dict[str, str] = {}

    def add(name: str, values: pd.Series | np.ndarray, policy: str) -> None:
        candidates[name] = np.clip(np.asarray(values, dtype=np.float64), 0.0, None)
        policies[name] = policy

    add("forecast_load_mw", test["forecast_load_mw"], "causal")
    if "actual_load_mw" in test.columns:
        add("actual_load_mw", test["actual_load_mw"], "diagnostic")
    add("solver_balance_mw", test[SOLVER_BALANCE_COLUMN], "causal")

    global_residual = float(train["forecast_net_import_residual_mw"].median())
    add(
        "forecast_load_minus_global_median_residual",
        test["forecast_load_mw"] - global_residual,
        "causal",
    )
    add(
        "forecast_load_minus_hour_of_week_median_residual",
        test["forecast_load_mw"]
        - _safe_group_median_prediction(train, test, ["hour_of_week"], "forecast_net_import_residual_mw"),
        "causal",
    )
    add(
        "forecast_load_minus_hour_of_day_median_residual",
        test["forecast_load_mw"]
        - _safe_group_median_prediction(train, test, ["hour"], "forecast_net_import_residual_mw"),
        "causal",
    )
    add(
        "forecast_load_minus_day_type_hour_median_residual",
        test["forecast_load_mw"]
        - _safe_group_median_prediction(train, test, ["is_weekend", "hour"], "forecast_net_import_residual_mw"),
        "causal",
    )
    if "forecast_net_import_residual_mw_lag_1" in test.columns:
        fallback = _safe_group_median_prediction(train, test, ["hour_of_week"], "forecast_net_import_residual_mw")
        add(
            "forecast_load_minus_lag_1_forecast_residual",
            test["forecast_load_mw"] - test["forecast_net_import_residual_mw_lag_1"].fillna(fallback),
            "causal",
        )
    if "forecast_net_import_residual_mw_lag_24" in test.columns:
        fallback = _safe_group_median_prediction(train, test, ["hour_of_week"], "forecast_net_import_residual_mw")
        add(
            "forecast_load_minus_lag_24_forecast_residual",
            test["forecast_load_mw"] - test["forecast_net_import_residual_mw_lag_24"].fillna(fallback),
            "causal",
        )
    if test_context_df is not None:
        nn_balance, nn_solve_df = _train_direct_balance_residual_model(
            train,
            test_context_df,
            use_weather=True,
            config=config,
        )
        nn_by_timestamp = pd.Series(
            nn_balance.to_numpy(dtype=np.float64),
            index=pd.to_datetime(nn_solve_df["timestamp_utc"], utc=True),
        )
        aligned = pd.to_datetime(test["timestamp_utc"], utc=True).map(nn_by_timestamp)
        add("direct_residual_model_balance", aligned.fillna(test[SOLVER_BALANCE_COLUMN]), "causal")
    add("oracle_total_balance", test[TARGET_COLUMNS].sum(axis=1), "diagnostic")
    return candidates, policies


def run_balance_target_diagnostics(
    system_hour_path: str | Path = "data/processed/nyiso_system_hour.parquet",
    *,
    config: ExperimentConfig | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, pd.DataFrame]]:
    config = resolve_experiment_config(config or ExperimentConfig())
    df = load_system_hour(system_hour_path).copy()
    if "forecast_net_import_residual_mw" not in df.columns:
        df["forecast_net_import_residual_mw"] = df["forecast_load_mw"] - df[TARGET_COLUMNS].sum(axis=1)
    df["forecast_net_import_residual_mw_lag_1"] = df["forecast_net_import_residual_mw"].shift(1)
    df["forecast_net_import_residual_mw_lag_24"] = df["forecast_net_import_residual_mw"].shift(24)
    train_df, val_df, test_df = chronological_split(df)
    train_df, val_df, test_df = add_solver_balance_targets(
        train_df,
        train_df,
        val_df,
        test_df,
        balance_target_policy=config.balance_target_policy,
    )
    target = test_df[TARGET_COLUMNS].to_numpy(dtype=np.float64)
    oracle_total = target.sum(axis=1)
    capacities = compute_available_capacities(train_df)
    ramp_limits = compute_empirical_ramp_limits(train_df, config.ramp_quantile) if config.use_ramp_constraints else None
    test_context = make_aligned_test_frame(train_df, test_df, config.resolved_history_window)
    candidates, policies = add_balance_target_candidates(
        train_df,
        test_df,
        config=config,
        test_context_df=test_context,
    )

    summary_rows = []
    hourly_rows = []
    diagnostics: dict[str, pd.DataFrame] = {}
    for name in [col for col in candidates.columns if col != "timestamp_utc"]:
        solve_df = test_df.copy()
        load_col = f"candidate_balance_{name}"
        solve_df[load_col] = candidates[name].to_numpy(dtype=np.float64)
        repair_pred, repair_diag = solve_repair_dispatch_many(
            solve_df,
            target,
            capacities,
            config,
            ramp_limits=ramp_limits,
            load_column=load_col,
        )
        diagnostics[name] = repair_diag
        candidate_balance = solve_df[load_col].to_numpy(dtype=np.float64)
        policy = policies.get(name, "causal")
        summary_rows.append(
            _balance_target_summary_row(
                name,
                candidate_balance,
                oracle_total,
                policy,
                repair_pred,
                target,
                repair_diag,
            )
        )
        hourly_rows.append(
            _balance_target_hourly_rows(
                name,
                candidate_balance,
                oracle_total,
                test_df,
                policy,
                repair_pred,
                target,
                repair_diag,
            )
        )

    summary = pd.DataFrame(summary_rows).sort_values(["causal_policy", "abs_error_mw"]).reset_index(drop=True)
    hourly = pd.concat(hourly_rows, ignore_index=True)
    return summary, hourly, diagnostics


def add_loss_experiment_balance_column(*frames: pd.DataFrame) -> tuple[pd.DataFrame, ...]:
    out = []
    for frame in frames:
        df = frame.copy()
        if "forecast_net_import_residual_mw_lag_1" not in df.columns:
            df["forecast_net_import_residual_mw_lag_1"] = df["forecast_net_import_residual_mw"].shift(1)
        lag_residual = df["forecast_net_import_residual_mw_lag_1"]
        lag_balance = df["forecast_load_mw"] - lag_residual
        df[LOSS_EXPERIMENT_BALANCE_COLUMN] = lag_balance.where(lag_residual.notna(), df[SOLVER_BALANCE_COLUMN])
        df[LOSS_EXPERIMENT_BALANCE_COLUMN] = df[LOSS_EXPERIMENT_BALANCE_COLUMN].clip(lower=0.0)
        out.append(df)
    return tuple(out)


def _predict_structured_cost_lp_with_costs(
    model: StructuredCostModel,
    test_context_df: pd.DataFrame,
    *,
    use_weather: bool,
    capacities: np.ndarray,
    config: ExperimentConfig,
    ramp_limits: RampLimits | None = None,
    load_column: str | None = None,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, np.ndarray, pd.DataFrame]:
    cols = feature_columns(use_weather, config.feature_mode, include_lag=config.include_lags)
    validate_feature_policy(cols, config.feature_mode)
    x_test, y_test, _, _, row_indices = make_window_matrix(
        test_context_df,
        features=cols,
        history_window=config.resolved_history_window,
        load_column=load_column or solver_load_column(config),
    )
    costs = model.predict_costs(x_test)
    solve_df = test_context_df.iloc[row_indices].copy()
    pred, diagnostics = solve_dispatch_many(
        solve_df,
        costs,
        capacities,
        config,
        ramp_limits=ramp_limits,
        load_column=load_column,
    )
    return pred, y_test, diagnostics, costs, solve_df


def _predict_residual_aware_cost_lp_with_costs(
    model: ResidualAwareCostModel,
    test_context_df: pd.DataFrame,
    *,
    use_weather: bool,
    capacities: np.ndarray,
    config: ExperimentConfig,
    ramp_limits: RampLimits | None = None,
    load_column: str | None = None,
    soft_balance_penalty: float = 0.1,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, np.ndarray, pd.DataFrame]:
    cols = feature_columns(use_weather, config.feature_mode, include_lag=config.include_lags)
    validate_feature_policy(cols, config.feature_mode)
    active_load_column = load_column or solver_load_column(config)
    x_test, y_test, _, _, row_indices = make_window_matrix(
        test_context_df,
        features=cols,
        history_window=config.resolved_history_window,
        load_column=active_load_column,
    )
    up_costs, down_costs = model.predict_adjustment_costs(x_test)
    solve_df = test_context_df.iloc[row_indices].copy()
    pred, diagnostics = solve_residual_aware_cost_dispatch_many(
        solve_df,
        up_costs,
        down_costs,
        capacities,
        config,
        soft_balance_penalty=soft_balance_penalty,
        ramp_limits=ramp_limits,
        load_column=active_load_column,
    )
    cost_matrix = np.concatenate([up_costs, down_costs], axis=1)
    return pred, y_test, diagnostics, cost_matrix, solve_df


def _residual_adjustment_fy_values(
    cost_matrix: np.ndarray,
    pred: np.ndarray,
    projected_oracle: np.ndarray,
    frame: pd.DataFrame,
) -> np.ndarray:
    costs = np.asarray(cost_matrix, dtype=np.float64)
    if costs.shape != (len(pred), len(TARGET_COLUMNS) * 2):
        raise ValueError(f"Expected residual cost matrix shape {(len(pred), len(TARGET_COLUMNS) * 2)}, received {costs.shape}")
    previous = np.vstack([_previous_generation_from_row(row) for _, row in frame.iterrows()])
    pred_up = np.maximum(pred - previous, 0.0)
    pred_down = np.maximum(previous - pred, 0.0)
    oracle_up = np.maximum(projected_oracle - previous, 0.0)
    oracle_down = np.maximum(previous - projected_oracle, 0.0)
    pred_adjust = np.concatenate([pred_up, pred_down], axis=1)
    oracle_adjust = np.concatenate([oracle_up, oracle_down], axis=1)
    return np.sum(costs * (oracle_adjust - pred_adjust), axis=1)


def _loss_experiment_summary_row(
    model_name: str,
    pred: np.ndarray,
    target: np.ndarray,
    projected_oracle: np.ndarray,
    balance_target: np.ndarray,
    diagnostics: pd.DataFrame | None = None,
    costs: np.ndarray | None = None,
    fenchel_young_values: np.ndarray | None = None,
) -> dict[str, float | str]:
    base = metrics(pred, target, diagnostics)
    diff = np.asarray(pred, dtype=np.float64) - np.asarray(projected_oracle, dtype=np.float64)
    if fenchel_young_values is not None:
        fy_values = np.asarray(fenchel_young_values, dtype=np.float64)
    elif costs is None:
        cost_matrix = np.tile(FIXED_FUEL_COSTS, (len(pred), 1))
        fy_values = np.sum(cost_matrix * (projected_oracle - pred), axis=1)
    else:
        cost_matrix = np.asarray(costs, dtype=np.float64)
        if cost_matrix.ndim == 1:
            cost_matrix = np.tile(cost_matrix, (len(pred), 1))
        if cost_matrix.shape[1] != pred.shape[1]:
            fy_values = np.full(len(pred), np.nan, dtype=np.float64)
        else:
            fy_values = np.sum(cost_matrix * (projected_oracle - pred), axis=1)
    return {
        "model": model_name,
        **base,
        "solution_l1_to_projected_oracle": float(np.mean(np.abs(diff))),
        "solution_l2_to_projected_oracle": float(np.mean(np.sqrt(np.mean(diff**2, axis=1)))),
        "fenchel_young_loss": float(np.nanmean(fy_values)),
        "objective_gap_under_learned_cost": float(np.nanmean(-fy_values)),
        "balance_target_abs_error_mw": float(np.mean(np.abs(balance_target - target.sum(axis=1)))),
        "success_rate": (
            float(diagnostics["solver_success"].mean())
            if diagnostics is not None and "solver_success" in diagnostics.columns and len(diagnostics)
            else 1.0
        ),
    }


def _loss_experiment_hourly_rows(
    model_name: str,
    pred: np.ndarray,
    target: np.ndarray,
    projected_oracle: np.ndarray,
    frame: pd.DataFrame,
    diagnostics: pd.DataFrame | None = None,
    costs: np.ndarray | None = None,
    fenchel_young_values: np.ndarray | None = None,
) -> pd.DataFrame:
    diff = np.asarray(pred, dtype=np.float64) - np.asarray(projected_oracle, dtype=np.float64)
    if fenchel_young_values is not None:
        fy_values = np.asarray(fenchel_young_values, dtype=np.float64)
    elif costs is None:
        cost_matrix = np.tile(FIXED_FUEL_COSTS, (len(pred), 1))
        fy_values = np.sum(cost_matrix * (projected_oracle - pred), axis=1)
    else:
        cost_matrix = np.asarray(costs, dtype=np.float64)
        if cost_matrix.ndim == 1:
            cost_matrix = np.tile(cost_matrix, (len(pred), 1))
        if cost_matrix.shape[1] != pred.shape[1]:
            fy_values = np.full(len(pred), np.nan, dtype=np.float64)
        else:
            fy_values = np.sum(cost_matrix * (projected_oracle - pred), axis=1)
    rows = pd.DataFrame(
        {
            "timestamp_utc": frame["timestamp_utc"].to_numpy(),
            "model": model_name,
            "oracle_total_mw": target.sum(axis=1),
            "projected_oracle_total_mw": projected_oracle.sum(axis=1),
            "predicted_total_mw": pred.sum(axis=1),
            "balance_target_mw": frame[LOSS_EXPERIMENT_BALANCE_COLUMN].to_numpy(dtype=np.float64),
            "solution_l1_to_projected_oracle": np.mean(np.abs(diff), axis=1),
            "solution_l2_to_projected_oracle": np.sqrt(np.mean(diff**2, axis=1)),
            "fenchel_young_loss": fy_values,
            "objective_gap_under_learned_cost": -fy_values,
        }
    )
    if diagnostics is not None:
        for col in [
            "solver_success",
            "balance_residual",
            "max_ramp_violation_mw",
            "runtime_seconds",
            "ens",
            "spill",
            "residual_supply",
        ]:
            if col in diagnostics.columns:
                rows[col] = diagnostics[col].to_numpy()
    return rows


"""
# OUTDATED
Legacy residual/repair experiment runners are commented out for the final
public SCUC proxy submission. The active path is run_zonal_cost_lto.py with
A5b/A6c/A7b stages. These old runners are preserved here only as commented
source text for historical reproducibility.

def run_loss_function_experiments(
    system_hour_path: str | Path = "data/processed/nyiso_system_hour.parquet",
    *,
    config: ExperimentConfig | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, pd.DataFrame]]:
    config = resolve_experiment_config(config or ExperimentConfig())
    df = load_system_hour(system_hour_path).copy()
    if "forecast_net_import_residual_mw" not in df.columns:
        df["forecast_net_import_residual_mw"] = df["forecast_load_mw"] - df[TARGET_COLUMNS].sum(axis=1)
    df["forecast_net_import_residual_mw_lag_1"] = df["forecast_net_import_residual_mw"].shift(1)
    train_df, val_df, test_df = chronological_split(df)
    train_df, val_df, test_df = add_solver_balance_targets(
        train_df,
        train_df,
        val_df,
        test_df,
        balance_target_policy=config.balance_target_policy,
    )
    train_df, val_df, test_df = add_loss_experiment_balance_column(train_df, val_df, test_df)
    loss_config = replace(
        config,
        load_balance_mode="forecast_load",
        solver_load_column=LOSS_EXPERIMENT_BALANCE_COLUMN,
        include_lags=True,
    )
    capacities = compute_available_capacities(train_df)
    ramp_limits = compute_empirical_ramp_limits(train_df, config.ramp_quantile) if config.use_ramp_constraints else None
    target = test_df[TARGET_COLUMNS].to_numpy(dtype=np.float64)
    balance_target = test_df[LOSS_EXPERIMENT_BALANCE_COLUMN].to_numpy(dtype=np.float64)
    projected_oracle, projected_diag = solve_repair_dispatch_many(
        test_df,
        target,
        capacities,
        loss_config,
        ramp_limits=ramp_limits,
        load_column=LOSS_EXPERIMENT_BALANCE_COLUMN,
    )

    rows = []
    hourly_rows = []
    diagnostics: dict[str, pd.DataFrame] = {}

    diagnostics["oracle_projected_reference"] = projected_diag
    rows.append(
        _loss_experiment_summary_row(
            "oracle_projected_reference",
            projected_oracle,
            target,
            projected_oracle,
            balance_target,
            projected_diag,
        )
    )
    hourly_rows.append(
        _loss_experiment_hourly_rows(
            "oracle_projected_reference",
            projected_oracle,
            target,
            projected_oracle,
            test_df,
            projected_diag,
        )
    )

    fixed_pred, fixed_diag = solve_dispatch_many(
        test_df,
        FIXED_FUEL_COSTS,
        capacities,
        loss_config,
        ramp_limits=ramp_limits,
        load_column=LOSS_EXPERIMENT_BALANCE_COLUMN,
    )
    diagnostics["fixed_cost_lp_reference"] = fixed_diag
    rows.append(
        _loss_experiment_summary_row(
            "fixed_cost_lp_reference",
            fixed_pred,
            target,
            projected_oracle,
            balance_target,
            fixed_diag,
            FIXED_FUEL_COSTS,
        )
    )
    hourly_rows.append(
        _loss_experiment_hourly_rows(
            "fixed_cost_lp_reference",
            fixed_pred,
            target,
            projected_oracle,
            test_df,
            fixed_diag,
            FIXED_FUEL_COSTS,
        )
    )

    test_context = make_aligned_test_frame(train_df, test_df, loss_config.resolved_history_window)
    for use_weather in [False, True]:
        suffix = "with_weather" if use_weather else "no_weather"
        direct_pred, direct_target, direct_solve_df = train_direct_model(
            train_df,
            test_context,
            use_weather=use_weather,
            config=loss_config,
            use_residual=True,
            return_solve_frame=True,
        )
        projected_aligned = projected_oracle[-len(direct_pred) :]
        balance_aligned = direct_solve_df[LOSS_EXPERIMENT_BALANCE_COLUMN].to_numpy(dtype=np.float64)
        model_name = f"direct_residual_nn_l2_{suffix}"
        rows.append(
            _loss_experiment_summary_row(
                model_name,
                direct_pred,
                direct_target,
                projected_aligned,
                balance_aligned,
                None,
            )
        )
        hourly_rows.append(
            _loss_experiment_hourly_rows(
                model_name,
                direct_pred,
                direct_target,
                projected_aligned,
                direct_solve_df,
                None,
            )
        )

        for structured_loss in ["candidate_softmax", "fenchel_young"]:
            run_config = replace(loss_config, structured_loss=structured_loss)
            model, history = train_structured_cost_model(
                train_df,
                use_weather=use_weather,
                capacities=capacities,
                config=run_config,
                ramp_limits=ramp_limits,
            )
            pred, y_test, diag, costs, solve_df = _predict_structured_cost_lp_with_costs(
                model,
                test_context,
                use_weather=use_weather,
                capacities=capacities,
                config=run_config,
                ramp_limits=ramp_limits,
                load_column=LOSS_EXPERIMENT_BALANCE_COLUMN,
            )
            model_name = f"structured_{structured_loss}_{suffix}"
            diag["structured_train_loss"] = history[-1] if history else np.nan
            diagnostics[model_name] = diag
            projected_aligned = projected_oracle[-len(pred) :]
            balance_aligned = solve_df[LOSS_EXPERIMENT_BALANCE_COLUMN].to_numpy(dtype=np.float64)
            rows.append(
                _loss_experiment_summary_row(
                    model_name,
                    pred,
                    y_test,
                    projected_aligned,
                    balance_aligned,
                    diag,
                    costs,
                )
            )
            hourly_rows.append(
                _loss_experiment_hourly_rows(
                    model_name,
                    pred,
                    y_test,
                    projected_aligned,
                    solve_df,
                    diag,
                    costs,
                )
            )

        residual_cost_model, residual_history = train_residual_aware_cost_model(
            train_df,
            use_weather=use_weather,
            capacities=capacities,
            config=loss_config,
            ramp_limits=ramp_limits,
            load_column=LOSS_EXPERIMENT_BALANCE_COLUMN,
            soft_balance_penalty=0.1,
        )
        pred, y_test, diag, costs, solve_df = _predict_residual_aware_cost_lp_with_costs(
            residual_cost_model,
            test_context,
            use_weather=use_weather,
            capacities=capacities,
            config=loss_config,
            ramp_limits=ramp_limits,
            load_column=LOSS_EXPERIMENT_BALANCE_COLUMN,
            soft_balance_penalty=0.1,
        )
        model_name = f"residual_aware_fenchel_young_{suffix}"
        diag["structured_train_loss"] = residual_history[-1] if residual_history else np.nan
        diagnostics[model_name] = diag
        projected_aligned = projected_oracle[-len(pred) :]
        balance_aligned = solve_df[LOSS_EXPERIMENT_BALANCE_COLUMN].to_numpy(dtype=np.float64)
        fy_values = _residual_adjustment_fy_values(costs, pred, projected_aligned, solve_df)
        rows.append(
            _loss_experiment_summary_row(
                model_name,
                pred,
                y_test,
                projected_aligned,
                balance_aligned,
                diag,
                costs,
                fenchel_young_values=fy_values,
            )
        )
        hourly_rows.append(
            _loss_experiment_hourly_rows(
                model_name,
                pred,
                y_test,
                projected_aligned,
                solve_df,
                diag,
                costs,
                fenchel_young_values=fy_values,
            )
        )

    summary = pd.DataFrame(rows)
    hourly = pd.concat(hourly_rows, ignore_index=True)
    return summary, hourly, diagnostics


def run_feasibility_gap_diagnostics(
    system_hour_path: str | Path = "data/processed/nyiso_system_hour.parquet",
    *,
    zone_hour_path: str | Path = "data/processed/nyiso_zone_hour.parquet",
    interface_hour_path: str | Path = "data/processed/nyiso_interface_hour.parquet",
    zonal_capacity_path: str | Path = "data/processed/nyiso_zonal_capacity.parquet",
    config: ExperimentConfig | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, pd.DataFrame]]:
    config = resolve_experiment_config(config or ExperimentConfig())
    df = load_system_hour(system_hour_path)
    zone_hour = load_zone_hour(zone_hour_path)
    interface_hour = load_interface_hour(interface_hour_path)
    zonal_capacity_table = load_zonal_capacity(zonal_capacity_path)
    train_df, val_df, test_df = chronological_split(df)
    train_df, val_df, test_df = add_solver_balance_targets(
        train_df,
        train_df,
        val_df,
        test_df,
        balance_target_policy=config.balance_target_policy,
    )
    target = test_df[TARGET_COLUMNS].to_numpy(dtype=np.float64)
    test_df = test_df.copy()
    test_df[ORACLE_TOTAL_COLUMN] = target.sum(axis=1)
    capacities = compute_available_capacities(train_df)
    ramp_limits = compute_empirical_ramp_limits(train_df, config.ramp_quantile) if config.use_ramp_constraints else None
    zonal_capacities = compute_zonal_capacities(zonal_capacity_table, capacities)
    system_availability_caps = compute_empirical_system_availability_caps(train_df, capacities)
    load_column = solver_load_column(config)

    variants: list[tuple[str, np.ndarray, pd.DataFrame | None]] = []
    diagnostics: dict[str, pd.DataFrame] = {}

    raw_diag = pd.DataFrame(
        {
            "timestamp_utc": test_df["timestamp_utc"],
            "solver_success": True,
            "balance_residual": np.nan,
            "max_ramp_violation_mw": np.nan,
            "ens": 0.0,
            "spill": 0.0,
            "residual_supply": 0.0,
            "runtime_seconds": 0.0,
        }
    )
    variants.append(("oracle_raw", target, raw_diag))
    diagnostics["oracle_raw"] = raw_diag

    pred, diag = solve_projection_dispatch_many(test_df, target, capacities, config, ramp_limits=None)
    variants.append(("nonnegative_capacity_only", pred, diag))
    diagnostics["nonnegative_capacity_only"] = diag

    pred, diag = solve_projection_dispatch_many(test_df, target, capacities, config, ramp_limits=ramp_limits)
    variants.append(("capacity_plus_ramp", pred, diag))
    diagnostics["capacity_plus_ramp"] = diag

    pred, diag = solve_repair_dispatch_many(
        test_df,
        target,
        capacities,
        config,
        ramp_limits=ramp_limits,
        load_column=load_column,
    )
    variants.append(("capacity_ramp_plus_solver_balance", pred, diag))
    diagnostics["capacity_ramp_plus_solver_balance"] = diag

    pred, diag = solve_repair_dispatch_many(
        test_df,
        target,
        capacities,
        config,
        ramp_limits=ramp_limits,
        load_column=ORACLE_TOTAL_COLUMN,
    )
    variants.append(("capacity_ramp_plus_oracle_total_balance", pred, diag))
    diagnostics["capacity_ramp_plus_oracle_total_balance"] = diag

    pred, diag = solve_zonal_dispatch_many(
        test_df,
        zone_hour,
        interface_hour,
        np.zeros(len(FUEL_CATEGORIES), dtype=np.float64),
        zonal_capacities,
        config,
        ramp_limits=ramp_limits,
        system_availability_caps=system_availability_caps,
        load_column=load_column,
        repair_predictions=target,
    )
    variants.append(("zonal_capacity_ramp_availability_solver_balance", pred, diag))
    diagnostics["zonal_capacity_ramp_availability_solver_balance"] = diag

    pred, diag = solve_zonal_dispatch_many(
        test_df,
        zone_hour,
        interface_hour,
        np.zeros(len(FUEL_CATEGORIES), dtype=np.float64),
        zonal_capacities,
        config,
        ramp_limits=ramp_limits,
        system_availability_caps=system_availability_caps,
        load_column=ORACLE_TOTAL_COLUMN,
        repair_predictions=target,
    )
    variants.append(("zonal_capacity_ramp_availability_oracle_total_balance", pred, diag))
    diagnostics["zonal_capacity_ramp_availability_oracle_total_balance"] = diag

    summary = pd.DataFrame([_projection_summary_row(name, pred, target, diag) for name, pred, diag in variants])
    hourly = pd.concat(
        [_projection_hourly_rows(name, pred, target, test_df) for name, pred, _ in variants],
        ignore_index=True,
    )
    return summary, hourly, diagnostics
"""


def _format_table_value(value: object) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, (float, np.floating)):
        if abs(float(value)) < 1e-4 and float(value) != 0.0:
            return f"{float(value):.2e}"
        return f"{float(value):.3f}"
    return str(value)


def format_results_table(results: pd.DataFrame, columns: list[str] | None = None) -> str:
    columns = columns or COMPACT_METRIC_COLUMNS
    display_cols = [col for col in columns if col in results.columns]
    rows = ["\t".join(display_cols)]
    for _, row in results[display_cols].iterrows():
        rows.append("\t".join(_format_table_value(row[col]) for col in display_cols))
    return "\n".join(rows)


def solver_diagnostics_summary(diagnostics: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for name, frame in diagnostics.items():
        row = {
            "model": name,
            "rows": int(len(frame)),
            "success_rate": float(frame["solver_success"].mean()) if "solver_success" in frame.columns and len(frame) else np.nan,
            "mean_abs_balance_residual": (
                float(frame["balance_residual"].abs().mean()) if "balance_residual" in frame.columns and len(frame) else np.nan
            ),
            "max_ramp_violation_mw": (
                float(frame["max_ramp_violation_mw"].max()) if "max_ramp_violation_mw" in frame.columns and len(frame) else np.nan
            ),
            "mean_interface_slack_mw": (
                float(frame["interface_slack_total_mw"].mean()) if "interface_slack_total_mw" in frame.columns and len(frame) else np.nan
            ),
            "mean_transfer_magnitude_mw": (
                float(frame["transfer_magnitude_mw"].mean()) if "transfer_magnitude_mw" in frame.columns and len(frame) else np.nan
            ),
            "mean_runtime_seconds": (
                float(frame["runtime_seconds"].mean()) if "runtime_seconds" in frame.columns and len(frame) else np.nan
            ),
        }
        rows.append(row)
    return pd.DataFrame(rows)


def repair_comparison_table(results: pd.DataFrame) -> pd.DataFrame:
    model = results["model"].astype(str)
    mask = model.str.contains("direct_neural_residual_|residual_repair_lp_|residual_soft_repair_lp_", regex=True)
    return results.loc[mask].reset_index(drop=True)


"""
# OUTDATED
Legacy zonal diagnostic and residual/repair full-period runners are commented
out for the final public SCUC proxy submission. Use run_zonal_cost_lto.py for
active A5b/A6c/A7b experiments.

def _split_for_diagnostics(df: pd.DataFrame) -> tuple[dict[str, pd.DataFrame], str]:
    splits = main_project_split(df)
    split_strategy = "fixed_year"
    if any(splits[name].empty for name in ["train", "validation", "test", "holdout"]):
        splits = adaptive_chronological_split(df)
        split_strategy = "adaptive_chronological"
    return splits, split_strategy


def _zonal_diagnostic_variants() -> list[dict[str, object]]:
    return [
        {
            "variant": "zonal_no_interface_no_availability_no_transfer",
            "use_interface": False,
            "use_availability": False,
            "transfer_penalty": 0.0,
        },
        {
            "variant": "zonal_interface_only",
            "use_interface": True,
            "use_availability": False,
            "transfer_penalty": 0.0,
        },
        {
            "variant": "zonal_availability_only",
            "use_interface": False,
            "use_availability": True,
            "transfer_penalty": 0.0,
        },
        {
            "variant": "zonal_interface_availability",
            "use_interface": True,
            "use_availability": True,
            "transfer_penalty": 0.0,
        },
        {
            "variant": "zonal_interface_availability_transfer_p1",
            "use_interface": True,
            "use_availability": True,
            "transfer_penalty": 1.0,
        },
        {
            "variant": "zonal_interface_availability_transfer_p5",
            "use_interface": True,
            "use_availability": True,
            "transfer_penalty": 5.0,
        },
        {
            "variant": "zonal_interface_availability_transfer_p20",
            "use_interface": True,
            "use_availability": True,
            "transfer_penalty": 20.0,
        },
    ]


def run_zonal_diagnostics(
    system_hour_path: str | Path = "data/processed/nyiso_system_hour.parquet",
    *,
    zone_hour_path: str | Path = "data/processed/nyiso_zone_hour.parquet",
    interface_hour_path: str | Path | None = "data/processed/nyiso_interface_hour.parquet",
    zonal_capacity_path: str | Path = "data/processed/nyiso_zonal_capacity.parquet",
    split: str = "test",
    max_hours: int | None = 500,
    config: ExperimentConfig | None = None,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame], pd.DataFrame]:
    config = resolve_experiment_config(config or ExperimentConfig())
    df = load_system_hour(system_hour_path)
    zone_hour = load_zone_hour(zone_hour_path)
    if interface_hour_path is not None and Path(interface_hour_path).exists():
        interface_hour = load_interface_hour(interface_hour_path)
    else:
        interface_hour = empty_interface_hour()
    zonal_capacity_table = load_zonal_capacity(zonal_capacity_path)

    splits, split_strategy = _split_for_diagnostics(df)
    train_df = splits["train"]
    val_df = splits["validation"]
    test_df = splits["test"]
    holdout_df = splits["holdout"]
    train_df, val_df, test_df, holdout_df = add_solver_balance_targets(
        train_df,
        train_df,
        val_df,
        test_df,
        holdout_df,
        balance_target_policy=config.balance_target_policy,
    )
    split_map = {
        "train": train_df,
        "validation": val_df,
        "val": val_df,
        "test": test_df,
        "test_2023_2024": test_df,
        "holdout": holdout_df,
        "holdout_2025": holdout_df,
    }
    if split not in split_map:
        raise ValueError(f"Unknown split '{split}'. Expected one of {sorted(split_map)}")
    eval_df = split_map[split].sort_values("timestamp_utc").reset_index(drop=True)
    if max_hours is not None and max_hours > 0:
        eval_df = eval_df.head(max_hours).copy()
    if eval_df.empty:
        raise ValueError(f"Selected split '{split}' is empty.")

    timestamps = set(eval_df["timestamp_utc"])
    zone_eval = zone_hour[zone_hour["timestamp_utc"].isin(timestamps)].copy()
    interface_eval = (
        interface_hour[interface_hour["timestamp_utc"].isin(timestamps)].copy()
        if "timestamp_utc" in interface_hour.columns and not interface_hour.empty
        else empty_interface_hour()
    )
    interface_data_available = not interface_eval.empty
    finite_interface_rows = 0
    if interface_data_available:
        finite_interface_rows = int((interface_eval.get("upper_is_finite", False) | interface_eval.get("lower_is_finite", False)).sum())

    load_column = solver_load_column(config)
    if load_column not in eval_df.columns:
        raise KeyError(f"Configured zonal load column is missing from selected split: {load_column}")
    capacities = compute_available_capacities(train_df)
    zonal_capacities = compute_zonal_capacities(zonal_capacity_table, capacities)
    ramp_limits = compute_empirical_ramp_limits(train_df, config.ramp_quantile) if config.use_ramp_constraints else None
    system_availability_caps = compute_empirical_system_availability_caps(train_df, capacities)
    target = eval_df[TARGET_COLUMNS].to_numpy(dtype=np.float64)

    rows: list[dict[str, object]] = []
    diagnostics: dict[str, pd.DataFrame] = {}
    for variant in _zonal_diagnostic_variants():
        variant_config = replace(
            config,
            use_zonal_interface_constraints=bool(variant["use_interface"]),
            use_zonal_availability_bounds=bool(variant["use_availability"]),
            transfer_penalty=float(variant["transfer_penalty"]),
        )
        pred, diag = solve_zonal_dispatch_many(
            eval_df,
            zone_eval,
            interface_eval if bool(variant["use_interface"]) else empty_interface_hour(),
            FIXED_FUEL_COSTS,
            zonal_capacities,
            variant_config,
            ramp_limits=ramp_limits,
            system_availability_caps=system_availability_caps,
            load_column=load_column,
        )
        name = str(variant["variant"])
        diagnostics[name] = diag
        row = {
            "variant": name,
            "split": split,
            "split_strategy": split_strategy,
            "rows": int(len(eval_df)),
            "use_interface": bool(variant["use_interface"]),
            "use_availability": bool(variant["use_availability"]),
            "transfer_penalty": float(variant["transfer_penalty"]),
            "interface_data_available": bool(interface_data_available),
            "finite_interface_rows": int(finite_interface_rows),
            "success_rate": float(diag["solver_success"].mean()) if len(diag) else np.nan,
            **metrics(pred, target, diag),
        }
        rows.append(row)

    summary = pd.DataFrame(rows)
    split_summary = pd.DataFrame(
        [
            {
                "split": name,
                "split_strategy": split_strategy,
                "rows": int(len(frame)),
                "start_utc": frame["timestamp_utc"].min() if len(frame) else pd.NaT,
                "end_utc": frame["timestamp_utc"].max() if len(frame) else pd.NaT,
            }
            for name, frame in [
                ("train", train_df),
                ("validation", val_df),
                ("test", test_df),
                ("holdout", holdout_df),
            ]
        ]
    )
    return summary, diagnostics, split_summary


def _evaluate_full_period_split(
    train_df: pd.DataFrame,
    history_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    *,
    split_name: str,
    capacities: np.ndarray,
    ramp_limits: RampLimits | None,
    config: ExperimentConfig,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    target = eval_df[TARGET_COLUMNS].to_numpy(dtype=np.float64)
    rows: list[dict[str, float | str]] = []
    diagnostics: dict[str, pd.DataFrame] = {}

    def add_row(model: str, pred: np.ndarray, diag: pd.DataFrame | None = None, model_target: np.ndarray | None = None) -> None:
        rows.append(
            {
                "split": split_name,
                "model": model,
                **metrics(pred, target if model_target is None else model_target, diag),
            }
        )

    rows.append(
        {
            "split": split_name,
            "model": "persistence_previous_hour",
            **metrics(persistence_baseline(history_df, eval_df), target),
        }
    )
    rows.append(
        {
            "split": split_name,
            "model": "persistence_previous_day_hour",
            **metrics(persistence_lag_baseline(history_df, eval_df, lag_hours=24), target),
        }
    )
    rows.append(
        {
            "split": split_name,
            "model": "persistence_previous_week_hour",
            **metrics(persistence_lag_baseline(history_df, eval_df, lag_hours=168), target),
        }
    )
    rows.append({"split": split_name, "model": "hour_of_week", **metrics(hour_of_week_baseline(train_df, eval_df), target)})

    test_context = make_aligned_test_frame(history_df, eval_df, config.resolved_history_window)
    for use_weather in [False, True]:
        suffix = "with_weather" if use_weather else "no_weather"
        direct_pred, direct_target, direct_solve_df = train_direct_model(
            train_df,
            test_context,
            use_weather=use_weather,
            config=config,
            use_residual=True,
            return_solve_frame=True,
        )
        add_row(f"direct_neural_residual_{suffix}_with_lag", direct_pred, model_target=direct_target)

        hard_pred, hard_diag = solve_repair_dispatch_many(
            direct_solve_df,
            direct_pred,
            capacities,
            config,
            ramp_limits=ramp_limits,
            load_column=LAG1_BALANCE_COLUMN,
        )
        hard_name = f"residual_repair_lp_{suffix}_lag1_balance"
        diagnostics[hard_name] = hard_diag
        add_row(hard_name, hard_pred, hard_diag, direct_target)

        for penalty in [0.1, 1.0]:
            soft_pred, soft_diag = solve_soft_repair_dispatch_many(
                direct_solve_df,
                direct_pred,
                capacities,
                config,
                soft_balance_penalty=penalty,
                ramp_limits=ramp_limits,
                load_column=LAG1_BALANCE_COLUMN,
            )
            penalty_label = str(penalty).replace(".", "p")
            soft_name = f"residual_soft_repair_lp_{suffix}_p{penalty_label}"
            diagnostics[soft_name] = soft_diag
            add_row(soft_name, soft_pred, soft_diag, direct_target)

    return pd.DataFrame(rows), diagnostics


def run_full_period_experiment(
    system_hour_path: str | Path = "data/processed/full/nyiso_system_hour.parquet",
    *,
    config: ExperimentConfig | None = None,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame], pd.DataFrame]:
    config = resolve_experiment_config(config or ExperimentConfig())
    df = load_system_hour(system_hour_path)
    splits = main_project_split(df)
    split_strategy = "fixed_year"
    if any(splits[name].empty for name in ["train", "validation", "test", "holdout"]):
        splits = adaptive_chronological_split(df)
        split_strategy = "adaptive_chronological"
    train_df = splits["train"]
    val_df = splits["validation"]
    test_df = splits["test"]
    holdout_df = splits["holdout"]
    train_df, val_df, test_df, holdout_df = add_solver_balance_targets(
        train_df,
        train_df,
        val_df,
        test_df,
        holdout_df,
        balance_target_policy=config.balance_target_policy,
    )
    if any(frame.empty for frame in [train_df, val_df, test_df, holdout_df]):
        sizes = {name: int(len(frame)) for name, frame in zip(["train", "validation", "test", "holdout"], [train_df, val_df, test_df, holdout_df])}
        raise ValueError(f"Full-period split contains an empty frame: {sizes}")

    capacities = compute_available_capacities(train_df)
    ramp_limits = compute_empirical_ramp_limits(train_df, config.ramp_quantile) if config.use_ramp_constraints else None
    split_summary = pd.DataFrame(
        [
            {
                "split": name,
                "split_strategy": split_strategy,
                "rows": int(len(frame)),
                "start_utc": frame["timestamp_utc"].min(),
                "end_utc": frame["timestamp_utc"].max(),
            }
            for name, frame in [
                ("train", train_df),
                ("validation", val_df),
                ("test", test_df),
                ("holdout", holdout_df),
            ]
        ]
    )

    test_results, test_diag = _evaluate_full_period_split(
        train_df,
        pd.concat([train_df, val_df], ignore_index=True).sort_values("timestamp_utc"),
        test_df,
        split_name="test_2023_2024",
        capacities=capacities,
        ramp_limits=ramp_limits,
        config=config,
    )
    holdout_results, holdout_diag = _evaluate_full_period_split(
        train_df,
        pd.concat([train_df, val_df, test_df], ignore_index=True).sort_values("timestamp_utc"),
        holdout_df,
        split_name="holdout_2025",
        capacities=capacities,
        ramp_limits=ramp_limits,
        config=config,
    )
    results = pd.concat([test_results, holdout_results], ignore_index=True)
    diagnostics = {
        **{f"test_2023_2024_{name}": frame for name, frame in test_diag.items()},
        **{f"holdout_2025_{name}": frame for name, frame in holdout_diag.items()},
    }
    return results, diagnostics, split_summary


def run_smoke_experiment(
    system_hour_path: str | Path = "data/processed/nyiso_system_hour.parquet",
    *,
    zone_hour_path: str | Path = "data/processed/nyiso_zone_hour.parquet",
    interface_hour_path: str | Path = "data/processed/nyiso_interface_hour.parquet",
    zonal_capacity_path: str | Path = "data/processed/nyiso_zonal_capacity.parquet",
    config: ExperimentConfig | None = None,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    config = resolve_experiment_config(config or ExperimentConfig())
    df = load_system_hour(system_hour_path)
    zone_hour = load_zone_hour(zone_hour_path)
    interface_hour = load_interface_hour(interface_hour_path)
    zonal_capacity_table = load_zonal_capacity(zonal_capacity_path)
    validate_feature_policy(feature_columns(False, config.feature_mode, include_lag=config.include_lags), config.feature_mode)
    train_df, val_df, test_df = chronological_split(df)
    train_df, val_df, test_df = add_solver_balance_targets(
        train_df,
        train_df,
        val_df,
        test_df,
        balance_target_policy=config.balance_target_policy,
    )
    load_column = solver_load_column(config)
    if load_column not in train_df.columns or load_column not in test_df.columns:
        raise KeyError(f"Configured solver load column is missing: {load_column}")
    capacities = compute_available_capacities(train_df)
    ramp_limits = (
        compute_empirical_ramp_limits(train_df, config.ramp_quantile)
        if config.use_ramp_constraints
        else None
    )
    target = test_df[TARGET_COLUMNS].to_numpy(dtype=np.float64)
    zonal_capacities = compute_zonal_capacities(zonal_capacity_table, capacities)
    system_availability_caps = compute_empirical_system_availability_caps(train_df, capacities)
    rows = []
    diagnostics: dict[str, pd.DataFrame] = {}

    rows.append({"model": "persistence_previous_hour", **metrics(persistence_baseline(train_df, test_df), target)})
    rows.append(
        {
            "model": "persistence_previous_day_hour",
            **metrics(persistence_lag_baseline(train_df, test_df, lag_hours=24), target),
        }
    )
    rows.append(
        {
            "model": "persistence_previous_week_hour",
            **metrics(persistence_lag_baseline(train_df, test_df, lag_hours=168), target),
        }
    )
    rows.append(
        {
            "model": "day_ahead_persistence_rollout",
            **metrics(day_ahead_persistence_rollout(train_df, test_df), target),
        }
    )
    rows.append({"model": "hour_of_week", **metrics(hour_of_week_baseline(train_df, test_df), target)})

    if config.run_system_fixed_lp or config.run_antiquated_pipelines:
        fixed_pred, fixed_diag = solve_dispatch_many(
            test_df,
            FIXED_FUEL_COSTS,
            capacities,
            config,
            ramp_limits=ramp_limits,
            load_column=load_column,
        )
        fixed_name = "fixed_cost_ramp_lp" if ramp_limits is not None else "fixed_cost_lp"
        diagnostics[fixed_name] = fixed_diag
        rows.append({"model": fixed_name, **metrics(fixed_pred, target, fixed_diag)})

    zonal_pred, zonal_diag = solve_zonal_dispatch_many(
        test_df,
        zone_hour,
        interface_hour,
        FIXED_FUEL_COSTS,
        zonal_capacities,
        config,
        ramp_limits=ramp_limits,
        system_availability_caps=system_availability_caps,
        load_column=load_column,
    )
    diagnostics["fixed_cost_zonal_interface_lp"] = zonal_diag
    rows.append({"model": "fixed_cost_zonal_interface_lp", **metrics(zonal_pred, target, zonal_diag)})

    history_window = config.resolved_history_window
    test_context = make_aligned_test_frame(train_df, test_df, history_window)
    for use_weather in [False, True]:
        lag_options = [True]
        if config.run_no_lag_direct_models or config.run_antiquated_pipelines:
            lag_options = [False, True]
        for include_lags in lag_options:
            run_config = replace(config, include_lags=include_lags)
            suffix = "with_weather" if use_weather else "no_weather"
            lag_suffix = "with_lag" if include_lags else "no_lag"
            direct_pred, direct_target, direct_solve_df = train_direct_model(
                train_df,
                test_context,
                use_weather=use_weather,
                config=run_config,
                use_residual=True,
                return_solve_frame=True,
            )
            rows.append({"model": f"direct_neural_residual_{suffix}_{lag_suffix}", **metrics(direct_pred, direct_target)})

            if include_lags:
                repair_balance_options = [
                    ("hour_of_week_balance", HOUR_OF_WEEK_BALANCE_COLUMN),
                    ("lag1_balance", LAG1_BALANCE_COLUMN),
                ]
                for balance_suffix, repair_load_column in repair_balance_options:
                    repair_pred, repair_diag = solve_repair_dispatch_many(
                        direct_solve_df,
                        direct_pred,
                        capacities,
                        run_config,
                        ramp_limits=ramp_limits,
                        load_column=repair_load_column,
                    )
                    repair_name = f"residual_repair_lp_{suffix}_{balance_suffix}"
                    diagnostics[repair_name] = repair_diag
                    rows.append({"model": repair_name, **metrics(repair_pred, direct_target, repair_diag)})

                    if repair_load_column == LAG1_BALANCE_COLUMN:
                        for penalty in run_config.soft_balance_penalty_grid:
                            soft_pred, soft_diag = solve_soft_repair_dispatch_many(
                                direct_solve_df,
                                direct_pred,
                                capacities,
                                run_config,
                                soft_balance_penalty=float(penalty),
                                ramp_limits=ramp_limits,
                                load_column=repair_load_column,
                            )
                            penalty_label = str(penalty).replace(".", "p")
                            soft_name = f"residual_soft_repair_lp_{suffix}_p{penalty_label}"
                            diagnostics[soft_name] = soft_diag
                            rows.append({"model": soft_name, **metrics(soft_pred, direct_target, soft_diag)})

                    zonal_repair_pred, zonal_repair_diag = solve_zonal_dispatch_many(
                        direct_solve_df,
                        zone_hour,
                        interface_hour,
                        np.zeros(len(FUEL_CATEGORIES), dtype=np.float64),
                        zonal_capacities,
                        run_config,
                        ramp_limits=ramp_limits,
                        system_availability_caps=system_availability_caps,
                        load_column=repair_load_column,
                        repair_predictions=direct_pred,
                    )
                    zonal_repair_name = f"residual_zonal_repair_lp_{suffix}_{balance_suffix}"
                    diagnostics[zonal_repair_name] = zonal_repair_diag
                    rows.append(
                        {
                            "model": zonal_repair_name,
                            **metrics(zonal_repair_pred, direct_target, zonal_repair_diag),
                        }
                    )

    if config.run_learned_cost_grid or config.run_antiquated_pipelines:
        for use_weather in [False, True]:
            lag_options = [True]
            if config.run_antiquated_pipelines:
                lag_options = [False, True]
            for include_lags in lag_options:
                run_config = replace(config, include_lags=include_lags)
                suffix = "with_weather" if use_weather else "no_weather"
                lag_suffix = "with_lag" if include_lags else "no_lag"
                model, history = train_structured_cost_model(
                    train_df,
                    use_weather=use_weather,
                    capacities=capacities,
                    config=run_config,
                    ramp_limits=ramp_limits,
                )
                lp_pred, lp_target, lp_diag = predict_structured_cost_lp(
                    model,
                    test_context,
                    use_weather=use_weather,
                    capacities=capacities,
                    config=run_config,
                    ramp_limits=ramp_limits,
                )
                lp_diag["structured_train_loss"] = history[-1] if history else np.nan
                diagnostics[f"learned_cost_lp_{suffix}_{lag_suffix}"] = lp_diag
                rows.append({"model": f"learned_cost_lp_{suffix}_{lag_suffix}", **metrics(lp_pred, lp_target, lp_diag)})

                zonal_lp_pred, zonal_lp_target, zonal_lp_diag = predict_structured_cost_zonal_lp(
                    model,
                    test_context,
                    use_weather=use_weather,
                    zone_hour=zone_hour,
                    interface_hour=interface_hour,
                    zonal_capacities=zonal_capacities,
                    config=run_config,
                    ramp_limits=ramp_limits,
                    system_availability_caps=system_availability_caps,
                )
                zonal_lp_diag["structured_train_loss"] = history[-1] if history else np.nan
                diagnostics[f"learned_cost_zonal_lp_{suffix}_{lag_suffix}"] = zonal_lp_diag
                rows.append(
                    {
                        "model": f"learned_cost_zonal_lp_{suffix}_{lag_suffix}",
                        **metrics(zonal_lp_pred, zonal_lp_target, zonal_lp_diag),
                    }
                )

    results = pd.DataFrame(rows)
    if {"direct_neural_no_weather", "direct_neural_with_weather"}.issubset(set(results["model"])):
        no_weather = results.loc[results["model"] == "direct_neural_no_weather", "overall_mae"].iloc[0]
        with_weather = results.loc[results["model"] == "direct_neural_with_weather", "overall_mae"].iloc[0]
        results["weather_direct_mae_improvement"] = no_weather - with_weather
    return results, diagnostics
"""

# OUTDATED
# Legacy module-level smoke entrypoint disabled for final submission cleanup.
# if __name__ == "__main__":
#     table, _ = run_smoke_experiment()
#     print(format_results_table(table))
