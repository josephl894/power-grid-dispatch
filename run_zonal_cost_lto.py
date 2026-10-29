from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from nyiso_experiments import (
    ANCILLARY_STRESS_FEATURE_COLUMNS,
    DA_CONSTRAINT_STRESS_FEATURE_COLUMNS,
    DAY_AHEAD_ANCILLARY_STRESS_FEATURE_COLUMNS,
    DAY_AHEAD_SCHEDULED_OUTAGE_STRESS_FEATURE_COLUMNS,
    ExperimentConfig,
    FUEL_CATEGORIES,
    LAG1_BALANCE_COLUMN,
    LOSS_EXPERIMENT_BALANCE_COLUMN,
    SCHEDULED_OUTAGE_STRESS_FEATURE_COLUMNS,
    TARGET_COLUMNS,
    THERMAL_DERATE_FUELS,
    ZONAL_BID_LOAD_FEATURE_COLUMNS,
    add_financial_cost_features,
    add_loss_experiment_balance_column,
    add_solver_balance_targets,
    adaptive_chronological_split,
    compute_available_capacities,
    compute_empirical_ramp_limits,
    compute_empirical_system_availability_caps,
    compute_zonal_capacities,
    empty_interface_hour,
    feature_columns,
    financial_cost_matrix_from_frame,
    format_results_table,
    load_interface_hour,
    load_system_hour,
    load_zonal_capacity,
    load_zone_hour,
    main_project_split,
    make_aligned_test_frame,
    metrics,
    predict_zonal_financial_cost_lto,
    predict_zonal_financial_oracle,
    train_zonal_financial_cost_model,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run cost-aware zonal Fenchel-Young LtO dispatch.")
    parser.add_argument("--processed-dir", default="data/processed/final_submission")
    parser.add_argument("--cost-input-dir", default="data/processed/cost_inputs")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--direct-epochs", type=int, default=40)
    parser.add_argument("--structured-epochs", type=int, default=1)
    parser.add_argument("--structured-lr", type=float, default=0.03)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--max-train-hours", type=int, default=720)
    parser.add_argument("--max-eval-hours", type=int, default=168)
    parser.add_argument("--split", choices=["validation", "test", "holdout", "all"], default="all")
    parser.add_argument("--zonal-capacity-source", choices=["legacy", "gold_book"], default="gold_book")
    parser.add_argument(
        "--proxy-stage",
        choices=[
            "a1",
            "a2",
            "a3",
            "a4",
            "a5a",
            "a5b",
            "a6a",
            "a6b",
            "a6c",
            "a6d",
            "a6e",
            "a6f",
            "a8a",
            "a8b",
            "a8c",
            "a8d",
            "a8e",
            "a7a",
            "a7b",
        ],
        default="a2",
    )
    parser.add_argument("--run-interface-off-ablation", action="store_true")
    parser.add_argument("--reserve-requirement-fraction", type=float, default=0.04)
    parser.add_argument("--reserve-slack-penalty", type=float, default=5e4)
    parser.add_argument("--reserve-penalty-min", type=float, default=5e3)
    parser.add_argument("--reserve-penalty-max", type=float, default=2e5)
    parser.add_argument("--reserve-penalty-lr", type=float, default=0.02)
    parser.add_argument("--reserve-penalty-l2", type=float, default=0.0)
    parser.add_argument("--reserve-fraction-min", type=float, default=0.0)
    parser.add_argument("--reserve-fraction-max", type=float, default=0.06)
    parser.add_argument("--reserve-fraction-lr", type=float, default=0.01)
    parser.add_argument("--reserve-fraction-l2", type=float, default=0.001)
    parser.add_argument("--thermal-derate-max", type=float, default=0.5)
    parser.add_argument("--thermal-derate-lr", type=float, default=0.05)
    parser.add_argument("--thermal-derate-l2", type=float, default=0.0)
    parser.add_argument("--thermal-derate-mean-penalty", type=float, default=0.0)
    parser.add_argument("--gate-p511a", type=float, default=1.0)
    parser.add_argument("--gate-p54a", type=float, default=1.0)
    parser.add_argument("--gate-p54c", type=float, default=1.0)
    parser.add_argument("--gate-p5", type=float, default=1.0)
    parser.add_argument("--gate-p59", type=float, default=1.0)
    parser.add_argument("--gate-weather", type=float, default=1.0)
    parser.add_argument("--gate-cost", type=float, default=1.0)
    return parser.parse_args()


def _choose_splits(df: pd.DataFrame) -> tuple[dict[str, pd.DataFrame], str]:
    fixed = main_project_split(df)
    if len(fixed["train"]) and len(fixed["test"]):
        return fixed, "fixed_year"
    return adaptive_chronological_split(df), "adaptive"


def _load_required_interface(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Build or copy nyiso_interface_hour.parquet before running with "
            "use_zonal_interface_constraints=True."
        )
    interface_hour = load_interface_hour(path)
    if interface_hour.empty:
        raise ValueError(
            f"{path} is empty. Interface-constrained zonal claims require finite P-32 interface data."
        )
    finite_rows = int((interface_hour["upper_is_finite"] | interface_hour["lower_is_finite"]).sum())
    if finite_rows == 0:
        raise ValueError(
            f"{path} has no finite interface limits. Interface-constrained zonal claims are not meaningful."
        )
    return interface_hour


def _zonal_capacity_path(processed_dir: Path, source: str) -> Path:
    if source == "legacy":
        return processed_dir / "nyiso_zonal_capacity.parquet"
    return processed_dir.parent / "scuc_proxy" / "zonal_fuel_capacity.parquet"


def _fuel_key(fuel: str) -> str:
    return fuel.lower().replace(" ", "_")


def _summary_row(
    model: str,
    pred: np.ndarray,
    target: np.ndarray,
    diagnostics: pd.DataFrame,
    *,
    oracle_diagnostics: pd.DataFrame | None = None,
    structured_train_loss: float = np.nan,
) -> dict[str, float | str]:
    row: dict[str, float | str] = {"model": model, **metrics(pred, target, diagnostics)}
    row["structured_train_loss"] = structured_train_loss
    row["mean_financial_dispatch_cost"] = float(diagnostics["financial_dispatch_cost"].mean())
    row["mean_financial_cost_per_mwh"] = float(diagnostics["financial_dispatch_cost_per_mwh"].mean())
    row["success_rate"] = float(diagnostics["solver_success"].mean())
    if oracle_diagnostics is not None:
        aligned = diagnostics[["timestamp_utc", "financial_dispatch_cost"]].merge(
            oracle_diagnostics[["timestamp_utc", "financial_dispatch_cost"]],
            on="timestamp_utc",
            suffixes=("", "_oracle"),
            how="inner",
        )
        gap = aligned["financial_dispatch_cost"] - aligned["financial_dispatch_cost_oracle"]
        row["mean_financial_cost_gap_to_oracle"] = float(gap.mean())
        row["p95_financial_cost_gap_to_oracle"] = float(gap.quantile(0.95))
    else:
        row["mean_financial_cost_gap_to_oracle"] = 0.0
        row["p95_financial_cost_gap_to_oracle"] = 0.0
    return row


def _hourly_frame(
    solve_df: pd.DataFrame,
    target: np.ndarray,
    oracle_pred: np.ndarray,
    lto_pred: np.ndarray,
    oracle_diag: pd.DataFrame,
    lto_diag: pd.DataFrame,
) -> pd.DataFrame:
    out = pd.DataFrame(
        {
            "timestamp_utc": solve_df["timestamp_utc"].to_numpy(),
            "timestamp_local": solve_df["timestamp_local"].to_numpy(),
            "solver_balance_mw": solve_df[LOSS_EXPERIMENT_BALANCE_COLUMN].to_numpy(dtype=np.float64),
        }
    )
    for idx, fuel in enumerate(FUEL_CATEGORIES):
        key = _fuel_key(fuel)
        out[f"actual_{key}_mw"] = target[:, idx]
        out[f"oracle_{key}_mw"] = oracle_pred[:, idx]
        out[f"zonal_financial_fy_lto_{key}_mw"] = lto_pred[:, idx]
    out["actual_total_mw"] = target.sum(axis=1)
    out["oracle_total_mw"] = oracle_pred.sum(axis=1)
    out["zonal_financial_fy_lto_total_mw"] = lto_pred.sum(axis=1)
    out["oracle_financial_dispatch_cost"] = oracle_diag["financial_dispatch_cost"].to_numpy(dtype=np.float64)
    out["zonal_financial_fy_lto_dispatch_cost"] = lto_diag["financial_dispatch_cost"].to_numpy(dtype=np.float64)
    out["financial_cost_gap_to_oracle"] = (
        out["zonal_financial_fy_lto_dispatch_cost"] - out["oracle_financial_dispatch_cost"]
    )
    return out


def _selected_splits(split: str) -> list[str]:
    if split == "all":
        return ["validation", "test", "holdout"]
    return [split]


def _interface_summary_rows(
    interface_hour: pd.DataFrame,
    split_frames: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    rows = []
    all_frames = {"all_loaded": pd.DataFrame({"timestamp_utc": interface_hour["timestamp_utc"].drop_duplicates()})}
    all_frames.update(split_frames)
    for split_name, frame in all_frames.items():
        timestamps = pd.to_datetime(frame["timestamp_utc"], utc=True).drop_duplicates()
        subset = interface_hour[interface_hour["timestamp_utc"].isin(timestamps)].copy()
        finite_mask = subset["upper_is_finite"] | subset["lower_is_finite"] if not subset.empty else pd.Series(dtype=bool)
        rows.append(
            {
                "split": split_name,
                "hours_requested": int(len(timestamps)),
                "hours_with_interface_rows": int(subset["timestamp_utc"].nunique()) if not subset.empty else 0,
                "interface_rows": int(len(subset)),
                "finite_limit_rows": int(finite_mask.sum()) if len(finite_mask) else 0,
                "interface_count": int(subset["interface_name"].nunique()) if not subset.empty else 0,
                "stress_mean": float(subset["stress_mean"].mean()) if "stress_mean" in subset.columns and not subset.empty else np.nan,
                "stress_max": float(subset["stress_max"].max()) if "stress_max" in subset.columns and not subset.empty else np.nan,
                "active_side_sentinel_count": float(subset["active_side_sentinel_count"].sum())
                if "active_side_sentinel_count" in subset.columns and not subset.empty
                else 0.0,
                "first_timestamp_utc": subset["timestamp_utc"].min() if not subset.empty else pd.NaT,
                "last_timestamp_utc": subset["timestamp_utc"].max() if not subset.empty else pd.NaT,
            }
        )
    return pd.DataFrame(rows)


def _feature_metadata(config: ExperimentConfig) -> pd.DataFrame:
    cols = feature_columns(
        True,
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
    return pd.DataFrame(
        {
            "feature": cols,
            "is_da_constraint_stress_feature": [col in DA_CONSTRAINT_STRESS_FEATURE_COLUMNS for col in cols],
            "is_scheduled_outage_stress_feature": [
                col in SCHEDULED_OUTAGE_STRESS_FEATURE_COLUMNS for col in cols
            ],
            "is_ancillary_stress_feature": [col in ANCILLARY_STRESS_FEATURE_COLUMNS for col in cols],
            "is_day_ahead_ancillary_stress_feature": [
                col in DAY_AHEAD_ANCILLARY_STRESS_FEATURE_COLUMNS for col in cols
            ],
            "is_day_ahead_scheduled_outage_stress_feature": [
                col in DAY_AHEAD_SCHEDULED_OUTAGE_STRESS_FEATURE_COLUMNS for col in cols
            ],
            "is_zonal_bid_load_feature": [col in ZONAL_BID_LOAD_FEATURE_COLUMNS for col in cols],
            "is_cost_feature": [col.startswith("financial_cost_") or col.startswith("fuel_price_") for col in cols],
            "is_lag_feature": ["_lag_" in col for col in cols],
        }
    )


def _da_constraint_feature_summary(system_df: pd.DataFrame) -> pd.DataFrame:
    cols = [col for col in DA_CONSTRAINT_STRESS_FEATURE_COLUMNS if col in system_df.columns]
    if not cols:
        return pd.DataFrame(columns=["feature", "count", "mean", "std", "min", "max"])
    summary = system_df[cols].describe().T.reset_index().rename(columns={"index": "feature"})
    return summary[["feature", "count", "mean", "std", "min", "max"]]


def _scheduled_outage_feature_summary(system_df: pd.DataFrame) -> pd.DataFrame:
    cols = [col for col in SCHEDULED_OUTAGE_STRESS_FEATURE_COLUMNS if col in system_df.columns]
    if not cols:
        return pd.DataFrame(columns=["feature", "count", "mean", "std", "min", "max"])
    summary = system_df[cols].describe().T.reset_index().rename(columns={"index": "feature"})
    return summary[["feature", "count", "mean", "std", "min", "max"]]


def _ancillary_feature_summary(system_df: pd.DataFrame) -> pd.DataFrame:
    cols = [col for col in ANCILLARY_STRESS_FEATURE_COLUMNS if col in system_df.columns]
    if not cols:
        return pd.DataFrame(columns=["feature", "count", "mean", "std", "min", "max"])
    summary = system_df[cols].describe().T.reset_index().rename(columns={"index": "feature"})
    return summary[["feature", "count", "mean", "std", "min", "max"]]


def _day_ahead_ancillary_feature_summary(system_df: pd.DataFrame) -> pd.DataFrame:
    cols = [col for col in DAY_AHEAD_ANCILLARY_STRESS_FEATURE_COLUMNS if col in system_df.columns]
    if not cols:
        return pd.DataFrame(columns=["feature", "count", "mean", "std", "min", "max"])
    summary = system_df[cols].describe().T.reset_index().rename(columns={"index": "feature"})
    return summary[["feature", "count", "mean", "std", "min", "max"]]


def _day_ahead_scheduled_outage_feature_summary(system_df: pd.DataFrame) -> pd.DataFrame:
    cols = [col for col in DAY_AHEAD_SCHEDULED_OUTAGE_STRESS_FEATURE_COLUMNS if col in system_df.columns]
    if not cols:
        return pd.DataFrame(columns=["feature", "count", "mean", "std", "min", "max"])
    summary = system_df[cols].describe().T.reset_index().rename(columns={"index": "feature"})
    return summary[["feature", "count", "mean", "std", "min", "max"]]


def _zonal_bid_load_feature_summary(system_df: pd.DataFrame) -> pd.DataFrame:
    cols = [col for col in ZONAL_BID_LOAD_FEATURE_COLUMNS if col in system_df.columns]
    if not cols:
        return pd.DataFrame(columns=["feature", "count", "mean", "std", "min", "max"])
    summary = system_df[cols].describe().T.reset_index().rename(columns={"index": "feature"})
    return summary[["feature", "count", "mean", "std", "min", "max"]]


def _thermal_derate_summary(diagnostics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for fuel in THERMAL_DERATE_FUELS:
        key = fuel.lower().replace(" ", "_")
        for col in [c for c in diagnostics.columns if c.startswith("derate_") and c.endswith(f"_{key}")]:
            parts = col.split("_")
            if len(parts) < 3:
                continue
            zone = parts[1]
            values = diagnostics[col].astype(float)
            rows.append(
                {
                    "zone_code": zone,
                    "fuel_category": fuel,
                    "mean_derate": float(values.mean()),
                    "max_derate": float(values.max()),
                    "p95_derate": float(values.quantile(0.95)),
                }
            )
    return pd.DataFrame(rows)


def _derate_stress_correlation(system_df: pd.DataFrame, diagnostics: pd.DataFrame) -> pd.DataFrame:
    if "mean_thermal_derate" not in diagnostics.columns:
        return pd.DataFrame(columns=["feature", "correlation_with_mean_thermal_derate"])
    aligned = diagnostics[["timestamp_utc", "mean_thermal_derate"]].merge(
        system_df[["timestamp_utc", *[c for c in DA_CONSTRAINT_STRESS_FEATURE_COLUMNS if c in system_df.columns]]],
        on="timestamp_utc",
        how="left",
    )
    rows = []
    for col in DA_CONSTRAINT_STRESS_FEATURE_COLUMNS:
        if col not in aligned.columns:
            continue
        corr = aligned["mean_thermal_derate"].corr(aligned[col])
        rows.append({"feature": col, "correlation_with_mean_thermal_derate": float(corr) if pd.notna(corr) else np.nan})
    return pd.DataFrame(rows)


def _reserve_penalty_summary(diagnostics: pd.DataFrame, config: ExperimentConfig) -> pd.DataFrame:
    if "reserve_slack_penalty" not in diagnostics.columns:
        return pd.DataFrame(columns=["metric", "value"])
    penalties = diagnostics["reserve_slack_penalty"].astype(float)
    rows = [
        {"metric": "mean", "value": float(penalties.mean())},
        {"metric": "p05", "value": float(penalties.quantile(0.05))},
        {"metric": "p50", "value": float(penalties.quantile(0.50))},
        {"metric": "p95", "value": float(penalties.quantile(0.95))},
        {"metric": "min", "value": float(penalties.min())},
        {"metric": "max", "value": float(penalties.max())},
        {
            "metric": "fraction_at_min",
            "value": float((penalties <= config.reserve_penalty_min + 1e-6).mean()),
        },
        {
            "metric": "fraction_at_max",
            "value": float((penalties >= config.reserve_penalty_max - 1e-6).mean()),
        },
    ]
    return pd.DataFrame(rows)


def _reserve_fraction_summary(diagnostics: pd.DataFrame, config: ExperimentConfig) -> pd.DataFrame:
    if "reserve_requirement_fraction" not in diagnostics.columns:
        return pd.DataFrame(columns=["metric", "value"])
    fractions = diagnostics["reserve_requirement_fraction"].astype(float)
    rows = [
        {"metric": "mean", "value": float(fractions.mean())},
        {"metric": "p05", "value": float(fractions.quantile(0.05))},
        {"metric": "p50", "value": float(fractions.quantile(0.50))},
        {"metric": "p95", "value": float(fractions.quantile(0.95))},
        {"metric": "min", "value": float(fractions.min())},
        {"metric": "max", "value": float(fractions.max())},
        {
            "metric": "fraction_at_min",
            "value": float((fractions <= config.reserve_fraction_min + 1e-8).mean()),
        },
        {
            "metric": "fraction_at_max",
            "value": float((fractions >= config.reserve_fraction_max - 1e-8).mean()),
        },
    ]
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    processed_dir = Path(args.processed_dir)
    output_dir = Path(args.output_dir) if args.output_dir else processed_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    system_df = load_system_hour(processed_dir / "nyiso_system_hour.parquet")
    system_df = add_financial_cost_features(system_df, cost_input_dir=args.cost_input_dir)
    zone_hour = load_zone_hour(processed_dir / "nyiso_zone_hour.parquet")
    interface_hour = _load_required_interface(processed_dir / "nyiso_interface_hour.parquet")
    zonal_capacity_table = load_zonal_capacity(_zonal_capacity_path(processed_dir, args.zonal_capacity_source))
    include_da_constraint_stress = args.proxy_stage in {
        "a3",
        "a4",
        "a5b",
        "a6a",
        "a6b",
        "a6c",
        "a6d",
        "a6e",
        "a6f",
        "a8a",
        "a8b",
        "a8c",
        "a8d",
        "a8e",
        "a7a",
        "a7b",
    }
    include_scheduled_outage_stress = args.proxy_stage in {
        "a5a",
        "a5b",
        "a6a",
        "a6b",
        "a6c",
        "a6d",
        "a6e",
        "a6f",
        "a8a",
        "a8b",
        "a8c",
        "a8d",
        "a8e",
        "a7a",
        "a7b",
    }
    include_ancillary_stress = args.proxy_stage in {"a6a", "a6c", "a7a", "a7b"}
    include_day_ahead_ancillary_stress = args.proxy_stage in {"a6d", "a6e", "a6f", "a8a", "a8b", "a8c", "a8d", "a8e"}
    include_day_ahead_scheduled_outage_stress = args.proxy_stage in {"a6e", "a6f", "a8a", "a8c", "a8d", "a8e"}
    include_zonal_bid_load_features = args.proxy_stage in {"a6f", "a8a", "a8e"}
    use_reserve_headroom_constraint = args.proxy_stage in {
        "a6b",
        "a6c",
        "a6d",
        "a6e",
        "a6f",
        "a8a",
        "a8b",
        "a8c",
        "a8d",
        "a8e",
        "a7a",
        "a7b",
    }
    use_learned_reserve_penalty = args.proxy_stage in {"a7a", "a8b", "a8c", "a8d", "a8e"}
    use_learned_reserve_fraction = args.proxy_stage in {"a7b", "a8b", "a8c", "a8d", "a8e"}
    use_learned_thermal_derates = args.proxy_stage in {"a4", "a8c", "a8d", "a8e"}
    thermal_derate_max = args.thermal_derate_max
    thermal_derate_lr = args.thermal_derate_lr
    thermal_derate_l2 = args.thermal_derate_l2
    thermal_derate_mean_penalty = args.thermal_derate_mean_penalty
    if args.proxy_stage in {"a8c", "a8d", "a8e"}:
        thermal_derate_max = min(thermal_derate_max, 0.20)
        thermal_derate_lr = min(thermal_derate_lr, 0.02)
        thermal_derate_l2 = max(thermal_derate_l2, 0.001)
        thermal_derate_mean_penalty = max(thermal_derate_mean_penalty, 0.05)

    splits, split_policy = _choose_splits(system_df)
    train_df, validation_df, test_df, holdout_df = add_solver_balance_targets(
        splits["train"],
        splits["train"],
        splits["validation"],
        splits["test"],
        splits["holdout"],
        balance_target_policy="lag_1_residual",
    )
    train_df, validation_df, test_df, holdout_df = add_loss_experiment_balance_column(
        train_df,
        validation_df,
        test_df,
        holdout_df,
    )
    config = ExperimentConfig(
        feature_mode="causal_day_ahead",
        load_balance_mode="forecast_load",
        solver_load_column=LOSS_EXPERIMENT_BALANCE_COLUMN,
        direct_epochs=args.direct_epochs,
        structured_epochs=args.structured_epochs,
        structured_lr=args.structured_lr,
        hidden_dim=args.hidden_dim,
        include_lags=True,
        include_cost_features=True,
        zonal_capacity_source=args.zonal_capacity_source,
        include_da_constraint_stress_features=include_da_constraint_stress,
        include_scheduled_outage_stress_features=include_scheduled_outage_stress,
        include_ancillary_stress_features=include_ancillary_stress,
        include_day_ahead_ancillary_stress_features=include_day_ahead_ancillary_stress,
        include_day_ahead_scheduled_outage_stress_features=include_day_ahead_scheduled_outage_stress,
        include_zonal_bid_load_features=include_zonal_bid_load_features,
        gate_p511a=args.gate_p511a,
        gate_p54a=args.gate_p54a,
        gate_p54c=args.gate_p54c,
        gate_p5=args.gate_p5,
        gate_p59=args.gate_p59,
        gate_weather=args.gate_weather,
        gate_cost=args.gate_cost,
        use_reserve_headroom_constraint=use_reserve_headroom_constraint,
        reserve_requirement_fraction=args.reserve_requirement_fraction,
        reserve_slack_penalty=args.reserve_slack_penalty,
        use_learned_reserve_penalty=use_learned_reserve_penalty,
        reserve_penalty_min=args.reserve_penalty_min,
        reserve_penalty_max=args.reserve_penalty_max,
        reserve_penalty_lr=args.reserve_penalty_lr,
        reserve_penalty_l2=args.reserve_penalty_l2,
        use_learned_reserve_fraction=use_learned_reserve_fraction,
        reserve_fraction_min=args.reserve_fraction_min,
        reserve_fraction_max=args.reserve_fraction_max,
        reserve_fraction_lr=args.reserve_fraction_lr,
        reserve_fraction_l2=args.reserve_fraction_l2,
        use_learned_thermal_derates=use_learned_thermal_derates,
        thermal_derate_max=thermal_derate_max,
        thermal_derate_lr=thermal_derate_lr,
        thermal_derate_l2=thermal_derate_l2,
        thermal_derate_mean_penalty=thermal_derate_mean_penalty,
        use_zonal_interface_constraints=True,
        use_zonal_availability_bounds=True,
    )
    config = replace(config, soft_balance_penalty_grid=[0.1])
    capacities = compute_available_capacities(train_df)
    ramp_limits = compute_empirical_ramp_limits(train_df, config.ramp_quantile) if config.use_ramp_constraints else None
    zonal_capacities = compute_zonal_capacities(zonal_capacity_table, capacities)
    system_availability_caps = compute_empirical_system_availability_caps(train_df, capacities)

    model, history = train_zonal_financial_cost_model(
        train_df,
        use_weather=True,
        zone_hour=zone_hour,
        interface_hour=interface_hour,
        zonal_capacities=zonal_capacities,
        config=config,
        ramp_limits=ramp_limits,
        system_availability_caps=system_availability_caps,
        load_column=LOSS_EXPERIMENT_BALANCE_COLUMN,
        max_train_samples=args.max_train_hours,
    )

    eval_map = {"validation": validation_df, "test": test_df, "holdout": holdout_df}
    interface_summary = _interface_summary_rows(
        interface_hour,
        {
            "train": train_df,
            "validation": validation_df,
            "test": test_df,
            "holdout": holdout_df,
        },
    )
    interface_summary.insert(0, "proxy_stage", args.proxy_stage)
    interface_summary.to_csv(output_dir / "zonal_cost_lto_interface_summary.csv", index=False)
    _feature_metadata(config).to_csv(output_dir / "zonal_cost_lto_feature_metadata.csv", index=False)
    if include_da_constraint_stress:
        _da_constraint_feature_summary(system_df).to_csv(
            output_dir / "zonal_cost_lto_da_constraint_feature_summary.csv",
            index=False,
        )
    if include_scheduled_outage_stress:
        _scheduled_outage_feature_summary(system_df).to_csv(
            output_dir / "zonal_cost_lto_scheduled_outage_feature_summary.csv",
            index=False,
        )
    if include_ancillary_stress:
        _ancillary_feature_summary(system_df).to_csv(
            output_dir / "zonal_cost_lto_ancillary_feature_summary.csv",
            index=False,
        )
    if include_day_ahead_ancillary_stress:
        _day_ahead_ancillary_feature_summary(system_df).to_csv(
            output_dir / "zonal_cost_lto_day_ahead_ancillary_feature_summary.csv",
            index=False,
        )
    if include_day_ahead_scheduled_outage_stress:
        _day_ahead_scheduled_outage_feature_summary(system_df).to_csv(
            output_dir / "zonal_cost_lto_day_ahead_scheduled_outage_feature_summary.csv",
            index=False,
        )
    if include_zonal_bid_load_features:
        _zonal_bid_load_feature_summary(system_df).to_csv(
            output_dir / "zonal_cost_lto_zonal_bid_load_feature_summary.csv",
            index=False,
        )

    summary_parts = []
    hourly_parts = []
    oracle_diag_parts = []
    lto_diag_parts = []
    cost_parts = []
    interface_ablation_parts = []
    for split_name in _selected_splits(args.split):
        eval_df = eval_map[split_name].sort_values("timestamp_utc").copy()
        if args.max_eval_hours > 0:
            eval_df = eval_df.head(args.max_eval_hours).copy()
        test_context = make_aligned_test_frame(train_df, eval_df, config.resolved_history_window)
        oracle_pred, target, oracle_diag, _, solve_df, _ = predict_zonal_financial_oracle(
            test_context,
            use_weather=True,
            zone_hour=zone_hour,
            interface_hour=interface_hour,
            zonal_capacities=zonal_capacities,
            config=config,
            ramp_limits=ramp_limits,
            system_availability_caps=system_availability_caps,
            load_column=LOSS_EXPERIMENT_BALANCE_COLUMN,
        )
        lto_pred, lto_target, lto_diag, learned_costs, lto_solve_df, _ = predict_zonal_financial_cost_lto(
            model,
            test_context,
            use_weather=True,
            zone_hour=zone_hour,
            interface_hour=interface_hour,
            zonal_capacities=zonal_capacities,
            config=config,
            ramp_limits=ramp_limits,
            system_availability_caps=system_availability_caps,
            load_column=LOSS_EXPERIMENT_BALANCE_COLUMN,
        )
        if not np.allclose(target, lto_target):
            raise ValueError(f"Oracle and LtO evaluation targets are misaligned for split {split_name}.")

        split_summary = pd.DataFrame(
            [
                _summary_row("zonal_financial_oracle_lp", oracle_pred, target, oracle_diag),
                _summary_row(
                    "zonal_financial_fenchel_young_lto",
                    lto_pred,
                    target,
                    lto_diag,
                    oracle_diagnostics=oracle_diag,
                    structured_train_loss=history[-1] if history else np.nan,
                ),
            ]
        )
        split_summary["split"] = split_name
        split_summary["split_policy"] = split_policy
        split_summary["proxy_stage"] = args.proxy_stage
        split_summary["zonal_capacity_source"] = args.zonal_capacity_source
        split_summary["interface_constraints_enabled"] = True
        split_summary["include_da_constraint_stress_features"] = include_da_constraint_stress
        split_summary["include_scheduled_outage_stress_features"] = include_scheduled_outage_stress
        split_summary["include_ancillary_stress_features"] = include_ancillary_stress
        split_summary["include_day_ahead_ancillary_stress_features"] = include_day_ahead_ancillary_stress
        split_summary["include_day_ahead_scheduled_outage_stress_features"] = include_day_ahead_scheduled_outage_stress
        split_summary["include_zonal_bid_load_features"] = include_zonal_bid_load_features
        split_summary["use_reserve_headroom_constraint"] = use_reserve_headroom_constraint
        split_summary["reserve_requirement_fraction"] = args.reserve_requirement_fraction
        split_summary["reserve_slack_penalty"] = args.reserve_slack_penalty
        split_summary["use_learned_reserve_penalty"] = use_learned_reserve_penalty
        split_summary["reserve_penalty_min"] = args.reserve_penalty_min
        split_summary["reserve_penalty_max"] = args.reserve_penalty_max
        split_summary["use_learned_reserve_fraction"] = use_learned_reserve_fraction
        split_summary["reserve_fraction_min"] = args.reserve_fraction_min
        split_summary["reserve_fraction_max"] = args.reserve_fraction_max
        split_summary["use_learned_thermal_derates"] = use_learned_thermal_derates
        split_summary["gate_p511a"] = args.gate_p511a
        split_summary["gate_p54a"] = args.gate_p54a
        split_summary["gate_p54c"] = args.gate_p54c
        split_summary["gate_p5"] = args.gate_p5
        split_summary["gate_p59"] = args.gate_p59
        split_summary["gate_weather"] = args.gate_weather
        split_summary["gate_cost"] = args.gate_cost
        split_summary["thermal_derate_max"] = thermal_derate_max
        split_summary["thermal_derate_lr"] = thermal_derate_lr
        split_summary["thermal_derate_l2"] = thermal_derate_l2
        split_summary["thermal_derate_mean_penalty"] = thermal_derate_mean_penalty
        split_summary["train_rows"] = len(train_df)
        split_summary["eval_rows"] = len(solve_df)
        split_summary["max_train_hours"] = args.max_train_hours
        split_summary["learned_cost_mean"] = float(np.mean(learned_costs))
        summary_parts.append(split_summary)

        split_hourly = _hourly_frame(lto_solve_df, target, oracle_pred, lto_pred, oracle_diag, lto_diag)
        split_hourly.insert(0, "split", split_name)
        hourly_parts.append(split_hourly)

        split_costs = pd.DataFrame(
            {
                "split": split_name,
                "timestamp_utc": lto_solve_df["timestamp_utc"].to_numpy(),
                "learned_cost_mean": learned_costs.reshape(len(learned_costs), -1).mean(axis=1),
                "learned_cost_min": learned_costs.reshape(len(learned_costs), -1).min(axis=1),
                "learned_cost_max": learned_costs.reshape(len(learned_costs), -1).max(axis=1),
                "financial_cost_mean": financial_cost_matrix_from_frame(lto_solve_df, zonal=True)
                .reshape(len(lto_solve_df), -1)
                .mean(axis=1),
            }
        )
        cost_parts.append(split_costs)
        oracle_diag_parts.append(oracle_diag.assign(split=split_name, model="zonal_financial_oracle_lp"))
        lto_diag_parts.append(lto_diag.assign(split=split_name, model="zonal_financial_fenchel_young_lto"))
        if use_learned_thermal_derates:
            _thermal_derate_summary(lto_diag).to_csv(
                output_dir / f"zonal_cost_lto_{split_name}_thermal_derate_summary.csv",
                index=False,
            )
        if use_learned_reserve_penalty:
            _reserve_penalty_summary(lto_diag, config).to_csv(
                output_dir / f"zonal_cost_lto_{split_name}_reserve_penalty_summary.csv",
                index=False,
            )
        if use_learned_reserve_fraction:
            _reserve_fraction_summary(lto_diag, config).to_csv(
                output_dir / f"zonal_cost_lto_{split_name}_reserve_fraction_summary.csv",
                index=False,
            )
            _derate_stress_correlation(system_df, lto_diag).to_csv(
                output_dir / f"zonal_cost_lto_{split_name}_derate_stress_correlation.csv",
                index=False,
            )

        if args.run_interface_off_ablation:
            off_config = replace(config, use_zonal_interface_constraints=False)
            off_oracle_pred, off_target, off_oracle_diag, _, _, _ = predict_zonal_financial_oracle(
                test_context,
                use_weather=True,
                zone_hour=zone_hour,
                interface_hour=empty_interface_hour(),
                zonal_capacities=zonal_capacities,
                config=off_config,
                ramp_limits=ramp_limits,
                system_availability_caps=system_availability_caps,
                load_column=LOSS_EXPERIMENT_BALANCE_COLUMN,
            )
            off_lto_pred, off_lto_target, off_lto_diag, _, _, _ = predict_zonal_financial_cost_lto(
                model,
                test_context,
                use_weather=True,
                zone_hour=zone_hour,
                interface_hour=empty_interface_hour(),
                zonal_capacities=zonal_capacities,
                config=off_config,
                ramp_limits=ramp_limits,
                system_availability_caps=system_availability_caps,
                load_column=LOSS_EXPERIMENT_BALANCE_COLUMN,
            )
            if not np.allclose(off_target, off_lto_target):
                raise ValueError(f"Interface-off oracle and LtO targets are misaligned for split {split_name}.")
            off_summary = pd.DataFrame(
                [
                    _summary_row("zonal_financial_oracle_lp_interface_off", off_oracle_pred, off_target, off_oracle_diag),
                    _summary_row(
                        "zonal_financial_fenchel_young_lto_interface_off",
                        off_lto_pred,
                        off_target,
                        off_lto_diag,
                        oracle_diagnostics=off_oracle_diag,
                        structured_train_loss=history[-1] if history else np.nan,
                    ),
                ]
            )
            off_summary["split"] = split_name
            off_summary["split_policy"] = split_policy
            off_summary["proxy_stage"] = args.proxy_stage
            off_summary["zonal_capacity_source"] = args.zonal_capacity_source
            off_summary["interface_constraints_enabled"] = False
            off_summary["include_da_constraint_stress_features"] = include_da_constraint_stress
            off_summary["include_scheduled_outage_stress_features"] = include_scheduled_outage_stress
            off_summary["include_ancillary_stress_features"] = include_ancillary_stress
            off_summary["include_day_ahead_ancillary_stress_features"] = include_day_ahead_ancillary_stress
            off_summary["include_day_ahead_scheduled_outage_stress_features"] = (
                include_day_ahead_scheduled_outage_stress
            )
            off_summary["include_zonal_bid_load_features"] = include_zonal_bid_load_features
            off_summary["use_reserve_headroom_constraint"] = use_reserve_headroom_constraint
            off_summary["reserve_requirement_fraction"] = args.reserve_requirement_fraction
            off_summary["reserve_slack_penalty"] = args.reserve_slack_penalty
            off_summary["use_learned_reserve_penalty"] = use_learned_reserve_penalty
            off_summary["reserve_penalty_min"] = args.reserve_penalty_min
            off_summary["reserve_penalty_max"] = args.reserve_penalty_max
            off_summary["use_learned_reserve_fraction"] = use_learned_reserve_fraction
            off_summary["reserve_fraction_min"] = args.reserve_fraction_min
            off_summary["reserve_fraction_max"] = args.reserve_fraction_max
            off_summary["use_learned_thermal_derates"] = use_learned_thermal_derates
            off_summary["train_rows"] = len(train_df)
            off_summary["eval_rows"] = len(off_oracle_diag)
            off_summary["max_train_hours"] = args.max_train_hours
            interface_ablation_parts.append(off_summary)

    summary = pd.concat(summary_parts, ignore_index=True)
    hourly = pd.concat(hourly_parts, ignore_index=True)
    costs = pd.concat(cost_parts, ignore_index=True)
    oracle_diag_out = pd.concat(oracle_diag_parts, ignore_index=True)
    lto_diag_out = pd.concat(lto_diag_parts, ignore_index=True)

    summary.to_csv(output_dir / "zonal_cost_lto_metrics.csv", index=False)
    hourly.to_csv(output_dir / "zonal_cost_lto_hourly.csv", index=False)
    oracle_diag_out.to_csv(output_dir / "zonal_cost_lto_oracle_diagnostics.csv", index=False)
    lto_diag_out.to_csv(output_dir / "zonal_cost_lto_lto_diagnostics.csv", index=False)
    costs.to_csv(output_dir / "zonal_cost_lto_learned_cost_summary.csv", index=False)
    if interface_ablation_parts:
        pd.concat(interface_ablation_parts, ignore_index=True).to_csv(
            output_dir / "zonal_cost_lto_interface_ablation_metrics.csv",
            index=False,
        )
    pd.DataFrame({"epoch": np.arange(1, len(history) + 1), "structured_train_loss": history}).to_csv(
        output_dir / "zonal_cost_lto_training_history.csv",
        index=False,
    )

    display_cols = [
        "split",
        "model",
        "overall_mae",
        "total_fuel_mix_mae",
        "mean_financial_dispatch_cost",
        "mean_financial_cost_gap_to_oracle",
        "success_rate",
        "mean_abs_balance_residual",
        "mean_interface_slack_mw",
        "mean_binding_interface_count",
        "mean_transfer_magnitude_mw",
        "mean_reserve_requirement_mw",
        "mean_reserve_headroom_mw",
        "mean_reserve_slack_mw",
        "max_reserve_slack_mw",
        "mean_reserve_slack_penalty",
        "p50_reserve_slack_penalty",
        "mean_reserve_requirement_fraction",
        "p50_reserve_requirement_fraction",
        "mean_thermal_derate",
        "max_thermal_derate",
        "mean_runtime_seconds",
    ]
    print(format_results_table(summary, display_cols))
    print(f"\nWrote zonal cost LtO outputs to {output_dir}")


if __name__ == "__main__":
    main()
