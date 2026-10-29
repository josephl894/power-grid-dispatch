from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from nyiso_processing import (
    add_ancillary_stress_features,
    add_constraint_stress_features,
    add_day_ahead_ancillary_stress_features,
    add_outage_stress_features,
    add_zonal_bid_load_features,
    build_gold_book_capacity_tables,
    load_ancillary_service_features_hour,
    load_constraint_features_hour,
    load_day_ahead_ancillary_service_features_hour,
    load_interface_hour,
    load_outage_features_hour,
    load_zonal_bid_load_features_hour,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build public-data NYISO SCUC proxy input tables.")
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--processed-dir", default="data/processed/scuc_proxy")
    parser.add_argument(
        "--interface-processed-dir",
        action="append",
        default=[],
        help="Processed directory that should receive nyiso_interface_hour.parquet. May be repeated.",
    )
    parser.add_argument(
        "--enhance-system-hour-dir",
        action="append",
        default=[],
        help="Processed directory whose nyiso_system_hour.parquet should receive enhanced P-511A features. May be repeated.",
    )
    parser.add_argument("--interface-start", default="2020-01-01")
    parser.add_argument("--interface-end", default="2025-12-31")
    return parser.parse_args()


def _write_parquet(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def _enhance_system_hour(raw_dir: Path, processed_dir: Path, start: str, end: str) -> None:
    system_path = processed_dir / "nyiso_system_hour.parquet"
    if not system_path.exists():
        raise FileNotFoundError(f"Missing {system_path}")
    system_hour = pd.read_parquet(system_path)
    system_hour["timestamp_utc"] = pd.to_datetime(system_hour["timestamp_utc"], utc=True)
    enhanced = load_constraint_features_hour(
        raw_dir,
        "P-511A Day-Ahead Limiting Constraints",
        prefix="da",
        start=start,
        end=end,
    )
    if not enhanced.empty:
        drop_cols = [col for col in enhanced.columns if col != "timestamp_utc" and col in system_hour.columns]
        system_hour = system_hour.drop(columns=drop_cols).merge(enhanced, on="timestamp_utc", how="left")
    ancillary = load_ancillary_service_features_hour(
        raw_dir,
        start=start,
        end=end,
    )
    if not ancillary.empty:
        drop_cols = [col for col in ancillary.columns if col != "timestamp_utc" and col in system_hour.columns]
        system_hour = system_hour.drop(columns=drop_cols).merge(ancillary, on="timestamp_utc", how="left")
    da_ancillary = load_day_ahead_ancillary_service_features_hour(
        raw_dir,
        start=start,
        end=end,
    )
    if not da_ancillary.empty:
        drop_cols = [col for col in da_ancillary.columns if col != "timestamp_utc" and col in system_hour.columns]
        system_hour = system_hour.drop(columns=drop_cols).merge(da_ancillary, on="timestamp_utc", how="left")
    da_scheduled = load_outage_features_hour(
        raw_dir,
        "P-54C Day-Ahead Scheduled Outages",
        prefix="da_scheduled",
        start=start,
        end=end,
    )
    if not da_scheduled.empty:
        drop_cols = [col for col in da_scheduled.columns if col != "timestamp_utc" and col in system_hour.columns]
        system_hour = system_hour.drop(columns=drop_cols).merge(da_scheduled, on="timestamp_utc", how="left")
    p59 = load_zonal_bid_load_features_hour(
        raw_dir,
        start=start,
        end=end,
    )
    if not p59.empty:
        drop_cols = [col for col in p59.columns if col != "timestamp_utc" and col in system_hour.columns]
        system_hour = system_hour.drop(columns=drop_cols).merge(p59, on="timestamp_utc", how="left")
    if "p59_energy_bid_load_mw" in system_hour.columns:
        system_hour["p59_bid_load_gap_to_forecast_mw"] = (
            system_hour["p59_energy_bid_load_mw"] - system_hour["forecast_load_mw"]
        )
    system_hour = add_constraint_stress_features(system_hour, "da")
    system_hour = add_outage_stress_features(system_hour, "scheduled")
    system_hour = add_outage_stress_features(system_hour, "da_scheduled")
    system_hour = add_ancillary_stress_features(system_hour)
    system_hour = add_day_ahead_ancillary_stress_features(system_hour)
    system_hour = add_zonal_bid_load_features(system_hour)
    numeric_cols = system_hour.select_dtypes(include="number").columns
    system_hour[numeric_cols] = system_hour[numeric_cols].ffill().bfill().fillna(0.0)
    _write_parquet(system_hour.sort_values("timestamp_utc"), system_path)
    feature_cols = [col for col in system_hour.columns if col.startswith("da_constraint_")]
    outage_cols = [col for col in system_hour.columns if col.startswith("scheduled_outage_")]
    ancillary_cols = [col for col in system_hour.columns if col.startswith("ancillary_")]
    da_ancillary_cols = [col for col in system_hour.columns if col.startswith("da_ancillary_")]
    p59_cols = [col for col in system_hour.columns if col.startswith("p59_")]
    print(
        f"\nEnhanced {system_path}: "
        f"da_constraint_feature_cols={len(feature_cols)}, "
        f"scheduled_outage_feature_cols={len(outage_cols)}, "
        f"ancillary_feature_cols={len(ancillary_cols)}, "
        f"da_ancillary_feature_cols={len(da_ancillary_cols)}, "
        f"p59_feature_cols={len(p59_cols)}"
    )


def main() -> None:
    args = parse_args()
    raw_dir = Path(args.raw_dir)
    processed_dir = Path(args.processed_dir)

    facilities, fuel_capacity, tech_capacity, validation = build_gold_book_capacity_tables(raw_dir)
    _write_parquet(facilities, processed_dir / "generator_facilities.parquet")
    _write_parquet(fuel_capacity, processed_dir / "zonal_fuel_capacity.parquet")
    _write_parquet(tech_capacity, processed_dir / "zonal_tech_capacity.parquet")
    validation.to_csv(processed_dir / "validation_checks.csv", index=False)

    print("SCUC proxy capacity tables:")
    print(f"generator_facilities\trows={facilities.shape[0]}\tcols={facilities.shape[1]}")
    print(f"zonal_fuel_capacity\trows={fuel_capacity.shape[0]}\tcols={fuel_capacity.shape[1]}")
    print(f"zonal_tech_capacity\trows={tech_capacity.shape[0]}\tcols={tech_capacity.shape[1]}")

    print("\nValidation checks:")
    print(validation.to_string(index=False))

    totals = fuel_capacity.groupby("fuel_category", as_index=False)["capacity_mw"].sum()
    print("\nSystem summer capacity by fuel:")
    print(totals.to_string(index=False))

    interface_targets = [Path(path) for path in args.interface_processed_dir]
    if interface_targets:
        interface_hour = load_interface_hour(raw_dir, start=args.interface_start, end=args.interface_end)
        if interface_hour.empty:
            raise RuntimeError("P-32 interface-hour build returned no rows.")
        for target_dir in interface_targets:
            target_path = target_dir / "nyiso_interface_hour.parquet"
            _write_parquet(interface_hour, target_path)
            finite_rows = int((interface_hour["upper_is_finite"] | interface_hour["lower_is_finite"]).sum())
            print(
                f"\nWrote {target_path}: rows={len(interface_hour)}, "
                f"finite_limit_rows={finite_rows}, "
                f"range={interface_hour['timestamp_utc'].min()} to {interface_hour['timestamp_utc'].max()}"
            )

    for target_dir in [Path(path) for path in args.enhance_system_hour_dir]:
        _enhance_system_hour(raw_dir, target_dir, args.interface_start, args.interface_end)


if __name__ == "__main__":
    main()
