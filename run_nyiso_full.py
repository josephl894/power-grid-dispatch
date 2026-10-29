# OUTDATED
# Legacy full-period residual/repair runner. Keep only for historical
# reproducibility; final SCUC proxy work should not extend this path.
from __future__ import annotations

import argparse
from pathlib import Path

from nyiso_experiments import (
    COMPACT_DIAGNOSTIC_COLUMNS,
    COMPACT_METRIC_COLUMNS,
    COMPACT_REPAIR_COMPARISON_COLUMNS,
    ExperimentConfig,
    format_results_table,
    repair_comparison_table,
    run_full_period_experiment,
    solver_diagnostics_summary,
)
from nyiso_processing import build_and_validate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run NYISO full-period or pilot chronological experiments.")
    parser.add_argument("--start", default="2016-01-01")
    parser.add_argument("--end", default="2025-12-31")
    parser.add_argument("--processed-dir", default="data/processed/full")
    parser.add_argument("--direct-epochs", type=int, default=40)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--skip-build", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.processed_dir)

    if args.skip_build:
        print(f"Skipping build; using processed tables in {output_dir}")
    else:
        outputs, checks = build_and_validate(
            start=args.start,
            end=args.end,
            processed_dir=output_dir,
            include_interface_hour=False,
        )

        print("Processed full-period tables:")
        for name, frame in outputs.items():
            print(f"{name}\trows={frame.shape[0]}\tcols={frame.shape[1]}")

        print("\nKey checks:")
        key_checks = [
            "weather_zones",
            "weather_duplicate_zone_hours",
            "weather_remaining_ghi_sentinel",
            "system_duplicate_hours",
            "system_rows",
            "target_negative_values",
            "zone_hour_zones",
            "zone_hour_duplicate_zone_hours",
            "zone_system_forecast_load_max_abs_diff",
            "interface_rows",
            "interface_finite_limit_abs_ge_9000",
            "zonal_capacity_rows",
        ]
        for key in key_checks:
            if key in checks:
                print(f"{key}\t{checks[key]}")

    config = ExperimentConfig(
        direct_epochs=args.direct_epochs,
        structured_epochs=1,
        candidate_count=4,
        hidden_dim=args.hidden_dim,
        soft_balance_penalty_grid=[0.1, 1.0],
    )
    results, diagnostics, split_summary = run_full_period_experiment(
        system_hour_path=output_dir / "nyiso_system_hour.parquet",
        config=config,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_dir / "nyiso_full_results.csv", index=False)
    split_summary.to_csv(output_dir / "nyiso_full_split_summary.csv", index=False)
    solver_summary = solver_diagnostics_summary(diagnostics)
    solver_summary.to_csv(output_dir / "nyiso_full_solver_diagnostics.csv", index=False)
    repair_results = repair_comparison_table(results)
    repair_results.to_csv(output_dir / "nyiso_full_repair_comparison.csv", index=False)
    results[results["split"] == "test_2023_2024"].to_csv(output_dir / "nyiso_full_test_results.csv", index=False)
    results[results["split"] == "holdout_2025"].to_csv(output_dir / "nyiso_full_holdout_results.csv", index=False)
    for name, frame in diagnostics.items():
        frame.to_csv(output_dir / f"nyiso_full_{name}_diagnostics.csv", index=False)

    print("\nSplit row counts:")
    print(format_results_table(split_summary, ["split", "split_strategy", "rows", "start_utc", "end_utc"]))

    result_columns = ["split", *COMPACT_METRIC_COLUMNS]
    print("\nFull-period results:")
    print(format_results_table(results, result_columns))

    repair_columns = ["split", *COMPACT_REPAIR_COMPARISON_COLUMNS]
    print("\nRepair comparison:")
    print(format_results_table(repair_results, repair_columns))

    print("\nSolver diagnostics:")
    print(format_results_table(solver_summary, COMPACT_DIAGNOSTIC_COLUMNS))


if __name__ == "__main__":
    main()
