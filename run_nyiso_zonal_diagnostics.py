# OUTDATED
# Legacy bounded zonal LP diagnostic runner used before the A5b/A6c/A7b public
# SCUC proxy path was selected.
from __future__ import annotations

import argparse
from pathlib import Path

from nyiso_experiments import (
    COMPACT_DIAGNOSTIC_COLUMNS,
    COMPACT_ZONAL_DIAGNOSTIC_COLUMNS,
    ExperimentConfig,
    format_results_table,
    run_zonal_diagnostics,
    solver_diagnostics_summary,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run bounded NYISO zonal LP diagnostic variants.")
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--split", default="test", choices=["train", "validation", "val", "test", "test_2023_2024", "holdout", "holdout_2025"])
    parser.add_argument("--max-hours", type=int, default=500)
    parser.add_argument("--direct-epochs", type=int, default=40)
    parser.add_argument("--hidden-dim", type=int, default=32)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    processed_dir = Path(args.processed_dir)
    interface_path = processed_dir / "nyiso_interface_hour.parquet"
    if not interface_path.exists():
        print(
            "Interface table not found in processed dir; interface variants will run with no finite interface bounds."
        )

    config = ExperimentConfig(
        direct_epochs=args.direct_epochs,
        structured_epochs=1,
        candidate_count=4,
        hidden_dim=args.hidden_dim,
    )
    summary, diagnostics, split_summary = run_zonal_diagnostics(
        system_hour_path=processed_dir / "nyiso_system_hour.parquet",
        zone_hour_path=processed_dir / "nyiso_zone_hour.parquet",
        interface_hour_path=interface_path,
        zonal_capacity_path=processed_dir / "nyiso_zonal_capacity.parquet",
        split=args.split,
        max_hours=args.max_hours,
        config=config,
    )

    processed_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(processed_dir / "nyiso_zonal_diagnostic_results.csv", index=False)
    solver_summary = solver_diagnostics_summary(diagnostics)
    solver_summary.to_csv(processed_dir / "nyiso_zonal_diagnostic_solver.csv", index=False)
    split_summary.to_csv(processed_dir / "nyiso_zonal_diagnostic_split_summary.csv", index=False)
    for name, frame in diagnostics.items():
        frame.to_csv(processed_dir / f"nyiso_zonal_diagnostic_{name}_hourly.csv", index=False)

    print("Split row counts:")
    print(format_results_table(split_summary, ["split", "split_strategy", "rows", "start_utc", "end_utc"]))

    print("\nZonal diagnostic results:")
    print(format_results_table(summary, COMPACT_ZONAL_DIAGNOSTIC_COLUMNS))

    print("\nSolver diagnostics:")
    print(format_results_table(solver_summary, COMPACT_DIAGNOSTIC_COLUMNS))

    if "interface_data_available" in summary.columns and not bool(summary["interface_data_available"].any()):
        print("\nNote: no processed interface-hour data was available for this run, so interface variants are equivalent to no finite interface constraints.")


if __name__ == "__main__":
    main()
