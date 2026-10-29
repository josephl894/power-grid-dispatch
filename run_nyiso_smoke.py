# OUTDATED
# Legacy January smoke workflow for the residual/repair experiments. The final
# public SCUC proxy path is A5b/A6c/A7b via run_zonal_cost_lto.py.
from __future__ import annotations

from pathlib import Path

from nyiso_experiments import (
    COMPACT_BALANCE_TARGET_COLUMNS,
    COMPACT_DIAGNOSTIC_COLUMNS,
    COMPACT_FEASIBILITY_COLUMNS,
    COMPACT_LOSS_EXPERIMENT_COLUMNS,
    COMPACT_REPAIR_COMPARISON_COLUMNS,
    ExperimentConfig,
    format_results_table,
    repair_comparison_table,
    run_balance_target_diagnostics,
    run_feasibility_gap_diagnostics,
    run_loss_function_experiments,
    run_smoke_experiment,
    solver_diagnostics_summary,
)
from nyiso_processing import build_and_validate


def main() -> None:
    outputs, checks = build_and_validate(start="2024-01-01", end="2024-01-31")
    print("Processed tables:")
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

    config = ExperimentConfig(direct_epochs=40, structured_epochs=1, candidate_count=4, hidden_dim=32)
    results, diagnostics = run_smoke_experiment(config=config)
    feasibility_summary, feasibility_hourly, _ = run_feasibility_gap_diagnostics(config=config)
    balance_summary, balance_hourly, _ = run_balance_target_diagnostics(config=config)
    loss_summary, loss_hourly, _ = run_loss_function_experiments(config=config)
    output_dir = Path("data/processed")
    output_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_dir / "nyiso_smoke_results.csv", index=False)
    for name, frame in diagnostics.items():
        frame.to_csv(output_dir / f"nyiso_smoke_{name}_diagnostics.csv", index=False)
    feasibility_summary.to_csv(output_dir / "nyiso_feasibility_gap_summary.csv", index=False)
    feasibility_hourly.to_csv(output_dir / "nyiso_feasibility_gap_hourly.csv", index=False)
    balance_summary.to_csv(output_dir / "nyiso_balance_target_summary.csv", index=False)
    balance_hourly.to_csv(output_dir / "nyiso_balance_target_hourly.csv", index=False)
    loss_summary.to_csv(output_dir / "nyiso_loss_experiment_summary.csv", index=False)
    loss_hourly.to_csv(output_dir / "nyiso_loss_experiment_hourly.csv", index=False)

    result_columns = None if config.display_metric_profile == "compact" else results.columns.tolist()
    print("\nSmoke results:")
    print(format_results_table(results, result_columns))

    print("\nRepair comparison:")
    print(format_results_table(repair_comparison_table(results), COMPACT_REPAIR_COMPARISON_COLUMNS))

    print("\nSolver diagnostics:")
    print(format_results_table(solver_diagnostics_summary(diagnostics), COMPACT_DIAGNOSTIC_COLUMNS))

    print("\nFeasibility gap diagnostics:")
    print(format_results_table(feasibility_summary, COMPACT_FEASIBILITY_COLUMNS))

    print("\nBalance target diagnostics:")
    print(format_results_table(balance_summary, COMPACT_BALANCE_TARGET_COLUMNS))

    print("\nLoss function experiments:")
    print(format_results_table(loss_summary, COMPACT_LOSS_EXPERIMENT_COLUMNS))


if __name__ == "__main__":
    main()
