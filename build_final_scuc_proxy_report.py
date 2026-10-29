from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


BASE_DIR = Path("data/processed/final_submission")
FIG_DIR = Path("figures/final_scuc_proxy")
TABLE_PATH = BASE_DIR / "final_scuc_proxy_ablation_table.csv"
APPENDIX_PATH = BASE_DIR / "final_scuc_proxy_appendix_table.csv"
MONTHLY_PATH = BASE_DIR / "final_scuc_proxy_monthly_diagnostics.csv"

RUNS = [
    {
        "stage": "A0",
        "label": "A0 Legacy capacity",
        "path": BASE_DIR / "zonal_cost_lto_ablation_A0_legacy_test_e3",
        "capacity_source": "legacy",
        "p511a": False,
        "p54a": False,
        "p6b": False,
        "reserve_proxy": False,
        "learned_reserve_fraction": False,
        "primary_role": "legacy baseline",
    },
    {
        "stage": "A1",
        "label": "A1 Gold Book capacity",
        "path": BASE_DIR / "zonal_cost_lto_ablation_A1_gold_book_test_e3",
        "capacity_source": "gold_book",
        "p511a": False,
        "p54a": False,
        "p6b": False,
        "reserve_proxy": False,
        "learned_reserve_fraction": False,
        "primary_role": "public fleet baseline",
    },
    {
        "stage": "A2",
        "label": "A2 P-32 interfaces",
        "path": BASE_DIR / "zonal_cost_lto_A2_interface_test_e3",
        "capacity_source": "gold_book",
        "p511a": False,
        "p54a": False,
        "p6b": False,
        "reserve_proxy": False,
        "learned_reserve_fraction": False,
        "primary_role": "interface baseline",
    },
    {
        "stage": "A3",
        "label": "A3 P-511A stress",
        "path": BASE_DIR / "zonal_cost_lto_A3_p511a_test_e3",
        "capacity_source": "gold_book",
        "p511a": True,
        "p54a": False,
        "p6b": False,
        "reserve_proxy": False,
        "learned_reserve_fraction": False,
        "primary_role": "constraint feature ablation",
    },
    {
        "stage": "A5b",
        "label": "A5b P-511A + P-54A",
        "path": BASE_DIR / "zonal_cost_lto_A5b_reference_test_e3",
        "capacity_source": "gold_book",
        "p511a": True,
        "p54a": True,
        "p6b": False,
        "reserve_proxy": False,
        "learned_reserve_fraction": False,
        "primary_role": "clean no-reserve baseline",
    },
    {
        "stage": "A6c",
        "label": "A6c P-6B + fixed reserve",
        "path": BASE_DIR / "zonal_cost_lto_A6c_frac02_train720_e7",
        "capacity_source": "gold_book",
        "p511a": True,
        "p54a": True,
        "p6b": True,
        "reserve_proxy": True,
        "learned_reserve_fraction": False,
        "primary_role": "primary financial/stress-aware candidate",
    },
    {
        "stage": "A7b",
        "label": "A7b learned reserve fraction",
        "path": BASE_DIR / "zonal_cost_lto_A7b_learned_reserve_fraction_train720_e7",
        "capacity_source": "gold_book",
        "p511a": True,
        "p54a": True,
        "p6b": True,
        "reserve_proxy": True,
        "learned_reserve_fraction": True,
        "primary_role": "fuel-mix realism candidate",
    },
]

APPENDIX_RUNS = [
    {
        "stage": "A4",
        "label": "A4 learned thermal derates",
        "path": BASE_DIR / "zonal_cost_lto_A4_derate_test_e3",
        "primary_role": "rejected derate ablation",
    },
    {
        "stage": "A4r",
        "label": "A4r regularized derates",
        "path": BASE_DIR / "zonal_cost_lto_A4r_derate_regularized_test_e3",
        "primary_role": "rejected derate ablation",
    },
    {
        "stage": "A7a",
        "label": "A7a learned reserve penalty",
        "path": BASE_DIR / "zonal_cost_lto_A7a_learned_reserve_penalty_train720_e7",
        "primary_role": "rejected reserve penalty ablation",
    },
]

FUEL_KEYS = [
    "dual_fuel",
    "natural_gas",
    "hydro",
    "nuclear",
    "wind",
    "other_renewables",
    "other_fossil_fuels",
]


def _safe(row: pd.Series, column: str, default=np.nan):
    return row[column] if column in row.index else default


def load_run_summary(spec: dict) -> dict:
    metrics_path = spec["path"] / "zonal_cost_lto_metrics.csv"
    if not metrics_path.exists():
        raise FileNotFoundError(metrics_path)
    metrics = pd.read_csv(metrics_path)
    oracle = metrics[metrics["model"].str.contains("oracle", na=False)].iloc[0]
    lto = metrics[metrics["model"].str.contains("fenchel_young", na=False)].iloc[0]
    row = {
        "stage": spec["stage"],
        "label": spec["label"],
        "primary_role": spec.get("primary_role", ""),
        "capacity_source": spec.get("capacity_source", _safe(lto, "zonal_capacity_source", "")),
        "interface_enabled": bool(_safe(lto, "interface_constraints_enabled", True)),
        "uses_p511a": spec.get("p511a", bool(_safe(lto, "include_da_constraint_stress_features", False))),
        "uses_p54a_scheduled_outage": spec.get(
            "p54a", bool(_safe(lto, "include_scheduled_outage_stress_features", False))
        ),
        "uses_p6b_ancillary": spec.get("p6b", bool(_safe(lto, "include_ancillary_stress_features", False))),
        "uses_reserve_proxy": spec.get("reserve_proxy", bool(_safe(lto, "use_reserve_headroom_constraint", False))),
        "learned_reserve_fraction": spec.get(
            "learned_reserve_fraction", bool(_safe(lto, "use_learned_reserve_fraction", False))
        ),
        "oracle_financial_cost_per_hour": float(_safe(oracle, "mean_financial_dispatch_cost")),
        "lto_financial_cost_per_hour": float(_safe(lto, "mean_financial_dispatch_cost")),
        "regret_to_oracle_per_hour": float(_safe(lto, "mean_financial_cost_gap_to_oracle")),
        "p95_regret_to_oracle_per_hour": float(_safe(lto, "p95_financial_cost_gap_to_oracle")),
        "overall_p63_mae_mw": float(_safe(lto, "overall_mae")),
        "total_fuel_mix_mae_mw": float(_safe(lto, "total_fuel_mix_mae")),
        "success_rate": float(_safe(lto, "success_rate")),
        "mean_balance_residual_mw": float(_safe(lto, "mean_abs_balance_residual")),
        "mean_interface_slack_mw": float(_safe(lto, "mean_interface_slack_mw")),
        "mean_binding_interface_count": float(_safe(lto, "mean_binding_interface_count")),
        "mean_transfer_magnitude_mw": float(_safe(lto, "mean_transfer_magnitude_mw")),
        "mean_reserve_requirement_mw": float(_safe(lto, "mean_reserve_requirement_mw")),
        "mean_reserve_headroom_mw": float(_safe(lto, "mean_reserve_headroom_mw")),
        "mean_reserve_slack_mw": float(_safe(lto, "mean_reserve_slack_mw")),
        "mean_reserve_requirement_fraction": float(_safe(lto, "mean_reserve_requirement_fraction")),
        "p50_reserve_requirement_fraction": float(_safe(lto, "p50_reserve_requirement_fraction")),
        "mean_runtime_seconds": float(_safe(lto, "mean_runtime_seconds")),
    }
    return row


def build_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    main = pd.DataFrame([load_run_summary(spec) for spec in RUNS])
    appendix = pd.DataFrame([load_run_summary(spec) for spec in APPENDIX_RUNS if (spec["path"] / "zonal_cost_lto_metrics.csv").exists()])
    main.to_csv(TABLE_PATH, index=False)
    appendix.to_csv(APPENDIX_PATH, index=False)
    return main, appendix


def load_monthly_diagnostics() -> pd.DataFrame:
    rows = []
    for spec in [r for r in RUNS if r["stage"] in {"A5b", "A6c", "A7b"}]:
        hourly_path = spec["path"] / "zonal_cost_lto_hourly.csv"
        df = pd.read_csv(hourly_path, parse_dates=["timestamp_utc"])
        df["month"] = df["timestamp_utc"].dt.tz_convert(None).dt.to_period("M").astype(str)
        for fuel in FUEL_KEYS:
            df[f"abs_error_{fuel}"] = (
                df[f"zonal_financial_fy_lto_{fuel}_mw"] - df[f"actual_{fuel}_mw"]
            ).abs()
        df["overall_abs_error"] = df[[f"abs_error_{fuel}" for fuel in FUEL_KEYS]].mean(axis=1)
        df["total_fuel_mix_abs_error"] = (df["zonal_financial_fy_lto_total_mw"] - df["actual_total_mw"]).abs()
        grouped = (
            df.groupby("month", as_index=False)
            .agg(
                mean_financial_regret=("financial_cost_gap_to_oracle", "mean"),
                p95_financial_regret=("financial_cost_gap_to_oracle", lambda x: x.quantile(0.95)),
                overall_p63_mae_mw=("overall_abs_error", "mean"),
                total_fuel_mix_mae_mw=("total_fuel_mix_abs_error", "mean"),
            )
        )
        grouped.insert(0, "stage", spec["stage"])
        grouped.insert(1, "label", spec["label"])
        rows.append(grouped)
    monthly = pd.concat(rows, ignore_index=True)
    monthly.to_csv(MONTHLY_PATH, index=False)
    return monthly


def _bar(ax, df: pd.DataFrame, y: str, title: str, ylabel: str, highlight: str | None = None) -> None:
    colors = ["#4c78a8" if stage != highlight else "#f58518" for stage in df["stage"]]
    ax.bar(df["stage"], df[y], color=colors)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.25)
    ax.tick_params(axis="x", rotation=0)


def save_plots(main: pd.DataFrame, monthly: pd.DataFrame) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid")

    fig, ax = plt.subplots(figsize=(9, 4.8))
    _bar(ax, main, "regret_to_oracle_per_hour", "Mean financial regret by public proxy stage", "$/hour", "A6c")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "01_financial_regret_by_stage.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.bar(main["stage"], main["lto_financial_cost_per_hour"], color="#4c78a8", label="LtO dispatch")
    ax.plot(main["stage"], main["oracle_financial_cost_per_hour"], color="#e45756", marker="o", label="Oracle LP")
    ax.set_title("Financial dispatch cost by public proxy stage")
    ax.set_ylabel("$/hour")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "02_financial_cost_by_stage.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4.8))
    _bar(ax, main, "total_fuel_mix_mae_mw", "Total fuel-mix MAE by public proxy stage", "MW", "A7b")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "03_total_fuel_mix_mae_by_stage.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4.8))
    _bar(ax, main, "overall_p63_mae_mw", "Overall P-63 fuel MAE by public proxy stage", "MW", "A7b")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "04_overall_p63_mae_by_stage.png", dpi=180)
    plt.close(fig)

    monthly_pivot = monthly.pivot(index="month", columns="stage", values="mean_financial_regret")
    fig, ax = plt.subplots(figsize=(12, 5))
    for stage in ["A5b", "A6c"]:
        ax.plot(monthly_pivot.index, monthly_pivot[stage], marker="o", linewidth=1.8, label=stage)
    ax.set_title("Monthly financial regret: no-reserve baseline vs stress-aware proxy")
    ax.set_ylabel("$/hour")
    ax.tick_params(axis="x", rotation=60)
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "05_monthly_regret_a5b_vs_a6c.png", dpi=180)
    plt.close(fig)

    monthly_fuel = monthly.pivot(index="month", columns="stage", values="total_fuel_mix_mae_mw")
    fig, ax = plt.subplots(figsize=(12, 5))
    for stage in ["A5b", "A6c", "A7b"]:
        ax.plot(monthly_fuel.index, monthly_fuel[stage], marker="o", linewidth=1.8, label=stage)
    ax.set_title("Monthly total fuel-mix MAE")
    ax.set_ylabel("MW")
    ax.tick_params(axis="x", rotation=60)
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "06_monthly_total_fuel_mix_mae.png", dpi=180)
    plt.close(fig)

    a7b_diag = RUNS[-1]["path"] / "zonal_cost_lto_lto_diagnostics.csv"
    diag = pd.read_csv(a7b_diag)
    if "reserve_requirement_fraction" in diag.columns:
        fractions = diag["reserve_requirement_fraction"].dropna()
        fig, ax = plt.subplots(figsize=(8, 4.8))
        ax.hist(fractions, bins=40, color="#54a24b", alpha=0.85)
        ax.axvline(fractions.mean(), color="#e45756", linestyle="--", label=f"mean={fractions.mean():.3f}")
        ax.axvline(fractions.median(), color="#f58518", linestyle="--", label=f"median={fractions.median():.3f}")
        ax.set_title("A7b learned reserve requirement fraction")
        ax.set_xlabel("Reserve fraction of solver load")
        ax.set_ylabel("Hours")
        ax.legend()
        fig.tight_layout()
        fig.savefig(FIG_DIR / "07_a7b_learned_reserve_fraction_distribution.png", dpi=180)
        plt.close(fig)


def main() -> None:
    main_table, appendix_table = build_tables()
    monthly = load_monthly_diagnostics()
    save_plots(main_table, monthly)
    print(f"Wrote {TABLE_PATH}")
    print(f"Wrote {APPENDIX_PATH}")
    print(f"Wrote {MONTHLY_PATH}")
    print(f"Wrote figures to {FIG_DIR}")
    print("\nMain table:")
    display_cols = [
        "stage",
        "lto_financial_cost_per_hour",
        "regret_to_oracle_per_hour",
        "overall_p63_mae_mw",
        "total_fuel_mix_mae_mw",
        "success_rate",
    ]
    print(main_table[display_cols].to_string(index=False))


if __name__ == "__main__":
    main()
