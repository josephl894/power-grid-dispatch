from __future__ import annotations

import argparse
import math
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from data_processing import _date_from_name, list_datatype_csvs, summarize_datatype_dir
from nyiso_processing import NY_TZ, ZONE_NAME_TO_CODE


MONTH_NAMES = [
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
]

P27_GEN_USECOLS = [
    "Masked Gen ID",
    "Date Time",
    "Duration",
    "Market",
    "Upper Oper Limit",
    "Emer Oper Limit",
    "Fixed Min Gen MW",
    "Fixed Min Gen Cost",
    "Dispatch MW1",
    "Dispatch MW2",
    "Dispatch MW3",
    "Dispatch MW4",
    "Dispatch MW5",
    "Dispatch MW6",
    "Dispatch MW7",
    "Dispatch MW8",
    "Dispatch MW9",
    "Dispatch MW10",
    "Dispatch MW11",
    "Dispatch MW12",
    "Dispatch $/MW1",
    "Dispatch $/MW2",
    "Dispatch $/MW3",
    "Dispatch $/MW4",
    "Dispatch $/MW5",
    "Dispatch $/MW6",
    "Dispatch $/MW7",
    "Dispatch $/MW8",
    "Dispatch $/MW9",
    "Dispatch $/MW10",
    "Dispatch $/MW11",
    "Dispatch $/MW12",
    "10 Min Non-Synch MW",
    "10 Min Non-Synch Cost",
    "10 Min Spin MW",
    "10 Min Spin Cost",
    "30 Min Non-Synch MW",
    "30 Min Non-Synch Cost",
    "30 Min Spin MW",
    "30 Min Spin Cost",
    "Regulation MW",
    "Regulation Cost",
    "Regulation Movement Cost",
]

P27_UC_NUMERIC_COLS = [
    "Min Run Time",
    "Min Down Time",
    "Max Stops per Day",
    "Start Up Notification",
    "Start Up bid Time 1",
    "Start Up bid Time 2",
    "Start Up bid Time 3",
    "Start Up bid Time 4",
    "Start Up bid Time 5",
    "Start Up bid Time 6",
    "Start Up bid Cost 1",
    "Start Up bid Cost 2",
    "Start Up bid Cost 3",
    "Start Up bid Cost 4",
    "Start Up bid Cost 5",
    "Start Up bid Cost 6",
]


@dataclass(frozen=True)
class CostBuildConfig:
    raw_dir: Path = Path("data/raw")
    processed_dir: Path = Path("data/processed/cost_inputs")
    p27_chunksize: int = 250_000
    max_p27_gen_months: int = 0


def _clean_column(column: object) -> str:
    text = str(column).strip()
    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text)
    return text


def _snake(text: str) -> str:
    text = _clean_column(text).lower()
    text = re.sub(r"[^a-z0-9]+", "_", text).strip("_")
    return text


def _to_numeric(series: pd.Series) -> pd.Series:
    cleaned = series.astype(str).str.strip()
    cleaned = cleaned.mask(cleaned.isin(["", ".", "--", "W"]), np.nan)
    return pd.to_numeric(cleaned, errors="coerce")


def _write_table(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".parquet":
        df.to_parquet(path, index=False)
    else:
        df.to_csv(path, index=False)


def _normalize_zone(value: object) -> str | None:
    if pd.isna(value):
        return None
    text = str(value).strip().upper()
    return ZONE_NAME_TO_CODE.get(text)


def _localize_to_utc(series: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(series, errors="coerce", format="mixed")
    try:
        local = parsed.dt.tz_localize(NY_TZ, ambiguous="infer", nonexistent="shift_forward")
    except Exception:
        local = parsed.dt.tz_localize(NY_TZ, ambiguous="NaT", nonexistent="shift_forward")
    return local.dt.tz_convert("UTC")


def _parse_p27_datetime(series: pd.Series) -> pd.Series:
    text = series.astype(str).str.strip()
    parsed = pd.to_datetime(text, format="%d%b%Y:%H:%M:%S", errors="coerce", utc=True)
    return parsed


def _month_from_file(path: Path) -> pd.Timestamp:
    file_date = _date_from_name(path)
    if file_date is None:
        raise ValueError(f"Cannot infer month from file name: {path}")
    return pd.Timestamp(file_date.replace(day=1), tz="UTC")


def _manifest_row(dataset: str, path: Path, frequency: str) -> dict[str, object]:
    if path.is_dir():
        summary = summarize_datatype_dir(path)
        return {
            "dataset": dataset,
            "path": str(path),
            "frequency": frequency,
            "file_count": summary["csv_files"],
            "first_date": summary["first_csv_date"],
            "last_date": summary["last_csv_date"],
            "missing_count": summary["missing_daily_file_count"],
            "labels": ",".join(summary.get("file_labels", [])),
        }
    return {
        "dataset": dataset,
        "path": str(path),
        "frequency": frequency,
        "file_count": 1 if path.exists() else 0,
        "first_date": None,
        "last_date": None,
        "missing_count": None,
        "labels": path.name,
    }


def _eia923_manifest_row(raw_dir: Path) -> dict[str, object]:
    path = raw_dir / "EIA-923"
    files = _eia923_schedule_files(raw_dir) if path.exists() else []
    years = []
    for file_path in files:
        match = re.search(r"f923_(\d{4})", str(file_path))
        if match:
            years.append(int(match.group(1)))
    return {
        "dataset": "eia923",
        "path": str(path),
        "frequency": "annual_workbooks",
        "file_count": len(files),
        "first_date": f"{min(years)}-01-01" if years else None,
        "last_date": f"{max(years)}-12-31" if years else None,
        "missing_count": None,
        "labels": "EIA923_Schedules_2_3_4_5_M_12_xlsx",
    }


def build_raw_manifest(raw_dir: str | Path = "data/raw") -> pd.DataFrame:
    raw_dir = Path(raw_dir)
    rows = [
        _manifest_row("p26_operator_initiated_commitments", raw_dir / "P-26", "daily"),
        _manifest_row("p27_gen_bids", raw_dir / "P-27-Gen", "monthly"),
        _manifest_row("p27_uc_data", raw_dir / "P-27-UC", "monthly"),
        _eia923_manifest_row(raw_dir),
        _manifest_row("atb_electricity", raw_dir / "ATBe.csv", "release_file"),
        _manifest_row(
            "eia_fossil_natural_gas_monthly",
            raw_dir / "Average_cost_of_fossil_fuels_for_electricity_generation_(per_Btu)_for_natural_gas_monthly.csv",
            "monthly",
        ),
        _manifest_row(
            "eia_fossil_petroleum_liquids_monthly",
            raw_dir / "Average_cost_of_fossil_fuels_for_electricity_generation_(per_Btu)_for_petroleum_liquids_monthly.csv",
            "monthly",
        ),
    ]
    for name in ["P-2B", "P-24B", "P-4A", "P-4B"]:
        path = raw_dir / name
        files = list(path.glob("**/*")) if path.exists() else []
        rows.append(
            {
                "dataset": name.lower(),
                "path": str(path),
                "frequency": "nyiso_market_csv",
                "file_count": len([p for p in files if p.is_file()]),
                "first_date": None,
                "last_date": None,
                "missing_count": None,
                "labels": "empty" if not any(p.is_file() for p in files) else "",
            }
        )
    return pd.DataFrame(rows)


def load_eia_fossil_monthly_prices(raw_dir: str | Path = "data/raw") -> pd.DataFrame:
    raw_dir = Path(raw_dir)
    specs = [
        (
            "natural_gas",
            raw_dir / "Average_cost_of_fossil_fuels_for_electricity_generation_(per_Btu)_for_natural_gas_monthly.csv",
        ),
        (
            "petroleum_liquids",
            raw_dir / "Average_cost_of_fossil_fuels_for_electricity_generation_(per_Btu)_for_petroleum_liquids_monthly.csv",
        ),
    ]
    frames = []
    for fuel, path in specs:
        df = pd.read_csv(path, skiprows=4)
        if df.shape[1] != 2:
            raise ValueError(f"Expected two columns in {path}, got {df.columns.tolist()}")
        df.columns = ["month_label", "price_dollars_per_mmbtu"]
        df["month"] = pd.to_datetime(df["month_label"], format="%b %Y", errors="coerce").dt.to_period("M").dt.to_timestamp()
        df["fuel_price_source"] = "eia_electric_power_monthly"
        df["fuel"] = fuel
        df["price_dollars_per_mmbtu"] = pd.to_numeric(df["price_dollars_per_mmbtu"], errors="coerce")
        df["price_missing"] = df["price_dollars_per_mmbtu"].isna()
        df["price_zero_or_negative"] = df["price_dollars_per_mmbtu"].fillna(1.0) <= 0.0
        frames.append(
            df[
                [
                    "month",
                    "fuel",
                    "price_dollars_per_mmbtu",
                    "price_missing",
                    "price_zero_or_negative",
                    "fuel_price_source",
                ]
            ]
        )
    out = pd.concat(frames, ignore_index=True)
    return out.sort_values(["fuel", "month"]).reset_index(drop=True)


def _find_excel_header_row(path: Path, sheet_name: str, required: set[str], max_rows: int = 12) -> int:
    preview = pd.read_excel(path, sheet_name=sheet_name, header=None, nrows=max_rows)
    for idx, row in preview.iterrows():
        values = {_clean_column(v) for v in row.dropna().tolist()}
        if required.issubset(values):
            return int(idx)
    raise ValueError(f"Could not find header row in {path} sheet {sheet_name} for {required}")


def _read_eia923_sheet(path: Path, sheet_name: str, required: set[str]) -> pd.DataFrame:
    header = _find_excel_header_row(path, sheet_name, required)
    df = pd.read_excel(path, sheet_name=sheet_name, header=header)
    df.columns = [_clean_column(c) for c in df.columns]
    return df


def _eia923_schedule_files(raw_dir: Path) -> list[Path]:
    return sorted((raw_dir / "EIA-923").glob("f923_*/EIA923_Schedules_2_3_4_5_M_12*.xlsx"))


def load_eia923_fuel_receipts(raw_dir: str | Path = "data/raw") -> pd.DataFrame:
    raw_dir = Path(raw_dir)
    frames = []
    for path in _eia923_schedule_files(raw_dir):
        df = _read_eia923_sheet(path, "Page 5 Fuel Receipts and Costs", {"YEAR", "MONTH", "Plant State"})
        keep = [
            "YEAR",
            "MONTH",
            "Plant Id",
            "Plant Name",
            "Plant State",
            "ENERGY_SOURCE",
            "FUEL_GROUP",
            "QUANTITY",
            "Average Heat Content",
            "FUEL_COST",
            "Regulated",
            "Balancing Authority Code",
        ]
        available = [c for c in keep if c in df.columns]
        df = df[available].copy()
        df.columns = [_snake(c) for c in df.columns]
        if "plant_state" not in df.columns:
            continue
        df = df[df["plant_state"].astype(str).str.strip().eq("NY")].copy()
        if df.empty:
            continue
        df["quantity"] = _to_numeric(df["quantity"])
        df["average_heat_content"] = _to_numeric(df.get("average_heat_content", pd.Series(np.nan, index=df.index)))
        df["fuel_cost_cents_per_mmbtu"] = _to_numeric(df.get("fuel_cost", pd.Series(np.nan, index=df.index)))
        df["fuel_cost_dollars_per_mmbtu"] = df["fuel_cost_cents_per_mmbtu"] / 100.0
        df["receipt_mmbtu"] = df["quantity"] * df["average_heat_content"]
        df["source_file"] = path.name
        df = df.drop(columns=[c for c in ["fuel_cost"] if c in df.columns])
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    out["month"] = pd.to_datetime(
        out["year"].astype("Int64").astype(str) + "-" + out["month"].astype("Int64").astype(str) + "-01",
        errors="coerce",
    )
    return out.sort_values(["month", "fuel_group", "energy_source", "plant_id"]).reset_index(drop=True)


def aggregate_eia923_fuel_costs(receipts: pd.DataFrame) -> pd.DataFrame:
    if receipts.empty:
        return pd.DataFrame()
    df = receipts.copy()
    df = df[df["fuel_cost_dollars_per_mmbtu"].notna() & (df["fuel_cost_dollars_per_mmbtu"] > 0)].copy()
    df["weight"] = df["receipt_mmbtu"].where(df["receipt_mmbtu"] > 0, df["quantity"])
    df["weight"] = df["weight"].where(df["weight"] > 0, 1.0)

    def weighted(group: pd.DataFrame) -> pd.Series:
        weights = group["weight"].to_numpy(dtype=np.float64)
        costs = group["fuel_cost_dollars_per_mmbtu"].to_numpy(dtype=np.float64)
        return pd.Series(
            {
                "eia923_receipt_rows": len(group),
                "eia923_total_receipt_mmbtu": float(np.nansum(group["receipt_mmbtu"])),
                "eia923_weighted_cost_dollars_per_mmbtu": float(np.average(costs, weights=weights)),
                "eia923_median_cost_dollars_per_mmbtu": float(np.nanmedian(costs)),
            }
        )

    return (
        df.groupby(["month", "fuel_group"], dropna=False)
        .apply(weighted, include_groups=False)
        .reset_index()
        .sort_values(["fuel_group", "month"])
    )


def load_eia923_heat_rates(raw_dir: str | Path = "data/raw") -> pd.DataFrame:
    raw_dir = Path(raw_dir)
    rows = []
    for path in _eia923_schedule_files(raw_dir):
        df = _read_eia923_sheet(path, "Page 1 Generation and Fuel Data", {"Plant Id", "Plant State"})
        if "Plant State" not in df.columns:
            continue
        df = df[df["Plant State"].astype(str).str.strip().eq("NY")].copy()
        if df.empty:
            continue
        for month_idx, month_name in enumerate(MONTH_NAMES, start=1):
            mmbtu_col = f"Elec_MMBtu {month_name}"
            netgen_col = f"Netgen {month_name}"
            if mmbtu_col not in df.columns or netgen_col not in df.columns:
                continue
            tmp = pd.DataFrame(
                {
                    "year": _to_numeric(df["YEAR"]).astype("Int64"),
                    "month_number": month_idx,
                    "reported_prime_mover": df.get("Reported Prime Mover"),
                    "reported_fuel_type_code": df.get("Reported Fuel Type Code"),
                    "mer_fuel_type_code": df.get("MER Fuel Type Code"),
                    "elec_mmbtu": _to_numeric(df[mmbtu_col]),
                    "net_generation_mwh": _to_numeric(df[netgen_col]),
                }
            )
            rows.append(tmp)
    if not rows:
        return pd.DataFrame()
    data = pd.concat(rows, ignore_index=True)
    data = data[(data["net_generation_mwh"] > 0) & (data["elec_mmbtu"] > 0)].copy()
    data["month"] = pd.to_datetime(
        data["year"].astype(str) + "-" + data["month_number"].astype(str) + "-01",
        errors="coerce",
    )
    grouped = (
        data.groupby(["month", "reported_prime_mover", "reported_fuel_type_code", "mer_fuel_type_code"], dropna=False)
        .agg(
            elec_mmbtu=("elec_mmbtu", "sum"),
            net_generation_mwh=("net_generation_mwh", "sum"),
            eia923_heat_rate_mmbtu_per_mwh=("elec_mmbtu", lambda s: np.nan),
            rows=("elec_mmbtu", "size"),
        )
        .reset_index()
    )
    grouped["eia923_heat_rate_mmbtu_per_mwh"] = grouped["elec_mmbtu"] / grouped["net_generation_mwh"]
    return grouped.sort_values(["month", "reported_fuel_type_code", "reported_prime_mover"]).reset_index(drop=True)


def load_atb_dispatch_assumptions(
    raw_dir: str | Path = "data/raw",
    *,
    scenario: str = "Moderate",
    target_year: int = 2026,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw_dir = Path(raw_dir)
    cols = [
        "atb_year",
        "technology",
        "techdetail",
        "display_name",
        "scenario",
        "core_metric_parameter",
        "core_metric_variable",
        "value",
    ]
    data = pd.read_csv(raw_dir / "ATBe.csv", usecols=cols, low_memory=False)
    data = data[data["core_metric_parameter"].isin(["Variable O&M", "Heat Rate", "Fuel"])].copy()
    data["core_metric_variable"] = pd.to_numeric(data["core_metric_variable"], errors="coerce")
    data = data[data["scenario"].eq(scenario)].copy()

    selectors = [
        ("natural_gas_combined_cycle", "Natural Gas", "CC", "NaturalGas_FE", "NG 2-on-1 Combined Cycle (F-Frame)"),
        ("natural_gas_combustion_turbine", "Natural Gas", "GT", "NaturalGas_FE", "NG Combustion Turbine (F-Frame)"),
        ("nuclear_large", "Nuclear", "NB/NP", "Nuclear", "Nuclear - Large"),
        ("biopower_dedicated", "Other Renewables", "CG/ST", "Biopower", "Dedicated"),
        ("pumped_storage_hydro", "Hydro", "PS", "Pumped Storage Hydropower", "NatlClass1"),
    ]
    rows = []
    for assumption_name, fuel_category, unit_type_proxy, technology, techdetail_contains in selectors:
        subset = data[
            data["technology"].eq(technology)
            & data["techdetail"].astype(str).str.contains(re.escape(techdetail_contains), case=False, na=False)
        ].copy()
        if subset.empty:
            continue
        for parameter in ["Variable O&M", "Heat Rate", "Fuel"]:
            candidates = subset[subset["core_metric_parameter"].eq(parameter)].copy()
            if candidates.empty:
                continue
            candidates["year_distance"] = (candidates["core_metric_variable"] - target_year).abs()
            chosen = candidates.sort_values(["year_distance", "core_metric_variable"]).iloc[0]
            rows.append(
                {
                    "assumption_name": assumption_name,
                    "fuel_category": fuel_category,
                    "unit_type_proxy": unit_type_proxy,
                    "technology": chosen["technology"],
                    "techdetail": chosen["techdetail"],
                    "scenario": scenario,
                    "requested_metric_year": target_year,
                    "selected_metric_year": int(chosen["core_metric_variable"]),
                    "core_metric_parameter": parameter,
                    "value": float(chosen["value"]),
                    "source": "ATBe.csv",
                }
            )
    curated = pd.DataFrame(rows)
    return data.sort_values(["technology", "techdetail", "core_metric_parameter", "core_metric_variable"]), curated


def load_p26_operator_commitments(raw_dir: str | Path = "data/raw") -> tuple[pd.DataFrame, pd.DataFrame]:
    files = list_datatype_csvs(Path(raw_dir) / "P-26")
    frames = []
    for path in files:
        frame = pd.read_csv(path, encoding="utf-8-sig")
        frame.columns = [_clean_column(c) for c in frame.columns]
        frame["source_file"] = path.name
        frames.append(frame)
    if not frames:
        return pd.DataFrame(), pd.DataFrame()
    data = pd.concat(frames, ignore_index=True)
    data["insert_timestamp_utc"] = _localize_to_utc(data["Insert Time"])
    data["event_start_utc"] = _localize_to_utc(data["Event Start Time"])
    data["event_end_utc"] = _localize_to_utc(data["Event End Time"])
    data["ptid"] = _to_numeric(data["PTID"]).astype("Int64")
    data["zone_code"] = data["Load Zone of Resource"].map(_normalize_zone)
    data["commitment_type"] = data["Commitment Type"].astype(str).str.strip()
    data["commitment_reason"] = data["Commitment Reason"].astype(str).str.strip()
    data["uol_mw"] = _to_numeric(data["UOL (of committed resource)"])
    data["mw_committed"] = _to_numeric(data["MW Committed"])
    data["duration_hours"] = (data["event_end_utc"] - data["event_start_utc"]).dt.total_seconds() / 3600.0
    data["valid_event"] = (
        data["event_start_utc"].notna()
        & data["event_end_utc"].notna()
        & (data["event_end_utc"] > data["event_start_utc"])
        & data["zone_code"].notna()
    )

    expanded_rows = []
    valid = data[data["valid_event"]].copy()
    for idx, row in valid.iterrows():
        start = row["event_start_utc"].floor("h")
        last = (row["event_end_utc"] - pd.Timedelta(nanoseconds=1)).floor("h")
        if pd.isna(start) or pd.isna(last) or last < start:
            continue
        for ts in pd.date_range(start, last, freq="h"):
            expanded_rows.append(
                {
                    "timestamp_utc": ts,
                    "zone_code": row["zone_code"],
                    "p26_event_row": int(idx),
                    "ptid": row["ptid"],
                    "commitment_type": row["commitment_type"],
                    "commitment_reason": row["commitment_reason"],
                    "uol_mw": row["uol_mw"],
                    "mw_committed": row["mw_committed"],
                }
            )
    if expanded_rows:
        expanded = pd.DataFrame(expanded_rows)
        expanded["is_daru"] = expanded["commitment_type"].astype(str).str.upper().str.contains("DARU", na=False)
        expanded["is_oom"] = expanded["commitment_type"].astype(str).str.upper().str.contains("OOM", na=False)
        expanded["is_constraint_management"] = expanded["commitment_reason"].astype(str).str.upper().str.contains(
            "CONSTRAINT", na=False
        )
        hourly = (
            expanded.groupby(["timestamp_utc", "zone_code"], as_index=False)
            .agg(
                p26_commitment_count=("p26_event_row", "nunique"),
                p26_resource_count=("ptid", "nunique"),
                p26_committed_mw=("mw_committed", "sum"),
                p26_uol_mw=("uol_mw", "sum"),
                p26_daru_count=("is_daru", "sum"),
                p26_oom_count=("is_oom", "sum"),
                p26_constraint_commit_count=("is_constraint_management", "sum"),
            )
            .sort_values(["timestamp_utc", "zone_code"])
        )
        hourly["p26_available"] = True
    else:
        hourly = pd.DataFrame()
    return data, hourly


def _combine_chunk_groups(chunks: list[pd.DataFrame]) -> pd.DataFrame:
    if not chunks:
        return pd.DataFrame()
    data = pd.concat(chunks, ignore_index=True)
    sum_cols = [c for c in data.columns if c.endswith("_sum") or c in {"row_count", "duration_hours"}]
    max_cols = [c for c in data.columns if c.endswith("_max")]
    min_cols = [c for c in data.columns if c.endswith("_min")]
    weighted_mean_specs = [
        ("uol_mean", "uol_sum"),
        ("emergency_oper_limit_mean", "emergency_oper_limit_sum"),
        ("fixed_min_gen_mw_mean", "fixed_min_gen_mw_sum"),
        ("fixed_min_gen_cost_mean", "fixed_min_gen_cost_sum"),
        ("dispatch_price_mean", "dispatch_price_mean_sum"),
        ("dispatch_price_curve_max_mean", "dispatch_price_curve_max_sum"),
        ("dispatch_price_curve_min_mean", "dispatch_price_curve_min_sum"),
        ("dispatch_mw_curve_max_mean", "dispatch_mw_curve_max_sum"),
        ("reserve_offer_mw_mean", "reserve_offer_mw_sum"),
        ("reserve_offer_cost_mean", "reserve_offer_cost_sum"),
        ("regulation_mw_mean", "regulation_mw_sum"),
        ("regulation_cost_mean", "regulation_cost_sum"),
    ]
    grouped = data.groupby(["timestamp_utc", "market"], as_index=False)
    base = grouped[sum_cols].sum()
    for col in max_cols:
        base[col] = grouped[col].max()[col]
    for col in min_cols:
        base[col] = grouped[col].min()[col]
    counts = base["row_count"].replace(0, np.nan)
    for out_col, sum_col in weighted_mean_specs:
        if sum_col in base.columns:
            base[out_col] = base[sum_col] / counts
    return base.sort_values(["timestamp_utc", "market"]).reset_index(drop=True)


def load_p27_gen_bid_hour(
    raw_dir: str | Path = "data/raw",
    *,
    chunksize: int = 250_000,
    max_months: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    files = sorted((Path(raw_dir) / "P-27-Gen").glob("*.csv"))
    if max_months > 0:
        files = files[:max_months]
    chunk_groups: list[pd.DataFrame] = []
    file_rows = []
    dispatch_price_cols = [f"Dispatch $/MW{i}" for i in range(1, 13)]
    dispatch_mw_cols = [f"Dispatch MW{i}" for i in range(1, 13)]
    reserve_mw_cols = ["10 Min Non-Synch MW", "10 Min Spin MW", "30 Min Non-Synch MW", "30 Min Spin MW"]
    reserve_cost_cols = ["10 Min Non-Synch Cost", "10 Min Spin Cost", "30 Min Non-Synch Cost", "30 Min Spin Cost"]
    for path in files:
        file_row_count = 0
        file_bad_timestamps = 0
        for chunk in pd.read_csv(path, usecols=lambda c: c in P27_GEN_USECOLS, chunksize=chunksize, low_memory=False):
            chunk.columns = [_clean_column(c) for c in chunk.columns]
            file_row_count += len(chunk)
            chunk["timestamp_utc"] = _parse_p27_datetime(chunk["Date Time"])
            file_bad_timestamps += int(chunk["timestamp_utc"].isna().sum())
            chunk["market"] = chunk["Market"].astype(str).str.strip()
            numeric_cols = [
                "Duration",
                "Upper Oper Limit",
                "Emer Oper Limit",
                "Fixed Min Gen MW",
                "Fixed Min Gen Cost",
                *dispatch_price_cols,
                *dispatch_mw_cols,
                *reserve_mw_cols,
                *reserve_cost_cols,
                "Regulation MW",
                "Regulation Cost",
                "Regulation Movement Cost",
            ]
            for col in numeric_cols:
                if col in chunk.columns:
                    chunk[col] = _to_numeric(chunk[col])
            price_frame = chunk[[c for c in dispatch_price_cols if c in chunk.columns]]
            mw_frame = chunk[[c for c in dispatch_mw_cols if c in chunk.columns]]
            reserve_mw_frame = chunk[[c for c in reserve_mw_cols if c in chunk.columns]]
            reserve_cost_frame = chunk[[c for c in reserve_cost_cols if c in chunk.columns]]
            chunk["dispatch_price_mean_row"] = price_frame.mean(axis=1, skipna=True)
            chunk["dispatch_price_max_row"] = price_frame.max(axis=1, skipna=True)
            chunk["dispatch_price_min_row"] = price_frame.min(axis=1, skipna=True)
            chunk["dispatch_mw_max_row"] = mw_frame.max(axis=1, skipna=True)
            chunk["reserve_offer_mw_row"] = reserve_mw_frame.sum(axis=1, skipna=True)
            chunk["reserve_offer_cost_mean_row"] = reserve_cost_frame.mean(axis=1, skipna=True)
            grouped = (
                chunk.dropna(subset=["timestamp_utc", "market"])
                .groupby(["timestamp_utc", "market"], as_index=False)
                .agg(
                    row_count=("Masked Gen ID", "size"),
                    duration_hours=("Duration", "sum"),
                    uol_sum=("Upper Oper Limit", "sum"),
                    emergency_oper_limit_sum=("Emer Oper Limit", "sum"),
                    fixed_min_gen_mw_sum=("Fixed Min Gen MW", "sum"),
                    fixed_min_gen_cost_sum=("Fixed Min Gen Cost", "sum"),
                    dispatch_price_mean_sum=("dispatch_price_mean_row", "sum"),
                    dispatch_price_curve_max_sum=("dispatch_price_max_row", "sum"),
                    dispatch_price_curve_min_sum=("dispatch_price_min_row", "sum"),
                    dispatch_price_curve_max_max=("dispatch_price_max_row", "max"),
                    dispatch_price_curve_min_min=("dispatch_price_min_row", "min"),
                    dispatch_mw_curve_max_sum=("dispatch_mw_max_row", "sum"),
                    dispatch_mw_curve_max_max=("dispatch_mw_max_row", "max"),
                    reserve_offer_mw_sum=("reserve_offer_mw_row", "sum"),
                    reserve_offer_cost_sum=("reserve_offer_cost_mean_row", "sum"),
                    regulation_mw_sum=("Regulation MW", "sum"),
                    regulation_cost_sum=("Regulation Cost", "sum"),
                )
            )
            chunk_groups.append(grouped)
        file_rows.append(
            {
                "source_file": path.name,
                "month": _month_from_file(path).date().isoformat(),
                "row_count": file_row_count,
                "bad_timestamp_count": file_bad_timestamps,
            }
        )
    return _combine_chunk_groups(chunk_groups), pd.DataFrame(file_rows)


def load_p27_uc_month(raw_dir: str | Path = "data/raw") -> pd.DataFrame:
    files = sorted((Path(raw_dir) / "P-27-UC").glob("*.csv"))
    rows = []
    for path in files:
        df = pd.read_csv(path, low_memory=False)
        df.columns = [_clean_column(c) for c in df.columns]
        for col in P27_UC_NUMERIC_COLS:
            if col in df.columns:
                df[col] = _to_numeric(df[col])
        startup_cost_cols = [c for c in [f"Start Up bid Cost {i}" for i in range(1, 7)] if c in df.columns]
        startup_time_cols = [c for c in [f"Start Up bid Time {i}" for i in range(1, 7)] if c in df.columns]
        startup_cost = df[startup_cost_cols].max(axis=1, skipna=True) if startup_cost_cols else pd.Series(np.nan, index=df.index)
        startup_time = df[startup_time_cols].max(axis=1, skipna=True) if startup_time_cols else pd.Series(np.nan, index=df.index)
        month = _month_from_file(path)
        rows.append(
            {
                "month": month.tz_convert(None),
                "p27_uc_resource_rows": len(df),
                "p27_uc_masked_gen_count": int(df["Masked Gen ID"].nunique()) if "Masked Gen ID" in df.columns else np.nan,
                "min_run_time_median": float(df["Min Run Time"].median()) if "Min Run Time" in df.columns else np.nan,
                "min_run_time_p90": float(df["Min Run Time"].quantile(0.90)) if "Min Run Time" in df.columns else np.nan,
                "min_down_time_median": float(df["Min Down Time"].median()) if "Min Down Time" in df.columns else np.nan,
                "min_down_time_p90": float(df["Min Down Time"].quantile(0.90)) if "Min Down Time" in df.columns else np.nan,
                "max_stops_per_day_median": float(df["Max Stops per Day"].median()) if "Max Stops per Day" in df.columns else np.nan,
                "startup_cost_median": float(startup_cost.median()),
                "startup_cost_p90": float(startup_cost.quantile(0.90)),
                "startup_time_median": float(startup_time.median()),
                "startup_time_p90": float(startup_time.quantile(0.90)),
                "source_file": path.name,
            }
        )
    return pd.DataFrame(rows).sort_values("month").reset_index(drop=True)


def _check(status: bool, name: str, value: object, detail: str = "") -> dict[str, object]:
    return {"check": name, "status": "PASS" if status else "WARN", "value": value, "detail": detail}


def validate_cost_inputs(outputs: dict[str, pd.DataFrame], manifest: pd.DataFrame) -> pd.DataFrame:
    rows = []
    manifest_lookup = manifest.set_index("dataset")
    rows.append(_check(True, "manifest_rows", len(manifest), "Raw-source manifest was created."))
    for dataset in ["p26_operator_initiated_commitments", "p27_gen_bids", "p27_uc_data"]:
        if dataset in manifest_lookup.index:
            row = manifest_lookup.loc[dataset]
            rows.append(_check(int(row["file_count"]) > 0, f"{dataset}_files_present", row["file_count"], str(row.to_dict())))

    p26 = outputs.get("p26_zone_hour", pd.DataFrame())
    rows.append(_check(not p26.empty, "p26_zone_hour_nonempty", len(p26)))
    if not p26.empty:
        rows.append(_check(p26.duplicated(["timestamp_utc", "zone_code"]).sum() == 0, "p26_no_duplicate_zone_hours", int(p26.duplicated(["timestamp_utc", "zone_code"]).sum())))
        rows.append(_check((p26["p26_committed_mw"].fillna(0) >= 0).all(), "p26_nonnegative_committed_mw", float(p26["p26_committed_mw"].min())))

    p27 = outputs.get("p27_gen_bid_hour", pd.DataFrame())
    rows.append(_check(not p27.empty, "p27_gen_bid_hour_nonempty", len(p27)))
    if not p27.empty:
        rows.append(_check(p27.duplicated(["timestamp_utc", "market"]).sum() == 0, "p27_gen_no_duplicate_market_hours", int(p27.duplicated(["timestamp_utc", "market"]).sum())))
        rows.append(_check((p27["uol_sum"].fillna(0) >= 0).all(), "p27_gen_nonnegative_uol", float(p27["uol_sum"].min())))

    uc = outputs.get("p27_uc_month", pd.DataFrame())
    rows.append(_check(not uc.empty, "p27_uc_month_nonempty", len(uc)))
    if not uc.empty:
        rows.append(_check(uc.duplicated(["month"]).sum() == 0, "p27_uc_no_duplicate_months", int(uc.duplicated(["month"]).sum())))

    fossil = outputs.get("eia_fossil_monthly_prices", pd.DataFrame())
    rows.append(_check(not fossil.empty, "eia_fossil_prices_nonempty", len(fossil)))
    if not fossil.empty:
        gas = fossil[fossil["fuel"].eq("natural_gas")]
        oil = fossil[fossil["fuel"].eq("petroleum_liquids")]
        rows.append(_check(gas["price_dollars_per_mmbtu"].notna().sum() >= 100, "eia_gas_price_coverage", int(gas["price_dollars_per_mmbtu"].notna().sum())))
        rows.append(_check(oil["price_dollars_per_mmbtu"].notna().sum() >= 40, "eia_petroleum_price_coverage_sparse_but_usable", int(oil["price_dollars_per_mmbtu"].notna().sum())))

    eia923 = outputs.get("eia923_monthly_fuel_cost", pd.DataFrame())
    rows.append(_check(not eia923.empty, "eia923_monthly_fuel_cost_nonempty", len(eia923)))
    if not eia923.empty:
        rows.append(_check((eia923["eia923_weighted_cost_dollars_per_mmbtu"] > 0).all(), "eia923_positive_weighted_cost", float(eia923["eia923_weighted_cost_dollars_per_mmbtu"].min())))

    atb = outputs.get("atb_curated_dispatch_assumptions", pd.DataFrame())
    rows.append(_check(not atb.empty, "atb_curated_nonempty", len(atb)))
    if not atb.empty:
        needed = {"natural_gas_combined_cycle", "natural_gas_combustion_turbine"}
        present = set(atb["assumption_name"].dropna())
        rows.append(_check(needed.issubset(present), "atb_has_gas_ct_and_cc", ",".join(sorted(present))))
        pivot = atb.pivot_table(index="assumption_name", columns="core_metric_parameter", values="value", aggfunc="first")
        if {"natural_gas_combined_cycle", "natural_gas_combustion_turbine"}.issubset(pivot.index) and "Heat Rate" in pivot.columns:
            cc = pivot.loc["natural_gas_combined_cycle", "Heat Rate"]
            ct = pivot.loc["natural_gas_combustion_turbine", "Heat Rate"]
            rows.append(_check(cc < ct, "atb_cc_heat_rate_lower_than_ct", f"cc={cc}, ct={ct}"))

    heat = outputs.get("eia923_heat_rates", pd.DataFrame())
    rows.append(_check(not heat.empty, "eia923_heat_rates_nonempty", len(heat)))
    if not heat.empty:
        plausible = heat["eia923_heat_rate_mmbtu_per_mwh"].between(3, 30)
        rows.append(_check(plausible.mean() > 0.90, "eia923_heat_rate_mostly_plausible", float(plausible.mean())))

    if not fossil.empty and not eia923.empty:
        gas_monthly = fossil[fossil["fuel"].eq("natural_gas")][["month", "price_dollars_per_mmbtu"]].copy()
        eia_gas = eia923[eia923["fuel_group"].eq("Natural Gas")][
            ["month", "eia923_weighted_cost_dollars_per_mmbtu"]
        ]
        merged = gas_monthly.merge(eia_gas, on="month", how="inner")
        if not merged.empty:
            diff = (merged["price_dollars_per_mmbtu"] - merged["eia923_weighted_cost_dollars_per_mmbtu"]).abs()
            rows.append(_check(diff.median() < 2.0, "eia_gas_monthly_reconciles_with_eia923_median", float(diff.median())))

    return pd.DataFrame(rows)


def build_cost_input_datasets(config: CostBuildConfig) -> tuple[dict[str, pd.DataFrame], pd.DataFrame, pd.DataFrame]:
    raw_dir = Path(config.raw_dir)
    processed_dir = Path(config.processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)

    manifest = build_raw_manifest(raw_dir)
    outputs: dict[str, pd.DataFrame] = {}

    outputs["eia_fossil_monthly_prices"] = load_eia_fossil_monthly_prices(raw_dir)
    receipts = load_eia923_fuel_receipts(raw_dir)
    outputs["eia923_fuel_receipts_ny"] = receipts
    outputs["eia923_monthly_fuel_cost"] = aggregate_eia923_fuel_costs(receipts)
    outputs["eia923_heat_rates"] = load_eia923_heat_rates(raw_dir)
    atb_relevant, atb_curated = load_atb_dispatch_assumptions(raw_dir)
    outputs["atb_relevant_cost_rows"] = atb_relevant
    outputs["atb_curated_dispatch_assumptions"] = atb_curated
    _, p26_hourly = load_p26_operator_commitments(raw_dir)
    outputs["p26_zone_hour"] = p26_hourly
    p27_gen_hour, p27_gen_file_summary = load_p27_gen_bid_hour(
        raw_dir,
        chunksize=config.p27_chunksize,
        max_months=config.max_p27_gen_months,
    )
    outputs["p27_gen_bid_hour"] = p27_gen_hour
    outputs["p27_gen_file_summary"] = p27_gen_file_summary
    outputs["p27_uc_month"] = load_p27_uc_month(raw_dir)

    validation = validate_cost_inputs(outputs, manifest)

    _write_table(manifest, processed_dir / "validation_manifest.csv")
    _write_table(validation, processed_dir / "validation_checks.csv")
    for name, frame in outputs.items():
        suffix = ".parquet" if name not in {"p27_gen_file_summary"} else ".csv"
        _write_table(frame, processed_dir / f"{name}{suffix}")
    return outputs, manifest, validation
