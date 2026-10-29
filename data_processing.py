from __future__ import annotations

import re
from calendar import monthrange
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

DATE_PREFIX_RE = re.compile(r"^(\d{8})(.*)$")
DEFAULT_TIMESTAMP_COLUMNS = ("Time Stamp", "Timestamp", "time_stamp", "timestamp")

def _date_from_name(path: str | Path) -> date | None:
    """Return the leading YYYYMMDD date from a file/folder name, if present."""
    match = DATE_PREFIX_RE.match(Path(path).name)
    if not match:
        return None
    try:
        return datetime.strptime(match.group(1), "%Y%m%d").date()
    except ValueError:
        return None

def _label_from_name(path: str | Path) -> str | None:
    """Return the suffix after a leading YYYYMMDD date, if present."""
    match = DATE_PREFIX_RE.match(Path(path).stem)
    if not match:
        return None
    return match.group(2)

def _iter_expected_days(start: date, end: date) -> Iterable[date]:
    current = start
    while current <= end:
        yield current
        current += timedelta(days=1)

def _last_day_of_month(month_start: date) -> date:
    return month_start.replace(day=monthrange(month_start.year, month_start.month)[1])

def _find_timestamp_column(columns: Iterable[str], requested: str | None = None) -> str | None:
    columns = list(columns)
    if requested is not None:
        if requested not in columns:
            raise ValueError(f"Requested timestamp column {requested!r} was not found.")
        return requested

    for candidate in DEFAULT_TIMESTAMP_COLUMNS:
        if candidate in columns:
            return candidate

    lowered = {column.lower(): column for column in columns}
    for key, original in lowered.items():
        if "time" in key and "stamp" in key:
            return original
    return None

def _read_csv(path: Path, encoding: str, **read_csv_kwargs: Any) -> pd.DataFrame:
    try:
        return pd.read_csv(path, encoding=encoding, **read_csv_kwargs)
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="latin1", **read_csv_kwargs)

def list_datatype_csvs(data_path: str | Path, recursive: bool = True) -> list[Path]:
    """
    Return CSV files in one datatype path sorted by date prefix and path.

    `data_path` may be either a directory such as
    `data/raw/P-2A Day-Ahead Market LBMP` or a single CSV file.
    """
    path = Path(data_path).expanduser()
    if path.is_file():
        if path.suffix.lower() != ".csv":
            raise ValueError(f"Expected a CSV file, got {path}")
        return [path]
    if not path.is_dir():
        raise FileNotFoundError(f"No such file or directory: {path}")

    pattern = "**/*.csv" if recursive else "*.csv"
    files = [p for p in path.glob(pattern) if not p.name.startswith(".")]
    return sorted(files, key=lambda p: (_date_from_name(p) or date.min, str(p)))

def summarize_datatype_dir(
    data_dir: str | Path,
    *,
    expected_start: str | date | None = None,
    expected_end: str | date | None = None,
) -> dict[str, Any]:
    """
    Summarize one raw datatype directory and report missing daily CSV dates.

    If `expected_start`/`expected_end` are omitted, the expected range is inferred
    from the first monthly folder through the last day of the last monthly folder.
    """
    data_dir = Path(data_dir).expanduser()
    if not data_dir.is_dir():
        raise NotADirectoryError(f"Expected a datatype directory, got {data_dir}")

    csv_files = list_datatype_csvs(data_dir)
    month_dirs = sorted(
        [p for p in data_dir.iterdir() if p.is_dir()],
        key=lambda p: (_date_from_name(p) or date.min, str(p)),
    )

    file_dates = [_date_from_name(path) for path in csv_files]
    found_dates = {day for day in file_dates if day is not None}
    month_dates = [_date_from_name(path) for path in month_dirs]
    month_dates = [day for day in month_dates if day is not None]

    if expected_start is None:
        expected_start_date = min(month_dates or found_dates) if (month_dates or found_dates) else None
    else:
        expected_start_date = pd.to_datetime(expected_start).date()

    if expected_end is None:
        expected_end_date = _last_day_of_month(max(month_dates)) if month_dates else None
    else:
        expected_end_date = pd.to_datetime(expected_end).date()

    missing_dates: list[str] = []
    if expected_start_date is not None and expected_end_date is not None:
        expected_days = set(_iter_expected_days(expected_start_date, expected_end_date))
        missing_dates = [day.strftime("%Y%m%d") for day in sorted(expected_days - found_dates)]

    return {
        "dataset": data_dir.name,
        "path": str(data_dir),
        "monthly_dirs": len(month_dirs),
        "csv_files": len(csv_files),
        "first_csv_date": min(found_dates).strftime("%Y%m%d") if found_dates else None,
        "last_csv_date": max(found_dates).strftime("%Y%m%d") if found_dates else None,
        "directory_labels": sorted({label for label in (_label_from_name(p.name) for p in month_dirs) if label}),
        "file_labels": sorted({label for label in (_label_from_name(p.name) for p in csv_files) if label}),
        "missing_daily_file_count": len(missing_dates),
        "missing_daily_files": missing_dates,
    }

def read_csv_file(
    csv_path: str | Path,
    *,
    parse_timestamps: bool = True,
    timestamp_column: str | None = None,
    encoding: str = "utf-8-sig",
    **read_csv_kwargs: Any,
) -> pd.DataFrame:
    """Read one standalone CSV file and optionally parse its timestamp column."""
    path = Path(csv_path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"No such CSV file: {path}")

    df = _read_csv(path, encoding=encoding, **read_csv_kwargs)
    if parse_timestamps:
        ts_col = _find_timestamp_column(df.columns, timestamp_column)
        if ts_col is not None:
            df[ts_col] = pd.to_datetime(df[ts_col], errors="coerce")
    return df

def read_datatype_dir(
    data_dir: str | Path,
    *,
    recursive: bool = True,
    parse_timestamps: bool = True,
    timestamp_column: str | None = None,
    add_source_columns: bool = False,
    sort: bool = True,
    encoding: str = "utf-8-sig",
    **read_csv_kwargs: Any,
) -> pd.DataFrame:
    """
    Read all CSVs for one datatype directory into one sequential DataFrame.

    Parameters
    ----------
    data_dir:
        Directory for one datatype, e.g.
        `data/raw/P-2A Day-Ahead Market LBMP`.
    recursive:
        When true, read CSVs from all nested monthly folders.
    parse_timestamps:
        Parse `Time Stamp`, `Timestamp`, or a requested `timestamp_column`.
    timestamp_column:
        Explicit timestamp column to parse and sort by. If omitted, the reader
        infers the common NYISO timestamp column names.
    add_source_columns:
        Add `_source_date`, `_source_file`, and `_source_month_dir` columns.
    sort:
        Sort the collated data by timestamp when available, then by source file
        date and original row order.
    encoding:
        CSV encoding passed to pandas. Falls back to latin1 on decode errors.
    **read_csv_kwargs:
        Additional keyword arguments passed to `pandas.read_csv`.
    """
    data_dir = Path(data_dir).expanduser()
    files = list_datatype_csvs(data_dir, recursive=recursive)
    if not files:
        raise FileNotFoundError(f"No CSV files found under {data_dir}")

    frames: list[pd.DataFrame] = []
    inferred_timestamp_column: str | None = timestamp_column

    for file_order, path in enumerate(files):
        frame = _read_csv(path, encoding=encoding, **read_csv_kwargs)
        file_date = _date_from_name(path)

        if parse_timestamps:
            ts_col = _find_timestamp_column(frame.columns, inferred_timestamp_column)
            if inferred_timestamp_column is None:
                inferred_timestamp_column = ts_col
            if ts_col is not None:
                frame[ts_col] = pd.to_datetime(frame[ts_col], errors="coerce")

        frame["_source_date_for_sort"] = pd.Timestamp(file_date) if file_date else pd.NaT
        frame["_source_file_order_for_sort"] = file_order
        frame["_source_row_order_for_sort"] = range(len(frame))

        if add_source_columns:
            frame["_source_date"] = pd.Timestamp(file_date) if file_date else pd.NaT
            frame["_source_file"] = str(path)
            frame["_source_month_dir"] = path.parent.name

        frames.append(frame)

    data = pd.concat(frames, ignore_index=True)

    if sort:
        sort_columns: list[str] = []
        if inferred_timestamp_column is not None and inferred_timestamp_column in data.columns:
            sort_columns.append(inferred_timestamp_column)
        sort_columns.extend(
            ["_source_date_for_sort", "_source_file_order_for_sort", "_source_row_order_for_sort"]
        )
        data = data.sort_values(sort_columns, kind="mergesort").reset_index(drop=True)

    return data.drop(
        columns=[
            "_source_date_for_sort",
            "_source_file_order_for_sort",
            "_source_row_order_for_sort",
        ]
    )

def read_raw_datatype(
    name_or_path: str | Path,
    *,
    raw_dir: str | Path = "data/raw",
    **kwargs: Any,
) -> pd.DataFrame:
    """
    Convenience wrapper that accepts either a datatype path or a name in raw_dir.

    Examples
    --------
    >>> p2a = read_raw_datatype("P-2A Day-Ahead Market LBMP")
    >>> rt = read_raw_datatype("data/raw/P-24A Real-Time Market LBMP")
    """
    path = Path(name_or_path).expanduser()
    if not path.exists():
        path = Path(raw_dir).expanduser() / str(name_or_path)

    if path.is_file():
        return read_csv_file(path, **kwargs)
    return read_datatype_dir(path, **kwargs)
