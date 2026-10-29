from __future__ import annotations

import argparse
import logging
import time
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd
import requests


API_URL = "https://power.larc.nasa.gov/api/temporal/hourly/point"
DEFAULT_START = "20160101"
DEFAULT_END = "20260301"
DEFAULT_OUTPUT_DIR = Path("data")
REQUEST_DELAY_SECONDS = 0.5
REQUEST_TIMEOUT_SECONDS = 60
MAX_RETRIES = 3

PARAMETERS = [
    "ALLSKY_SFC_SW_DWN",
    "WS50M",
    "WD50M",
    "T2M",
    "PRECTOT",
]

ZONES = {
    "A": (42.692316, -78.560877),  # WEST
    "B": (42.654772, -77.641387),  # GENESE
    "C": (42.650333, -76.500678),  # CENTRL
    "D": (44.425418, -73.870140),  # NORTH
    "E": (43.499764, -74.825769),  # MHK VL
    "F": (43.103300, -73.877211),  # CAPITL
    "G": (41.636849, -74.251350),  # HUD VL
    "H": (41.185668, -73.797881),  # MILLWD
    "I": (40.977470, -73.783271),  # DUNWOD
    "J": (40.682526, -73.904697),  # N.Y.C.
    "K": (40.858629, -72.996361),  # LONGIL
}


def year_chunks(start_yyyymmdd: str, end_yyyymmdd: str) -> list[tuple[int, str, str]]:
    """Split an inclusive YYYYMMDD date range into one inclusive chunk per year."""
    start_date = pd.to_datetime(start_yyyymmdd, format="%Y%m%d").date()
    end_date = pd.to_datetime(end_yyyymmdd, format="%Y%m%d").date()
    if start_date > end_date:
        raise ValueError("start date must be on or before end date")

    chunks: list[tuple[int, str, str]] = []
    for year in range(start_date.year, end_date.year + 1):
        chunk_start = max(start_date, date(year, 1, 1))
        chunk_end = min(end_date, date(year, 12, 31))
        chunks.append(
            (
                year,
                chunk_start.strftime("%Y%m%d"),
                chunk_end.strftime("%Y%m%d"),
            )
        )
    return chunks


def build_request_params(
    latitude: float,
    longitude: float,
    start_yyyymmdd: str,
    end_yyyymmdd: str,
) -> dict[str, str | float]:
    """Build NASA POWER request parameters for one point and date chunk."""
    return {
        "parameters": ",".join(PARAMETERS),
        "community": "RE",
        "longitude": longitude,
        "latitude": latitude,
        "start": start_yyyymmdd,
        "end": end_yyyymmdd,
        "format": "JSON",
        "time-standard": "UTC",
    }


def request_json_with_retries(
    session: requests.Session,
    params: dict[str, str | float],
    *,
    max_retries: int = MAX_RETRIES,
) -> dict[str, Any] | None:
    """Return API JSON, retrying transient request failures before skipping."""
    for attempt in range(1, max_retries + 1):
        try:
            response = session.get(API_URL, params=params, timeout=REQUEST_TIMEOUT_SECONDS)
            response.raise_for_status()
            payload = response.json()
            if "messages" in payload:
                logging.warning("NASA POWER message: %s", payload["messages"])
            return payload
        except requests.RequestException as exc:
            logging.warning(
                "Request failed on attempt %s/%s for lat=%s lon=%s start=%s end=%s: %s",
                attempt,
                max_retries,
                params.get("latitude"),
                params.get("longitude"),
                params.get("start"),
                params.get("end"),
                exc,
            )
        except ValueError as exc:
            logging.warning(
                "Could not parse JSON for lat=%s lon=%s start=%s end=%s: %s",
                params.get("latitude"),
                params.get("longitude"),
                params.get("start"),
                params.get("end"),
                exc,
            )
            return None

        if attempt < max_retries:
            time.sleep(2**attempt)

    logging.warning(
        "Skipping failed request for lat=%s lon=%s start=%s end=%s",
        params.get("latitude"),
        params.get("longitude"),
        params.get("start"),
        params.get("end"),
    )
    return None


def parse_power_payload(payload: dict[str, Any], zone: str) -> pd.DataFrame:
    """Convert a NASA POWER hourly response into a tidy DataFrame."""
    try:
        parameter_data = payload["properties"]["parameter"]
    except KeyError as exc:
        raise ValueError("NASA POWER response is missing properties.parameter") from exc

    frames: list[pd.Series] = []
    for parameter in PARAMETERS:
        values = parameter_data.get(parameter)
        if values is None:
            logging.warning("Parameter %s is missing for zone %s", parameter, zone)
            continue
        frames.append(pd.Series(values, name=parameter, dtype="float64"))

    if not frames:
        return pd.DataFrame(columns=["datetime", "zone", *PARAMETERS])

    df = pd.concat(frames, axis=1)
    df.index = df.index.astype(str)
    df.insert(0, "datetime", pd.to_datetime(df.index, format="%Y%m%d%H", utc=True))
    df.insert(1, "zone", zone)
    for parameter in PARAMETERS:
        if parameter not in df.columns:
            df[parameter] = pd.NA
    return df[["datetime", "zone", *PARAMETERS]]


def fetch_zone_year(
    session: requests.Session,
    zone: str,
    latitude: float,
    longitude: float,
    year: int,
    start_yyyymmdd: str,
    end_yyyymmdd: str,
) -> pd.DataFrame:
    """Fetch and parse one NYISO zone for one calendar-year chunk."""
    logging.info("Fetching zone %s for %s (%s to %s)", zone, year, start_yyyymmdd, end_yyyymmdd)
    params = build_request_params(latitude, longitude, start_yyyymmdd, end_yyyymmdd)
    payload = request_json_with_retries(session, params)
    if payload is None:
        return pd.DataFrame(columns=["datetime", "zone", *PARAMETERS])
    try:
        return parse_power_payload(payload, zone)
    except ValueError as exc:
        logging.warning("Skipping zone %s for %s due to malformed payload: %s", zone, year, exc)
        return pd.DataFrame(columns=["datetime", "zone", *PARAMETERS])


def save_zone_weather(
    zone: str,
    frames: list[pd.DataFrame],
    output_dir: Path,
) -> Path:
    """Combine yearly frames, sort chronologically, and save one CSV for a zone."""
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{zone}_weather.csv"

    if frames:
        zone_df = pd.concat(frames, ignore_index=True)
    else:
        zone_df = pd.DataFrame(columns=["datetime", "zone", *PARAMETERS])

    zone_df = (
        zone_df.drop_duplicates(subset=["datetime", "zone"])
        .sort_values("datetime")
        .reset_index(drop=True)
    )
    zone_df.to_csv(output_path, index=False)
    logging.info("Saved %s rows to %s", len(zone_df), output_path)
    return output_path


def download_weather(start: str, end: str, output_dir: Path) -> list[Path]:
    """Download all configured zones and save one combined CSV per zone."""
    chunks = year_chunks(start, end)
    saved_paths: list[Path] = []

    with requests.Session() as session:
        for zone, (latitude, longitude) in ZONES.items():
            zone_frames: list[pd.DataFrame] = []
            for year, chunk_start, chunk_end in chunks:
                frame = fetch_zone_year(
                    session,
                    zone,
                    latitude,
                    longitude,
                    year,
                    chunk_start,
                    chunk_end,
                )
                if not frame.empty:
                    zone_frames.append(frame)
                time.sleep(REQUEST_DELAY_SECONDS)

            saved_paths.append(save_zone_weather(zone, zone_frames, output_dir))

    return saved_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download hourly NASA POWER weather data for NYISO zones A-K."
    )
    parser.add_argument("--start", default=DEFAULT_START, help="Inclusive start date as YYYYMMDD.")
    parser.add_argument("--end", default=DEFAULT_END, help="Inclusive end date as YYYYMMDD.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where per-zone CSV files will be written.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging verbosity.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(message)s",
    )
    saved_paths = download_weather(args.start, args.end, args.output_dir)
    logging.info("Completed weather download for %s zones.", len(saved_paths))


if __name__ == "__main__":
    main()
