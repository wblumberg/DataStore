"""Command line ingest and maintenance utilities for observation SQL storage."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .sql_store import ObservationInsert, ObservationSQLStore, parse_time_like, to_iso_utc

LIGHTNING_COLUMNS = [
    "year",
    "month",
    "day",
    "hour",
    "minute",
    "second",
    "nanosecond",
    "lat",
    "lon",
    "peak_current",
    "multiplicity",
    "num_sensors",
    "dof",
    "error_ellipse_angle",
    "error_ellipse_semi_major_axis",
    "error_ellipse_semi_minor_axis",
    "chi_squared",
    "rise_time",
    "peak_to_zero_time",
    "max_rate_of_rise",
    "cloud_indicator",
    "angle_indicator",
    "signal_indicator",
    "timing_indicator",
]


def _default_db_path() -> Path:
    return Path("/data/store/point/observations.sqlite")


def _resolve_db_path(raw: str | None) -> Path:
    if not raw:
        return _default_db_path()
    return Path(raw).expanduser().resolve()


def _read_text_file(path: Path) -> str:
    if path.suffix.lower() == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            return handle.read()
    return path.read_text(encoding="utf-8")


def _extract_time_from_filename(path: Path) -> datetime | None:
    stem = path.name
    for token in stem.replace(".", "_").split("_"):
        if len(token) >= 12 and token[:12].isdigit():
            dt = parse_time_like(token[:12])
            if dt is not None:
                return dt

    chunks = stem.replace(".", "_").split("_")
    for idx in range(len(chunks) - 1):
        left = chunks[idx]
        right = chunks[idx + 1]
        if len(left) == 8 and len(right) >= 4 and left.isdigit() and right[:4].isdigit():
            dt = parse_time_like(f"{left}_{right[:4]}")
            if dt is not None:
                return dt

    return None


def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _extract_platform_id(record: dict[str, Any]) -> str | None:
    for key in (
        "id",
        "station",
        "station_id",
        "stationId",
        "icao",
        "callsign",
        "platform_id",
        "name",
    ):
        value = record.get(key)
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return None


def _extract_lat_lon(record: dict[str, Any]) -> tuple[float | None, float | None]:
    if "coord" in record and isinstance(record["coord"], dict):
        coord = record["coord"]
        lat = _to_float(coord.get("lat") or coord.get("latitude"))
        lon = _to_float(coord.get("lon") or coord.get("longitude"))
        return lat, lon

    if "geometry" in record and isinstance(record["geometry"], dict):
        geom = record["geometry"]
        if geom.get("type") == "Point" and isinstance(geom.get("coordinates"), list):
            coords = geom["coordinates"]
            if len(coords) >= 2:
                lon = _to_float(coords[0])
                lat = _to_float(coords[1])
                return lat, lon

    lat = _to_float(record.get("lat") or record.get("latitude"))
    lon = _to_float(record.get("lon") or record.get("longitude"))
    return lat, lon


def _extract_elevation(record: dict[str, Any]) -> float | None:
    for key in ("elevation_m", "elevation", "elev", "altitude_m", "altitude"):
        value = _to_float(record.get(key))
        if value is not None:
            return value
    return None


def _extract_obs_time(record: dict[str, Any], fallback_time: datetime | None) -> datetime:
    for key in (
        "obs_time",
        "observation_time",
        "time",
        "valid_time",
        "datetime",
        "timestamp",
        "ts",
    ):
        parsed = parse_time_like(record.get(key))
        if parsed is not None:
            return parsed

    if fallback_time is not None:
        return fallback_time

    return datetime.now(tz=timezone.utc)


def _flatten_record(raw: dict[str, Any]) -> dict[str, Any]:
    """Normalize common nested observation payloads into one dictionary."""
    out: dict[str, Any] = {}

    if "data" in raw and isinstance(raw["data"], dict):
        out.update(raw["data"])
        for key, value in raw.items():
            if key not in ("data", "coord"):
                out[key] = value
        if "coord" in raw and isinstance(raw["coord"], dict):
            out["coord"] = raw["coord"]
        return out

    if raw.get("type") == "Feature":
        props = raw.get("properties") if isinstance(raw.get("properties"), dict) else {}
        out.update(props)
        if "geometry" in raw:
            out["geometry"] = raw["geometry"]
        return out

    return dict(raw)


def _records_from_json_payload(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, dict) and payload.get("type") == "FeatureCollection":
        features = payload.get("features") if isinstance(payload.get("features"), list) else []
        return [entry for entry in features if isinstance(entry, dict)]

    if isinstance(payload, list):
        return [entry for entry in payload if isinstance(entry, dict)]

    if isinstance(payload, dict):
        return [payload]

    return []


def _records_from_gempak_payload(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [entry for entry in payload if isinstance(entry, dict)]

    if isinstance(payload, dict):
        for key in ("features", "records", "stations", "data"):
            items = payload.get(key)
            if isinstance(items, list):
                return [entry for entry in items if isinstance(entry, dict)]
        return [payload]

    return []


def _to_python_scalar(value: Any) -> Any:
    if hasattr(value, "item") and callable(getattr(value, "item")):
        try:
            return value.item()
        except Exception:
            return value
    return value


def _clean_gempak_number(value: Any) -> Any:
    scalar = _to_python_scalar(value)
    number = _to_float(scalar)
    if number is None:
        if isinstance(scalar, str):
            text = scalar.strip()
            return None if text == "" else text
        return scalar
    if number <= -9998 or number >= 9.9e19:
        return None
    return number


def _extract_gempak_lat_lon(record: dict[str, Any]) -> tuple[float | None, float | None]:
    geometry = record.get("geometry") if isinstance(record.get("geometry"), dict) else None
    if geometry and geometry.get("type") == "Point" and isinstance(geometry.get("coordinates"), list):
        coords = geometry.get("coordinates")
        if len(coords) >= 2:
            lon = _to_float(coords[0])
            lat = _to_float(coords[1])
            return lat, lon

    props = record.get("properties") if isinstance(record.get("properties"), dict) else {}
    lat = _to_float(props.get("latitude") or props.get("lat"))
    lon = _to_float(props.get("longitude") or props.get("lon"))
    return lat, lon


def _extract_gempak_obs_time(record: dict[str, Any], fallback_time: datetime | None) -> datetime | None:
    props = record.get("properties") if isinstance(record.get("properties"), dict) else {}
    candidates = (
        props.get("date_time"),
        props.get("datetime"),
        props.get("valid_time"),
        props.get("time"),
        record.get("date_time"),
        record.get("datetime"),
        record.get("valid_time"),
        record.get("time"),
    )

    for candidate in candidates:
        parsed = parse_time_like(candidate)
        if parsed is not None:
            return parsed

    return fallback_time


def parse_gempak_sfjson_records(
    *,
    records: list[dict[str, Any]],
    obs_type: str,
    source_name: str | None,
    fallback_time: datetime | None,
) -> list[ObservationInsert]:
    parsed_rows: list[ObservationInsert] = []
    skipped_missing_time = 0

    for record in records:
        if not isinstance(record, dict):
            continue

        props_raw = record.get("properties") if isinstance(record.get("properties"), dict) else {}
        values_raw = record.get("values") if isinstance(record.get("values"), dict) else {}

        props: dict[str, Any] = {}
        for key, value in props_raw.items():
            key_text = str(key)
            scalar = _to_python_scalar(value)
            if isinstance(scalar, str):
                scalar = scalar.strip()
            if scalar in (None, ""):
                continue
            props[key_text] = scalar

        values: dict[str, Any] = {}
        for key, value in values_raw.items():
            values[str(key)] = _clean_gempak_number(value)

        obs_time = _extract_gempak_obs_time(record, fallback_time)
        if obs_time is None:
            skipped_missing_time += 1
            continue

        lat, lon = _extract_gempak_lat_lon(record)
        elev = _to_float(
            props.get("elevation_m")
            or props.get("elevation")
            or props.get("elev")
            or props.get("station_elevation")
        )

        platform_id = _extract_platform_id(props) or _extract_platform_id(record)

        payload: dict[str, Any] = dict(values)
        for key, value in props.items():
            if key in ("latitude", "longitude", "lat", "lon"):
                continue
            if key in payload:
                payload[f"prop_{key}"] = value
            else:
                payload[key] = value

        parsed_rows.append(
            ObservationInsert(
                obs_type=obs_type,
                obs_time=obs_time,
                platform_id=platform_id,
                lat=lat,
                lon=lon,
                elevation_m=elev,
                source_name=source_name,
                payload=payload,
            )
        )

    if skipped_missing_time:
        print(
            f"WARNING: skipped {skipped_missing_time} GEMPAK records with no valid observation time",
            file=sys.stderr,
        )

    return parsed_rows


def parse_gempak_sfjson_file(
    *,
    path: Path,
    obs_type: str,
    source_name: str | None,
    fallback_time: datetime | None,
) -> list[ObservationInsert]:
    text = _read_text_file(path)
    if not text.strip():
        return []

    payload = json.loads(text)
    records = _records_from_gempak_payload(payload)
    return parse_gempak_sfjson_records(
        records=records,
        obs_type=obs_type,
        source_name=source_name,
        fallback_time=fallback_time,
    )


def parse_gempak_surface_file(
    *,
    path: Path,
    obs_type: str,
    source_name: str | None,
    fallback_time: datetime | None,
    gempak_country: str | None,
    gempak_date_time: str | None,
) -> list[ObservationInsert]:
    try:
        from gempakio import GempakSurface  # type: ignore[import-not-found]
    except ImportError:
        try:
            from metpy.io import GempakSurface  # type: ignore[import-not-found]
        except ImportError as exc:
            raise RuntimeError(
                "GEMPAK ingest requires 'gempakio' (preferred) or 'metpy' to be installed"
            ) from exc

    kwargs: dict[str, Any] = {}
    if gempak_country:
        kwargs["country"] = gempak_country
    if gempak_date_time:
        kwargs["date_time"] = gempak_date_time

    reader = GempakSurface(str(path))
    records_payload = reader.sfjson(**kwargs)
    records = _records_from_gempak_payload(records_payload)

    return parse_gempak_sfjson_records(
        records=records,
        obs_type=obs_type,
        source_name=source_name,
        fallback_time=fallback_time,
    )


def parse_json_like_file(
    *,
    path: Path,
    obs_type: str,
    source_name: str | None,
    fallback_time: datetime | None,
) -> list[ObservationInsert]:
    text = _read_text_file(path)
    if not text.strip():
        return []

    payload = json.loads(text)
    records = _records_from_json_payload(payload)

    out: list[ObservationInsert] = []
    for record in records:
        flat = _flatten_record(record)
        lat, lon = _extract_lat_lon(flat)
        elev = _extract_elevation(flat)
        platform_id = _extract_platform_id(flat)
        obs_time = _extract_obs_time(flat, fallback_time)

        out.append(
            ObservationInsert(
                obs_type=obs_type,
                obs_time=obs_time,
                platform_id=platform_id,
                lat=lat,
                lon=lon,
                elevation_m=elev,
                source_name=source_name,
                payload=flat,
            )
        )

    return out


def parse_lightning_txt_file(
    *,
    path: Path,
    obs_type: str,
    source_name: str | None,
    fallback_time: datetime | None,
) -> list[ObservationInsert]:
    text = _read_text_file(path)
    if not text.strip():
        return []

    parsed_rows: list[ObservationInsert] = []
    reader = csv.reader(text.splitlines(), delimiter="\t")
    min_cols = len(LIGHTNING_COLUMNS)
    with_record_type_cols = min_cols + 1
    skipped_short_rows = 0
    skipped_bad_time_rows = 0

    for raw_row in reader:
        if not raw_row:
            continue

        cols = [entry.strip() for entry in raw_row]
        if not any(cols):
            continue

        record_type: str | None = None
        extra_fields: list[str] = []

        if len(cols) >= with_record_type_cols:
            record_type = cols[0]
            cols = cols[1:]

        if len(cols) < min_cols:
            skipped_short_rows += 1
            continue

        if len(cols) > min_cols:
            extra_fields = cols[min_cols:]
            cols = cols[:min_cols]

        row = {name: cols[i] for i, name in enumerate(LIGHTNING_COLUMNS)}

        try:
            dt = datetime(
                year=int(row["year"]),
                month=int(row["month"]),
                day=int(row["day"]),
                hour=int(row["hour"]),
                minute=int(row["minute"]),
                second=int(row["second"]),
                tzinfo=timezone.utc,
            )
        except (TypeError, ValueError):
            if fallback_time is None:
                skipped_bad_time_rows += 1
                continue
            dt = fallback_time

        payload: dict[str, Any] = {}
        for key, value in row.items():
            if value is None or value == "":
                continue
            number = _to_float(value)
            payload[key] = number if number is not None else value

        if record_type not in (None, ""):
            payload["record_type"] = _to_float(record_type) if _to_float(record_type) is not None else record_type

        if extra_fields:
            payload["extra_fields"] = [
                (_to_float(value) if _to_float(value) is not None else value)
                for value in extra_fields
                if value not in (None, "")
            ]

        parsed_rows.append(
            ObservationInsert(
                obs_type=obs_type,
                obs_time=dt,
                platform_id=None,
                lat=_to_float(row.get("lat")),
                lon=_to_float(row.get("lon")),
                elevation_m=None,
                source_name=source_name,
                payload=payload,
            )
        )

    if skipped_short_rows or skipped_bad_time_rows:
        print(
            f"WARNING: {path.name} skipped rows: short={skipped_short_rows}, bad_time={skipped_bad_time_rows}",
            file=sys.stderr,
        )

    return parsed_rows


def parse_input_file(
    *,
    path: Path,
    obs_type: str,
    source_name: str | None,
    assume_time: datetime | None,
    file_format: str,
    gempak_country: str | None = None,
    gempak_date_time: str | None = None,
) -> list[ObservationInsert]:
    gempak_time = parse_time_like(gempak_date_time) if gempak_date_time else None
    if gempak_date_time and gempak_time is None:
        raise ValueError(f"Cannot parse --gempak-date-time '{gempak_date_time}'")

    fallback_time = gempak_time or assume_time or _extract_time_from_filename(path)

    fmt = file_format.lower().strip()
    if fmt == "auto":
        suffixes = [suffix.lower() for suffix in path.suffixes]
        if ".txt" in suffixes or path.suffix.lower() in (".ltg", ".ltng", ".tsv"):
            fmt = "lightning_txt"
        elif path.suffix.lower() in (".sfc", ".gem"):
            fmt = "gempak_surface"
        else:
            fmt = "json"

    if fmt == "lightning_txt":
        return parse_lightning_txt_file(
            path=path,
            obs_type=obs_type,
            source_name=source_name,
            fallback_time=fallback_time,
        )

    if fmt in ("json", "geojson"):
        return parse_json_like_file(
            path=path,
            obs_type=obs_type,
            source_name=source_name,
            fallback_time=fallback_time,
        )

    if fmt == "gempak_sfjson":
        return parse_gempak_sfjson_file(
            path=path,
            obs_type=obs_type,
            source_name=source_name,
            fallback_time=fallback_time,
        )

    if fmt == "gempak_surface":
        return parse_gempak_surface_file(
            path=path,
            obs_type=obs_type,
            source_name=source_name,
            fallback_time=fallback_time,
            gempak_country=gempak_country,
            gempak_date_time=gempak_date_time,
        )

    raise ValueError(f"Unsupported --format '{file_format}'")


def _split_csv(value: str | None) -> list[str] | None:
    if value is None:
        return None
    parts = [entry.strip() for entry in value.split(",") if entry.strip()]
    return parts or None


def _print_json(data: Any) -> None:
    print(json.dumps(data, indent=2, sort_keys=True, default=str))


def cmd_init(args: argparse.Namespace) -> int:
    db_path = _resolve_db_path(args.db)
    store = ObservationSQLStore(db_path)
    store.initialize()
    print(f"Initialized observation database: {db_path}")
    return 0


def cmd_ingest(args: argparse.Namespace) -> int:
    db_path = _resolve_db_path(args.db)
    store = ObservationSQLStore(db_path)
    store.initialize()

    assume_time = parse_time_like(args.assume_time) if args.assume_time else None
    if args.assume_time and assume_time is None:
        print(f"ERROR: cannot parse --assume-time '{args.assume_time}'", file=sys.stderr)
        return 2

    obs_type = args.obs_type.strip().upper()
    all_rows: list[ObservationInsert] = []

    for raw in args.input:
        path = Path(raw).expanduser().resolve()
        if not path.exists():
            print(f"WARNING: skipping missing file {path}", file=sys.stderr)
            continue

        try:
            parsed_rows = parse_input_file(
                path=path,
                obs_type=obs_type,
                source_name=args.source_name or path.name,
                assume_time=assume_time,
                file_format=args.format,
                gempak_country=args.gempak_country,
                gempak_date_time=args.gempak_date_time,
            )
        except Exception as exc:
            print(f"ERROR: failed parsing {path}: {exc}", file=sys.stderr)
            return 2

        all_rows.extend(parsed_rows)
        print(f"Parsed {len(parsed_rows)} records from {path}")

    result = store.insert_many(all_rows)
    _print_json({"db": str(db_path), "obs_type": obs_type, **result})
    return 0


def cmd_list_times(args: argparse.Namespace) -> int:
    db_path = _resolve_db_path(args.db)
    store = ObservationSQLStore(db_path)
    store.initialize()

    obs_types = _split_csv(args.obs_types)
    times = store.list_unique_times(obs_types=obs_types, limit=args.limit)
    _print_json(
        {
            "db": str(db_path),
            "obs_types": obs_types,
            "count": len(times),
            "times": times,
        }
    )
    return 0


def cmd_query(args: argparse.Namespace) -> int:
    db_path = _resolve_db_path(args.db)
    store = ObservationSQLStore(db_path)
    store.initialize()

    center = parse_time_like(args.center) if args.center else None
    if args.center and center is None:
        print(f"ERROR: cannot parse --center '{args.center}'", file=sys.stderr)
        return 2

    obs_types = _split_csv(args.obs_types)
    parameters = _split_csv(args.parameters)
    result = store.query_observations(
        obs_types=obs_types,
        center_time=center,
        minutes_before=args.minutes_before,
        minutes_after=args.minutes_after,
        latest_only=args.latest_only,
        prefer_most_data=args.prefer_most_data,
        parameter_names=parameters,
        bin_minutes=args.bin_minutes,
        max_rows=args.max_rows,
    )
    _print_json(result)
    return 0


def cmd_delete(args: argparse.Namespace) -> int:
    db_path = _resolve_db_path(args.db)
    store = ObservationSQLStore(db_path)
    store.initialize()

    obs_types = _split_csv(args.obs_types)
    source_names = _split_csv(args.source_names)
    before = parse_time_like(args.before) if args.before else None
    after = parse_time_like(args.after) if args.after else None

    if args.before and before is None:
        print(f"ERROR: cannot parse --before '{args.before}'", file=sys.stderr)
        return 2
    if args.after and after is None:
        print(f"ERROR: cannot parse --after '{args.after}'", file=sys.stderr)
        return 2

    if not obs_types and not source_names and before is None and after is None and not args.force:
        print("ERROR: refusing full-table delete without filters. Add --force to allow.", file=sys.stderr)
        return 2

    if not obs_types and not source_names and before is None and after is None and args.force:
        deleted = store.delete_rows(
            obs_types=None,
            source_names=None,
            before=datetime(9999, 12, 31, tzinfo=timezone.utc),
            after=datetime(1970, 1, 1, tzinfo=timezone.utc),
        )
    else:
        deleted = store.delete_rows(
            obs_types=obs_types,
            source_names=source_names,
            before=before,
            after=after,
        )

    _print_json(
        {
            "db": str(db_path),
            "deleted": deleted,
            "obs_types": obs_types,
            "source_names": source_names,
            "before": to_iso_utc(before) if before else None,
            "after": to_iso_utc(after) if after else None,
        }
    )
    return 0


def cmd_prune(args: argparse.Namespace) -> int:
    db_path = _resolve_db_path(args.db)
    store = ObservationSQLStore(db_path)
    store.initialize()

    obs_types = _split_csv(args.obs_types)
    deleted = store.prune_older_than(keep_days=args.keep_days, obs_types=obs_types)
    _print_json(
        {
            "db": str(db_path),
            "keep_days": args.keep_days,
            "obs_types": obs_types,
            "deleted": deleted,
        }
    )
    return 0


def cmd_vacuum(args: argparse.Namespace) -> int:
    db_path = _resolve_db_path(args.db)
    store = ObservationSQLStore(db_path)
    store.initialize()
    store.vacuum()
    print(f"VACUUM complete: {db_path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Ingest and maintain SQL-backed point observations.",
    )
    parser.add_argument(
        "--db",
        default=str(_default_db_path()),
        help="Path to SQLite DB file (default: /data/store/point/observations.sqlite)",
    )

    sub = parser.add_subparsers(dest="command", required=True)

    p_init = sub.add_parser("init", help="Create DB schema and indexes")
    p_init.set_defaults(func=cmd_init)

    p_ingest = sub.add_parser("ingest", help="Ingest observations from files")
    p_ingest.add_argument("--obs-type", required=True, help="Observation type label, e.g. METAR")
    p_ingest.add_argument(
        "--input",
        nargs="+",
        required=True,
        help="Input files (.json/.geojson/.json.gz, lightning txt, or GEMPAK)",
    )
    p_ingest.add_argument(
        "--format",
        default="auto",
        choices=["auto", "json", "geojson", "lightning_txt", "gempak_sfjson", "gempak_surface"],
        help="Input parser format",
    )
    p_ingest.add_argument("--source-name", default=None, help="Source label stored with each row")
    p_ingest.add_argument(
        "--assume-time",
        default=None,
        help="Fallback observation time if record/file has none (ISO or YYYYMMDD_HHMM)",
    )
    p_ingest.add_argument(
        "--gempak-country",
        default=None,
        help="Optional country filter passed to GempakSurface.sfjson()",
    )
    p_ingest.add_argument(
        "--gempak-date-time",
        default=None,
        help="GEMPAK DATTIM selector (e.g. 202603162005) and fallback obs time",
    )
    p_ingest.set_defaults(func=cmd_ingest)

    p_times = sub.add_parser("list-times", help="List unique observation times in DB")
    p_times.add_argument("--obs-types", default=None, help="Comma-separated observation types")
    p_times.add_argument("--limit", type=int, default=200)
    p_times.set_defaults(func=cmd_list_times)

    p_query = sub.add_parser("query", help="Query observations")
    p_query.add_argument("--obs-types", default=None, help="Comma-separated observation types")
    p_query.add_argument("--center", default=None, help="Reference time (ISO or key)")
    p_query.add_argument("--minutes-before", type=int, default=0)
    p_query.add_argument("--minutes-after", type=int, default=0)
    p_query.add_argument("--latest-only", action="store_true")
    p_query.add_argument(
        "--prefer-most-data",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Choose densest snapshot when latest-only is enabled (default: true)",
    )
    p_query.add_argument("--bin-minutes", type=int, default=0)
    p_query.add_argument("--parameters", default=None, help="Comma-separated payload keys to include")
    p_query.add_argument("--max-rows", type=int, default=50000)
    p_query.set_defaults(func=cmd_query)

    p_delete = sub.add_parser("delete", help="Delete observations by type/time filters")
    p_delete.add_argument("--obs-types", default=None, help="Comma-separated observation types")
    p_delete.add_argument(
        "--source-names",
        default=None,
        help="Comma-separated source_name values (usually input filenames)",
    )
    p_delete.add_argument("--before", default=None, help="Delete rows with obs_time <= this time")
    p_delete.add_argument("--after", default=None, help="Delete rows with obs_time >= this time")
    p_delete.add_argument("--force", action="store_true", help="Allow full-table delete")
    p_delete.set_defaults(func=cmd_delete)

    p_prune = sub.add_parser("prune", help="Delete rows older than keep-days")
    p_prune.add_argument("--keep-days", type=int, required=True)
    p_prune.add_argument("--obs-types", default=None, help="Comma-separated observation types")
    p_prune.set_defaults(func=cmd_prune)

    p_vacuum = sub.add_parser("vacuum", help="Run SQLite VACUUM")
    p_vacuum.set_defaults(func=cmd_vacuum)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


__all__ = [
    "LIGHTNING_COLUMNS",
    "build_parser",
    "main",
    "parse_input_file",
    "parse_json_like_file",
    "parse_lightning_txt_file",
]


if __name__ == "__main__":
    raise SystemExit(main())
