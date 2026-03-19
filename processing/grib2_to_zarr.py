from __future__ import annotations

import argparse
import importlib
import json
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable


SANITIZE_RE = re.compile(r"[^0-9A-Za-z_]+")


def _optional_numpy() -> Any | None:
    try:
        import numpy as np  # type: ignore
    except ImportError:
        return None
    return np


def _require_numpy() -> Any:
    np = _optional_numpy()
    if np is None:
        raise ImportError("numpy is required. Install with: pip install numpy")
    return np


@dataclass
class FileConversionResult:
    input_file: Path
    output_store: Path
    messages_seen: int
    messages_written: int
    messages_failed: int
    variable_names: list[str]
    errors: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "input_file": str(self.input_file),
            "output_store": str(self.output_store),
            "messages_seen": self.messages_seen,
            "messages_written": self.messages_written,
            "messages_failed": self.messages_failed,
            "variable_names": self.variable_names,
            "errors": self.errors,
        }


def _iso_utc(dt: datetime) -> str:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    else:
        dt = dt.astimezone(timezone.utc)
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def _safe_get(obj: Any, names: Iterable[str], default: Any = None) -> Any:
    for name in names:
        if not hasattr(obj, name):
            continue
        try:
            value = getattr(obj, name)
        except Exception:
            continue

        if callable(value):
            try:
                value = value()
            except TypeError:
                continue
            except Exception:
                continue

        if value is not None:
            return value
    return default


def _as_datetime(value: Any) -> datetime | None:
    np = _optional_numpy()

    if value is None:
        return None

    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)

    if np is not None and isinstance(value, np.datetime64):
        try:
            ts = value.astype("datetime64[ns]").astype(np.int64) / 1_000_000_000
            return datetime.fromtimestamp(ts, tz=timezone.utc)
        except Exception:
            return None

    if isinstance(value, (int, float)):
        try:
            return datetime.fromtimestamp(float(value), tz=timezone.utc)
        except Exception:
            return None

    if isinstance(value, str):
        raw = value.strip()
        if not raw:
            return None

        if raw.endswith("Z"):
            raw = raw[:-1] + "+00:00"

        try:
            parsed = datetime.fromisoformat(raw)
            if parsed.tzinfo is None:
                return parsed.replace(tzinfo=timezone.utc)
            return parsed.astimezone(timezone.utc)
        except ValueError:
            pass

        compact = raw.replace("_", "")
        for fmt in (
            "%Y%m%d%H%M%S",
            "%Y%m%d%H%M",
            "%Y%m%d%H",
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%d %H:%M",
        ):
            try:
                return datetime.strptime(compact, fmt).replace(tzinfo=timezone.utc)
            except ValueError:
                continue

    return None


def _parse_forecast_hour(raw: Any) -> int | None:
    np = _optional_numpy()

    if raw is None:
        return None

    if isinstance(raw, timedelta):
        return int(round(raw.total_seconds() / 3600.0))

    if np is not None and isinstance(raw, np.timedelta64):
        try:
            hours = raw / np.timedelta64(1, "h")
            return int(round(float(hours)))
        except Exception:
            return None

    if np is not None and isinstance(raw, np.integer):
        return int(raw)

    if isinstance(raw, int):
        return int(raw)

    if np is not None and isinstance(raw, np.floating):
        return int(round(float(raw)))

    if isinstance(raw, float):
        return int(round(float(raw)))

    text = str(raw).strip()
    if not text:
        return None

    if "-" in text:
        parts = [p for p in text.split("-") if p.strip()]
        if parts:
            text = parts[-1].strip()

    match = re.search(r"-?\d+(?:\.\d+)?", text)
    if not match:
        return None

    try:
        return int(round(float(match.group(0))))
    except ValueError:
        return None


def _sanitize_token(raw: Any, default: str = "var") -> str:
    token = "" if raw is None else str(raw).strip()
    token = SANITIZE_RE.sub("_", token).strip("_").lower()
    if not token:
        token = default
    if token[0].isdigit():
        token = f"v_{token}"
    return token


def _to_jsonable(value: Any) -> Any:
    np = _optional_numpy()

    if np is not None and isinstance(value, (np.integer, np.floating)):
        return value.item()
    if np is not None and isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, datetime):
        return _iso_utc(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    return value


def _build_level_label(level_type: Any, level_value: Any) -> str:
    np = _optional_numpy()

    level_type_token = _sanitize_token(level_type, default="level")
    if level_value is None:
        return level_type_token

    if np is not None and isinstance(level_value, np.integer):
        level_value_token = str(int(level_value))
    elif isinstance(level_value, int):
        level_value_token = str(int(level_value))
    elif np is not None and isinstance(level_value, np.floating):
        level_value_token = f"{float(level_value):g}".replace(".", "p").replace("-", "m")
    elif isinstance(level_value, float):
        level_value_token = f"{float(level_value):g}".replace(".", "p").replace("-", "m")
    else:
        level_value_token = _sanitize_token(level_value, default="value")

    return f"{level_type_token}_{level_value_token}"


def _extract_message_metadata(msg: Any, message_index: int) -> dict[str, Any]:
    short_name = _safe_get(
        msg,
        (
            "shortName",
            "short_name",
            "parameterShortName",
            "abbr",
            "name",
        ),
        default=f"var_{message_index}",
    )
    long_name = _safe_get(
        msg,
        (
            "fullName",
            "parameterName",
            "long_name",
            "name",
        ),
        default=short_name,
    )
    units = _safe_get(
        msg,
        (
            "units",
            "parameterUnits",
            "unit",
        ),
        default="unknown",
    )

    level_type = _safe_get(
        msg,
        (
            "typeOfLevel",
            "typeOfFirstFixedSurface",
            "firstFixedSurfaceType",
            "verticalLevelType",
        ),
    )
    level_value = _safe_get(
        msg,
        (
            "level",
            "scaledValueOfFirstFixedSurface",
            "valueOfFirstFixedSurface",
            "firstFixedSurfaceValue",
        ),
    )

    reference_time = _as_datetime(
        _safe_get(
            msg,
            (
                "refDate",
                "analDate",
                "referenceDate",
                "time",
                "dataDate",
            ),
        )
    )
    valid_time = _as_datetime(
        _safe_get(
            msg,
            (
                "validDate",
                "validTime",
                "valid_time",
            ),
        )
    )

    forecast_time_raw = _safe_get(
        msg,
        (
            "forecastTime",
            "leadTime",
            "fcstTime",
            "stepRange",
            "step",
        ),
    )
    forecast_hour = _parse_forecast_hour(forecast_time_raw)

    if valid_time is None and reference_time is not None and forecast_hour is not None:
        valid_time = reference_time + timedelta(hours=forecast_hour)

    ni = _safe_get(
        msg,
        (
            "Nx",
            "Ni",
            "nx",
            "numberOfPointsAlongAParallel",
        ),
    )
    nj = _safe_get(
        msg,
        (
            "Ny",
            "Nj",
            "ny",
            "numberOfPointsAlongAMeridian",
        ),
    )

    metadata: dict[str, Any] = {
        "message_index": message_index,
        "short_name": _sanitize_token(short_name, default=f"var_{message_index}"),
        "long_name": str(long_name),
        "units": str(units),
        "level_type": _to_jsonable(level_type),
        "level_value": _to_jsonable(level_value),
        "vertical_level_label": _build_level_label(level_type, level_value),
        "reference_time": _iso_utc(reference_time) if reference_time else None,
        "valid_time": _iso_utc(valid_time) if valid_time else None,
        "forecast_time_raw": _to_jsonable(forecast_time_raw),
        "forecast_hour": forecast_hour,
        "grid_type": _to_jsonable(
            _safe_get(
                msg,
                (
                    "gridType",
                    "gridDefinitionTemplate",
                    "gridDefinitionTemplateNumber",
                ),
            )
        ),
        "ni": int(ni) if ni is not None else None,
        "nj": int(nj) if nj is not None else None,
    }

    for key in (
        "discipline",
        "parameterCategory",
        "parameterNumber",
        "productDefinitionTemplateNumber",
        "typeOfGeneratingProcess",
        "generatingProcessIdentifier",
    ):
        value = _safe_get(msg, (key,))
        if value is not None:
            metadata[key] = _to_jsonable(value)

    return metadata


def _read_message_array(msg: Any) -> Any:
    np = _require_numpy()

    raw = None

    for attr in ("data", "values"):
        if not hasattr(msg, attr):
            continue
        try:
            value = getattr(msg, attr)
        except Exception:
            continue
        if callable(value):
            try:
                value = value()
            except TypeError:
                continue
            except Exception:
                continue
        if value is not None:
            raw = value
            break

    if raw is None:
        raise ValueError("GRIB2 message has no readable data/values payload")

    array = np.asarray(raw, dtype=np.float32)
    if array.ndim == 0:
        raise ValueError("GRIB2 message payload was scalar; expected a grid")

    return array


def _infer_chunks(shape: tuple[int, ...], chunk_y: int, chunk_x: int) -> tuple[int, ...]:
    if len(shape) == 1:
        return (max(1, min(shape[0], chunk_x)),)

    if len(shape) == 2:
        return (
            max(1, min(shape[0], chunk_y)),
            max(1, min(shape[1], chunk_x)),
        )

    leading = [1 for _ in shape[:-2]]
    trailing = [
        max(1, min(shape[-2], chunk_y)),
        max(1, min(shape[-1], chunk_x)),
    ]
    return tuple(leading + trailing)


def _build_variable_name(metadata: dict[str, Any], include_level_in_name: bool) -> str:
    base = _sanitize_token(metadata.get("short_name"), default="var")
    if not include_level_in_name:
        return base

    level = _sanitize_token(metadata.get("vertical_level_label"), default="level")
    return f"{base}_{level}"


def _try_write_lat_lon(
    root: Any,
    msg: Any,
    *,
    chunk_y: int,
    chunk_x: int,
) -> bool:
    np = _require_numpy()

    latlons_fn = getattr(msg, "latlons", None)
    if latlons_fn is None or not callable(latlons_fn):
        return False

    try:
        lats, lons = latlons_fn()
    except Exception:
        return False

    lat_arr = np.asarray(lats, dtype=np.float32)
    lon_arr = np.asarray(lons, dtype=np.float32)

    if lat_arr.shape != lon_arr.shape:
        return False

    if lat_arr.ndim not in (1, 2):
        return False

    chunks = _infer_chunks(tuple(lat_arr.shape), chunk_y, chunk_x)

    root.create_dataset("lat", data=lat_arr, overwrite=True, chunks=chunks)
    root.create_dataset("lon", data=lon_arr, overwrite=True, chunks=chunks)

    root.attrs.setdefault("lat_min", float(np.nanmin(lat_arr)))
    root.attrs.setdefault("lat_max", float(np.nanmax(lat_arr)))
    root.attrs.setdefault("lon_min", float(np.nanmin(lon_arr)))
    root.attrs.setdefault("lon_max", float(np.nanmax(lon_arr)))

    if lat_arr.ndim == 2:
        nj, ni = lat_arr.shape
        root.attrs.setdefault("nj", int(nj))
        root.attrs.setdefault("ni", int(ni))
        root.attrs.setdefault("grid_type", "curvilinear")
    else:
        nj = lat_arr.shape[0]
        ni = lon_arr.shape[0]
        root.attrs.setdefault("nj", int(nj))
        root.attrs.setdefault("ni", int(ni))
        root.attrs.setdefault("grid_type", "plate_carree")

    return True


def _build_compressor(name: str, clevel: int) -> Any:
    if name == "none":
        return None

    try:
        numcodecs = importlib.import_module("numcodecs")
    except ImportError as exc:
        raise ImportError(
            "numcodecs is required for compressed output. Install with: pip install numcodecs"
        ) from exc

    Blosc = getattr(numcodecs, "Blosc")

    return Blosc(cname=name, clevel=clevel, shuffle=Blosc.BITSHUFFLE)


def convert_grib2_file_to_zarr(
    input_file: Path,
    output_store: Path,
    *,
    overwrite: bool,
    chunk_y: int,
    chunk_x: int,
    compressor_name: str,
    compression_level: int,
    include_level_in_name: bool,
    write_lat_lon: bool,
    stop_on_error: bool,
) -> FileConversionResult:
    try:
        grib2io = importlib.import_module("grib2io")
    except ImportError as exc:
        raise ImportError(
            "grib2io is required. Install with: pip install grib2io"
        ) from exc

    try:
        zarr = importlib.import_module("zarr")
    except ImportError as exc:
        raise ImportError(
            "zarr is required. Install with: pip install zarr"
        ) from exc

    if output_store.exists() and not overwrite:
        raise FileExistsError(f"Output store already exists: {output_store}")

    output_store.parent.mkdir(parents=True, exist_ok=True)

    root = zarr.open_group(str(output_store), mode="w")
    variables_group = root.require_group("variables")
    compressor = _build_compressor(compressor_name, compression_level)

    messages_seen = 0
    messages_written = 0
    messages_failed = 0
    variable_names: list[str] = []
    errors: list[str] = []

    name_counts: dict[str, int] = {}
    valid_times: list[datetime] = []
    forecast_hours: list[int] = []
    cycle_time: str | None = None
    lat_lon_written = False

    grib_stream = grib2io.open(str(input_file))

    try:
        for message_index, msg in enumerate(grib_stream, start=1):
            messages_seen += 1
            try:
                metadata = _extract_message_metadata(msg, message_index)
                data_array = _read_message_array(msg)
                chunks = _infer_chunks(tuple(data_array.shape), chunk_y, chunk_x)

                base_name = _build_variable_name(metadata, include_level_in_name)
                occurrence = name_counts.get(base_name, 0) + 1
                name_counts[base_name] = occurrence
                variable_name = base_name if occurrence == 1 else f"{base_name}__m{occurrence:03d}"

                dataset = variables_group.create_dataset(
                    variable_name,
                    data=data_array,
                    chunks=chunks,
                    compressor=compressor,
                    overwrite=True,
                )

                dataset.attrs.update({
                    key: _to_jsonable(value)
                    for key, value in metadata.items()
                })
                dataset.attrs["source_file"] = input_file.name
                dataset.attrs["shape"] = list(data_array.shape)
                dataset.attrs["dtype"] = str(data_array.dtype)

                variable_names.append(variable_name)
                messages_written += 1

                valid_dt = _as_datetime(metadata.get("valid_time"))
                if valid_dt is not None:
                    valid_times.append(valid_dt)

                if metadata.get("forecast_hour") is not None:
                    forecast_hours.append(int(metadata["forecast_hour"]))

                if cycle_time is None and metadata.get("reference_time"):
                    cycle_time = str(metadata["reference_time"])

                if write_lat_lon and not lat_lon_written:
                    lat_lon_written = _try_write_lat_lon(
                        root,
                        msg,
                        chunk_y=chunk_y,
                        chunk_x=chunk_x,
                    )

            except Exception as exc:
                messages_failed += 1
                errors.append(f"message {message_index}: {exc}")
                if stop_on_error:
                    raise

    finally:
        if hasattr(grib_stream, "close"):
            try:
                grib_stream.close()
            except Exception:
                pass

    root.attrs.update(
        {
            "converter": "backend/processing/grib2_to_zarr.py",
            "converted_at": _iso_utc(datetime.now(timezone.utc)),
            "source_file": input_file.name,
            "source_path": str(input_file.resolve()),
            "messages_seen": messages_seen,
            "messages_written": messages_written,
            "messages_failed": messages_failed,
            "variable_count": len(variable_names),
            "variables": variable_names,
            "chunk_y": chunk_y,
            "chunk_x": chunk_x,
            "compressor": compressor_name,
            "compression_level": compression_level,
            "include_level_in_name": include_level_in_name,
        }
    )

    if cycle_time is not None:
        root.attrs["cycle_time"] = cycle_time

    if valid_times:
        root.attrs["valid_time_min"] = _iso_utc(min(valid_times))
        root.attrs["valid_time_max"] = _iso_utc(max(valid_times))

    if forecast_hours:
        root.attrs["forecast_hour_min"] = int(min(forecast_hours))
        root.attrs["forecast_hour_max"] = int(max(forecast_hours))

    return FileConversionResult(
        input_file=input_file,
        output_store=output_store,
        messages_seen=messages_seen,
        messages_written=messages_written,
        messages_failed=messages_failed,
        variable_names=variable_names,
        errors=errors,
    )


def _discover_grib_files(input_path: Path, patterns: list[str], recursive: bool) -> list[Path]:
    if input_path.is_file():
        return [input_path]

    files: list[Path] = []
    for pattern in patterns:
        iterator = input_path.rglob(pattern) if recursive else input_path.glob(pattern)
        files.extend([path for path in iterator if path.is_file()])

    unique = sorted({path.resolve() for path in files})
    return [Path(path) for path in unique]


def _resolve_output_store(
    input_file: Path,
    *,
    input_root: Path,
    output_root: Path,
    single_input: bool,
) -> Path:
    if single_input:
        if output_root.suffix == ".zarr":
            return output_root
        return output_root / f"{input_file.stem}.zarr"

    rel = input_file.relative_to(input_root)
    return (output_root / rel).with_suffix(".zarr")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Convert GRIB2 files to chunked Zarr stores by reading each GRIB2 message "
            "one-by-one with grib2io."
        )
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Input GRIB2 file or directory of GRIB2 files.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help=(
            "Output Zarr store path (for single file) or output directory (for directory input). "
            "Default: next to input file or inside input directory."
        ),
    )
    parser.add_argument(
        "--pattern",
        default="*.grib2,*.grb2,*.grib,*.grb",
        help="Comma-separated glob patterns used when --input is a directory.",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recursively scan subdirectories when --input is a directory.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output Zarr stores if they already exist.",
    )
    parser.add_argument(
        "--chunk-y",
        type=int,
        default=256,
        help="Chunk size for the Y dimension.",
    )
    parser.add_argument(
        "--chunk-x",
        type=int,
        default=256,
        help="Chunk size for the X dimension.",
    )
    parser.add_argument(
        "--compressor",
        default="zstd",
        choices=["none", "zstd", "lz4", "blosclz"],
        help="Compressor for Zarr arrays.",
    )
    parser.add_argument(
        "--compression-level",
        type=int,
        default=5,
        help="Compression level for the selected compressor.",
    )
    parser.add_argument(
        "--include-level-in-name",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include vertical level identifier in output variable names.",
    )
    parser.add_argument(
        "--write-lat-lon",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Attempt to write lat/lon coordinate arrays to each output store.",
    )
    parser.add_argument(
        "--stop-on-error",
        action="store_true",
        help="Stop conversion immediately on the first failing message/file.",
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    input_path = Path(args.input).expanduser().resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")

    if args.output is None:
        output_root = input_path.parent if input_path.is_file() else input_path
    else:
        output_root = Path(args.output).expanduser().resolve()

    patterns = [p.strip() for p in args.pattern.split(",") if p.strip()]
    files = _discover_grib_files(input_path, patterns=patterns, recursive=args.recursive)

    if not files:
        raise FileNotFoundError(
            f"No files found under {input_path} with patterns: {patterns}"
        )

    single_input = input_path.is_file()
    results: list[FileConversionResult] = []
    file_failures = 0

    for file_path in files:
        out_store = _resolve_output_store(
            file_path,
            input_root=input_path if not single_input else file_path.parent,
            output_root=output_root,
            single_input=single_input,
        )

        try:
            result = convert_grib2_file_to_zarr(
                file_path,
                out_store,
                overwrite=args.overwrite,
                chunk_y=args.chunk_y,
                chunk_x=args.chunk_x,
                compressor_name=args.compressor,
                compression_level=args.compression_level,
                include_level_in_name=args.include_level_in_name,
                write_lat_lon=args.write_lat_lon,
                stop_on_error=args.stop_on_error,
            )
            results.append(result)
            print(
                json.dumps(
                    {
                        "status": "ok",
                        "input_file": str(file_path),
                        "output_store": str(out_store),
                        "messages_seen": result.messages_seen,
                        "messages_written": result.messages_written,
                        "messages_failed": result.messages_failed,
                    },
                    indent=2,
                )
            )
        except Exception as exc:
            file_failures += 1
            error_payload = {
                "status": "error",
                "input_file": str(file_path),
                "output_store": str(out_store),
                "error": str(exc),
            }
            print(json.dumps(error_payload, indent=2))
            if args.stop_on_error:
                raise

    total_seen = sum(r.messages_seen for r in results)
    total_written = sum(r.messages_written for r in results)
    total_failed_messages = sum(r.messages_failed for r in results)

    summary = {
        "files_requested": len(files),
        "files_succeeded": len(results),
        "files_failed": file_failures,
        "messages_seen": total_seen,
        "messages_written": total_written,
        "messages_failed": total_failed_messages,
    }
    print(json.dumps({"summary": summary}, indent=2))

    if file_failures > 0:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
