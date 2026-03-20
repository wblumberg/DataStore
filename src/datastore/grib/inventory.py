"""Variable inventory helpers for GRIB2-based model data."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import grib2io

from ..core.models import VariableInfo


def get_variable_type(msg) -> str:
    """Determine the logical variable type from a GRIB2 message."""
    if hasattr(msg, "statisticalProcess") and msg.statisticalProcess is not None:
        process_type = msg.statisticalProcess
        time_range = getattr(msg, "timeRangeOfStatisticalProcess", "")
        unit_of_time_range = getattr(msg, "unitOfTimeRangeOfStatisticalProcess", "")
        unit_of_time_range = unit_of_time_range.split("-")[1] if "-" in unit_of_time_range else unit_of_time_range

        if time_range and unit_of_time_range == "1":
            time_range += "h"

        if "Average" in process_type:
            return f"average_{time_range}h" if time_range else "average"
        if "Accumulation" in process_type:
            return f"accum_{time_range}h" if time_range else "accum"
        if "Maximum" in process_type:
            return f"max_{time_range}h" if time_range else "max"
        if "Minimum" in process_type:
            return f"min_{time_range}h" if time_range else "min"
        return f"stat_{process_type}_{time_range}" if time_range else f"stat_{process_type}"

    if hasattr(msg, "stepRange") and msg.stepRange:
        return f"accum_{msg.stepRange}"

    return "instant"


def inventory_variables(files: Sequence[Path], sample_limit: int = 5) -> list[VariableInfo]:
    """Inventory unique variables from a sample of GRIB2 files."""
    variables: dict[str, VariableInfo] = {}

    for path in list(files)[:sample_limit]:
        try:
            with grib2io.open(str(path)) as grib_file:
                for msg in grib_file:
                    variable_name = msg.shortName
                    if variable_name in variables:
                        continue

                    variables[variable_name] = VariableInfo(
                        name=variable_name,
                        long_name=getattr(msg, "fullName", variable_name),
                        units=getattr(msg, "units", ""),
                        level_type=_level_type(msg),
                        level1=_surface_value(msg, "typeOfFirstFixedSurface", "valueOfFirstFixedSurface"),
                        level2=_surface_value(msg, "typeOfSecondFixedSurface", "valueOfSecondFixedSurface"),
                        type=get_variable_type(msg),
                    )
        except Exception as exc:
            print(f"Error reading {path}: {exc}")

    return list(variables.values())


def load_variables_db(variables_file: Path) -> list[VariableInfo]:
    with open(variables_file) as handle:
        payload = json.load(handle)
    return [VariableInfo.from_dict(item) for item in payload]


def save_variables_db(variables: Sequence[VariableInfo], output_file: Path) -> None:
    payload = [variable.to_dict() for variable in variables]
    with open(output_file, "w") as handle:
        json.dump(payload, handle, indent=2)


def _level_type(msg) -> str:
    surface_type = getattr(msg, "typeOfFirstFixedSurface", None)
    if surface_type == 103:
        return "HGHT"
    if surface_type == 100:
        return "PRES"
    if surface_type == 20:
        return "TEMP"
    return ""


def _surface_value(msg, surface_attr: str, value_attr: str) -> str:
    surface_type = getattr(msg, surface_attr, 255)
    if surface_type == 255:
        return ""
    if hasattr(msg, value_attr):
        return str(int(getattr(msg, value_attr)))
    return ""
