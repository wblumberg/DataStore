"""Download and convert ECMWF high-resolution forecast data."""

from __future__ import annotations

import argparse
import io
import os
import re
import subprocess
from datetime import datetime, timedelta

MODELS = [
    {
        "outdir": "/home/gblumberg/data/gempak/model/ecmwf_hr",
        "gribdir": "/home/gblumberg/data/base/model/ecmwf_hr",
        "model": "ifs",
        "product": "oper",
        "gempak_pattern": "YYYYMMDDHHfFFF_ecmwfhr.gem",
        "model_ext": "_ecmwfhr.gem",
        "fxx": range(0, 243, 3),
        "params": None,
    },
]


def round_down_to_00_or_12(dt: datetime) -> datetime:
    dt = dt.replace(minute=0, second=0, microsecond=0)
    return dt.replace(hour=0 if dt.hour < 12 else 12)


def extract_forecast_hour(filename: str) -> int | None:
    match = re.search(r"f(\d{2,})", filename)
    if match:
        return int(match.group(1))
    return None


def grib_to_gempak(filename: str) -> None:
    import pygrib

    with pygrib.open(filename) as grib_messages:
        memfile = io.BytesIO()
        for message in grib_messages:
            if message.level == "mean sea level":
                message.parameterNumber = 198
            memfile.write(message.tostring())
        grib2_bytes = memfile.getvalue()

    proc = subprocess.Popen(
        [
            "/home/gblumberg/GEMPAK7/os/linux64/bin/dcgrib2",
            "-v",
            "1",
            "-e",
            "GEMTBL=/home/gblumberg/gemtbls/",
            "/home/gblumberg/data/gempak/model/ecmwfhr/ecmwfhr_YYYYMMDDHHfFFF.gem",
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    stdout, stderr = proc.communicate(input=grib2_bytes)
    print("dcgrib2 stdout:", stdout.decode())
    if stderr:
        print("dcgrib2 stderr:", stderr.decode())


def convert_grib2(filename: str) -> None:
    cmd = ["grib_set", "-r", "-w", "packingType=grid_ccsds", "-s", "packingType=grid_simple", filename, "OUT.grib2"]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print("Error converting GRIB2 file:", result.stderr)
    else:
        print("GRIB2 file converted successfully.")


def process_model(model_cfg: dict) -> None:
    from herbie import FastHerbie

    model = model_cfg["model"]
    product = model_cfg["product"]
    gribdir = model_cfg["gribdir"]

    print(f"\n=== Processing {model.upper()} {product} ===")
    latest_run = round_down_to_00_or_12(datetime.utcnow() - timedelta(hours=8))
    print(latest_run)

    grid_name = gribdir.split("/")[-1]
    new_filename = f"{gribdir}/{grid_name}_{latest_run:%Y%m%d%H}f00.grib2"
    if os.path.exists(new_filename):
        return

    downloader = FastHerbie(
        [latest_run],
        model=model,
        product=product,
        fxx=model_cfg["fxx"],
        save_dir=model_cfg["gribdir"],
        priority=["ecmwf", "aws", "azure"],
        verbose=True,
        max_threads=10,
    )

    if model_cfg["params"] is not None:
        downloader.download(model_cfg["params"], verbose=False)
    else:
        downloader.download(verbose=False)

    for herbie_object in downloader.objects:
        path_to_file = str(herbie_object.get_localFilePath())
        print(f"Processing file: {path_to_file}")
        convert_grib2(path_to_file)
        grib_to_gempak("OUT.grib2")
        old_dir = gribdir + "/" + model + "/"

    if len(downloader.objects) > 0:
        os.system(f"rm -rf {old_dir}")
        os.system("rm OUT.grib2")


def main(argv: list[str] | None = None) -> int:
    _ = argparse.ArgumentParser().parse_args(argv)
    for cfg in MODELS:
        process_model(cfg)
    return 0


__all__ = [
    "MODELS",
    "convert_grib2",
    "extract_forecast_hour",
    "grib_to_gempak",
    "main",
    "process_model",
    "round_down_to_00_or_12",
]
