import os
import subprocess
from herbie.latest import HerbieLatest
from herbie import FastHerbie
import numpy as np
from datetime import datetime, timedelta
import glob
import pygrib
import io
import subprocess

MODELS = [
    # GEMPAK path, model, product (Herbie values), GEMPAK file pattern (no {} for date/fhour)
    {
        "outdir": "/home/gblumberg/data/gempak/model/ecmwf_hr",
        "gribdir": "/home/gblumberg/data/base/model/ecmwf_hr",
        "model": "ifs",
        "product": "oper",
        "gempak_pattern": "YYYYMMDDHHfFFF_ecmwfhr.gem",
        "model_ext": "_ecmwfhr.gem",
        "fxx": range(0,243,3),
        "params": None,
    },
]

def round_down_to_00_or_12(dt):
    dt = dt.replace(minute=0, second=0, microsecond=0)
    if dt.hour < 12:
        return dt.replace(hour=0)
    else:
        return dt.replace(hour=12)

import re

def extract_forecast_hour(filename):
    """
    Extracts the forecast hour from a filename.

    The forecast hour is assumed to be denoted by 'f' followed by two or more digits (e.g., f00, f03, f120).
    This function returns the hour as an integer, or None if not found.

    Args:
        filename (str): The filename to extract from.

    Returns:
        int or None: The forecast hour, or None if not found.
    """
    match = re.search(r'f(\d{2,})', filename)
    if match:
        return int(match.group(1))
    return None

def grib_to_gempak(fn):
    # 1. Read existing messages
    with pygrib.open(fn) as grbs:
        # Get all grib messages and make them a string we'll pass to dcgrib2
        memfile = io.BytesIO()
        for msg in grbs:
            # Correct the MEAN SEA LEVEL PRESSURE GRIB2 MESSAGE
            if msg.level == "mean sea level":
                msg.parameterNumber = 198
            memfile.write(msg.tostring())
        grib2_bytes = memfile.getvalue()

    # 5. Pipe to dcgrib2
    proc = subprocess.Popen(
        ['/home/gblumberg/GEMPAK7/os/linux64/bin/dcgrib2', '-v', '1', '-e', 'GEMTBL=/home/gblumberg/gemtbls/', '/home/gblumberg/data/gempak/model/ecmwfhr/ecmwfhr_YYYYMMDDHHfFFF.gem'],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    stdout, stderr = proc.communicate(input=grib2_bytes)

    print("dcgrib2 stdout:", stdout.decode())
    if stderr:
        print("dcgrib2 stderr:", stderr.decode())

def convert_grib2(fn):
    cmd = ['wgrib2', fn, '-set_grib_type', 'c3', '-grib_out', 'OUT.grib2']
    cmd = ['grib_set', '-r', '-w', 'packingType=grid_ccsds', '-s', 'packingType=grid_simple', fn, 'OUT.grib2']
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print("Error converting GRIB2 file:", result.stderr)
    else:
        print("GRIB2 file converted successfully.")

def process_model(model_cfg):
    model = model_cfg["model"]
    product = model_cfg["product"]
    gribdir = model_cfg["gribdir"]

    print(f"\n=== Processing {model.upper()} {product} ===")
    #H = HerbieLatest(model=model, product=product)
    dt_utc = datetime.utcnow() - timedelta(hours=8)

    latest_run = round_down_to_00_or_12(dt_utc)
    print(latest_run)
    #stop
    grid_name = gribdir.split('/')[-1]
    new_fn = f"{gribdir}/{grid_name}_{latest_run:%Y%m%d%H}f00.grib2"
    if os.path.exists(new_fn):
        return
    FH = FastHerbie([latest_run], model=model, product=product, fxx=model_cfg['fxx'], \ 
            save_dir=model_cfg['gribdir'], priority=['ecmwf', 'aws', 'azure'], \ 
            verbose=True, max_threads=10)
    #print(f"Latest run: {H.valid_date:%Y%m%d %HZ}, F:{fhr}")

    if model_cfg['params'] is not None:
        path = FH.download(model_cfg['params'], verbose=False)
    else:
        path = FH.download(verbose=False)

    for H in FH.objects:
        path2file = str(H.get_localFilePath())
        filename = H.get_localFilePath().name
        print(f"Processing file: {path2file}")
        convert_grib2(path2file)
        grib_to_gempak('OUT.grib2')
        
        old_dir = gribdir + '/' + model + '/'

    if len(FH.objects) > 0:
        os.system(f"rm -rf {old_dir}")
        os.system('rm OUT.grib2 ')

if __name__ == "__main__":
    for cfg in MODELS:
        process_model(cfg)
