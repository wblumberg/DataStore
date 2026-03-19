#!/home/gblumberg/miniforge3/envs/gemgen/bin/python
import os
import sys
import subprocess
import numpy as np
from datetime import datetime, timedelta
import glob
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests

MODELS = [
    {
        "outdir": "/home/gblumberg/data/gempak/model/hiresw",
        "gribdir": "/home/gblumberg/data/base/model/hiresw_conusfv3",
        "model": "hiresw",
        "product": "fv3_2p5km",
        "nssl_link": "https://data.nssl.noaa.gov/thredds/fileServer/FRDD/HREF/{date:%Y}/{date:%Y%m%d}/hiresw_conusfv3_{date:%Y%m%d%H}f{fxx:03d}.grib2",
        "gempak_pattern": "YYYYMMDDHHfFFF_hiresw_conusfv3.gem",
        "model_ext": "_hiresw_conusfv3.gem",
        "fxx": range(0, 61, 1),
        "params": None,
        "ready_time": (datetime(1900, 1, 1, 2, 50, 0), datetime(1900, 1, 1, 14, 50, 0)),
    },
    {
        "outdir": "/home/gblumberg/data/gempak/model/hiresw",
        "gribdir": "/home/gblumberg/data/base/model/hiresw_conusarw",
        "model": "hiresw",
        "product": "arw_2p5km",
        "nssl_link": "https://data.nssl.noaa.gov/thredds/fileServer/FRDD/HREF/{date:%Y}/{date:%Y%m%d}/hiresw_conusarw_{date:%Y%m%d%H}f{fxx:03d}.grib2",
        "gempak_pattern": "YYYYMMDDHHfFFF_hiresw_conusarw.gem",
        "model_ext": "_hires_conusarw.gem",
        "fxx": range(0, 49, 1),
        "params": None,
        "ready_time": (datetime(1900, 1, 1, 3, 15, 0), datetime(1900, 1, 1, 15, 15, 0)),
    },
    {
        "outdir": "/home/gblumberg/data/gempak/model/namnest",
        "gribdir": "/home/gblumberg/data/base/model/namnest",
        "model": "nam",
        "product": "conusnest.hiresf",
        "nssl_link": "https://data.nssl.noaa.gov/thredds/fileServer/FRDD/HREF/{date:%Y}/{date:%Y%m%d}/nam_conusnest_{date:%Y%m%d%H}f{fxx:03d}.grib2",
        "gempak_pattern": "YYYYMMDDHHfFFF_namnest.gem",
        "model_ext": "_namnest.gem",
        "fxx": range(0, 61, 1),
        "params": None,
        "ready_time": (datetime(1900, 1, 1, 2, 40, 0), datetime(1900, 1, 1, 14, 40, 0)),
    },
    {
        "outdir": "/home/gblumberg/data/gempak/model/wrf4nssl",
        "gribdir": "/home/gblumberg/data/base/model/wrf4nssl",
        "model": "wrf4nssl",
        "product": None,
        "nssl_link": "https://data.nssl.noaa.gov/thredds/fileServer/FRDD/HREF/{date:%Y}/{date:%Y%m%d}/hiresw_conusnssl_{date:%Y%m%d%H}f{fxx:03d}.grib2",
        "gempak_pattern": "YYYYMMDDHHfFFF_wrf4nssl.gem",
        "model_ext": "_nsslwrf.gem",
        "fxx": range(0, 49, 1),
        "params": None,
    },
    {
        "outdir": "/home/gblumberg/data/gempak/model/hrrr",
        "gribdir": "/home/gblumberg/data/base/model/hrrr",
        "model": "hrrr",
        "product": "sfc",
        "nssl_link": "https://data.nssl.noaa.gov/thredds/fileServer/FRDD/HREF/{date:%Y}/{date:%Y%m%d}/hrrr_ncep_{date:%Y%m%d%H}f{fxx:03d}.grib2",
        "gempak_pattern": "YYYYMMDDHHfFFF_hrrr.gem",
        "model_ext": "_hrrr.gem",
        "fxx": range(0, 49, 1),
        "params": None,
        "ready_time": (datetime(1900, 1, 1, 1, 50, 0), datetime(1900, 1, 1, 13, 50, 0)),
    },
]


def round_down_to_00_or_12(dt):
    dt = dt.replace(minute=0, second=0, microsecond=0)
    #dt = datetime(2025,7,1,0,0,0) - timedelta(seconds=3600*12)
    #dt = datetime(2025,7,2,0,0,0)
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

def build_urls(cfg, date=None):
    if date is None:
        date = datetime.utcnow()
    urls = []
    for hour in cfg["hours"]:
        run_date = date.replace(hour=hour, minute=0, second=0, microsecond=0)
        for fxx in cfg["fxx"]:
            url = cfg["url_template"].format(date=run_date, hour=hour, fxx=fxx)
            urls.append(url)
    return urls

def download_file(url, cfg):
    filename = os.path.basename(url)
    
    parts = filename.split("_")[-1]
    out_dir = cfg['gribdir']
    fn_ext = out_dir.split('/')[-1]
    filename = f"{fn_ext}_{parts}"
    filename = filename.replace('f0', 'f')

    #print(out_dir, filename, url)
    dest_path = os.path.join(out_dir, filename)
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        with open(dest_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        print(f"Downloaded: {dest_path}")
        return dest_path
    except Exception as exc:
        print(f"Failed: {url} ({exc})")
        return None

def build_urls(cfg, date=None):
    if date is None:
        date = datetime.utcnow()
        run_date = round_down_to_00_or_12(date)
    urls = []
    #print("CFG:", cfg)
    for fxx in cfg["fxx"]:
        url = cfg["nssl_link"].format(date=run_date, fxx=fxx)
        # A fix because on Jan 20, 2026 I found that the NSSL Thredds directory was accidentally putting
        # 2026 HREF runs into the 2025 folder.
        #print("Replacing the HREF/2026/ with HREF/2025/ due to NSSL Thredds bug.")
        #url = url.replace('HREF/2026/', 'HREF/2025/')
        # On March 18, 2026, I noticed this wasn't an issue anymore.
        urls.append(url)
    return urls

def process_model(model_cfg, max_workers=4):
    model = model_cfg["model"]
    gribdir = model_cfg["gribdir"]
    nssl_path = model_cfg['nssl_link']

    #save_dir = os.path.join(base_grib_dir, f"{model}_{product}")
    #os.makedirs(save_dir, exist_ok=True)
    #os.makedirs(outdir, exist_ok=True)

    print(f"\n=== Downloading {model.upper()} ===")
    urls = build_urls(model_cfg) 
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(download_file, url, model_cfg) for url in urls]
        for future in as_completed(futures):
            _ = future.result()

if __name__ == "__main__":
    model = sys.argv[1]
    for cfg in MODELS:
        if model in cfg['gempak_pattern']:
            process_model(cfg)
