import os
import gc
import numpy as np
from climada_petals.hazard import TCForecast

def ms_to_knots(ms):
    return int(round(float(ms) * 1.94384)) if ms is not None and not np.isnan(ms) else 0

def format_lat(lat):
    try:
        lat = float(lat)
    except Exception:
        return "   0N"
    if np.isnan(lat):
        return "   0N"
    hemi = 'N' if lat >= 0 else 'S'
    val = int(round(abs(lat) * 10))
    return f"{val:03d}{hemi}"

def format_lon(lon):
    try:
        lon = float(lon)
    except Exception:
        return "   0E"
    if np.isnan(lon):
        return "   0E"
    hemi = 'E' if lon >= 0 else 'W'
    val = int(round(abs(lon) * 10))
    return f"{val:04d}{hemi}"

def get_basin_code2(sid):
    if "L" in sid:
        return "AL"
    elif "W" in sid:
        return "WP"
    elif "C" in sid:
        return "CP"
    elif "E" in sid:
        return "EP"
    elif "P" in sid:
        return "SP"
    elif "A" in sid:
        return "IO"
    elif "B" in sid:
        return "BB"
    elif "U" in sid:
        return "AU"
    elif "S" in sid:
        return "SI"
    elif "X" in sid:
        return "XX"
    return "XX"

def get_tc_type(cat):
    cat = str(cat).lower()
    if 'tropical storm' in cat: return 'TS'
    if 'typhoon' in cat: return 'TY'
    if 'hurricane' in cat: return 'HU'
    if 'tropical depression' in cat: return 'TD'
    if 'extratropical' in cat: return 'EX'
    if 'post' in cat: return 'PT'
    if 'subtropical' in cat: return 'SS'
    return 'XX'

def to_np_datetime(val):
    # Normalize run_datetime to numpy.datetime64 if possible
    if val is None:
        return None
    try:
        return np.datetime64(val)
    except Exception:
        try:
            # If it's already numpy datetime-like
            return np.datetime64(str(val))
        except Exception:
            return None

def main():
    tc_fcast = TCForecast()
    tc_fcast.fetch_ecmwf()

    out_dir = "/data/gempak/storm/enstrack/"
    os.makedirs(out_dir, exist_ok=True)

    # determine a single output filename using the run datetime from the first dataset
    dtg = None
    try:
        if tc_fcast.data:
            first_ds = tc_fcast.data[0]
            first_attrs = getattr(first_ds, "attrs", {}) or {}
            raw_run = first_attrs.get('run_datetime', None)
            run_dt0 = to_np_datetime(raw_run)
            if run_dt0 is None:
                try:
                    run_dt0 = first_ds['time'].isel(time=0).values
                except Exception:
                    run_dt0 = None
            if run_dt0 is not None:
                dtg = str(run_dt0).replace('-', '').replace(':', '').replace('T', '')[:10]
    except Exception:
        dtg = None

    # fallback to current hour if we couldn't determine dtg
    if dtg is None:
        now = np.datetime64('now').astype('datetime64[h]')
        dtg = str(now).replace('-', '').replace(':', '').replace('T', '')[:10]

    out_fname = f"cyclone_{dtg}"
    out_path = os.path.join(out_dir, out_fname)

    # Open the output file once and stream records to it to avoid large memory usage.
    with open(out_path, 'w') as fout:
        # Iterate datasets (ensemble members) and write records directly.
        for ds in tc_fcast.data:
            try:
                attrs = ds.attrs
                sid = str(attrs.get('sid', '01')).strip()
                cyclone_num = ''.join(filter(str.isdigit, sid)) or '01'
                basin = get_basin_code2(sid)
                raw_run = attrs.get('run_datetime', None)
                run_dt = to_np_datetime(raw_run)
                # If run_datetime wasn't provided, use first forecast time as the run reference
                if run_dt is None:
                    run_dt = ds['time'].isel(time=0).values

                # Format dtg as YYYYMMDDHH from run_dt (not used for filename here, filename set above)
                dtg_local = str(run_dt).replace('-', '').replace(':', '').replace('T', '')[:10]

                # TECHNUM + TECH building
                technum = int(attrs.get('ensemble_number', 1)) - 1
                technum_str = str(max(0, technum)).zfill(2)
                tech = "EC00" if technum == 0 else f"EP{technum_str}"

                category = attrs.get('category', 'XX')
                name = attrs.get('name', sid)

                n_times = ds['time'].sizes.get('time', None) or len(ds['time'])

                for i in range(n_times):
                    # Access scalars using isel to avoid creating full arrays
                    time_i = ds['time'].isel(time=i).values
                    # Compute tau in hours between forecast time and run_dt
                    try:
                        delta_hours = int((np.datetime64(time_i) - np.datetime64(run_dt)) / np.timedelta64(1, 'h'))
                    except Exception:
                        # Fallback: use index*6 if time-step info missing
                        delta_hours = i * 6

                    # Extract position and fields safely
                    try:
                        lat_i = ds['lat'].isel(time=i).item()
                    except Exception:
                        lat_i = np.nan
                    try:
                        lon_i = ds['lon'].isel(time=i).item()
                    except Exception:
                        lon_i = np.nan

                    try:
                        wind_i = ds['max_sustained_wind'].isel(time=i).item()
                    except Exception:
                        wind_i = np.nan
                    try:
                        pres_i = ds['central_pressure'].isel(time=i).item()
                    except Exception:
                        pres_i = np.nan

                    atcf_lat = format_lat(lat_i)
                    atcf_lon = format_lon(lon_i)
                    vmax = ms_to_knots(wind_i)
                    mslp = int(round(pres_i)) if not np.isnan(pres_i) else 0
                    tc_type = get_tc_type(category)

                    row = [
                        basin,                      # BASIN
                        cyclone_num.zfill(2),       # CY
                        dtg_local,                  # YYYYMMDDHH
                        f"{max(0, technum):02d}",   # TECHNUM/MIN (string to preserve leading zeros)
                        tech,                       # TECH
                        f"{int(delta_hours):3d}",   # TAU
                        atcf_lat,                   # LatN/S
                        atcf_lon,                   # LonE/W
                        f"{vmax:3d}",               # VMAX (knots)
                        f"{mslp:5d}",               # MSLP (mb)
                        tc_type,                    # TY
                        ' 34',                      # RAD (set to 34kt, minimal)
                        'AAA',                      # WINDCODE (full circle)
                        '  0', '  0', '  0', '  0', # RAD1-4 (not defined)
                        '   0', '   0',             # POUTER, ROUTER (not defined)
                        '  0',                      # RMW (not defined)
                        '  0',                      # GUSTS (not defined)
                        '  0',                      # EYE (not defined)
                        '  0',
                    ]
                    # Write this record immediately
                    fout.write(','.join(str(x) for x in row) + '\n')

            finally:
                # Release resources for this dataset
                close_fn = getattr(ds, 'close', None)
                if callable(close_fn):
                    try:
                        close_fn()
                    except Exception:
                        pass
                try:
                    del ds
                except Exception:
                    pass
                gc.collect()

if __name__ == "__main__":
    main()


