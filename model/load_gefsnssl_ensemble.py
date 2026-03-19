"""
Load the latest GEFS post-processed ensemble forecasts from the NSSL THREDDS server
and merge into a 4D (member, time, lat, lon) xarray Dataset.
"""

import datetime
import warnings
import requests
import xarray as xr
import numpy as np

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
BASE_URL = (
    "https://data.nssl.noaa.gov/thredds/dodsC/FRDD/GEFS-v12/"
    "GEFS_v12_forecasts/GEFS_op"
)
CATALOG_BASE_URL = (
    "https://data.nssl.noaa.gov/thredds/catalog/FRDD/GEFS-v12/"
    "GEFS_v12_forecasts/GEFS_op"
)

# GEFS members: c00 = control, p01–p30 = perturbation members
CONTROL_MEMBER = ["c00"]
PERTURBATION_MEMBERS = [f"p{i:02d}" for i in range(1, 31)]
ALL_MEMBERS = CONTROL_MEMBER + PERTURBATION_MEMBERS

# Run cycles to check, most recent first
RUN_HOURS = ["12", "00"]


# ---------------------------------------------------------------------------
# Helper: find the latest available run
# ---------------------------------------------------------------------------
def get_latest_run(
    lookback_days: int = 2,
) -> tuple[str, str]:
    """
    Walk backwards through recent dates/cycles and return the first
    (date_str, cycle_str) for which a control-member file exists on the server.

    Returns
    -------
    date_str : str   e.g. "20260225"
    cycle_str : str  e.g. "00"
    """
    today = datetime.datetime.utcnow().date()
    for delta in range(lookback_days + 1):
        check_date = today - datetime.timedelta(days=delta)
        date_str = check_date.strftime("%Y%m%d")
        for cycle in RUN_HOURS:
            run_id = f"{date_str}_{cycle}Z"
            # Use the THREDDS catalog endpoint to test existence (fast HEAD check)
            catalog_url = f"{CATALOG_BASE_URL}/{run_id}/catalog.html"
            try:
                resp = requests.head(catalog_url, timeout=10)
                if resp.status_code == 200:
                    print(f"Latest available run: {run_id}")
                    return date_str, cycle
            except requests.RequestException:
                continue
    raise RuntimeError(
        f"Could not find a valid GEFS run in the last {lookback_days} days."
    )


# ---------------------------------------------------------------------------
# Helper: build OPeNDAP URL for a single member
# ---------------------------------------------------------------------------
def build_member_url(date_str: str, cycle: str, member: str) -> str:
    """
    Build the OPeNDAP URL for one GEFS member.

    Parameters
    ----------
    date_str : str   e.g. "20260225"
    cycle    : str   e.g. "00"
    member   : str   e.g. "p01" or "c00"
    """
    run_id = f"{date_str}_{cycle}Z"
    init_str = f"{date_str}{cycle}"  # e.g. "2026022500"
    filename = f"convective_parms_{init_str}_{member}_f000-f384.nc"
    return f"{BASE_URL}/{run_id}/{member}/{filename}"


# ---------------------------------------------------------------------------
# Helper: open a single member dataset (lazy)
# ---------------------------------------------------------------------------
def open_member(url: str, member: str) -> xr.Dataset | None:
    """
    Open one member's OPeNDAP dataset lazily.
    Returns None if the file is unreachable.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # suppress PyDAP protocol warnings
        try:
            ds = xr.open_dataset(url, engine="pydap", chunks={})
            # Tag with member coordinate so we can concat later
            ds = ds.expand_dims(dim="member").assign_coords(member=[member])
            return ds
        except Exception as exc:
            print(f"  WARNING: could not open {member} – {exc}")
            return None


# ---------------------------------------------------------------------------
# Main: load all members and concatenate
# ---------------------------------------------------------------------------
def load_gefs_ensemble(
    date_str: str | None = None,
    cycle: str | None = None,
    members: list[str] | None = None,
) -> xr.Dataset:
    """
    Load GEFS ensemble forecasts into a 4-D Dataset with dimensions
    (member, time, lat, lon).

    Parameters
    ----------
    date_str : str or None
        Init date as "YYYYMMDD". If None, the latest available run is used.
    cycle : str or None
        Cycle hour as "00" or "12". If None, the latest available run is used.
    members : list[str] or None
        List of member IDs to load, e.g. ["c00", "p01", "p02"].
        Defaults to all 31 members (c00 + p01–p30).

    Returns
    -------
    xr.Dataset  with dims (member, time, lat, lon)
    """
    if date_str is None or cycle is None:
        date_str, cycle = get_latest_run()

    if members is None:
        members = ALL_MEMBERS

    datasets = []
    print(f"Opening {len(members)} member(s) for {date_str}_{cycle}Z ...")
    for member in members:
        url = build_member_url(date_str, cycle, member)
        print(f"  {member}: {url}")
        ds = open_member(url, member)
        if ds is not None:
            datasets.append(ds)

    if not datasets:
        raise RuntimeError("No member datasets could be opened.")

    print(f"\nConcatenating {len(datasets)} member(s) along 'member' dimension ...")
    ensemble = xr.concat(datasets, dim="member")

    print("Done!  Ensemble dataset:")
    print(ensemble)
    return ensemble


# ---------------------------------------------------------------------------
# Example usage
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Auto-detect the latest run:
    ens = load_gefs_ensemble()

    # --- Or pin a specific run: ---
    # ens = load_gefs_ensemble(date_str="20260225", cycle="00")

    # --- Or load only a subset of members: ---
    # ens = load_gefs_ensemble(members=["c00", "p01", "p02", "p03"])

    # Example ensemble statistics (computed lazily until .compute() is called)
    ens_mean   = ens.mean(dim="member")
    ens_std    = ens.std(dim="member")
    #ens_median = ens.median(dim="member")

    # Probability that, e.g., 2-m temperature exceeds 300 K
    prob_t2m_gt_300 = (ens["t2m"] > 300).mean(dim="member") * 100  # percent

    print("\nEnsemble mean t2m (first time step, degrees K):")
    print(ens_mean["t2m"].isel(time=0).values)
