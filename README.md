# Weather Datastore

This repository is an installable Python package and a script-first workflow toolkit for weather model ingest and ensemble postprocessing.

## Is This a Python Package You Need to Install?

Short answer: yes, this is a Python package.

Package root:
- [pyproject.toml](pyproject.toml)
- [src/datastore](src/datastore)

You can use it in two modes:

1. Script-first mode (no pip install required)
- Use wrappers in [utils](utils) and [scripts](scripts).
- These wrappers insert [src](src) into Python path at runtime.

2. Package mode (recommended for external projects)
- Install in editable mode from repo root:

```bash
python -m pip install -e .
```

After that, import directly:

```python
from datastore.pipelines.member_store import build_member_store
```

## Your Three Requirements

### 1) Keep Download Scripts External

That is fully supported. The datastore package does not require you to use in-repo downloaders.

Contract for ingestion step:
- Input is local GRIB2 files under one or more directories.
- Member and cycle metadata are inferred from file naming patterns.

Use your existing external downloader to stage files, then run member conversion and postprocessing from this repo.

Relevant modules:
- [src/datastore/pipelines/member_store.py](src/datastore/pipelines/member_store.py)
- [src/datastore/pipelines/main_build_ensemble.py](src/datastore/pipelines/main_build_ensemble.py)

### 2) Convert Each HREF Member to Its Own Zarr Store

This is exactly what member store pipeline does.

Operational pattern:

1. Build variable inventory once and list discovered members:

```bash
python utils/main_build_ensemble.py \
  --input-root /path/to/href/grib2 \
  --patterns "*.grib2" \
  --variables-db ./variables.json
```

2. Build one member Zarr per member:

Installed/editable package mode:

```bash
python -m datastore.pipelines.member_store \
  --input-root /path/to/href/grib2 \
  --patterns "*.grib2" \
  --variables-db ./variables.json \
  --member m01 \
  --output-dir /path/to/member_zarr \
  --max-lags 4
```

Script-first mode without installation:

```bash
PYTHONPATH=src python -m datastore.pipelines.member_store \
  --input-root /path/to/href/grib2 \
  --patterns "*.grib2" \
  --variables-db ./variables.json \
  --member m01 \
  --output-dir /path/to/member_zarr \
  --max-lags 4
```

Repeat step 2 for each member.

Output pattern:
- /path/to/member_zarr/m01.zarr
- /path/to/member_zarr/m02.zarr
- ...

### 3) Streamlined Variable Postprocessing

You asked for:
- max hourly UH over 4 hours,
- mean 2m temperature,
- mean 2m dewpoint,
- neighborhood probability of UH exceedance over the same 4 hours.

A dedicated helper script now exists for exactly this workflow:
- [scripts/postprocess_href_products.py](scripts/postprocess_href_products.py)

Run example:

```bash
python scripts/postprocess_href_products.py \
  --member-zarrs /data/members/m01.zarr /data/members/m02.zarr /data/members/m03.zarr \
  --output-zarr /data/diagnostics/href/derived_products.zarr \
  --uh-var uh \
  --temp-var tmp2m \
  --dewpoint-var dpt2m \
  --uh-threshold 75 \
  --time-window-steps 4 \
  --radius-x 10 \
  --radius-y 10 \
  --system href \
  --run-id 2026031900 \
  --manifest-path /data/manifests/href/2026031900/manifest.json \
  --validate-contract
```

Notes:
- Variable names depend on your inventory and GRIB short names. Validate with your generated variables database.
- If your model uses different names, override --uh-var, --temp-var, --dewpoint-var.

## What the Helper Produces

From input member stores, it writes one diagnostics Zarr with:

- <uh_var>_4h_member_max
- <temp_var>_mean
- <dewpoint_var>_mean
- <uh_var>_gt_<threshold>_nprob_4h

Implementation references:
- [scripts/postprocess_href_products.py](scripts/postprocess_href_products.py)
- [src/datastore/diagnostics/ensemble.py](src/datastore/diagnostics/ensemble.py)
- [src/datastore/pipelines/postprocess_ensemble.py](src/datastore/pipelines/postprocess_ensemble.py)

## End-to-End Flow Summary

1. External downloader stages GRIB2 files.
2. Member conversion writes one Zarr per member.
3. Postprocess helper derives operational products for UH, temperature, and dewpoint.
4. Optional manifest and contract checks record and validate outputs.

For profile-based orchestration, see:
- [src/datastore/workflows/orchestrator.py](src/datastore/workflows/orchestrator.py)
- [config/profiles/hpc_operational_example.json](config/profiles/hpc_operational_example.json)

## Combining Lagged and Non-Lagged Member Stores

If you have directories like:

- `20260320.00.href_members/`
- `20260319.12.href_members/`

and each contains per-member stores such as:

- `wrf4nssl.zarr`
- `hiresw_arw.zarr`
- `hiresw_fv3.zarr`
- `namnest.zarr`
- `hrrr.zarr`

then a plain xarray multi-file open is not enough to construct a lagged ensemble correctly.

Why:

1. `xarray.open_mfdataset(..., combine="by_coords")` combines by matching coordinates, not by your meteorological concept of “lagged member”.
2. Stores from different cycles usually have overlapping valid times, so xarray will not reliably infer that they should become extra members.
3. If both cycles use the same member label, such as `wrf4nssl`, then even explicit concatenation will create duplicate member coordinate values unless you rename them or add lag metadata.

Recommended approach:

1. Open each `.zarr` store individually with `xr.open_zarr(...)`.
2. Parse the source cycle from the parent directory name.
3. Assign a unique member coordinate per store, for example `wrf4nssl_lag00` and `wrf4nssl_lag12`.
4. Add `source_cycle` or `lag_hours` as coordinates on the member dimension.
5. Concatenate explicitly with `xr.concat(datasets, dim="member")`.

This repository uses that explicit pattern in [src/datastore/pipelines/postprocess_ensemble.py](src/datastore/pipelines/postprocess_ensemble.py#L13), where member stores are opened one-by-one and then concatenated across the member dimension.

A package helper now exists for this exact task:
- [src/datastore/pipelines/lagged_members.py](src/datastore/pipelines/lagged_members.py)

Lazy loading example:

```python
from datastore.pipelines import load_lagged_member_ensemble

href = load_lagged_member_ensemble(
  [
    "/data/20260320.00.href_members",
    "/data/20260319.12.href_members",
  ],
  variables=["uh", "t2m", "td2m"],
  chunks="auto",
  join="inner",
)
```

What the helper does:

1. Opens each store lazily with `xr.open_zarr(..., chunks="auto")`.
2. Optionally filters to just the requested data variables before concatenation.
3. Infers the source cycle from the parent directory name.
4. Renames duplicate members into unique lagged members such as `wrf4nssl_lag00` and `wrf4nssl_lag12`.
5. Adds member-dimension coordinates:
   - `base_member`
   - `source_cycle`
   - `lag_hours`
   - `source_store`
6. Concatenates only along `member`, leaving the data lazy/dask-backed.

If you pass `variables=[...]`, the helper keeps only those data variables and preserves the needed coordinates.

Example pattern:

```python
from pathlib import Path
import pandas as pd
import xarray as xr

paths = sorted(Path("/data").glob("*.href_members/*.zarr"))
latest_cycle = pd.Timestamp("2026-03-20T00:00:00Z")

datasets = []
for path in paths:
  cycle_token = path.parent.name.replace(".href_members", "")
  cycle = pd.to_datetime(cycle_token, format="%Y%m%d.%H", utc=True)
  lag_hours = int((latest_cycle - cycle).total_seconds() / 3600)

  ds = xr.open_zarr(path, consolidated=True)
  ds = ds.assign_coords(
    member=[f"{path.stem}_lag{lag_hours:02d}"],
    source_cycle=("member", [cycle_token]),
    lag_hours=("member", [lag_hours]),
  )
  datasets.append(ds)

href = xr.concat(datasets, dim="member", join="inner")
```

Notes:

- In this repo, member-store `time` coordinates represent valid times, so `join="inner"` is usually the right choice when you want only the common overlap window across lagged cycles.
- Use `join="outer"` only if you want the union of all valid times and are willing to carry `NaN` values where a lagged member does not exist.
