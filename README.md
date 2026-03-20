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

```bash
python -m datastore.pipelines.member_store \
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
