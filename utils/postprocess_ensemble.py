"""
Post-process ensemble: load member Zarrs and compute statistics.
"""

import json
from pathlib import Path
from typing import List

import xarray as xr
import numpy as np

from inventory_variables import VariableInfo

def load_variables_db(variables_file: Path) -> List[VariableInfo]:
    """Load variables from JSON database."""
    with open(variables_file) as f:
        var_dicts = json.load(f)
    return [
        VariableInfo(
            name=d['name'],
            long_name=d['long_name'],
            units=d['units'],
            level=d['level'],
            type=d['type']
        )
        for d in var_dicts
    ]

def load_ensemble(member_zarrs: List[Path], variables: List[VariableInfo]) -> xr.Dataset:
    """Load and concatenate member Zarrs into ensemble Dataset."""
    datasets = []
    for zarr_path in member_zarrs:
        ds = xr.open_zarr(str(zarr_path))
        # Add member coordinate
        member_name = zarr_path.stem  # Assume directory name is member
        ds = ds.expand_dims(member=[member_name])
        datasets.append(ds)
    
    # Concatenate along member dimension
    ensemble = xr.concat(datasets, dim='member')
    return ensemble

def compute_ensemble_stats(ensemble: xr.Dataset) -> xr.Dataset:
    """Compute ensemble statistics like mean, spread, etc."""
    stats = {}
    
    for var_name in ensemble.data_vars:
        var_data = ensemble[var_name]
        
        # Mean
        stats[f"{var_name}_mean"] = var_data.mean(dim='member')
        
        # Spread (standard deviation)
        stats[f"{var_name}_spread"] = var_data.std(dim='member')
        
        # Other stats can be added here
        # e.g., probability of exceeding threshold, etc.
    
    return xr.Dataset(stats, coords=ensemble.coords)

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--member-zarrs', nargs='+', required=True, help='Member Zarr directories')
    parser.add_argument('--variables-db', required=True, help='Variables database JSON file')
    parser.add_argument('--output-stats', required=True, help='Output Zarr for ensemble stats')
    args = parser.parse_args()

    import logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

    # Load variables
    variables = load_variables_db(Path(args.variables_db))
    logger.info(f"Loaded {len(variables)} variables")

    # Load ensemble
    member_zarrs = [Path(p) for p in args.member_zarrs]
    ensemble = load_ensemble(member_zarrs, variables)
    logger.info(f"Loaded ensemble with {len(ensemble.member)} members, {len(ensemble.time)} times")

    # Compute stats
    stats = compute_ensemble_stats(ensemble)
    logger.info("Computed ensemble statistics")

    # Save stats
    stats.to_zarr(args.output_stats, mode='w', consolidated=True)
    logger.info(f"Saved ensemble stats to {args.output_stats}")

if __name__ == '__main__':
    main()