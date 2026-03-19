"""
Inventory variables and save to a central database.
"""

import json
from pathlib import Path
from typing import List

import grib2io

from discover_grib import discover_files
from inventory_variables import VariableInfo, inventory_variables

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-root', nargs='+', required=True, help='Input root directories')
    parser.add_argument('--patterns', nargs='*', default=['*.grib2'], help='File patterns')
    parser.add_argument('--exclude-dirs', nargs='*', default=[], help='Directories to exclude')
    parser.add_argument('--output-variables', required=True, help='Output JSON file for variables')
    args = parser.parse_args()

    # Discover sample files
    files = discover_files(args.input_root, args.patterns, args.exclude_dirs)
    sample_files = files[:30]  # More samples for better inventory

    # Inventory variables
    variables = inventory_variables(sample_files)

    # Save to JSON
    var_dicts = [
        {
            'name': v.name,
            'long_name': v.long_name,
            'units': v.units,
            'level1': v.level1,
            'level2': v.level2,
            'level_type': v.level_type,
            'type': v.type
        }
        for v in variables
    ]
    with open(args.output_variables, 'w') as f:
        json.dump(var_dicts, f, indent=2)

    print(f"Inventory complete. {len(variables)} variables saved to {args.output_variables}")

if __name__ == '__main__':
    main()
