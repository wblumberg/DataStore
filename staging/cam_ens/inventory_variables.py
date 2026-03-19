"""
Inventory variables from GRIB2 files.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List

import grib2io

@dataclass
class VariableInfo:
    name: str
    long_name: str
    units: str
    level_type: str
    level1: str
    level2: str
    type: str

def get_variable_type(msg) -> str:
    """Determine the variable type from GRIB2 message."""
    #print(msg.statisticalProcess)
    if hasattr(msg, 'statisticalProcess') and msg.statisticalProcess is not None:
        # Statistical product
        proc_type = msg.statisticalProcess
        time_range = getattr(msg, 'timeRangeOfStatisticalProcess', '')
        unit_of_time_range = getattr(msg, 'unitOfTimeRangeOfStatisticalProcess', '')
        unit_of_time_range = unit_of_time_range.split('-')[1] if '-' in unit_of_time_range else unit_of_time_range
        
        if time_range and unit_of_time_range == "1":
            time_range += "h"
        
        # print(proc_type, time_range, unit_of_time_range)
        if "Average" in proc_type:
            return f'average_{time_range}h' if time_range else 'average'
        elif "Accumulation" in proc_type:
            return f'accum_{time_range}h' if time_range else 'accum'
        elif "Maximum" in proc_type:
            return f'max_{time_range}h' if time_range else 'max'
        elif "Minimum" in proc_type:
            return f'min_{time_range}h' if time_range else 'min'
        else:
            return f'stat_{proc_type}_{time_range}' if time_range else f'stat_{proc_type}'
    else:
        # Non-statistical
        if hasattr(msg, 'stepRange') and msg.stepRange:
            return f'accum_{msg.stepRange}'
        else:
            return 'instant'

def inventory_variables(files: List[Path]) -> List[VariableInfo]:
    """Inventory all variables from the GRIB2 files."""
    variables_dict = {}
    
    for path in files[:5]:  # Sample first few files
        try:
            with grib2io.open(str(path)) as f:
                for msg in f:
                    var_name = msg.shortName
                    if var_name not in variables_dict:
                        var_type = get_variable_type(msg)
                        if msg.typeOfFirstFixedSurface == 103:
                            LEVEL_TYPE = 'HGHT'
                        elif msg.typeOfFirstFixedSurface == 100:
                            LEVEL_TYPE = 'PRES'
                        elif msg.typeOfFirstFixedSurface == 20:
                            LEVEL_TYPE = 'TEMP'
                        else:
                            LEVEL_TYPE = None
                            
                        if msg.typeOfFirstFixedSurface == 255:
                            level1 = ''
                        else:
                            level1 = str(int(msg.valueOfFirstFixedSurface)) if hasattr(msg, 'valueOfFirstFixedSurface') else ''
                        if msg.typeOfSecondFixedSurface == 255:
                            level2 = ''
                        else:
                            level2 = str(int(msg.valueOfSecondFixedSurface)) if hasattr(msg, 'valueOfSecondFixedSurface') else ''
                            
                        variables_dict[var_name] = VariableInfo(
                            name=var_name,
                            long_name=getattr(msg, 'fullName', var_name),
                            units=getattr(msg, 'units', ''),
                            level_type = LEVEL_TYPE,
                            level1=level1,
                            level2=level2,
                            type=var_type
                        )
        except Exception as e:
            print(f"Error reading {path}: {e}")
    
    return list(variables_dict.values())