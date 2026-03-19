"""
Adjust accumulated variables for lagged members.
"""

from datetime import datetime
from typing import List

from discover_grib import ForecastFile
from inventory_variables import VariableInfo

def adjust_accumulated_for_lagged(file: ForecastFile, base_cycle: datetime, variables: List[VariableInfo]) -> None:
    """Adjust accumulated variables for lagged members.
    
    TODO: Implement subtraction of accumulated variables.
    For lagged members, for accumulated fields, subtract the accumulated value
    from the lagged cycle's initial forecast to make it relative to the base cycle.
    """
    # Placeholder
    pass