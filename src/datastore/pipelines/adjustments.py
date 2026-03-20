"""Adjustment hooks for lagged-member workflows."""

from __future__ import annotations

from datetime import datetime
from typing import Sequence

from ..core.models import ForecastFile, VariableInfo


def adjust_accumulated_for_lagged(file: ForecastFile, base_cycle: datetime, variables: Sequence[VariableInfo]) -> None:
    """Placeholder for accumulated-field lag adjustment logic."""
    _ = (file, base_cycle, variables)
