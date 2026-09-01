"""Single-channel record analysis."""

from dcio.analysis.bursts import (
    bursts_from_record,
    extract_burst_intervals,
    extract_bursts,
)
from dcio.analysis.record import (
    Periods,
    SingleChannelRecord,
    from_scn,
    impose_resolution,
    set_periods,
)

__all__ = [
    "Periods",
    "SingleChannelRecord",
    "from_scn",
    "impose_resolution",
    "set_periods",
    "extract_bursts",
    "extract_burst_intervals",
    "bursts_from_record",
]
