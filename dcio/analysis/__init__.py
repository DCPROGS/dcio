"""Single-channel record analysis."""

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
]
